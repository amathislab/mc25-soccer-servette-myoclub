# Example usage: uv run router.py --ball-pos 0.5 0.0 1.2 --ball-vel 2.0 -1.5 0.3

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple


class RouterMLP(nn.Module):

    def __init__(self, input_dim=6, hidden_dims=[256, 256, 128], num_classes=135):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExpertRouter:
    """
    Expert router for ball state -> expert zone mapping.

    Attributes:
        model: Neural network for classification
        scaler: StandardScaler for input normalization
        label_encoder: LabelEncoder for expert ID decoding
    """

    def __init__(self, model, scaler, label_encoder, device="cpu"):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.device = device

        self.model.eval()
        self.model.to(device)

    @classmethod
    def load(cls, router_path: str, device: str = "cpu") -> "ExpertRouter":
        """
        Load trained router from pickle file.

        Args:
            router_path: Path to router .pkl file
            device: Device for inference ("cpu" or "cuda")

        Returns:
            ExpertRouter instance
        """
        router_path = Path(router_path)

        if not router_path.exists():
            raise FileNotFoundError(f"Router not found: {router_path}")

        # Load router package
        with open(router_path, "rb") as f:
            package = pickle.load(f)

        # Reconstruct model from saved config
        config = package["model_config"]
        model = RouterMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
        )
        model.load_state_dict(package["model_state_dict"])

        return cls(
            model=model,
            scaler=package["scaler"],
            label_encoder=package["label_encoder"],
            device=device,
        )

    def predict(
        self,
        ball_state: Union[list, np.ndarray],
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict expert zone for given ball state.

        Args:
            ball_state: Ball state [x, y, z, vx, vy, vz] (6D array or list)
            return_confidence: Whether to return prediction confidence

        Returns:
            expert_name: Expert zone name (e.g., "1_2_far_low")
            confidence: Prediction confidence (only if return_confidence=True)
        """
        ball_state = np.asarray(ball_state, dtype=np.float32)

        # Ensure 2D shape (1, 6)
        if ball_state.ndim == 1:
            ball_state = ball_state.reshape(1, -1)

        # Normalize input
        ball_state_normalized = self.scaler.transform(ball_state)

        # Convert to tensor
        x = torch.FloatTensor(ball_state_normalized).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        # Decode prediction
        pred_idx = pred_idx.cpu().item()
        confidence = confidence.cpu().item()
        expert_name = self.label_encoder.classes_[pred_idx]

        if return_confidence:
            return expert_name, confidence
        return expert_name


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query expert router")
    parser.add_argument(
        "--router",
        default="../router/router_135way.pkl",
        help="Path to router pickle file"
    )
    parser.add_argument(
        "--ball-pos",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Ball position"
    )
    parser.add_argument(
        "--ball-vel",
        type=float,
        nargs=3,
        required=True,
        metavar=("VX", "VY", "VZ"),
        help="Ball velocity"
    )
    args = parser.parse_args()

    # Load router
    print(f"Loading router from {args.router}...")
    router = ExpertRouter.load(args.router)

    # Prepare ball state
    ball_state = args.ball_pos + args.ball_vel

    # Predict
    expert_name, confidence = router.predict(ball_state, return_confidence=True)

    # Print result
    print(f"\nBall position: {args.ball_pos}")
    print(f"Ball velocity: {args.ball_vel}")
    print(f" Expert: {expert_name}")
    print(f" Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()