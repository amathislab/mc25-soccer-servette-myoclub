import os
import pickle
import time

import copy
import numpy as np
import torch
import torch.nn as nn

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)


class PolicyNetwork(nn.Module):
    """Neural network for policy distillation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: list = [2048, 1024, 512, 256]):
        super().__init__()

        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        return self.network(obs)


class EnvShell:

    def __init__(self, stub):

        action_len = unpack_for_grpc(
            stub.get_action_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )

        obs_len = unpack_for_grpc(
            stub.get_observation_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )
        self.observation_space = gym.spaces.Box(shape=(obs_len,), high=1e6, low=-1e6)
        self.action_space = gym.spaces.Box(shape=(action_len,), high=10.0, low=-10.0)
        print("Action Space", self.action_space)
        print("Observation Space", self.observation_space)
        # TODO case for remapping of [-1 1] -> [0 1]


class Policy:

    def __init__(self, env):
        self.action_space = env.action_space

        self.device = torch.device("cpu")  # EvalAI uses CPU
        self.model = PolicyNetwork(
            obs_dim=1292,
            action_dim=290,
            hidden_layers=[2048, 1024, 1024, 512, 512, 256]
        ).to(self.device)

        # Load model weights - try multiple possible paths
        possible_paths = [
            "best_model.pth",                    # Docker container (./agent copied to /)
            "agent/best_model.pth",              # Local development
            os.path.join(os.path.dirname(__file__), "best_model.pth"),  # Same directory as script
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded trained model from {model_path}")
            self.use_trained_model = True
        else:
            print(f"Model file not found. Tried: {possible_paths}")
            raise FileNotFoundError(f"Model file not found. Tried: {possible_paths}")

    def __call__(self, obs):
        if not self.use_trained_model:
            return self.action_space.sample()

        # convert to tensor
        if isinstance(obs, tuple):
            obs = obs[0]  # Take observation from (obs, info) tuple
        obs = np.asarray(obs, dtype=np.float32).flatten()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get action from model, action before sigmoid to get activation in [0, 1]
        with torch.no_grad():
            action = self.model(obs_tensor).squeeze(0).cpu().numpy()
            
        return action

custom_obs_keys = [      
    'internal_qpos',
    'internal_qvel',
    'grf',
    'torso_angle',
    'pelvis_angle',
    'model_root_pos',
    'model_root_vel',
    'muscle_length',
    'muscle_velocity',
    'muscle_force',
    'ball_pos',
    'goal_bounds',
]


time.sleep(60) # DO NOT REMOVE. Required for EvalAI processing

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    channel = grpc.insecure_channel("environment:8086")
else:
    channel = grpc.insecure_channel("localhost:8086")

stub = evaluation_pb2_grpc.EnvironmentStub(channel)
env_shell = EnvShell(stub)
# policy = deprl.load_baseline(env_shell)
policy = Policy(env_shell)

# Preparing the dictionary of environment keys
custom_environment_varibles = {'obs_keys':custom_obs_keys, 'normalize_act':True}

# Setting the keys to the environment
stub.set_environment_keys(
    evaluation_pb2.Package(SerializedEntity=pack_for_grpc(custom_environment_varibles))
).SerializedEntity

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"LOCO-SOCCER: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = unpack_for_grpc(
        stub.reset(
            evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
        ).SerializedEntity
    )

    counter = 0

    while not flag_trial:
    #for t in range(1000):
        print(
            f"Trial: {trial}, Iteration: {counter} flag_trial: {flag_trial} flat_completed: {flat_completed}"
        )
        # Just to be sure
        if isinstance(obs, tuple):
            obs = obs[0]
        action = policy(obs)
        base = unpack_for_grpc(
            stub.act_on_environment(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
            ).SerializedEntity
        )
        # print(f" \t \t after step: {base['feedback'][1:4]}")
        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]
        # print(
        #     f" \t \t after step: flag_trial: {flag_trial} flat_completed: {flat_completed}"
        # )
        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
