import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym
from stable_baselines3 import PPO

from utils import RemoteConnection

"""
Define your custom observation keys here
"""
custom_obs_keys = [
    'pelvis_pos',
    'body_qpos',
    'body_qvel',
    'ball_pos',
    'ball_vel',
    'paddle_pos',
    'paddle_vel',
    'paddle_ori',
    'reach_err',
    'touching_info',
    'act',
]

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class Policy:

    def __init__(self, env):
        self.action_space = env.action_space

        # Load trained PPO policy from best_model
        possible_paths = [
            "best_model_beta_p2.zip",
            "agent/best_model_beta_p2.zip",
            os.path.join(os.path.dirname(__file__), "best_model_beta_p2.zip"),
        ]

        vecnorm_paths = [
            "vecnormalize_beta_p2.pkl",
            "agent/vecnormalize_beta_p2.pkl",
            os.path.join(os.path.dirname(__file__), "vecnormalize_beta_p2.pkl"),
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        vecnorm_path = None
        for path in vecnorm_paths:
            if os.path.exists(path):
                vecnorm_path = path
                break

        if model_path and vecnorm_path:
            model = PPO.load(model_path, device="cpu")
            self.policy = model.policy

            with open(vecnorm_path, "rb") as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False

            print(f"Loaded trained PPO policy from {model_path}")
            self.use_trained_model = True
        else:
            print("Model files not found. Using random policy.")
            self.use_trained_model = False
            self.policy = None
            self.vecnorm = None

    def __call__(self, obs):
        if not self.use_trained_model:
            return self.action_space.sample()

        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32).flatten()

        norm_obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.policy.predict(norm_obs, deterministic=False)

        return action

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

policy = Policy(rc)

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"PINGPONG: Start Resetting the environment and get 1st obs of iter {trial}")

    obs = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        ################################################
        ## Using trained PPO policy
        action = policy(obs)
        ################################################

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
