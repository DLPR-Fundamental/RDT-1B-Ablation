import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Any, Dict
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from mani_skill.utils import common, gym_utils   # noqa: F401

class ManiSkillEnvWrapper:
    def __init__(self, env_id: str, obs_mode='rgb', render_mode='rgb_array',
                 control_mode='pd_joint_pos', reward_mode='dense',
                 sim_backend='auto', max_steps=400, shader='default'):
        try:
            self.env = gym.make(
                env_id,
                obs_mode=obs_mode,
                control_mode=control_mode,
                render_mode=render_mode,
                reward_mode=reward_mode,
                sensor_configs=dict(shader_pack=shader),
                human_render_camera_configs=dict(shader_pack=shader),
                viewer_camera_configs=dict(shader_pack=shader),
                sim_backend=sim_backend,
                max_episode_steps=max_steps,
            )
        except TypeError:
            self.env = gym.make(
                env_id,
                obs_mode=obs_mode,
                control_mode=control_mode,
                render_mode=render_mode,
                reward_mode=reward_mode,
                sensor_configs=dict(shader_pack=shader),
                human_render_camera_configs=dict(shader_pack=shader),
                viewer_camera_configs=dict(shader_pack=shader),
                sim_backend=sim_backend,
            )
        self.max_steps = max_steps

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()