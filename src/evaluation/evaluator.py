from __future__ import annotations
import os
from collections import deque
from typing import Any, Dict
from PIL import Image
import cv2
import tqdm
import imageio

from src.benchmark.maniskill.wrapper import ManiSkillEnvWrapper
from src.evaluation.adapter.base import BasePolicyAdapter
from src.utility.helper import _extract_rgb
class Evaluator:
    def __init__(self, env: ManiSkillEnvWrapper, policy: BasePolicyAdapter,
                 num_episodes=10, max_steps=400, live_view=False, save_video=False, video_dir='demos'):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.live_view = live_view
        self.save_video = save_video
        self.video_dir = video_dir
        if save_video and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

    def run(self, base_seed=20241201):
        success_count = 0
        episodes_info = []

        for ep in tqdm.trange(self.num_episodes):
            steps, success, video_frames = self.policy.run_episode(
                self.env, max_steps=self.max_steps, base_seed=base_seed,
                episode_idx=ep, live_view=self.live_view, save_video=self.save_video
            )

            if self.save_video and video_frames:
                path = os.path.join(self.video_dir, f'episode_{ep+1}.mp4')
                writer = imageio.get_writer(path, fps=20)
                for f in video_frames:
                    writer.append_data(f)
                writer.close()

            episodes_info.append({'episode': ep+1, 'success': success, 'steps': steps})
            print(f"Episode {ep+1}: success={success}, steps={steps}")
            if success:
                success_count += 1

        success_rate = success_count / self.num_episodes * 100
        print(f"Success rate: {success_rate:.2f}%")
        return {'success_rate': success_rate, 'episodes': episodes_info}