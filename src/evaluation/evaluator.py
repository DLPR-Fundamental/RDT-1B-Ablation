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
            bundle = self.env.reset(seed=ep+base_seed)
            obs_window = deque([bundle['rgb']]*self.policy.required_frames, maxlen=self.policy.required_frames)
            proprio = bundle['proprio']
            self.policy.reset()
            done = False
            steps = 0
            video_frames = []

            while steps < self.max_steps and not done:
                obs = {'proprio': proprio, 'images': [Image.fromarray(f) if f is not None else None for f in list(obs_window)]}
                actions_seq = self.policy.infer(obs)
                # step loop
                for act in actions_seq:
                    next_bundle, reward, terminated, truncated, info = self.env.step(act)
                    proprio = next_bundle['proprio']
                    if next_bundle['rgb'] is not None:
                        obs_window.append(next_bundle['rgb'])
                    if self.live_view and next_bundle['rgb'] is not None:
                        cv2.imshow('Live', next_bundle['rgb'][..., ::-1])
                        cv2.waitKey(1)
                    if self.save_video and next_bundle['rgb'] is not None:
                        video_frames.append(next_bundle['rgb'])
                    steps += 1
                    if terminated or truncated:
                        if info.get('success', False):
                            success_count += 1
                        done = True
                        break

            # save video
            if self.save_video and video_frames:
                path = os.path.join(self.video_dir, f'episode_{ep+1}.mp4')
                writer = imageio.get_writer(path, fps=20)
                for f in video_frames:
                    writer.append_data(f)
                writer.close()

            episodes_info.append({'episode': ep+1, 'success': info.get('success', False), 'steps': steps})
            print(f"Episode {ep+1}: success={info.get('success', False)}, steps={steps}")

        success_rate = success_count/self.num_episodes*100
        print(f"Success rate: {success_rate:.2f}%")
        return {'success_rate': success_rate, 'episodes': episodes_info}