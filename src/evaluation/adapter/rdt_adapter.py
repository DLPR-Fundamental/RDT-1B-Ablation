from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from src.evaluation.adapter.base import BasePolicyAdapter

class RDTPolicyAdapter(BasePolicyAdapter):
    required_frames = 2  # RDT 모델 입력 프레임 수

    def __init__(self, config: dict, pretrained_path=None, text_embed=None,
                 device='cuda', dtype=torch.float16, quant_mode='none', action_downsample=1):
        self.device = device
        self.dtype = dtype
        self.action_downsample = action_downsample
        self.text_embed = text_embed
        self.policy = self._load_policy(config, pretrained_path)

    def _load_policy(self, config, pretrained_path):
        from src.model.policy.rdt.maniskill_model import create_model
        policy = create_model(args=config, dtype=self.dtype, pretrained=pretrained_path)
        # device 이동
        if hasattr(policy, 'policy'):
            policy.policy = policy.policy.to(self.device, dtype=self.dtype)
        if getattr(policy, 'vision_model', None):
            policy.vision_model = policy.vision_model.to(self.device, dtype=self.dtype)
        if getattr(policy, 'text_model', None):
            policy.text_model = policy.text_model.to(self.device, dtype=self.dtype)
        policy.reset()
        return policy

    def reset(self):
        self.policy.reset()

    def infer(self, obs: dict) -> np.ndarray:
        # proprio
        proprio_t = torch.tensor(np.asarray(obs['proprio']).ravel(),
                                device=self.device,
                                dtype=self.dtype).unsqueeze(0)

        # images: PIL/ndarray -> tensor, device 이동
        images = obs['images'][:self.required_frames]
        images_t = []
        for img in images:
            if img is None:
                images_t.append(None)
            elif isinstance(img, np.ndarray):
                images_t.append(torch.tensor(img, device=self.device, dtype=self.dtype).permute(2,0,1).unsqueeze(0))
            elif isinstance(img, Image.Image):
                t = torch.tensor(np.array(img), device=self.device, dtype=self.dtype).permute(2,0,1).unsqueeze(0)
                images_t.append(t)
            else:
                raise TypeError(f"Unknown image type: {type(img)}")

        # text embed device/dtype
        text_embed = None
        if self.text_embed is not None:
            text_embed = self.text_embed.to(self.device, dtype=self.dtype) if isinstance(self.text_embed, torch.Tensor) else None

        # 추론
        if self.device == 'cuda':
            with torch.cuda.amp.autocast(dtype=self.dtype):
                actions = self.policy.step(proprio_t, images_t, text_embed).squeeze(0)
        else:
            actions = self.policy.step(proprio_t, images_t, text_embed).squeeze(0)

        if self.action_downsample > 1:
            actions = actions[::self.action_downsample]

        return actions.detach().cpu().numpy()
