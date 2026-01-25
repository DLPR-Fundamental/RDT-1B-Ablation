import os
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from collections import deque

from src.evaluation.adapter.base import BasePolicyAdapter
from src.utility.helper import _extract_rgb

class RDTPolicyAdapter(BasePolicyAdapter):
    required_history = 2        # RDT-1B model minimum history length
    required_slots = 6          # RDT-1B model required slots (3 cam × 2 frame)
    policy_name = "rdt-1b"

    def __init__(self, 
                 config: dict, 
                 text_embed=None, 
                 pretrained_model_path=None, 
                 pretrained_text_encoder_name_or_path=None, 
                 pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
                 device='cuda', dtype=torch.float16, action_downsample=4):
        self.device = device
        self.dtype = dtype
        self.action_downsample = action_downsample
        self.text_embed = text_embed.to(device, dtype) if text_embed is not None else None
        self.policy = self._load_policy(config, pretrained_model_path, 
                                       pretrained_text_encoder_name_or_path,
                                       pretrained_vision_encoder_name_or_path)

    def reset(self):
        self.policy.reset()

    def infer(self, obs: dict) -> np.ndarray:
        images_pil = obs['images']
        proprio_t = obs['state']

        text_embed = self.text_embed

        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda'), dtype=self.dtype):
            actions = self.policy.step(proprio_t, images_pil, text_embed).squeeze(0)

        if self.action_downsample > 1:
            actions = actions[::self.action_downsample]

        return actions.detach().cpu().numpy()

    def _pad_to_rdt_format(self, images: List) -> List:
        """required_history=2 → required_slots=6 reshape, only for rdt-1b"""
        padded = []
        for img in images:
            padded.extend([img, None, None])
        while len(padded) < self.required_slots:
            padded.append(None)
        return padded[:self.required_slots]

    def _extract_first_obs(self, bundle, env):
        return _extract_rgb(env.render(), bundle)

    def _extract_obs(self, bundle, env):
        return _extract_rgb(env.render(), bundle) if env.render() is not None else None

    def _get_proprio(self, bundle):
        return bundle['agent']['qpos'][:, :-1]

    def _prepare_obs(self, obs_window, proprio):
        images_padded = self._pad_to_rdt_format(obs_window)
        images_pil = [Image.fromarray(img) if img is not None else None for img in images_padded]
        return {'images': images_pil, 'state': proprio}

    def _load_policy(self, config, pretrained_model_path, 
                     pretrained_text_encoder_name_or_path,
                     pretrained_vision_encoder_name_or_path):
        from src.model.policy.rdt.maniskill_model import create_model
        policy = create_model(
            args=config, 
            dtype=self.dtype, 
            pretrained=pretrained_model_path,
            pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        )
        if hasattr(policy, 'policy'):
            policy.policy = policy.policy.to(self.device, dtype=self.dtype)
        if getattr(policy, 'vision_model', None):
            policy.vision_model = policy.vision_model.to(self.device, dtype=self.dtype)
        if getattr(policy, 'text_model', None):
            policy.text_model = policy.text_model.to(self.device, dtype=self.dtype)
        policy.reset()
        return policy

    def _load_lang_embed(env_id: str, explicit_path: Optional[str]):
        candidates = []
        if explicit_path is not None:
            candidates.append(explicit_path)
        candidates += [
            f'./text_embed_{env_id}.pt',
            f'lang_embeds/text_embed_{env_id}.pt',
            f'data/text_embed_{env_id}.pt',
        ]
        for p in candidates:
            if p and os.path.exists(p):
                print(f"[INFO] Using precomputed language embedding: {p}")
                emb = torch.load(p, map_location="cpu")
                # (L, D) → (1, L, D) 형태 보정
                if isinstance(emb, torch.Tensor) and emb.ndim == 2:
                    emb = emb.unsqueeze(0)
                return emb
        raise FileNotFoundError(
            f"Precomputed language embedding not found. "
            f"Pass --lang_embeddings_path or place text_embed_{env_id}.pt at repo root."
        )