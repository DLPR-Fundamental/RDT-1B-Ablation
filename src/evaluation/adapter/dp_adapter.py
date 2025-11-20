from __future__ import annotations
import numpy as np
import torch

from src.evaluation.adapter.base import BasePolicyAdapter
from src.utility.helper import _extract_rgb

class DiffusionPolicyAdapter(BasePolicyAdapter):
    required_frames = 1  # DP는 single frame 사용 가능

    def __init__(self, checkpoint_path, device='cuda', state_min=None, state_max=None, action_min=None, action_max=None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.policy = self._load_policy(checkpoint_path)
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max

    def _load_policy(self, checkpoint_path):
        import dill, hydra
        from src.model.policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir='.')
        workspace.load_payload(payload)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        return policy.to(self.device).eval()

    def infer(self, obs: dict) -> np.ndarray:
        with torch.no_grad():
            out = self.policy.predict_action(obs)
            actions = out['action_pred'].squeeze(0)
            if self.action_min is not None:
                actions = (actions + 1) / 2 * (torch.tensor(self.action_max)-torch.tensor(self.action_min)) + torch.tensor(self.action_min)
            return actions.detach().cpu().numpy()