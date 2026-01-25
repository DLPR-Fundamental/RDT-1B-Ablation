import os
import numpy as np

import torch
from collections import deque

from src.evaluation.adapter.base import BasePolicyAdapter
from src.utility.helper import _extract_rgb

class DiffusionPolicyAdapter(BasePolicyAdapter):
    required_history = 2
    policy_name = "diffusion_policy"

    DATA_STAT = {'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0], 'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033], 'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0], 'action_std': [0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999], 'action_mean': [-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494]}

    def __init__(self, checkpoint_path, output_dir, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.policy = self._load_policy(checkpoint_path, output_dir, device)

        self.state_min = torch.tensor(self.DATA_STAT['state_min']).cuda()
        self.state_max = torch.tensor(self.DATA_STAT['state_max']).cuda()
        self.action_min = torch.tensor(self.DATA_STAT['action_min']).cuda()
        self.action_max = torch.tensor(self.DATA_STAT['action_max']).cuda()

    def _load_policy(self, checkpoint_path, output_dir, device):
        import dill, hydra
        from src.model.policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace
        # load checkpoint
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
    
    def infer(self, obs: dict) -> np.ndarray:
        out = self.policy.predict_action(obs)
        actions = out['action_pred'].squeeze(0)
        actions = (actions + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        actions = actions[:8] 
        return actions.detach().cpu().numpy()

    def _extract_first_obs(self, bundle, env):
        return self._extract_obs(bundle, env)

    def _extract_obs(self, bundle, env):
        img = env.render().cuda().float()
        
        img = img.permute(0, 3, 1, 2)
        proprio = self._get_proprio(bundle)

        return {
            "head_cam": img,
            "agent_pos": proprio
        }
    
    def _get_proprio(self, bundle):
        proprio = bundle['agent']['qpos'][:].cuda()
        proprio = (proprio - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        return proprio
    
    def _prepare_obs(self, obs_window, proprio):
        return obs_window[-1]

    def _show_live(self, img):
        import cv2
        img_tensor = img["head_cam"]

        if torch.is_tensor(img_tensor):
            img = img_tensor.detach().cpu().numpy()

        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[0] != img.shape[-1]:
            img = img.transpose(1,2,0)

        img = np.clip(img, 0, 255).astype(np.uint8)

        cv2.imshow(f'{self.policy_name} Live', img[..., ::-1])
        cv2.waitKey(1)