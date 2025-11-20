import numpy as np

class BasePolicyAdapter:
    """
    infer(obs: dict) -> np.ndarray
    - obs = {'proprio': np.ndarray, 'images': List[PIL.Image]}
    """
    required_frames: int = 2  # default, 정책마다 다르게 설정 가능

    def reset(self):
        pass

    def infer(self, obs: dict) -> np.ndarray:
        raise NotImplementedError