from typing import Any, Optional
import numpy as np
import torch

def _to_numpy_rgb(x: Any) -> Optional[np.ndarray]:
    if torch.is_tensor(x):
        arr = x.detach().cpu()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        arr = arr.numpy()
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return None

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return None
    return None


def _find_first_frame(x: Any) -> Optional[np.ndarray]:
    arr = _to_numpy_rgb(x)
    if arr is not None:
        return arr
    if isinstance(x, (list, tuple)):
        for item in x:
            arr = _find_first_frame(item)
            if arr is not None:
                return arr
    if isinstance(x, dict):
        for v in x.values():
            arr = _find_first_frame(v)
            if arr is not None:
                return arr
    return None


def _extract_rgb(frame: Any, obs: Optional[dict] = None) -> np.ndarray:
    arr = _find_first_frame(frame)
    if arr is not None:
        return arr
    if obs is not None:
        arr = _find_first_frame(obs)
        if arr is not None:
            return arr
    raise RuntimeError(
        "Could not extract an RGB ndarray from env.render()/obs. "
        "Try obs_mode='rgb' and render_mode='rgb_array'."
    )

