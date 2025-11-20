from __future__ import annotations
import argparse
import os
import random
import numpy as np
import torch
import traceback

try:
    from src.model.policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace  # type: ignore
except Exception:
    RobotWorkspace = None

try:
    from src.model.policy.rdt.maniskill_model import create_model  # type: ignore
except Exception:
    create_model = None

from src.benchmark.maniskill.wrapper import ManiSkillEnvWrapper
from src.evaluation.evaluator import Evaluator
from src.evaluation.adapter.dp_adapter import DiffusionPolicyAdapter
from src.evaluation.adapter.rdt_adapter import RDTPolicyAdapter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dp', 'rdt'], required=True, help='dp | rdt')
    parser.add_argument('--env', default='PickCube-v1')
    parser.add_argument('--obs-mode', default='rgb')
    parser.add_argument('--render-mode', default='rgb_array')
    parser.add_argument('--num-traj', type=int, default=25)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--sim-backend', type=str, default='auto')
    parser.add_argument('--shader', type=str, default='default')
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--live-view', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    # RDT specific
    parser.add_argument('--action-downsample', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--lang-embeddings-path', type=str, default=None, help='Preprocessed text embeddings path for RDT')

    return parser.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seeds(args.random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 🔹 한 번만 선언

    env_wrapper = ManiSkillEnvWrapper(
        env_id=args.env,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        max_steps=args.max_steps,
        shader=args.shader,
    )

    if args.model == 'dp':
        if args.pretrained_path is None:
            raise RuntimeError('Diffusion Policy 체크포인트 경로를 --pretrained_path 로 지정하세요')
        policy = DiffusionPolicyAdapter(
            checkpoint_path=args.pretrained_path,
            device=device
        )

    elif args.model == 'rdt':  # 🔹 RDT 전용 옵션만 전달
        config = {}
        dtype = torch.float16 if args.dtype == 'fp16' else (torch.bfloat16 if args.dtype == 'bf16' else torch.float32)

        text_embed = None
        if args.lang_embeddings_path is not None:
            text_embed = torch.load(args.lang_embeddings_path, map_location=device)

        policy = RDTPolicyAdapter(
            config=config,
            pretrained_path=args.pretrained_path,
            text_embed=text_embed,
            device=device,
            dtype=dtype,
            action_downsample=args.action_downsample  # 🔹 RDT 전용 옵션
        )
    else:
        raise RuntimeError(f'알 수 없는 모델: {args.model}')

    evaluator = Evaluator(
        env=env_wrapper,
        policy=policy,
        num_episodes=args.num_traj,
        max_steps=args.max_steps,
        live_view=args.live_view,
        save_video=args.save_video
    )

    try:
        results = evaluator.run(base_seed=20241201)
    except Exception:
        print("예외 발생! 전체 트레이스백을 출력합니다:")
        traceback.print_exc()
        raise   # 원하면 재발생시키지 않고 return 하도록 바꿀 수 있음
    
    summary_path = f"eval_summary_{args.model}_{os.path.basename(args.pretrained_path or 'none')}.txt"
    with open(summary_path, 'a') as f:
        f.write(f"env={args.env},seed={args.random_seed},num={args.num_traj},success_rate={results['success_rate']:.2f}\n")
