import argparse
import os
import random
import numpy as np
import yaml
import torch
import traceback

from src.benchmark.maniskill.wrapper import ManiSkillEnvWrapper
from src.evaluation.evaluator import Evaluator
from src.evaluation.adapter.dp_adapter import DiffusionPolicyAdapter
from src.evaluation.adapter.rdt_adapter import RDTPolicyAdapter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dp', 'rdt'], required=True, help='dp | rdt')
    parser.add_argument('--env', default='PickCube-v1', required=True)
    parser.add_argument('--obs-mode', default='rgb')
    parser.add_argument('--render-mode', default='rgb_array')
    parser.add_argument('--num-traj', type=int, default=25)
    parser.add_argument('--pretrained_path', type=str, required=True)
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
    print(f"Random seed set to {args.random_seed}")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    env_wrapper = ManiSkillEnvWrapper(
        env_id=args.env,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        max_steps=args.max_steps,
        shader=args.shader,
    )
    print(f"Environment '{args.env}' initialized with obs_mode='{args.obs_mode}', render_mode='{args.render_mode}'")

    if args.model == 'diffusion_policy' or args.model == 'dp':
        print(f"Using Diffusion Policy model with checkpoint: {args.pretrained_path}")
        policy = DiffusionPolicyAdapter(
            checkpoint_path=args.pretrained_path,
            output_dir='.',
            device=device
        )
    elif args.model == 'rdt':
        print(f"Using RDT model with checkpoint: {args.pretrained_path}")
        import yaml
        with open('configs/base.yaml', "r") as fp:
            config = yaml.safe_load(fp)
        dtype = torch.float16 if args.dtype == 'fp16' else (torch.bfloat16 if args.dtype == 'bf16' else torch.float32)

        if args.lang_embeddings_path is None:
            raise ValueError("--lang-embeddings-path required for RDT model...")

        policy = RDTPolicyAdapter(
            config=config,
            text_embed=RDTPolicyAdapter._load_lang_embed(args.env, args.lang_embeddings_path),
            pretrained_model_path=args.pretrained_path,
            device=device,
            dtype=dtype,
            action_downsample=args.action_downsample
        )
        print(f"Using RDT model with checkpoint: {args.pretrained_path}")
    else:
        raise RuntimeError(f'Unknown model : {args.model}')

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
        print("Exception occurred! Printing full traceback:")
        traceback.print_exc()
        return
    
    summary_path = f"eval_summary_{args.model}_{os.path.basename(args.pretrained_path or 'none')}.txt"
    with open(summary_path, 'a') as f:
        f.write(f"env={args.env},seed={args.random_seed},num={args.num_traj},success_rate={results['success_rate']:.2f}\n")


if __name__ == '__main__':
    main()