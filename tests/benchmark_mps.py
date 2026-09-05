"""Short MiniMind MPS training benchmark; it does not save checkpoints."""

import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_autocast_context, get_grad_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=340)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--num-hidden-layers', type=int, default=8)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available')

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=max(args.seq_len, 512),
    )
    model = MiniMindForCausalLM(config).to('mps').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = get_grad_scaler('mps', args.dtype)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device='mps'
    )

    durations = []
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        torch.mps.synchronize()
        started = time.perf_counter()
        with get_autocast_context('mps', args.dtype):
            loss = model(input_ids, labels=input_ids).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        if step >= args.warmup_steps:
            durations.append(elapsed)

    mean_seconds = sum(durations) / len(durations)
    tokens_per_second = args.batch_size * args.seq_len / mean_seconds
    print(f'mean_step_seconds={mean_seconds:.3f}')
    print(f'tokens_per_second={tokens_per_second:.1f}')
    print(f'loss={loss.detach().float().item():.4f}')


if __name__ == '__main__':
    main()
