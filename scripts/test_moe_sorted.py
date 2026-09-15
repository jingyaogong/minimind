"""MoE 排序分发的正确性测试 / correctness tests for the sorted MoE dispatch.

本仓库没有 pytest / CI，所以这里不引入任何测试框架，直接 `python scripts/test_moe_sorted.py`
运行，失败返回非零。

reference_masked_forward 是改动之前逐专家做 mask 的写法，保留在这里作为参考实现：
它每次运行都会被执行，并断言与当前实现逐位相同，因此既能当作对照读，也不会随代码腐烂。
"""
import os
import sys
import traceback
import warnings

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.model_minimind import MiniMindConfig, MOEFeedForward


class Skip(Exception): pass


# ============================== 参考实现 / reference ==============================

def reference_masked_forward(mod, x):
    """改动前的逐专家掩码写法，逐位对照用。

    每个专家都要 3 次 device→host 同步：mask.any() 的取值、nonzero() 的输出长度、
    布尔索引的结果长度，全都得先回到主机才知道。
    """
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.view(-1, hidden_dim)
    scores = F.softmax(mod.gate(x_flat), dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=mod.config.num_experts_per_tok, dim=-1, sorted=False)
    if mod.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    y = torch.zeros_like(x_flat)
    for i, expert in enumerate(mod.experts):
        mask = (topk_idx == i)
        if mask.any():
            token_idx = mask.any(dim=-1).nonzero().flatten()
            weight = topk_weight[mask].view(-1, 1)
            y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
        elif mod.training:
            y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
    return y.view(batch_size, seq_len, hidden_dim)


# ============================== 工具 / helpers ==============================

def build(seed=0, **kw):
    base = dict(hidden_size=64, use_moe=True, num_experts=4, num_experts_per_tok=1, moe_intermediate_size=128)
    base.update(kw)
    torch.manual_seed(seed)
    return MOEFeedForward(MiniMindConfig(**base))


def force_routing(mod, assign, hidden_size=64):
    """构造 x 使得 token t 被路由到 assign[t]：gate 取单位阵，x 在目标维上给大值。"""
    with torch.no_grad():
        mod.gate.weight.zero_()
        for i in range(mod.config.num_experts): mod.gate.weight[i, i] = 1.0
    x = torch.zeros(1, len(assign), hidden_size)
    for t, a in enumerate(assign): x[0, t, a] = 10.0
    return x


# ============================== 用例 / tests ==============================

def test_bit_identical_to_masked_reference():
    """与掩码参考实现对照：前向和参数梯度在所有 k 下逐位相同；输入梯度在 k<=2 下逐位相同。

    k>=3 时输入梯度只能做到 1 ulp 以内，这不是 bug：每个 token 要把 k 份专家贡献加起来，
    浮点加法不满足结合律，(a+b)+c 与 a+(b+c) 可以差 1 ulp；两种写法的求和顺序不同。
    k<=2 时任何顺序都精确相等，所以那里仍然要求 torch.equal。实测最大相对偏差 ~1.2e-7
    （fp32 eps ≈ 1.2e-7），这里的阈值 1e-6 只是留一点余量。
    """
    for e, k in ((1, 1), (2, 1), (4, 1), (4, 2), (4, 3), (4, 4), (8, 2), (8, 3), (8, 8)):
        mod = build(seed=3, num_experts=e, num_experts_per_tok=k)
        torch.manual_seed(21)
        x = torch.randn(3, 11, 64)
        for train in (True, False):
            mod.train(train)
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya = reference_masked_forward(mod, xa)
            ya.sum().backward()
            ga = {n: (p.grad.clone() if p.grad is not None else None) for n, p in mod.named_parameters()}
            mod.zero_grad(set_to_none=True)
            yb = mod(xb)
            yb.sum().backward()
            assert torch.equal(ya, yb), f'E={e} k={k} train={train}: 前向不是逐位相同'
            if k <= 2:
                assert torch.equal(xa.grad, xb.grad), f'E={e} k={k} train={train}: 输入梯度不是逐位相同'
            else:  # 求和顺序不同，只能要求 1 ulp 量级
                rel = (xa.grad - xb.grad).abs().max() / xa.grad.abs().max()
                assert rel < 1e-6, f'E={e} k={k} train={train}: 输入梯度相对偏差 {rel:.2e} 超出浮点求和顺序可解释的范围'
            for n, p in mod.named_parameters():
                assert (ga[n] is None) == (p.grad is None), f'{n}: 梯度有无不一致'
                if p.grad is not None:
                    assert torch.equal(ga[n], p.grad), f'E={e} k={k} train={train} {n}: 参数梯度不是逐位相同'
            mod.zero_grad(set_to_none=True)


def test_edge_cases():
    """零token专家 / 全部token到同一专家 / batch 小于专家数。"""
    for name, assign in {'零token专家': [0, 1, 0, 1, 0, 1], '全部到专家0': [0] * 6, 'batch<专家数': [1, 2]}.items():
        mod = build(seed=5); mod.train()
        x = force_routing(mod, assign)
        assert torch.equal(reference_masked_forward(mod, x), mod(x)), f'{name}: 与掩码参考实现不一致'


def test_zero_token_expert_still_gets_grad():
    """DDP 以 find_unused_parameters=False 构造，空专家也必须出现在自动求导图里。"""
    mod = build(seed=5); mod.train()
    mod(force_routing(mod, [0] * 6)).sum().backward()
    missing = [n for n, p in mod.named_parameters() if p.grad is None]
    assert not missing, f'{missing} 没有梯度 -> DDP(find_unused_parameters=False) 会报错'


def test_aux_loss_and_state_dict_unchanged():
    """aux_loss 与 state_dict 结构不受本次改动影响。"""
    mod = build(seed=7); mod.train()
    torch.manual_seed(11)
    x = torch.randn(2, 16, 64)
    mod(x); got = mod.aux_loss.clone()
    scores = F.softmax(mod.gate(x.view(-1, 64)), dim=-1)
    _, topk_idx = torch.topk(scores, k=1, dim=-1, sorted=False)
    load = F.one_hot(topk_idx, 4).float().mean(0)
    want = (load * scores.mean(0)).sum() * 4 * mod.config.router_aux_loss_coef
    assert torch.equal(got, want), 'aux_loss 变了'
    expect = {'gate.weight'} | {f'experts.{i}.{p}_proj.weight' for i in range(4) for p in ('gate', 'up', 'down')}
    assert set(mod.state_dict()) == expect, 'state_dict 键名变了，旧 checkpoint 将无法 strict 加载'


def test_host_sync_count():
    """本次改动的核心：整层同步次数恒为 1，不随专家数增长。"""
    if not torch.cuda.is_available(): raise Skip('需要 CUDA')

    def syncs(fn, x):
        fn(x); torch.cuda.synchronize()  # 预热，避免把首次分配算进去
        torch.cuda.set_sync_debug_mode('warn')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always'); fn(x)
        torch.cuda.set_sync_debug_mode('default')
        return sum('sync' in str(r.message).lower() for r in w)

    for e in (4, 8, 16):
        mod = build(seed=3, num_experts=e).cuda().train()
        x = torch.randn(2, 64, 64, device='cuda')
        ref, now = syncs(lambda t: reference_masked_forward(mod, t), x), syncs(mod, x)
        assert ref >= 3 * e, f'E={e}: 掩码写法应约 3E={3 * e} 次同步，实测 {ref}'
        assert now == 1, f'E={e}: 排序写法应恒为 1 次同步，实测 {now}'


def test_cuda_bit_identical():
    """CUDA 上 fp32 与 bf16 autocast 同样逐位相同——排序改变了 kernel 的分块方式。"""
    if not torch.cuda.is_available(): raise Skip('需要 CUDA')
    for e, k in ((4, 1), (8, 2)):
        mod = build(seed=3, num_experts=e, num_experts_per_tok=k).cuda().train()
        torch.manual_seed(21)
        x = torch.randn(4, 128, 64, device='cuda')
        assert torch.equal(reference_masked_forward(mod, x), mod(x)), f'E={e} k={k}: CUDA fp32 不一致'
        with torch.autocast('cuda', dtype=torch.bfloat16):
            assert torch.equal(reference_masked_forward(mod, x), mod(x)), f'E={e} k={k}: CUDA bf16 不一致'


TESTS = [test_bit_identical_to_masked_reference, test_edge_cases, test_zero_token_expert_still_gets_grad,
         test_aux_loss_and_state_dict_unchanged, test_host_sync_count, test_cuda_bit_identical]


def main():
    print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}\n')
    fails = skips = 0
    for t in TESTS:
        try:
            t(); print(f'  PASS  {t.__name__}')
        except Skip as s:
            skips += 1; print(f'  SKIP  {t.__name__} ({s})')
        except Exception:
            fails += 1; print(f'  FAIL  {t.__name__}'); traceback.print_exc()
    print(f'\n{len(TESTS) - fails - skips} passed, {fails} failed, {skips} skipped')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
