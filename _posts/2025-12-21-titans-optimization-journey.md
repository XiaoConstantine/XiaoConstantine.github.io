---
layout: post
title: "Titans Part 2: Optimizing Memory Updates and Adaptive Learning"
---

This is a follow-up to my [previous post on implementing Titans](https://xiaocui.me/2025/12/20/implementing-titans-memory/). After getting the basic implementation working, I dove deeper into performance optimization with Claude Code as a pair programmer. This post covers the journey from a working but slow implementation to something that trains efficiently on commodity GPUs.

The Problem: Memory Update Bottleneck
-------------------------------------

After training for ~27k iterations on OpenWebText, I hit a wall. The loss plateaued around 3.8 and profiling revealed why - the memory update mechanism was consuming a disproportionate amount of GPU time:

```
_fused_weight_update_kernel    450ms   15.84%
triton_poi_fused_sigmoid       510ms   17.92%
```

Nearly 34% of CUDA time was spent on Titans memory operations. For a 130M parameter model, this overhead was unacceptable. The MFU (Model FLOPs Utilization) was stuck at 3.39%.

The Culprit: Per-Token Gradient Tensors
---------------------------------------

Looking at the original implementation, the bottleneck was clear. For each forward pass, we were computing per-token gradients that materialized a massive tensor:

```python
# Old path: outer product for each token
dW = torch.einsum('btc,bth->btch', d_pred, h1)  # [B, T, C, H]
```

With batch=2, sequence=512, hidden=768, expansion=2:
- Tensor size: 2 × 512 × 768 × 1536 = **1.2 billion elements**
- Memory: ~4.8GB just for gradients

This was the memory killer. Every forward pass allocated gigabytes of temporary memory, causing GPU memory pressure and slow kernel launches.

The Fix: Aggregated Gradients
-----------------------------

The insight came from staring at the equation. We don't need per-token gradients - we need the *sum* over tokens for the weight update:

```python
# New path: aggregate directly (no per-token storage!)
dW = torch.einsum('btc,bth->bch', d_pred, h1)  # [B, C, H]
```

Same information, but:
- Tensor size: 2 × 768 × 1536 = **2.4 million elements**
- Memory: ~9.6MB

**500x memory reduction.** The momentum update becomes an approximation (mean gradient instead of per-token EMA), but for typical momentum values (0.9), the difference is negligible.

```python
def aggregated_gradient_memory_update(
    keys, values, weights, momentum,
    lr, mom_coef, decay
):
    # Forward pass
    h1 = silu(keys @ W0.T)
    pred = h1 @ W1.T

    # Backward - aggregate over T
    d_pred = (2.0 / C) * (pred - values)
    dW1 = torch.einsum('btc,bth->bch', d_pred, h1)  # Sum over T
    dW0 = torch.einsum('bth,btc->bhc', dh1_pre, keys)

    # Momentum with mean gradient
    scale = 1.0 / T
    new_mom = mom_coef * momentum + (1 - mom_coef) * dW * scale
    new_weights = (1 - decay) * weights - lr * new_mom

    return new_weights, new_mom
```

Adding Adaptive Memory Parameters
---------------------------------

The Titans paper mentions adaptive learning rates - per-token learned parameters for lr, momentum, and decay. My initial implementation used fixed hyperparameters. Time to match the paper.

The idea: let the model learn *how* to update its memory based on the input:

```python
class NeuralMemory(nn.Module):
    def __init__(self, config):
        # ... existing code ...

        if config.adaptive_memory:
            self.to_lr = nn.Linear(n_embd, 1)
            self.to_momentum = nn.Linear(n_embd, 1)
            self.to_decay = nn.Linear(n_embd, 1)

    def update(self, x, state):
        if self.adaptive:
            # Per-token adaptive parameters via learned projections
            adaptive_lr = torch.sigmoid(self.to_lr(x)) * self.lr_max
            adaptive_momentum = torch.sigmoid(self.to_momentum(x))
            adaptive_decay = torch.sigmoid(self.to_decay(x))
```

The sigmoid bounds keep parameters in sensible ranges:
- LR: (0, lr_max) - typically 0.01
- Momentum: (0, 1) - higher = more memory persistence
- Decay: (0, 1) - higher = faster forgetting

A subtle bug emerged: the default `_init_weights` function was resetting my carefully initialized biases to zero. The fix required special-casing the adaptive projection layers:

```python
def _init_weights(self, module):
    match module:
        case NeuralMemory() if module.adaptive:
            # Preserve adaptive bias initialization
            nn.init.constant_(module.to_momentum.bias, 2.0)  # sigmoid(2)≈0.88
            nn.init.constant_(module.to_decay.bias, -4.0)    # sigmoid(-4)≈0.02
        case nn.Linear():
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

Optimizer Bottleneck: Fused AdamW
---------------------------------

With memory updates optimized, a new bottleneck appeared. Profiling showed:

```
Optimizer.step#AdamW.step    425ms   41.17%
aten::_foreach_mul_           87ms    8.43%
aten::_foreach_addcdiv_       81ms    7.84%
aten::_foreach_lerp_          64ms    6.18%
```

The optimizer was taking 41% of GPU time! AdamW was using 7+ separate foreach operations instead of a single fused kernel.

The fix was embarrassingly simple:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    fused=True  # Single CUDA kernel
)
```

Results after enabling fused optimizer:

| Metric | Before | After |
|--------|--------|-------|
| Optimizer time | 426ms (41%) | 170ms (17%) |
| Total CUDA time | 1,034ms | 787ms |
| MFU | 6.3% | 10.8% |

A 60% reduction in optimizer overhead, just by adding `fused=True`.

Profiling Deep Dive
-------------------

Building a proper profiling setup was essential. I added a kernel category summary to quickly identify bottleneck categories:

```
Category             Time (ms)    %
----------------------------------------
other                2053.51      50.6
optimizer            680.86       16.8
matmul/gemm          535.45       13.2
triton/fused         458.69       11.3
elementwise          173.29       4.3
memory_ops           153.29       3.8
attention            0.00         0.0
```

The high "other" category revealed torch.compile overhead - the `CompiledFxGraph` wrapper operations. Not much to optimize there, but knowing where time goes prevents chasing phantom bottlenecks. Memory operations are now just 3.8% thanks to our aggregated gradient approach.

Checkpoint Compatibility
------------------------

A practical challenge: resuming training from old checkpoints with new model code. The adaptive memory projections add 6 new parameters that old checkpoints don't have.

The solution checks the actual state_dict rather than trusting config:

```python
# Check if checkpoint actually has adaptive memory weights
ckpt_has_adaptive = any(
    "to_lr" in k or "to_momentum" in k or "to_decay" in k
    for k in state_dict
)

if upgrade_to_adaptive:
    # Load with strict=False to allow missing keys
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print(f"New parameters initialized: {len(missing)} keys")
```

This lets you upgrade a trained model to use adaptive memory mid-training. The core 128M weights are preserved; only the 6 small projection weights start fresh.

Results
-------

After all optimizations, training a 128M Titans model on L4:

| Metric | Initial | Optimized |
|--------|---------|-----------|
| MFU | 3.4% | 10.8% |
| Time/iter | ~2400ms | ~1950ms |
| Memory overhead | 4.8GB | 9.6MB |
| Optimizer overhead | 41% | 17% |

The throughput improvement came from:
1. Larger batch sizes (2 → 4) enabled by memory reduction
2. Fused optimizer kernels
3. Better GPU utilization

Current training on OpenWebText:
- Iteration: 29k+
- Train loss: 3.55
- Validation loss: 3.57
- Training with adaptive memory enabled
- Steady loss descent with no overfitting (train/val gap < 0.02)

Lessons From This Round
-----------------------

1. **Profile at the kernel level**: High-level timing hides the real story. The optimizer taking 41% was invisible without detailed profiling.

2. **Memory bandwidth matters**: The 500x gradient memory reduction wasn't about avoiding OOM - it was about reducing memory pressure that slowed everything else.

3. **Check the defaults**: `fused=True` for AdamW exists but isn't default. Same for `torch.compile` on attention. Read the docs for performance flags.

4. **Aggregation beats iteration**: When you can sum instead of iterate, do it. The per-token gradient loop was algorithmically correct but computationally wasteful.

5. **Checkpoint evolution**: Plan for model architecture changes. Strict loading breaks iteration; flexible loading with smart defaults enables continuous improvement.

What's Next
-----------

- Training to convergence with adaptive memory
- Comparison: fixed vs adaptive memory parameters
- Longer context experiments (2k, 4k, 8k tokens)
- Potential MAG (Memory as Gate) implementation

Code remains at [github.com/XiaoConstantine/nanogpt-titans](https://github.com/XiaoConstantine/nanogpt-titans).
