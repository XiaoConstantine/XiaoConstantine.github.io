---
layout: post
title: "Titans Part 3: HOPE Architecture - From Paper to Reality (and Back)"
---

This is Part 3 of my Titans implementation series. [Part 1](https://xiaocui.me/2025/12/20/implementing-titans-memory/) covered the basic memory mechanism. [Part 2](https://xiaocui.me/2025/12/21/titans-optimization-journey/) focused on performance optimization. This post tackles the full HOPE (Hierarchical Optimized Parallel Encoding) architecture - the multi-level memory system that makes Titans truly interesting - and the hard lessons learned when evaluation results didn't match expectations.

What HOPE Adds to Basic Titans
------------------------------

The basic Titans memory is a single MLP that updates via gradient descent during inference. HOPE extends this with three key components:

**1. Continuum Memory System (CMS)**

Instead of one memory, CMS uses multiple memory modules updating at different frequencies:

```python
# Level 0: Updates every segment (working memory)
# Level 1: Updates every 4 segments (episodic memory)
# Level 2: Updates every 16 segments (semantic memory)
cms_update_frequencies = (1, 4, 16)
```

This mirrors how biological memory works - fast-updating short-term memory feeding into slower long-term consolidation. The paper draws parallels to gamma/beta/theta brainwaves.

**2. Position-Dependent Gating**

Rather than a global gate that applies uniformly, HOPE uses a small MLP to produce per-token gate values:

```python
class PositionDependentGate(nn.Module):
    def __init__(self, dim, init_bias=0.0):
        self.linear1 = nn.Linear(dim, dim // 4)
        self.linear2 = nn.Linear(dim // 4, 1)
        # Initialize to sigmoid(init_bias) output
        self.linear2.bias = nn.Parameter(torch.tensor([init_bias]))

    def forward(self, x):
        h = F.silu(self.linear1(x))
        return torch.sigmoid(self.linear2(h))  # [B, T, 1]
```

This lets the model decide *per token* whether to use memory. Critical for tasks like needle-in-haystack where most tokens should ignore memory, but a few specific tokens need it.

**3. Internal Loss**

The memory module gets its own training signal independent of the language modeling loss:

```
L_internal = ||M(keys) - values||^2
```

This teaches the template weights to store and retrieve patterns, separate from how the gate learns to use memory.

MLX Implementation for Apple Silicon
------------------------------------

After validating the PyTorch implementation, I ported everything to MLX for Apple Silicon. The motivation: fast iteration on my MacBook without cloud GPU costs.

The architecture maps cleanly to MLX:

```python
class MLXContinuumMemorySystem(nn.Module):
    def __init__(self, dim, num_levels=3, update_frequencies=(1, 4, 16)):
        self.memories = [
            MLXNeuralMemory(dim) for _ in range(num_levels)
        ]
        # Learnable weights for combining levels
        self.level_weights = mx.zeros((num_levels,))

    def update(self, hidden_states, state):
        new_level_states = []
        for i, (mem, freq) in enumerate(zip(self.memories, self.update_frequencies)):
            # Only update at the right frequency
            if state.step % freq == 0:
                new_state = mem.update(hidden_states, state.level_states[i])
            else:
                new_state = state.level_states[i]
            new_level_states.append(new_state)
        return MLXCMSState(level_states=new_level_states, step=state.step + 1)
```

The multi-level retrieval combines outputs with learned softmax weights:

```python
def __call__(self, hidden_states, state):
    weights = mx.softmax(self.level_weights)
    level_outputs = [mem(hidden_states, state.level_states[i])
                     for i, mem in enumerate(self.memories)]
    stacked = mx.stack(level_outputs, axis=0)
    return mx.sum(weights.reshape(-1, 1, 1, 1) * stacked, axis=0)
```

The Failure: Memory Isn't Helping
---------------------------------

Here's where theory met reality. After fine-tuning Qwen2-0.5B with HOPE memory layers, the evaluation results were disappointing:

- **Perplexity**: Nearly identical with/without memory
- **Gate values**: Collapsing toward zero during training
- **Long-context tasks**: No improvement over baseline

The memory was theoretically correct. Tests passed. Gradients flowed. But downstream performance showed no benefit.

I spent days debugging:
- Verified memory state persisted across segments
- Checked gate initialization (started at sigmoid(0) = 0.5)
- Added gate regularization to prevent collapse
- Tried different memory layers (6, 12, 18)

Nothing moved the needle.

Root Cause: "Titans Revisited"
------------------------------

The answer came from a critical analysis paper: [Titans Revisited](https://arxiv.org/abs/2510.09551). Key finding:

> "Memory updates alone proved insufficient for meaningful test-time learning when the backbone is frozen... a mismatch between the frozen backbone input projections into key-value space and how the memory evolves. Without joint adaptation, the integration of new information is limited."

This hit home. My implementation:

1. **Froze the entire Qwen backbone** - only TITANS layers trained
2. **Memory weights used stop_gradient** - they update via test-time learning, not backprop
3. **Internal loss was disabled by default** - template weights got zero training signal

The architecture was correct, but the training dynamics were broken.

The Fixes
---------

**Fix 1: Enable Internal Loss**

The template weights that initialize memory need a training signal:

```python
# config.py - old defaults
use_internal_loss: bool = False
internal_loss_weight: float = 1e-4

# New defaults
use_internal_loss: bool = True
internal_loss_weight: float = 0.1  # 1000x stronger
```

Without internal loss, the memory MLP starts random and stays random. The test-time updates adjust from a bad starting point.

**Fix 2: Stronger Internal Loss Weight**

1e-4 was too weak. The LM loss (~3.5) dominated, and internal loss contributions (~0.001) vanished in the gradient noise. Bumping to 0.1 gives memory a real training signal.

**Fix 3: Unfreeze Adjacent Backbone Layers**

Per the "Titans Revisited" insight, memory evolves but the backbone's projections feeding it stay frozen. The solution: unfreeze transformer layers adjacent to memory insertion points.

```python
def get_layers_to_unfreeze(memory_layers, num_backbone_layers, radius):
    """Unfreeze layers within `radius` of each memory layer."""
    layers_to_unfreeze = set()
    for mem_idx in memory_layers:
        for offset in range(-radius, radius + 1):
            layer_idx = mem_idx + offset
            if 0 <= layer_idx < num_backbone_layers:
                layers_to_unfreeze.add(layer_idx)
    return layers_to_unfreeze
```

With `--unfreeze_backbone_layers 1`, layers 11, 12, 13 train alongside memory (if memory is at layer 12). This allows the backbone's key/value projections to adapt to the evolving memory system.

Training with lower LR for backbone (0.1x base) prevents catastrophic forgetting while enabling adaptation.

What the Research Says
----------------------

Diving deeper into the [Nested Learning](https://www.k-a.in/nl.html) framework that underlies HOPE:

**Memory as Associative Memory Module**: The paper recasts momentum in optimizers as an optimization problem. Adam emerges as the optimal form when you solve for "what's the best way to compress gradient information across time?"

**Frequency-Based Hierarchy**: Components ordered by update frequency naturally emerge in optimal learning systems. Fast components handle immediate context; slow components consolidate long-term patterns.

**Delta Gradient Descent**: An alternative memory mechanism with adaptive information retention:
```
W_{t+1} = W_t[I - η'_t x_t x_t^T] - η'_t ∇L ⊗ x_t
```
The first term handles selective forgetting of outdated information.

The key insight: current LLMs have "anterograde amnesia" - frozen long-term knowledge without mechanisms to incorporate new experiences. HOPE addresses this, but only if the training setup allows memory to actually learn.

Lessons Learned
---------------

1. **Defaults matter enormously**: A disabled-by-default internal loss meant memory never trained. Always audit defaults for experimental features.

2. **Read the critical papers**: The original Titans paper is optimistic. "Titans Revisited" identifies real failure modes. Both perspectives are necessary.

3. **Frozen backbone isn't free**: Fine-tuning only adapter layers is memory-efficient but creates representation mismatches. The backbone's projections need to adapt too.

4. **Test-time learning needs good initialization**: Memory that updates during inference is only as good as its starting point. If template weights are random, test-time updates start from chaos.

5. **Gate collapse is a symptom, not the cause**: I spent too long adding gate regularization. The gate collapsed because memory was useless, not the other way around.

6. **Multi-frequency helps less than expected**: CMS with 3 levels didn't significantly outperform single-level memory in my experiments. The paper shows benefits at longer contexts (32K+), which I haven't tested yet.

Results After the Fix
---------------------

After enabling internal loss and training the projections, comparison against baseline:

| Metric | Baseline (no TITANS) | TITANS (trained) | Improvement |
|--------|----------------------|------------------|-------------|
| Overall PPL | 24.89 | 21.43 | **-13.9%** |
| Early PPL | 23.84 | 21.90 | -8.1% |
| Middle PPL | 26.07 | 21.89 | -16.0% |
| Late PPL | 24.76 | 20.64 | **-16.6%** |
| Early→Late trend | -3.8% (worse) | +5.7% (better) | Reversed |

Key findings:

1. **TITANS improves all positions**: Every segment shows lower perplexity.
2. **Biggest gains at late positions**: 16.6% improvement confirms memory accumulates useful context.
3. **Trend reversal**: Baseline degrades as context grows (-3.8%); TITANS improves (+5.7%).
4. **14% overall improvement**: From 24.89 to 21.43 perplexity.

The baseline getting *worse* at later positions (-3.8%) while TITANS gets *better* (+5.7%) is exactly what "learning to memorize at test time" should produce. Memory provides compounding benefits as context accumulates.

The internal loss fix was critical - without training the key/value/query projections, the memory couldn't learn meaningful associations.

Updated Training Recipe
-----------------------

For fine-tuning a base model with HOPE memory:

```bash
uv run python -m nanogpt_titans.train_mlx \
    --model_name Qwen/Qwen2-0.5B \
    --max_steps 1000 \
    --segment_len 1024 \
    --memory_layers "8,12,16" \
    --use_internal_loss \
    --internal_loss_weight 0.1 \
    --unfreeze_backbone_layers 1 \
    --gate_min_value 0.15 \
    --gate_reg_weight 1.0
```

Key flags:
- `--use_internal_loss`: Train memory templates
- `--internal_loss_weight 0.1`: Strong memory signal
- `--unfreeze_backbone_layers 1`: Let backbone adapt
- `--gate_min_value 0.15`: Prevent complete gate collapse

What's Next
-----------

- Longer context evaluation (8K, 16K, 32K) where memory benefits should be clearer
- Comparison: frozen vs partially-unfrozen backbone
- Needle-in-haystack benchmarks to test retrieval
- Investigating Delta Gradient Descent as alternative to momentum-based updates

The journey from paper to working implementation revealed gaps that no amount of code review could catch. Only end-to-end evaluation exposed the training dynamics issues. Sometimes you have to run the experiment to understand what the equations really mean.

Code at [github.com/XiaoConstantine/nanogpt-titans](https://github.com/XiaoConstantine/nanogpt-titans).
