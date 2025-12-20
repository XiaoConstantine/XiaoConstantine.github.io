---
layout: post
title: Implementing Titans - Learning to Memorize at Test Time
---

Context engineering has become extreme intriguing to me recently, as hands on building couple of agentic platform project, one asepct of context engineering is memory and continous learning, Manus shared an excellent learning: [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) from harness perspective,
Google Research dropped a paper in January 2025 that caught my attention: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663). The core idea is elegant - give transformers a learnable memory that updates during inference, not just training. I decided to implement it from scratch to understand it deeply.

This post documents the journey, 0 hands on experience training model to get hands on, working to optimize training for performance: understanding the algorithm, building a working implementation, validating correctness, and optimizing for GPU performance - starting on Kaggle's free T4x2 GPUs and eventually moving to an L4.

The Core Insight
----------------

Standard transformers have a fundamental limitation: their "memory" is just the KV cache, which grows linearly with sequence length. Process a 100K token document, and you're storing 100K key-value pairs. Titans proposes something different: a neural network that *learns* what to remember.

The architecture has three memory types working together:

1. **Short-term memory**: Standard attention over recent tokens (sliding window)
2. **Long-term memory**: A small MLP that gets updated token-by-token based on "surprise"
3. **Persistent memory**: Learned parameters that act as constant context

The magic is in the long-term memory. It's not a fixed cache - it's weights that evolve during inference using gradient descent.

Understanding Surprise-Based Updates
------------------------------------

The key equation from the paper defines how memory updates:

```
M_t = M_{t-1} - η * ∇L(M_{t-1}; x_t) + momentum_t
```

Where:
- `M_t` is the memory MLP weights at time t
- `η` is a learned learning rate (per-parameter)
- `∇L` is the gradient of "surprise" - how poorly the memory predicted the current token
- `momentum_t` accumulates past gradients with exponential decay

The surprise is measured as MSE between what the memory retrieved and what it should have retrieved. High surprise = large gradient = big memory update. Boring, predictable tokens barely change the memory.

This creates an elegant information filter: the memory naturally learns to store surprising, potentially useful information while forgetting predictable patterns.

Implementation Challenges
-------------------------

**Challenge 1: Per-token gradients**

The paper requires computing gradients for each token independently. PyTorch's autograd computes gradients for the whole batch. I needed `vmap` (vectorized map) to parallelize per-token gradient computation:

```python
def per_token_grad(memory_params, single_x, single_q):
    pred = memory_forward(memory_params, single_q)
    loss = F.mse_loss(pred, single_x)
    return torch.autograd.grad(loss, memory_params)

# Vectorize over batch and sequence dimensions
batched_grad = torch.vmap(torch.vmap(per_token_grad))(params, x, queries)
```

**Challenge 2: Momentum across time**

The momentum term accumulates across the entire sequence. The naive approach is a for-loop:

```python
momentum = prev_momentum
for t in range(seq_len):
    momentum = beta * momentum + grad[t]
    updates[t] = momentum
```

This is painfully slow on GPU. The fix: recognize this as a parallel scan (prefix sum with custom operator). I implemented it using `torch.cumsum` with a trick:

```python
# Compute beta^(t-i) weights for all positions
powers = beta ** torch.arange(seq_len)
weights = powers.unsqueeze(0) / powers.unsqueeze(1)  # [t, i] = beta^(t-i)
weights = torch.tril(weights)  # Zero out future positions

# Parallel momentum: momentum[t] = sum_i(beta^(t-i) * grad[i])
momentum = torch.einsum('ti,btd->btd', weights, grads)
```

**Challenge 3: Memory causality**

The trickiest part: when retrieving from memory at token t, you must use memory state *before* seeing token t. The paper handles this by processing in segments:

```
Segment 1: tokens [0:128]   → retrieve using M_init, update to M_1
Segment 2: tokens [128:256] → retrieve using M_1, update to M_2
...
```

Within a segment, all tokens retrieve from the same memory state. This was the source of several bugs during development.

Validating Against the Paper
----------------------------

After getting the implementation running, I wrote tests to verify each component matched the paper's algorithm:

1. **Memory loss is MSE**: Verified the gradient computation matched `∇_M ||M(q) - v||²`
2. **Momentum accumulation**: Checked that momentum correctly persists across segments
3. **Weight update direction**: Confirmed updates use gradient *descent* (negative gradient)
4. **Retrieval causality**: Ensured no future information leaks into past token representations

The test suite has 59 tests covering edge cases: single tokens, batch size one, exact segment boundaries, zero gradients, and numerical stability.

One subtle bug I caught: the initial implementation used `M_t` for both retrieval and update computation at time t. The paper specifies using `M_{t-1}` for retrieval. Small difference, significant impact on training dynamics.

Hardware Journey
----------------

I started development on Kaggle's free tier: two Tesla T4 GPUs with 16GB VRAM each. The T4 is a solid workhorse but has limitations:

- **No bfloat16**: T4 only supports float16, which is less numerically stable
- **15GB usable VRAM**: After PyTorch overhead, you get ~15GB per GPU
- **Older architecture**: Turing (2018) vs newer Ampere/Ada

For initial development and validation, T4x2 was sufficient. I could train a 31M parameter model with batch size 8 and sequence length 1024. But scaling to 130M parameters hit VRAM limits.

The move to an L4 (24GB VRAM, Ada Lovelace architecture) unlocked:
- bfloat16 training for better stability
- Larger batch sizes
- Room for the 130M model with fused optimizations

Optimization Journey (with Claude)
----------------------------------

Here's where things got fun. I used Claude Code as a pair programmer for the optimization phase. The workflow looked like this:

1. Run training with PyTorch profiler
2. Share the profile output with Claude
3. Claude identifies bottlenecks and suggests fixes
4. Implement, profile again, repeat

This turned out to be surprisingly effective. Claude could parse profiler output, spot patterns I missed, and suggest optimizations from its knowledge of Triton, CUDA, and ML systems papers.

For example, when I shared this profile snippet:
```
aten::cat                    10.7ms    5.7%
aten::copy_                   8.2ms    4.4%
```

Claude immediately flagged the `torch.cat` as suspicious - it's creating new tensors instead of writing to pre-allocated buffers. The fix was straightforward once identified.

The first working version on T4 was slow. Profiling revealed the bottlenecks:

| Operation | Time | Issue |
|-----------|------|-------|
| `aten::cat` | 10.7ms | Concatenating memory context with input |
| `fused_weight_update` | 60.5ms | Triton kernel for parameter updates |
| `flex_attention` | 11.1ms | PyTorch's compiled attention |

**Optimization 1: Eliminate torch.cat**

The original code concatenated memory context, persistent memory, and input tokens:

```python
context = torch.cat([mem_context, persist, x], dim=1)  # Slow!
```

Fix: pre-allocate the buffer and copy in-place:

```python
context = torch.empty(b, prefix_len + t, c, device=x.device, dtype=x.dtype)
context[:, :num_longterm] = mem_context
context[:, num_longterm:prefix_len] = persist
context[:, prefix_len:] = x
```

Result: `aten::cat` disappeared from the profile.

**Optimization 2: Fused Linear Cross-Entropy**

I asked Claude to research how other projects handle cross-entropy at scale. It dug through [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), [Unsloth](https://github.com/unslothai/unsloth), and tinygrad, then synthesized the key insight: don't materialize the full logits tensor.

For a 130M parameter model:

- Full logits: `batch × seq × vocab` = 8 × 1024 × 50304 × 4 bytes = **1.6 GB**
- Fused approach: process in 1024-token chunks = **~200 MB peak**

The implementation uses a Triton kernel for cross-entropy on each chunk:

```python
for start in range(0, N, chunk_size):
    chunk_hidden = hidden[start:end]
    chunk_logits = chunk_hidden @ weight.T + bias
    chunk_losses = triton_cross_entropy(chunk_logits, chunk_targets)
    total_loss += chunk_losses.sum()
```

The 64x memory reduction allows training larger models on the same GPU.

**Optimization 3: Triton Kernels**

Writing Triton kernels from scratch is tedious. Claude helped here too - I'd describe what I wanted ("fused weight update with momentum and decay") and it would generate the kernel, explain the memory access patterns, and suggest block sizes.

Custom Triton kernels for hot paths:

- **Fused weight update**: Combines momentum, learning rate, and weight decay in one kernel
- **Cross-entropy forward**: Numerically stable log-softmax + NLL
- **LayerNorm**: Fused mean/variance computation
- **Linear + SiLU**: Fused matmul with activation

The Triton kernels avoid memory round-trips that kill performance on GPU. When one kernel had a bug (wrong reduction dimension), Claude could trace through the indexing math and spot the issue faster than I could.

**Optimization 4: FlexAttention**

PyTorch's `flex_attention` with prefix-LM masking. Memory context and persistent memory tokens can attend to each other bidirectionally, while input tokens attend causally. This matches the MAC (Memory as Context) architecture from the paper.

Results
-------

**On Kaggle T4x2 (31M model):**

| Metric | Value |
|--------|-------|
| Parameters | 31M |
| Context length | 1024 |
| Segment length | 128 |
| Batch size | 8 |
| VRAM usage | ~12GB per GPU |
| Training throughput | ~8k tokens/sec |

The T4 setup was good for validation and debugging. I could iterate quickly on the algorithm implementation without waiting for expensive GPU time.

**On L4 (130M model):**

| Metric | Value |
|--------|-------|
| Parameters | 130M |
| Context length | 1024 |
| Segment length | 128 |
| Memory MLP size | 768 → 512 → 768 |
| Batch size | 8 |
| VRAM usage | ~18GB |
| Training throughput | ~12k tokens/sec |

The memory system shows the expected behavior:
- Surprise decreases over time as the model learns what to expect
- Memory updates are larger for rare tokens
- Long-range dependencies improve compared to baseline GPT

Lessons Learned
---------------

1. **Read the paper twice**: The first pass gives intuition, the second reveals implementation details hiding in subscripts and footnotes.

2. **Test at the algorithm level**: Don't just test "does it run" - test "does momentum accumulate correctly across segments". These tests caught real bugs.

3. **Profile before optimizing**: My intuition about bottlenecks was often wrong. The `torch.cat` overhead surprised me - it looked innocent but was 10ms per forward pass.

4. **Memory vs compute tradeoffs**: The fused cross-entropy adds ~10ms of compute overhead but saves 1.4GB of memory. Worth it for larger models, not for small ones.

5. **Triton is worth learning**: Writing custom kernels for hot paths gives 2-10x speedups. The learning curve is steep but pays off.

6. **Start cheap, scale up**: Kaggle's free T4x2 was perfect for algorithm validation. Once the implementation was correct, I moved to L4 for scaling experiments. Don't pay for expensive GPUs while debugging.

7. **AI pair programming works for optimization**: Claude Code was genuinely useful for the profiling loop. It can parse profiler output, knows common CUDA/Triton patterns, and can research how other projects solved similar problems. The iteration speed was noticeably faster than solo debugging.

What's Next
-----------

- Scaling experiments: How does memory capacity need to grow with model size?
- Longer sequences: Testing on 32K+ contexts where the memory advantage should be most apparent
- Different architectures: The paper proposes MAC, MAG, and MAL variants - currently only MAC is implemented

Code is at [github.com/XiaoConstantine/nanogpt-titans](https://github.com/XiaoConstantine/nanogpt-titans).
