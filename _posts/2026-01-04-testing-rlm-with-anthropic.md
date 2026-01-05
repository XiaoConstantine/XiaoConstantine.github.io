---
layout: post
title: "Exploring Recursive Language Models"
---

I've been exploring [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601), a new inference paradigm from MIT's OASYS lab. The core idea is compelling: instead of forcing LLMs to process massive contexts in a single forward pass, give them a Python REPL and let them programmatically interact with the data.

The Problem RLM Solves
----------------------

Even with 200K+ context windows, LLMs struggle with certain long-context tasks. The issue isn't just attention - it's that some tasks require *computation* over data, not just pattern matching.

Consider: "Count how many items in this 100K-token dataset have label 'incorrect'."

A baseline LLM response:
```
I need to classify each of the 10 literal interpretations as either
correct or incorrect.

1. "The n... [truncated - never reaches answer]
```

The model starts *explaining* what it would do instead of *doing* it.

The RLM Paradigm
----------------

RLM replaces `llm.completion(prompt)` with a REPL-based interaction loop:

1. **Context as Variable**: The input is loaded as a Python variable the model can access
2. **Code Execution**: Model writes Python to examine/process the data
3. **Recursive Calls**: Model can invoke `lm(prompt)` for sub-queries on chunks
4. **Explicit Termination**: `FINAL(answer)` signals completion

```python
from rlm import RLM

rlm = RLM(
    backend="anthropic",
    backend_kwargs={"model_name": "claude-sonnet-4-5-20250929"},
    environment="local",
    max_depth=1,
    max_iterations=30,
)

result = rlm.completion(context, root_prompt="Find the secret code")
print(result.response)  # Direct answer, not explanation
```

The `max_depth` parameter controls recursion. At depth 0, the model can spawn sub-LM calls. Those sub-calls (depth 1) become regular LLM calls without further recursion.

Empirical Findings: OOLONG Benchmark
------------------------------------

I ran experiments on the OOLONG benchmark (long-context understanding from HuggingFace) comparing RLM vs direct prompting with Claude Sonnet 4.5.

**Results (10 tasks):**

| Metric | Baseline | RLM |
|--------|----------|-----|
| Real Accuracy | 0% | 100% |
| Avg Latency | 7.5s | 30s |
| Input Tokens | 7.5K | 162K |
| Output Tokens | 3.6K | 10.4K |

The baseline reported 70% accuracy, but every "correct" answer was a false positive - the model was explaining the task, not answering it, and evaluation was matching keywords in explanations.

**Every baseline response followed this pattern:**
```
I need to classify each of the 10 literal interpretations...
[proceeds to explain methodology, never produces answer]
```

**Every RLM response produced structured output:**
```
Label: incorrect
```
```
Answer: 7
```
```
User: 44106
```

The REPL forces execution over explanation.

How RLM Trajectories Work
-------------------------

Looking at the execution logs, a typical RLM trajectory:

**Iteration 1**: Model examines context structure
```python
lines = context.split('\n')
print(f"Found {len(lines)} lines")
print(lines[0])  # Sample first line
```

**Iteration 2-3**: Model parses and classifies
```python
results = []
for line in lines:
    parts = line.split(' <--> ')
    # Classification logic
    results.append(label)
```

**Iteration 4**: Model aggregates and answers
```python
count = sum(1 for r in results if r == 'incorrect')
FINAL(f"Answer: {count}")
```

The model operates as a programmer, not an oracle.

Model Behavior Differences: Haiku vs Sonnet
-------------------------------------------

Testing with Claude Haiku on the S-NIAH (needle-in-haystack) benchmark revealed a striking pattern (somewhat expected):

**S-NIAH Results (10 tasks):**

| Metric | Baseline | RLM |
|--------|----------|-----|
| Accuracy | 100% | 40% |
| Avg Latency | 1.2s | 7.5s |

Wait - RLM made Haiku *worse*? Yes. The baseline Haiku directly answered simple retrieval questions. But RLM Haiku couldn't follow the REPL format.

**Expected RLM output:**
```
162934
```

**Actual Haiku RLM output:**
```markdown
# The Special Number

The special number is **162934**.

## Why It's Special

According to the text...
```

Haiku defaulted to its conversational training - producing markdown-formatted explanations with headers, bold text, and prose. It found the answer but wrapped it in unparseable formatting.

More examples from the same run:

```markdown
# Hidden Message Analysis

## Key Finding

There **is** a hidden message explicitly stated in the te...
```

```markdown
# Puzzle Identification and Answer

**Puzzle Mentioned:**
"The answer to the puzzle is: constellatio...
```

The model knows the answer but can't resist explaining it. Every wrong answer contained the correct value buried in prose.

**Sonnet reliably:**
- Wrote valid, executable Python
- Used the `context` variable correctly
- Produced parseable structured outputs
- Followed system prompt's REPL conventions

RLM requires the model to operate as a "code agent" - understanding it's in a programmatic environment, not a conversational one. Haiku's stronger conversational instincts override the REPL context, making it paradoxically worse at simple tasks when given programmatic tools.

Technical Deep-Dive: Implementation Bugs
----------------------------------------

Exploring the codebase revealed several edge cases:

**1. FINAL pattern in code blocks**

The original regex matched `FINAL()` anywhere:
```python
final_pattern = r"FINAL\((.*?)\)"  # Too greedy
```

This caused premature termination when models wrote code referencing the pattern:
```markdown
```python
# Prepare for FINAL_VAR(result)
result = process_data()
```
```

Fix: anchor to line start:
```python
final_pattern = r"^\s*FINAL\((.*?)\)"
```

**2. Streaming required for Anthropic**

Anthropic's API requires streaming for outputs exceeding ~21K tokens - a non-obvious constraint that causes silent failures without proper handling.

The Token Economics
-------------------

RLM's accuracy improvement comes at a cost:

- **4x latency** - REPL iteration overhead
- **21x input tokens** - Recursive context reprocessing
- **3x output tokens** - Code generation + execution

For tasks where baseline accuracy is effectively 0%, the tradeoff is justified. The baseline doesn't just perform worse - it *can't answer at all*.

When to Use RLM
---------------

RLM shines when tasks require:
1. **Counting/aggregation** over large datasets
2. **Multi-step computation** that can't be done in-head
3. **Structured extraction** from noisy contexts
4. **Verification** of intermediate results

RLM is overkill for:
1. Simple retrieval (needle-in-haystack with one needle)
2. Summarization tasks
3. Anything a baseline LLM can answer directly

Key Insights
------------

1. **Baseline LLMs explain; RLM executes** - The fundamental difference. Direct prompting on complex tasks triggers analysis mode.

2. **Instruction-following is critical** - The model must maintain REPL context across iterations. Weaker models lose this frame.

3. **Evaluation metrics lie** - Substring matching in explanations creates false positives. Always inspect actual outputs.

4. **Programmers, not oracles** - RLM reframes LLMs as agents that write code to answer questions, rather than pattern-matching to produce answers.

The paradigm shift is subtle but significant: instead of asking "what is the answer?", RLM asks "what code would compute the answer?" For tasks requiring computation over data, this is the right abstraction.

---

*RLM: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) | Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)*
