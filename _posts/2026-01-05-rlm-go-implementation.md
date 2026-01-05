---
layout: post
title: "Exploring RLM Part 2: Context Engineering for Coding Agents"
---

In [Part 1](/2026/01/04/testing-rlm-with-anthropic.html), I explored the RLM paradigm - giving LLMs a REPL to programmatically interact with data instead of processing massive contexts in a single forward pass. This post covers how I've been applying RLM and other techniques to manage context in my day-to-day work with coding agents like Claude Code.

The Context Problem
-------------------

After months of using Claude Code daily, context management has become my primary productivity constraint. A typical session involves:

- Reading multiple source files for understanding
- Analyzing test outputs and error logs
- Cross-referencing documentation
- Iterating on implementations

Each of these consumes context. A 200K token window sounds generous until you're debugging a failing integration test with logs, source files, and stack traces all competing for space.

The symptoms are familiar: the agent starts forgetting earlier context, makes inconsistent suggestions, or the session hits auto-summarization and loses nuance. The goal isn't just fitting more in - it's using context *efficiently*.

Technique 1: Subagents for Isolation
------------------------------------

Claude Code's Task tool spawns subagents - isolated contexts that execute specific tasks and return summarized results. This is the most impactful technique I've adopted.

**Before subagents:**
```
Main context: [file1] + [file2] + [file3] + [exploration] + [analysis] = bloated
```

**With subagents:**
```
Main context: [high-level task]
  └─ Subagent 1: [file1 exploration] → returns: "Found X at line 42"
  └─ Subagent 2: [file2 analysis] → returns: "Pattern Y detected"
Main context: [summaries only]
```

The key insight: subagents carry their own context window. When they complete, only their *conclusions* merge into the main conversation - not all the files they read to reach those conclusions.

I now reflexively use subagents for:
- **Codebase exploration**: "Find where authentication is handled"
- **Code review**: Spawn a reviewer agent that returns findings, not full file contents
- **Parallel searches**: Multiple agents searching different areas simultaneously

The pattern has become: **explore in subagents, synthesize in main context**.

Technique 2: RLM for Large File Processing
-------------------------------------------

Some tasks require processing files too large to efficiently handle even in subagents. Log analysis, dataset exploration, multi-file aggregation - these benefit from RLM's programmatic approach.

I built [rlm-go](https://github.com/XiaoConstantine/rlm-go), a Go implementation that integrates with Claude Code as a skill. The architecture keeps the large context *outside* Claude's context window entirely:

```
┌──────────────────────────────────────────────┐
│              rlm-go Process                  │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐    │
│  │    Yaegi     │───►│   LLM Client     │    │
│  │  Interpreter │    │                  │    │
│  │              │    └──────────────────┘    │
│  │ - context    │                            │
│  │ - Query()    │◄── direct function call    │
│  │ - fmt.*      │                            │
│  └──────────────┘                            │
└──────────────────────────────────────────────┘
         │
         ▼ returns only: final answer
┌──────────────────────────────────────────────┐
│              Claude Code                     │
│  [main conversation context - file never     │
│   loaded, only sees RLM output]              │
└──────────────────────────────────────────────┘
```

Claude Code invokes `/rlm` with a file path and query. The rlm-go process loads the file, lets an LLM iteratively explore it via code execution, and returns only the final answer. The main Claude Code context never sees the raw file.

**Measured token savings:**

| Context Size | Direct Read | RLM | Savings |
|--------------|-------------|-----|---------|
| 5KB | 29,184 | 30,089 | -3% (overhead) |
| 716KB | 54,429 | 32,983 | **40%** |

The crossover point is around 50KB. Below that, RLM's overhead (system prompts, iteration messages) exceeds savings. Above it, the programmatic approach wins decisively.

**When I use /rlm:**
- Analyzing CI logs after test failures
- Exploring large JSON/JSONL datasets
- Aggregating patterns across multiple log files
- Any "find all X in this large file" task

Technique 3: Strategic File Reading
-----------------------------------

Not every file needs full reading. Claude Code's Read tool accepts `offset` and `limit` parameters. For large files where I only need specific sections:

```
Read first 100 lines: Read(file, limit=100)
Read lines 500-600: Read(file, offset=500, limit=100)
```

Combined with Grep to locate relevant sections first, this avoids loading entire files when only fragments matter.

The pattern: **grep to locate, read to extract** - not "read everything and search in-context."

Technique 4: Explicit Context Boundaries
----------------------------------------

When starting complex tasks, I've started being explicit about what context to preserve:

- "Analyze this file, then forget it - I only need the conclusion"
- "Search for X, summarize findings, we won't need the raw results"
- "Start a subagent for this exploration"

This signals to the agent (and myself) what's reference material versus persistent state. It's a mental model shift: treating context as a resource to manage, not infinite scratch space.

Putting It Together
-------------------

My current workflow for a complex debugging session:

1. **Spawn exploration subagent**: "Find all files related to authentication"
2. **Subagent returns**: Summary of 5 relevant files with line references
3. **Read specific sections**: Only the functions/classes identified, not full files
4. **For large logs**: `/rlm logs/auth.log "find authentication failures and their causes"`
5. **RLM returns**: Categorized findings without the 500KB log in context
6. **Synthesize in main context**: Work with summaries and conclusions

The main context stays focused on the *task*, not the *exploration*. Files are processed once in isolated contexts, and only conclusions persist.

Installation
------------

For the RLM skill:

```bash
# Install rlm-go
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/rlm-go/main/install.sh | bash

# Install Claude Code skill
~/.local/bin/rlm install-claude-code
```

Key Takeaways
-------------

1. **Subagents are the biggest win** - Isolating exploration from synthesis keeps main context clean. Use them aggressively.

2. **RLM for large files** - When context is too large even for subagents, offload to RLM. The 40% token savings on large files compound over a session.

3. **Read strategically** - Grep first, read sections, avoid full-file loads when possible.

4. **Context is a resource** - Treat the context window like memory. Be explicit about what to keep and what to discard.

5. **Match technique to scale** - Small files: read directly. Medium exploration: subagents. Large files: RLM. Each has its place.

The goal isn't maximum context utilization - it's maximum *useful* context. These techniques help keep the signal-to-noise ratio high as sessions grow complex.

---

*rlm-go: [github.com/XiaoConstantine/rlm-go](https://github.com/XiaoConstantine/rlm-go) | Part 1: [Exploring Recursive Language Models](/2026/01/04/testing-rlm-with-anthropic.html)*
