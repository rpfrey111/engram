# Engram Working Model: LLM Integration Design

**Date:** 2026-03-03
**Status:** Approved
**Builds on:** `2026-03-03-engram-design.md` (core architecture)

## Goal

Take the existing Engram Rust framework and make it a **working memory layer for a real LLM**. A conversational agent that can ingest knowledge AND remember conversation history — episodic + semantic memory working together.

## What's Built (Phases 1-3)

- MemoryNode, Edge, Constellation, Cue types
- GraphStore (in-memory HashMap)
- Vector similarity search (cosine)
- Spreading activation retrieval (BFS)
- Salience router (5 strategies: Deep/Targeted/Standard/Light/Skip)
- Context compiler (token budget aware)
- Consolidation (decay + LTP)
- Working memory (salience-based eviction)
- Engine orchestrator (bag-of-words placeholder embedding)
- Python SDK (basic PyO3 bindings)

## What's Missing for a Working Model

| Component | Design Doc Section | Brain Analog | Priority |
|---|---|---|---|
| Embedder trait + Ollama impl | Sensory Buffer | Sensory cortex | 1 |
| LLMProvider trait + Ollama impl | (new) | Prefrontal cortex | 2 |
| Sensory Buffer (deep encoding) | Memory System 1 | Sensory register | 3 |
| Persistence (serde to disk) | (implied by consolidation) | Sleep consolidation | 4 |
| CLI chat loop + episodic memory | Deliverable 1 | Consciousness loop | 5 |
| File ingestion pipeline | Deliverable 1 | Perceptual learning | 6 |
| Claude Code skill | Deliverable 2 | Teaching | 7 |

---

## Architecture

### Trait Abstractions

Following the design doc's embedded-first philosophy — pluggable backends with Ollama as default.

```rust
// The "Sensory Cortex" — transforms raw input into neural representations
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimensions(&self) -> usize;
}

// The "Prefrontal Cortex" — reasoning/language generation
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, prompt: &str, context: &LLMContext) -> Result<String>;
}
```

Default implementations: `OllamaEmbedder` (nomic-embed-text or mxbai-embed-large) and `OllamaLLM` (user's pulled model). Both call Ollama's HTTP API.

### Sensory Buffer (Deep Encoding)

Following Craik & Lockhart's levels of processing — input gets deeply encoded, not just embedded:

```
Input text
    → Generate embedding (Embedder trait)
    → Extract entities (named patterns, capitalized words)
    → Detect content type (code? conversation? fact?)
    → Score initial salience (novelty vs. existing graph)
    → Create edges to similar existing nodes
    → Assign encoding context (session, timestamp, source)
    → Output: fully-formed MemoryNode ready for graph insertion
```

Replaces the current `ingest()` which only does bag-of-words + insert.

### Persistence

The brain doesn't lose memories on sleep. Following Born & Wilhelm:

- Graph serialization: serde JSON to disk on shutdown, load on startup
- Storage path: `~/.engram/` directory with `graph.json` and `metadata.json`
- Incremental saves: after consolidation cycles (mimics sleep-based consolidation)

### Full Pipeline

```
USER INPUT
    │
    ▼
┌─────────────────────────────────────────────┐
│ SENSORY BUFFER                              │
│  embed(text) → extract entities → score     │
│  salience → classify content → create node  │
│  → store in graph with edges                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ SALIENCE ROUTER                             │
│  Score query salience → check working       │
│  memory coverage → select strategy          │
│  (Deep/Targeted/Standard/Light/Skip)        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ RETRIEVAL (Spreading Activation)            │
│  Find seeds (vector similarity) →           │
│  spread activation through graph →          │
│  assemble constellation (focal + context)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ CONTEXT COMPILER                            │
│  Token budget allocation → focal memories   │
│  at appropriate abstraction → relationship  │
│  map → confidence/coverage/gaps             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ LLM PROVIDER (Ollama)                       │
│  System prompt + compiled context +         │
│  user query → LLM generates response        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ EPISODIC ENCODING                           │
│  Store conversation turn as episodic        │
│  memory → link to query nodes (temporal +   │
│  contextual edges)                          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
RESPONSE TO USER

        ┌─────────────────────────────┐
        │ CONSOLIDATION (Background)  │
        │  LTP: strengthen co-active  │
        │  Decay: weaken unused       │
        │  Persist: save to disk      │
        └─────────────────────────────┘
```

### CLI Chat Agent

```
$ engram chat
🧠 Engram v0.1.0 | Memory: 847 nodes | Model: llama3.2

You: Tell me about quantum computing
[retrieval: Standard | seeds: 3 | spread: 12 nodes | 340 tokens | confidence: 0.72]