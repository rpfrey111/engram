# Engram: Neuromorphic Memory Architecture for LLMs

**Date:** 2026-03-03
**Status:** Approved
**Author:** Ryan Frey (vision) + Claude (architecture)

## Problem Statement

Current RAG and LLM data interaction is fundamentally broken compared to how the human brain handles memory and retrieval:

- **Flat retrieval**: cosine similarity returns disconnected chunks with no relationships
- **No memory layers**: everything is either in context or gone
- **No consolidation**: data sits as raw chunks forever, never compressed or reorganized
- **No salience**: every query gets the same retrieval depth regardless of importance
- **No schema compression**: repetitive patterns stored in full every time
- **No forgetting**: storage grows forever, retrieval quality degrades
- **No prediction gating**: retrieves on every query even when the model already has the answer

## Vision

Build a general-purpose, open-source memory architecture for LLMs that mirrors the human brain's proven mechanisms. Every component maps 1:1 to a neuroscience principle.

## Deliverables

1. **Engram library**: Rust core with Python and TypeScript SDKs
2. **Claude Code skill**: teaching brain-inspired approaches to RAG/compression/LLM data design

## Tech Stack

- **Rust core**: memory engine (graph + vector + consolidation). Brain demands speed and parallelism.
- **Python SDK**: primary interface (AI community)
- **TypeScript SDK**: secondary interface (web developers)
- **Embedded-first**: working memory + engram index run in-process. Cortical store can be local or cloud. Consolidation runs as background process.

---

## Architecture

### The Memory Node (the "Neuron")

The fundamental unit. Every piece of data becomes a memory node, rich and multi-dimensional.

```
MemoryNode {
    // Identity
    id:             UUID
    created_at:     Timestamp
    last_accessed:  Timestamp
    access_count:   u32

    // Content
    content:        String              // raw content
    embedding:      Vec<f32>            // vector representation
    content_type:   Enum [text, code, conversation, event, fact, skill, entity]

    // Encoding Context (where/when/who at encoding time)
    source:         String
    session_id:     Option<UUID>
    encoding_state: HashMap<String, f32>

    // Associations (weighted edges to other nodes)
    edges: Vec<Edge> {
        target_id:      UUID
        relation_type:  Enum [semantic, temporal, causal, contextual, hierarchical]
        weight:         f32     // 0.0 to 1.0, strengthened by co-activation (LTP)
        created_at:     Timestamp
    }

    // Importance (somatic marker equivalent)
    salience:       f32         // 0.0 to 1.0
    decay_rate:     f32         // how fast this memory fades without reinforcement

    // Abstraction hierarchy
    abstraction_level: Enum [raw, chunk, summary, schema]
    parent_id:      Option<UUID>
    children_ids:   Vec<UUID>
}
```

**Design decisions from neuroscience:**
- **Somatic markers become salience scores** (Damasio): high-salience nodes resist forgetting and get retrieved preferentially
- **Encoding specificity built in** (Tulving): encoding_state captures context at storage time for context-matched retrieval
- **Typed edges**: semantic, temporal, causal, contextual, hierarchical. Each is a different retrieval pathway.
- **Abstraction hierarchy**: raw, chunk, summary, schema. Linked via parent/children.

---

### The Five Memory Systems

#### 1. Sensory Buffer (Input Processor)

**Brain:** Sensory register. Holds raw input for milliseconds.

- Capacity: unlimited (streaming)
- Lifetime: milliseconds
- Job: parse input, generate embedding, extract entities/relationships, assign initial salience, create memory node, pass to salience router
- Always encodes deeply (Craik & Lockhart levels of processing)

#### 2. Working Memory (Active Buffer)

**Brain:** Prefrontal cortex working memory (Cowan's 4 +/- 1 chunks).

- Capacity: 4-7 memory constellations
- Lifetime: duration of current task/query
- Storage: in-process RAM
- Latency: sub-millisecond

Subsystems (from Baddeley's model):
- **Semantic buffer** (phonological loop): meaning/language
- **Structural buffer** (visuospatial sketchpad): relationships, schemas, code structure
- **Integration buffer** (episodic buffer): binds everything for the LLM

When full, lowest-salience item deactivates back to engram index.

#### 3. Engram Index (Hippocampus)

**Brain:** Hippocampus. Fast encoding, associative binding, rapid retrieval. NOT the permanent store.

- Capacity: tens of thousands of nodes
- Lifetime: session to days (before consolidation)
- Storage: in-process or sidecar, graph + vector hybrid
- Latency: single-digit milliseconds

The heart of the system. Stores nodes in an associative graph with typed, weighted edges. Retrieval uses spreading activation, not cosine similarity.

#### 4. Cortical Store (Long-term Memory)

**Brain:** Neocortex. Permanent, compressed, organized.

- Capacity: millions of nodes
- Lifetime: permanent (until pruned)
- Storage: local disk or cloud
- Latency: tens of milliseconds

Holds consolidated memories at multiple abstraction levels. Frequently co-activated nodes merge into schemas. Maps to semantic memory (facts) and episodic memory (experiences).

#### 5. Procedural Store (Skill Memory)

**Brain:** Basal ganglia + cerebellum. Automatic learned patterns.

- Capacity: thousands of patterns
- Lifetime: permanent
- Storage: compiled pattern index
- Latency: sub-millisecond

After seeing the same retrieval pattern repeatedly, encodes it as a pre-computed path. Fires instantly, bypassing full spreading activation. The "System 1" of retrieval.

---

### The Retrieval Engine (Spreading Activation)

#### Step 1: Multi-dimensional Query Encoding

```
RetrievalCue {
    semantic:       Vec<f32>            // embedding
    temporal:       TimeRange           // when relevant
    entities:       Vec<String>         // who/what mentioned
    context:        HashMap<String, f32> // current state
    intent:         Enum [recall, recognize, explore, verify]
    salience_floor: f32                 // minimum importance threshold
}
```

Intent modulates retrieval strategy (brain's PFC modulates hippocampal retrieval).

#### Step 2: Seed Activation

Four simultaneous matching strategies:
- Semantic match (vector similarity)
- Entity match (exact overlap)
- Temporal match (time window)
- Context match (encoding specificity)

Nodes matching on multiple dimensions get stronger seed activation.

#### Step 3: Spreading Activation

From seed nodes, activation spreads outward along edges:
- Activation diminishes by edge weight and depth
- Edge types get relevance boosts based on query intent
- Nodes reached by multiple paths accumulate activation (pattern completion)
- Depth limited to 2-3 hops typically
- Supports inhibitory (negative) edges for contradiction handling

#### Step 4: Constellation Assembly

Output is a MemoryConstellation, not a flat list:

```
MemoryConstellation {
    focal_nodes:    Vec<MemoryNode>     // highest activation
    context_nodes:  Vec<MemoryNode>     // supporting context
    relationships:  Vec<Edge>           // how they connect
    schemas:        Vec<Schema>         // activated patterns
    confidence:     f32                 // retrieval confidence
    coverage:       f32                 // how much of cue is addressed
}
```

#### Step 5: Prediction Gate

Before any retrieval, checks if working memory already covers the query:
- Coverage > 85%: skip retrieval entirely
- Coverage > 50%: targeted retrieval for gaps only
- Coverage < 50%: full retrieval

Estimated 40-60% reduction in retrieval calls for conversational contexts.

---

### The Consolidation Engine (Artificial Sleep)

Runs as background process during low-activity periods. Five operations:

#### 1. Replay and Transfer (hippocampus to cortex)

- Finds engram index nodes older than 1 hour with 2+ accesses
- Creates cortical store entries with higher-abstraction versions
- Links to existing schemas
- Reorganizes, not just copies

#### 2. Long-Term Potentiation (strengthen frequent paths)

- Finds edges co-activated during recent retrievals
- Strengthens weights (capped at 1.0)
- System literally learns associations from usage

#### 3. Long-Term Depression + Forgetting (prune the weak)

- Applies time-based decay weighted by salience
- High-salience nodes decay slowly, low-salience decay fast
- Prunes edges below minimum threshold
- Raw data below threshold: deleted (summary preserved). Higher abstractions: archived.

#### 4. Schema Formation (pattern abstraction)

- Finds clusters of similar nodes (5+ members, 0.8+ similarity)
- Creates or updates schema templates
- Replaces full content with delta from template
- Estimated 10-50x compression for repetitive data

#### 5. Procedural Extraction (skill learning)

- Analyzes retrieval log for recurring patterns
- Encodes optimal retrieval paths as pre-computed procedures
- Fires instantly on pattern match, bypassing full search

#### Schedule

| Operation | Frequency |
|---|---|
| Replay and transfer | Every 1 hour or on idle |
| Strengthen paths | Every 1 hour (with replay) |
| Decay and prune | Every 24 hours |
| Schema formation | Every 24 hours |
| Procedural extraction | Every 24 hours |

All operations idempotent and safely interruptible.

---

### The Salience Router (Attention System)

Maps to brain's three attention networks (Posner):
- **Alerting** (norepinephrine): salience scoring (novelty, urgency, emotional weight, entity importance)
- **Orienting** (acetylcholine): gap analysis against working memory
- **Executive** (dopamine): retrieval depth selection

#### Retrieval Strategies

| Salience | Coverage | Strategy |
|---|---|---|
| High, > 0.7 | Low, < 0.3 | Deep: 4 hops, 50 nodes, schemas + cortical |
| High, > 0.7 | Partial, 0.3-0.7 | Targeted: 2 hops, 20 nodes, gap-focused |
| Low, < 0.3 | Low, < 0.3 | Standard: 2 hops, 15 nodes |
| Any | High, > 0.85 | Skip: working memory handles it |
| Moderate | Moderate | Light: 1 hop, 10 nodes |

#### Context Switch Detection

- Divergence > 0.8 from working memory focus: full context switch (archive + clear)
- Divergence > 0.4: partial shift (deprioritize stale items)
- Below 0.4: same topic, continue

Prevents attention residue (Leroy) from polluting context.

#### Metacognitive Output

Every retrieval includes confidence, coverage, and identified gaps. The LLM knows what Engram doesn't know, preventing hallucination.

---

### The Context Compiler (Token Efficiency)

Three compression strategies:

#### 1. Schema Compression (Bartlett + Schank/Abelson)

Recurring patterns stored as template + deltas. Example: 3 similar support tickets become one pattern description with per-incident deltas. 75%+ token reduction.

#### 2. Hierarchical Abstraction (Tulving)

Pick abstraction level based on relevance:
- Relevance > 0.8: full raw content
- Relevance > 0.5: summary
- Relevance > 0.2: one-line schema reference
- Below 0.2: omit

#### 3. Relational Compression (Gestalt)

Encode relationships as structured context maps instead of disconnected chunks. More information in fewer tokens.

#### Compilation Pipeline

Works within a token budget:
1. Relationship map first (highest information density)
2. Active schemas (compressed patterns)
3. Focal memories at appropriate abstraction level
4. Context memories as summaries if budget allows
5. Metadata: confidence, coverage, gaps (always included)

#### Estimated Compression

| Scenario | RAG tokens | Engram tokens | Reduction |
|---|---|---|---|
| Simple fact recall | 500-1500 | 50-200 | 70-90% |
| Multi-document question | 2000-5000 | 300-800 | 75-85% |
| Recurring pattern (schemas) | 2000-10000 | 200-500 | 90-95% |
| Exploratory/broad query | 3000-8000 | 500-1500 | 60-80% |

---

## Complete System Flow

```
INPUT
  |
  v
Sensory Buffer -------- encodes multi-dimensional memory nodes
  |
  v
Salience Router ------- scores importance, picks retrieval strategy
  |
  v
Prediction Gate ------- skip retrieval if working memory suffices
  |
  v
Working Memory -------- 4-7 active constellations, sub-ms
  |
  v
Engram Index ---------- fast associative graph, spreading activation
  |
  v
Cortical Store -------- compressed long-term, multi-abstraction
  |
  v
Procedural Store ------ learned retrieval patterns, auto-fire
  |
  v
Context Compiler ------ schema + hierarchical + relational compression
  |
  v
OUTPUT TO LLM --------- structured context with relationships,
                        confidence, and gaps

    +---------------------------+
    |   CONSOLIDATION ENGINE    |  (background, "sleep")
    |                           |
    |  Replay & Transfer        |  hippocampus -> cortex
    |  LTP (strengthen)         |  frequent paths get stronger
    |  LTD + Forgetting         |  unused paths decay
    |  Schema Formation         |  patterns get abstracted
    |  Procedural Extraction    |  skills get compiled
    +---------------------------+
```

## Brain-to-System Mapping Reference

| Brain Mechanism | Researcher | Engram Component |
|---|---|---|
| Sensory register | Atkinson & Shiffrin | Sensory Buffer |
| Working memory (4 +/- 1) | Baddeley, Cowan | Working Memory |
| Hippocampal binding | Tulving, Squire | Engram Index |
| Neocortical storage | Squire & Alvarez | Cortical Store |
| Procedural memory | basal ganglia research | Procedural Store |
| Spreading activation | Collins & Loftus | Retrieval Engine |
| Somatic markers | Damasio | Salience scores |
| Encoding specificity | Tulving & Thomson | Encoding context matching |
| Levels of processing | Craik & Lockhart | Deep encoding in sensory buffer |
| Sleep consolidation | Born & Wilhelm | Consolidation engine |
| LTP | Bliss & Lomo | Path strengthening |
| LTD + synaptic homeostasis | Tononi & Cirelli | Decay and pruning |
| Schemas/scripts | Bartlett, Schank & Abelson | Schema formation + compression |
| Predictive processing | Friston | Prediction gate |
| Salience network | Posner, Menon | Salience router |
| Attention residue | Leroy | Context switch detection |
| Metacognition | Flavell | Confidence/coverage/gaps output |
| Chunking | Miller, Simon | Multi-level abstraction |
| Forgetting curve | Ebbinghaus | Adaptive decay |
| Dual process (System 1/2) | Kahneman | Procedural shortcircuit vs full retrieval |
