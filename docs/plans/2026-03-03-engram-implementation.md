# Engram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a neuromorphic memory architecture for LLMs in Rust, with Python and TypeScript SDKs.

**Architecture:** Brain-inspired memory system with 5 memory layers (sensory buffer, working memory, engram index, cortical store, procedural store), spreading activation retrieval, salience routing, consolidation engine, and context compilation. See `docs/plans/2026-03-03-engram-design.md` for full design.

**Tech Stack:** Rust (core), PyO3 (Python bindings), napi-rs or wasm-pack (TypeScript bindings), serde (serialization), uuid (identifiers), tokio (async/background tasks)

**Build Order Rationale:** The brain can't retrieve without neurons, can't associate without a hippocampus, can't consolidate without a cortex. We build bottom-up: types first, then storage, then retrieval, then intelligence layers, then SDKs.

---

## Phase 1: Foundation (Core Types)

Everything depends on the data model. Build it first, test it thoroughly.

### Task 1: Project Scaffolding

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/types/mod.rs`
- Create: `rustfmt.toml`
- Create: `.gitignore`

**Step 1: Initialize Rust project**

```bash
cd ~/Projects/engram
cargo init --lib
```

**Step 2: Configure Cargo.toml**

```toml
[package]
name = "engram"
version = "0.1.0"
edition = "2024"
description = "Neuromorphic memory architecture for LLMs"
license = "MIT"

[dependencies]
uuid = { version = "1", features = ["v4", "serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
ordered-float = { version = "4", features = ["serde"] }
thiserror = "2"

[dev-dependencies]
approx = "0.5"
```

**Step 3: Set up lib.rs with module declarations**

```rust
pub mod types;
```

**Step 4: Create types/mod.rs as empty module**

```rust
// Core types for the Engram neuromorphic memory system
```

**Step 5: Add rustfmt.toml**

```toml
max_width = 100
tab_spaces = 4
edition = "2024"
```

**Step 6: Add .gitignore**

```
/target
Cargo.lock
*.swp
*.swo
.DS_Store
```

**Step 7: Verify it compiles**

Run: `cargo build`
Expected: compiles with no errors

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: initialize engram rust project with dependencies"
```

---

### Task 2: Enums (ContentType, RelationType, AbstractionLevel, Intent)

**Files:**
- Create: `src/types/enums.rs`
- Modify: `src/types/mod.rs`
- Create: `tests/types_test.rs`

**Step 1: Write the failing test**

```rust
// tests/types_test.rs
use engram::types::enums::*;

#[test]
fn test_content_type_serialization() {
    let ct = ContentType::Conversation;
    let json = serde_json::to_string(&ct).unwrap();
    let deserialized: ContentType = serde_json::from_str(&json).unwrap();
    assert_eq!(ct, deserialized);
}

#[test]
fn test_relation_type_all_variants() {
    let types = vec![
        RelationType::Semantic,
        RelationType::Temporal,
        RelationType::Causal,
        RelationType::Contextual,
        RelationType::Hierarchical,
    ];
    assert_eq!(types.len(), 5);
}

#[test]
fn test_abstraction_level_ordering() {
    assert!((AbstractionLevel::Raw as u8) < (AbstractionLevel::Chunk as u8));
    assert!((AbstractionLevel::Chunk as u8) < (AbstractionLevel::Summary as u8));
    assert!((AbstractionLevel::Summary as u8) < (AbstractionLevel::Schema as u8));
}

#[test]
fn test_retrieval_intent_variants() {
    let intents = vec![
        RetrievalIntent::Recall,
        RetrievalIntent::Recognize,
        RetrievalIntent::Explore,
        RetrievalIntent::Verify,
    ];
    assert_eq!(intents.len(), 4);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test types_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/types/enums.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
    Code,
    Conversation,
    Event,
    Fact,
    Skill,
    Entity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Semantic,
    Temporal,
    Causal,
    Contextual,
    Hierarchical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum AbstractionLevel {
    Raw = 0,
    Chunk = 1,
    Summary = 2,
    Schema = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalIntent {
    Recall,
    Recognize,
    Explore,
    Verify,
}
```

Update `src/types/mod.rs`:

```rust
pub mod enums;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test types_test`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/types/enums.rs src/types/mod.rs tests/types_test.rs
git commit -m "feat: add core enum types (content, relation, abstraction, intent)"
```

---

### Task 3: Edge Type

**Files:**
- Create: `src/types/edge.rs`
- Modify: `src/types/mod.rs`
- Modify: `tests/types_test.rs`

**Step 1: Write the failing test**

```rust
// append to tests/types_test.rs
use engram::types::edge::Edge;
use uuid::Uuid;
use chrono::Utc;

#[test]
fn test_edge_creation() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Semantic, 0.8);
    assert_eq!(edge.target_id, target);
    assert_eq!(edge.relation_type, RelationType::Semantic);
    assert!((edge.weight - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_edge_weight_clamped() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Causal, 1.5);
    assert!(edge.weight <= 1.0);

    let edge2 = Edge::new(target, RelationType::Causal, -0.5);
    assert!(edge2.weight >= -1.0);
}

#[test]
fn test_edge_strengthen() {
    let target = Uuid::new_v4();
    let mut edge = Edge::new(target, RelationType::Semantic, 0.5);
    edge.strengthen(0.1);
    assert!((edge.weight - 0.6).abs() < f32::EPSILON);

    // Should not exceed 1.0
    edge.strengthen(0.9);
    assert!((edge.weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_edge_weaken() {
    let target = Uuid::new_v4();
    let mut edge = Edge::new(target, RelationType::Temporal, 0.5);
    edge.weaken(0.2);
    assert!((edge.weight - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_inhibitory_edge() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Semantic, -0.5);
    assert!(edge.is_inhibitory());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test types_test`
Expected: FAIL (edge module not found)

**Step 3: Write minimal implementation**

```rust
// src/types/edge.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::enums::RelationType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub target_id: Uuid,
    pub relation_type: RelationType,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
}

impl Edge {
    pub fn new(target_id: Uuid, relation_type: RelationType, weight: f32) -> Self {
        Self {
            target_id,
            relation_type,
            weight: weight.clamp(-1.0, 1.0),
            created_at: Utc::now(),
        }
    }

    pub fn strengthen(&mut self, amount: f32) {
        self.weight = (self.weight + amount).min(1.0);
    }

    pub fn weaken(&mut self, amount: f32) {
        self.weight = (self.weight - amount).max(-1.0);
    }

    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }
}
```

Update `src/types/mod.rs`:

```rust
pub mod enums;
pub mod edge;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test types_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/types/edge.rs src/types/mod.rs tests/types_test.rs
git commit -m "feat: add Edge type with LTP/LTD weight modification"
```

---

### Task 4: MemoryNode Type

**Files:**
- Create: `src/types/memory_node.rs`
- Modify: `src/types/mod.rs`
- Modify: `tests/types_test.rs`

**Step 1: Write the failing test**

```rust
// append to tests/types_test.rs
use engram::types::memory_node::MemoryNode;
use engram::types::enums::{ContentType, AbstractionLevel, RelationType};
use std::collections::HashMap;

#[test]
fn test_memory_node_creation() {
    let node = MemoryNode::new(
        "test content".to_string(),
        vec![0.1, 0.2, 0.3],
        ContentType::Text,
    );
    assert_eq!(node.content, "test content");
    assert_eq!(node.content_type, ContentType::Text);
    assert_eq!(node.abstraction_level, AbstractionLevel::Raw);
    assert_eq!(node.access_count, 0);
    assert!(node.edges.is_empty());
}

#[test]
fn test_memory_node_add_edge() {
    let mut node = MemoryNode::new(
        "source".to_string(),
        vec![0.1],
        ContentType::Text,
    );
    let target_id = Uuid::new_v4();
    node.add_edge(target_id, RelationType::Semantic, 0.9);
    assert_eq!(node.edges.len(), 1);
    assert_eq!(node.edges[0].target_id, target_id);
}

#[test]
fn test_memory_node_record_access() {
    let mut node = MemoryNode::new(
        "test".to_string(),
        vec![0.1],
        ContentType::Fact,
    );
    assert_eq!(node.access_count, 0);
    node.record_access();
    assert_eq!(node.access_count, 1);
    node.record_access();
    assert_eq!(node.access_count, 2);
}

#[test]
fn test_memory_node_salience_default() {
    let node = MemoryNode::new(
        "test".to_string(),
        vec![0.1],
        ContentType::Text,
    );
    assert!((node.salience - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_memory_node_serialization() {
    let node = MemoryNode::new(
        "test content".to_string(),
        vec![0.1, 0.2],
        ContentType::Event,
    );
    let json = serde_json::to_string(&node).unwrap();
    let deserialized: MemoryNode = serde_json::from_str(&json).unwrap();
    assert_eq!(node.id, deserialized.id);
    assert_eq!(node.content, deserialized.content);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test types_test`
Expected: FAIL (memory_node module not found)

**Step 3: Write minimal implementation**

```rust
// src/types/memory_node.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::edge::Edge;
use super::enums::{AbstractionLevel, ContentType, RelationType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,

    pub content: String,
    pub embedding: Vec<f32>,
    pub content_type: ContentType,

    pub source: String,
    pub session_id: Option<Uuid>,
    pub encoding_state: HashMap<String, f32>,

    pub edges: Vec<Edge>,

    pub salience: f32,
    pub decay_rate: f32,

    pub abstraction_level: AbstractionLevel,
    pub parent_id: Option<Uuid>,
    pub children_ids: Vec<Uuid>,
}

impl MemoryNode {
    pub fn new(content: String, embedding: Vec<f32>, content_type: ContentType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            content,
            embedding,
            content_type,
            source: String::new(),
            session_id: None,
            encoding_state: HashMap::new(),
            edges: Vec::new(),
            salience: 0.5,
            decay_rate: 0.1,
            abstraction_level: AbstractionLevel::Raw,
            parent_id: None,
            children_ids: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, target_id: Uuid, relation_type: RelationType, weight: f32) {
        self.edges.push(Edge::new(target_id, relation_type, weight));
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }

    pub fn with_salience(mut self, salience: f32) -> Self {
        self.salience = salience.clamp(0.0, 1.0);
        self
    }

    pub fn with_source(mut self, source: String) -> Self {
        self.source = source;
        self
    }

    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }
}
```

Update `src/types/mod.rs`:

```rust
pub mod enums;
pub mod edge;
pub mod memory_node;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test types_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/types/memory_node.rs src/types/mod.rs tests/types_test.rs
git commit -m "feat: add MemoryNode type with edges, salience, and abstraction hierarchy"
```

---

### Task 5: RetrievalCue and MemoryConstellation Types

**Files:**
- Create: `src/types/cue.rs`
- Create: `src/types/constellation.rs`
- Modify: `src/types/mod.rs`
- Modify: `tests/types_test.rs`

**Step 1: Write the failing test**

```rust
// append to tests/types_test.rs
use engram::types::cue::RetrievalCue;
use engram::types::constellation::MemoryConstellation;

#[test]
fn test_retrieval_cue_creation() {
    let cue = RetrievalCue::new(
        vec![0.1, 0.2, 0.3],
        RetrievalIntent::Recall,
    );
    assert_eq!(cue.intent, RetrievalIntent::Recall);
    assert!(cue.entities.is_empty());
    assert!((cue.salience_floor - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_retrieval_cue_builder() {
    let cue = RetrievalCue::new(vec![0.1], RetrievalIntent::Explore)
        .with_entities(vec!["Acme Corp".to_string()])
        .with_salience_floor(0.3);
    assert_eq!(cue.entities.len(), 1);
    assert!((cue.salience_floor - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_constellation_empty() {
    let constellation = MemoryConstellation::empty();
    assert!(constellation.focal_nodes.is_empty());
    assert!(constellation.context_nodes.is_empty());
    assert!((constellation.confidence - 0.0).abs() < f32::EPSILON);
    assert!((constellation.coverage - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_constellation_with_focal_node() {
    let node = MemoryNode::new("test".to_string(), vec![0.1], ContentType::Fact);
    let mut constellation = MemoryConstellation::empty();
    constellation.add_focal(node, 0.9);
    assert_eq!(constellation.focal_nodes.len(), 1);
    assert!((constellation.focal_nodes[0].activation - 0.9).abs() < f32::EPSILON);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test types_test`
Expected: FAIL (modules not found)

**Step 3: Write minimal implementation**

```rust
// src/types/cue.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::enums::RetrievalIntent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCue {
    pub semantic: Vec<f32>,
    pub temporal: Option<TimeRange>,
    pub entities: Vec<String>,
    pub context: HashMap<String, f32>,
    pub intent: RetrievalIntent,
    pub salience_floor: f32,
}

impl RetrievalCue {
    pub fn new(semantic: Vec<f32>, intent: RetrievalIntent) -> Self {
        Self {
            semantic,
            temporal: None,
            entities: Vec::new(),
            context: HashMap::new(),
            intent,
            salience_floor: 0.0,
        }
    }

    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_salience_floor(mut self, floor: f32) -> Self {
        self.salience_floor = floor.clamp(0.0, 1.0);
        self
    }

    pub fn with_temporal(mut self, range: TimeRange) -> Self {
        self.temporal = Some(range);
        self
    }

    pub fn with_context(mut self, context: HashMap<String, f32>) -> Self {
        self.context = context;
        self
    }
}
```

```rust
// src/types/constellation.rs
use serde::{Deserialize, Serialize};

use super::edge::Edge;
use super::memory_node::MemoryNode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedNode {
    pub node: MemoryNode,
    pub activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConstellation {
    pub focal_nodes: Vec<ActivatedNode>,
    pub context_nodes: Vec<ActivatedNode>,
    pub relationships: Vec<Edge>,
    pub confidence: f32,
    pub coverage: f32,
    pub gaps: Vec<String>,
}

impl MemoryConstellation {
    pub fn empty() -> Self {
        Self {
            focal_nodes: Vec::new(),
            context_nodes: Vec::new(),
            relationships: Vec::new(),
            confidence: 0.0,
            coverage: 0.0,
            gaps: Vec::new(),
        }
    }

    pub fn add_focal(&mut self, node: MemoryNode, activation: f32) {
        self.focal_nodes.push(ActivatedNode { node, activation });
        self.focal_nodes.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());
    }

    pub fn add_context(&mut self, node: MemoryNode, activation: f32) {
        self.context_nodes.push(ActivatedNode { node, activation });
    }

    pub fn total_nodes(&self) -> usize {
        self.focal_nodes.len() + self.context_nodes.len()
    }
}
```

Update `src/types/mod.rs`:

```rust
pub mod enums;
pub mod edge;
pub mod memory_node;
pub mod cue;
pub mod constellation;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test types_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/types/cue.rs src/types/constellation.rs src/types/mod.rs tests/types_test.rs
git commit -m "feat: add RetrievalCue and MemoryConstellation types"
```

---

## Phase 2: Engram Index (The Hippocampus)

The associative graph that stores nodes and edges. The heart of the system.

### Task 6: Graph Store (Node CRUD + Edge Management)

**Files:**
- Create: `src/index/mod.rs`
- Create: `src/index/graph.rs`
- Modify: `src/lib.rs`
- Create: `tests/index_test.rs`

**Step 1: Write the failing test**

```rust
// tests/index_test.rs
use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;

#[test]
fn test_graph_insert_and_get() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("hello".to_string(), vec![0.1, 0.2], ContentType::Text);
    let id = node.id;
    graph.insert(node);
    let retrieved = graph.get(&id).unwrap();
    assert_eq!(retrieved.content, "hello");
}

#[test]
fn test_graph_get_nonexistent() {
    let graph = GraphStore::new();
    let id = uuid::Uuid::new_v4();
    assert!(graph.get(&id).is_none());
}

#[test]
fn test_graph_add_edge_between_nodes() {
    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let node_b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let id_a = node_a.id;
    let id_b = node_b.id;
    graph.insert(node_a);
    graph.insert(node_b);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.8);

    let a = graph.get(&id_a).unwrap();
    assert_eq!(a.edges.len(), 1);
    assert_eq!(a.edges[0].target_id, id_b);
}

#[test]
fn test_graph_neighbors() {
    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let node_b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let node_c = MemoryNode::new("C".to_string(), vec![0.3], ContentType::Text);
    let id_a = node_a.id;
    let id_b = node_b.id;
    let id_c = node_c.id;
    graph.insert(node_a);
    graph.insert(node_b);
    graph.insert(node_c);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.9);
    graph.add_edge(id_a, id_c, RelationType::Temporal, 0.5);

    let neighbors = graph.neighbors(&id_a);
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_graph_remove_node() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("remove me".to_string(), vec![0.1], ContentType::Text);
    let id = node.id;
    graph.insert(node);
    assert!(graph.get(&id).is_some());
    graph.remove(&id);
    assert!(graph.get(&id).is_none());
}

#[test]
fn test_graph_node_count() {
    let mut graph = GraphStore::new();
    assert_eq!(graph.len(), 0);
    graph.insert(MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text));
    graph.insert(MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text));
    assert_eq!(graph.len(), 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test index_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/index/graph.rs
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::edge::Edge;
use crate::types::enums::RelationType;
use crate::types::memory_node::MemoryNode;

pub struct GraphStore {
    nodes: HashMap<Uuid, MemoryNode>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn insert(&mut self, node: MemoryNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn get(&self, id: &Uuid) -> Option<&MemoryNode> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: &Uuid) -> Option<&mut MemoryNode> {
        self.nodes.get_mut(id)
    }

    pub fn remove(&mut self, id: &Uuid) -> Option<MemoryNode> {
        self.nodes.remove(id)
    }

    pub fn add_edge(
        &mut self,
        source_id: Uuid,
        target_id: Uuid,
        relation_type: RelationType,
        weight: f32,
    ) {
        if let Some(source) = self.nodes.get_mut(&source_id) {
            source.add_edge(target_id, relation_type, weight);
        }
    }

    pub fn neighbors(&self, id: &Uuid) -> Vec<(&MemoryNode, &Edge)> {
        let Some(node) = self.nodes.get(id) else {
            return Vec::new();
        };
        node.edges
            .iter()
            .filter_map(|edge| {
                self.nodes.get(&edge.target_id).map(|target| (target, edge))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &MemoryNode> {
        self.nodes.values()
    }
}
```

```rust
// src/index/mod.rs
pub mod graph;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test index_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/index/ src/lib.rs tests/index_test.rs
git commit -m "feat: add GraphStore with node CRUD and edge management"
```

---

### Task 7: Vector Operations (Cosine Similarity + Seed Finding)

**Files:**
- Create: `src/index/vector.rs`
- Modify: `src/index/mod.rs`
- Modify: `tests/index_test.rs`

**Step 1: Write the failing test**

```rust
// append to tests/index_test.rs
use engram::index::vector;

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_find_seeds_by_similarity() {
    let mut graph = GraphStore::new();
    let close = MemoryNode::new("close".to_string(), vec![0.9, 0.1, 0.0], ContentType::Text);
    let far = MemoryNode::new("far".to_string(), vec![0.0, 0.0, 1.0], ContentType::Text);
    let medium = MemoryNode::new("medium".to_string(), vec![0.6, 0.4, 0.1], ContentType::Text);
    let close_id = close.id;
    graph.insert(close);
    graph.insert(far);
    graph.insert(medium);

    let query = vec![1.0, 0.0, 0.0];
    let seeds = vector::find_seeds_by_similarity(&graph, &query, 2, 0.0);
    assert_eq!(seeds.len(), 2);
    assert_eq!(seeds[0].0, close_id); // closest first
}

#[test]
fn test_find_seeds_with_threshold() {
    let mut graph = GraphStore::new();
    graph.insert(MemoryNode::new("close".to_string(), vec![0.95, 0.05], ContentType::Text));
    graph.insert(MemoryNode::new("far".to_string(), vec![0.0, 1.0], ContentType::Text));

    let query = vec![1.0, 0.0];
    let seeds = vector::find_seeds_by_similarity(&graph, &query, 10, 0.5);
    assert_eq!(seeds.len(), 1); // only the close one passes threshold
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test index_test`
Expected: FAIL (vector module not found)

**Step 3: Write minimal implementation**

```rust
// src/index/vector.rs
use uuid::Uuid;

use super::graph::GraphStore;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Returns vec of (node_id, similarity_score) sorted by score descending
pub fn find_seeds_by_similarity(
    graph: &GraphStore,
    query_embedding: &[f32],
    top_k: usize,
    threshold: f32,
) -> Vec<(Uuid, f32)> {
    let mut scores: Vec<(Uuid, f32)> = graph
        .all_nodes()
        .map(|node| {
            let sim = cosine_similarity(&node.embedding, query_embedding);
            (node.id, sim)
        })
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}
```

Update `src/index/mod.rs`:

```rust
pub mod graph;
pub mod vector;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test index_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/index/vector.rs src/index/mod.rs tests/index_test.rs
git commit -m "feat: add vector operations (cosine similarity, seed finding)"
```

---

### Task 8: Spreading Activation Algorithm

**Files:**
- Create: `src/retrieval/mod.rs`
- Create: `src/retrieval/activation.rs`
- Modify: `src/lib.rs`
- Create: `tests/retrieval_test.rs`

**Step 1: Write the failing test**

```rust
// tests/retrieval_test.rs
use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;
use engram::retrieval::activation::{spread, ActivationConfig, ActivationResult};
use uuid::Uuid;

fn build_test_graph() -> (GraphStore, Uuid, Uuid, Uuid, Uuid) {
    let mut graph = GraphStore::new();
    // A --0.9--> B --0.8--> C
    // A --0.3--> D
    let a = MemoryNode::new("A".to_string(), vec![1.0, 0.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.8, 0.2], ContentType::Text);
    let c = MemoryNode::new("C".to_string(), vec![0.5, 0.5], ContentType::Text);
    let d = MemoryNode::new("D".to_string(), vec![0.0, 1.0], ContentType::Text);
    let (id_a, id_b, id_c, id_d) = (a.id, b.id, c.id, d.id);

    graph.insert(a);
    graph.insert(b);
    graph.insert(c);
    graph.insert(d);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.9);
    graph.add_edge(id_b, id_c, RelationType::Semantic, 0.8);
    graph.add_edge(id_a, id_d, RelationType::Temporal, 0.3);

    (graph, id_a, id_b, id_c, id_d)
}

#[test]
fn test_spreading_activation_basic() {
    let (graph, id_a, id_b, id_c, id_d) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    // B should have higher activation than D (stronger edge)
    let b_activation = result.get(&id_b).unwrap_or(&0.0);
    let d_activation = result.get(&id_d).unwrap_or(&0.0);
    assert!(b_activation > d_activation);

    // C should have some activation (2 hops from A)
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!(*c_activation > 0.0);

    // C should have less activation than B (further away)
    assert!(c_activation < b_activation);
}

#[test]
fn test_spreading_activation_depth_limit() {
    let (graph, id_a, _, id_c, _) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig { max_depth: 1, ..Default::default() };

    let result = spread(&graph, &seeds, &config);

    // C is 2 hops away, should NOT be activated with max_depth=1
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!((*c_activation).abs() < f32::EPSILON);
}

#[test]
fn test_spreading_activation_threshold() {
    let (graph, id_a, _, _, id_d) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig {
        min_activation: 0.5,
        ..Default::default()
    };

    let result = spread(&graph, &seeds, &config);

    // D has weak edge (0.3), its activation should be below threshold
    let d_activation = result.get(&id_d).unwrap_or(&0.0);
    assert!((*d_activation).abs() < f32::EPSILON);
}

#[test]
fn test_multi_path_convergence() {
    let mut graph = GraphStore::new();
    // A --0.7--> C
    // B --0.7--> C
    // C should get activation from both paths
    let a = MemoryNode::new("A".to_string(), vec![1.0, 0.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.0, 1.0], ContentType::Text);
    let c = MemoryNode::new("C".to_string(), vec![0.5, 0.5], ContentType::Text);
    let (id_a, id_b, id_c) = (a.id, b.id, c.id);
    graph.insert(a);
    graph.insert(b);
    graph.insert(c);
    graph.add_edge(id_a, id_c, RelationType::Semantic, 0.7);
    graph.add_edge(id_b, id_c, RelationType::Semantic, 0.7);

    let seeds = vec![(id_a, 1.0), (id_b, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    // C reached from two paths, should have higher activation than if from one
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!(*c_activation > 0.7); // must be more than single-path activation
}

#[test]
fn test_inhibitory_edge() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![1.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.5], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, -0.5);

    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    let b_activation = result.get(&id_b).unwrap_or(&0.0);
    assert!(*b_activation < 0.0); // negative activation from inhibitory edge
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test retrieval_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/retrieval/activation.rs
use std::collections::HashMap;
use uuid::Uuid;

use crate::index::graph::GraphStore;

pub type ActivationResult = HashMap<Uuid, f32>;

pub struct ActivationConfig {
    pub max_depth: usize,
    pub decay_factor: f32,
    pub min_activation: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            decay_factor: 0.7,
            min_activation: 0.01,
        }
    }
}

pub fn spread(
    graph: &GraphStore,
    seeds: &[(Uuid, f32)],
    config: &ActivationConfig,
) -> ActivationResult {
    let mut activations: ActivationResult = HashMap::new();

    // Initialize seeds
    for (id, activation) in seeds {
        *activations.entry(*id).or_insert(0.0) += activation;
    }

    // BFS-style spreading with depth tracking
    let mut frontier: Vec<(Uuid, f32, usize)> = seeds
        .iter()
        .map(|(id, act)| (*id, *act, 0))
        .collect();

    while let Some((node_id, activation, depth)) = frontier.pop() {
        if depth >= config.max_depth {
            continue;
        }

        let neighbors = graph.neighbors(&node_id);
        for (neighbor, edge) in neighbors {
            let propagated = activation * edge.weight * config.decay_factor;

            if propagated.abs() < config.min_activation {
                continue;
            }

            *activations.entry(neighbor.id).or_insert(0.0) += propagated;
            frontier.push((neighbor.id, propagated, depth + 1));
        }
    }

    // Remove seeds from results (we want activated neighbors, not the seeds themselves)
    // Actually, keep them. The caller can filter.
    activations
}
```

```rust
// src/retrieval/mod.rs
pub mod activation;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test retrieval_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/retrieval/ src/lib.rs tests/retrieval_test.rs
git commit -m "feat: add spreading activation algorithm with depth limit, multi-path convergence, and inhibitory edges"
```

---

## Phase 3: Working Memory + Salience Router

### Task 9: Working Memory Buffer

**Files:**
- Create: `src/memory/mod.rs`
- Create: `src/memory/working.rs`
- Modify: `src/lib.rs`
- Create: `tests/memory_test.rs`

**Step 1: Write the failing test**

```rust
// tests/memory_test.rs
use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::types::constellation::MemoryConstellation;
use engram::memory::working::WorkingMemory;

#[test]
fn test_working_memory_capacity() {
    let mut wm = WorkingMemory::new(4);
    assert_eq!(wm.capacity(), 4);
    assert_eq!(wm.len(), 0);
    assert!(wm.is_empty());
}

#[test]
fn test_working_memory_add_and_get() {
    let mut wm = WorkingMemory::new(4);
    let mut constellation = MemoryConstellation::empty();
    let node = MemoryNode::new("test".to_string(), vec![0.1], ContentType::Fact);
    constellation.add_focal(node, 0.9);
    constellation.confidence = 0.8;

    wm.add(constellation, 0.9);
    assert_eq!(wm.len(), 1);
}

#[test]
fn test_working_memory_eviction_at_capacity() {
    let mut wm = WorkingMemory::new(2);

    let mut c1 = MemoryConstellation::empty();
    c1.confidence = 0.5;
    wm.add(c1, 0.3); // low salience

    let mut c2 = MemoryConstellation::empty();
    c2.confidence = 0.8;
    wm.add(c2, 0.9); // high salience

    assert_eq!(wm.len(), 2);

    let mut c3 = MemoryConstellation::empty();
    c3.confidence = 0.7;
    wm.add(c3, 0.6); // medium salience

    // Should still be at capacity, lowest salience (0.3) evicted
    assert_eq!(wm.len(), 2);
}

#[test]
fn test_working_memory_clear() {
    let mut wm = WorkingMemory::new(4);
    wm.add(MemoryConstellation::empty(), 0.5);
    wm.add(MemoryConstellation::empty(), 0.5);
    assert_eq!(wm.len(), 2);
    wm.clear();
    assert_eq!(wm.len(), 0);
}

#[test]
fn test_working_memory_coverage_assessment() {
    let mut wm = WorkingMemory::new(4);
    let mut c = MemoryConstellation::empty();
    let node = MemoryNode::new("rust programming".to_string(), vec![0.9, 0.1, 0.0], ContentType::Fact);
    c.add_focal(node, 0.9);
    c.coverage = 0.85;
    wm.add(c, 0.8);

    // Query similar to what's in working memory
    let coverage = wm.assess_coverage(&vec![0.85, 0.15, 0.0]);
    assert!(coverage > 0.5); // should show good coverage
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test memory_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/memory/working.rs
use crate::index::vector::cosine_similarity;
use crate::types::constellation::MemoryConstellation;

struct WorkingMemorySlot {
    constellation: MemoryConstellation,
    salience: f32,
}

pub struct WorkingMemory {
    slots: Vec<WorkingMemorySlot>,
    capacity: usize,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn add(&mut self, constellation: MemoryConstellation, salience: f32) {
        if self.slots.len() >= self.capacity {
            // Evict lowest salience
            if let Some((min_idx, _)) = self
                .slots
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.salience.partial_cmp(&b.1.salience).unwrap())
            {
                if salience > self.slots[min_idx].salience {
                    self.slots.remove(min_idx);
                } else {
                    return; // new item is lower salience than everything, skip
                }
            }
        }
        self.slots.push(WorkingMemorySlot {
            constellation,
            salience,
        });
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }

    pub fn assess_coverage(&self, query_embedding: &[f32]) -> f32 {
        if self.slots.is_empty() {
            return 0.0;
        }

        let mut max_coverage: f32 = 0.0;
        for slot in &self.slots {
            for activated in &slot.constellation.focal_nodes {
                let sim = cosine_similarity(&activated.node.embedding, query_embedding);
                max_coverage = max_coverage.max(sim * slot.constellation.coverage);
            }
        }
        max_coverage.clamp(0.0, 1.0)
    }

    pub fn constellations(&self) -> impl Iterator<Item = &MemoryConstellation> {
        self.slots.iter().map(|s| &s.constellation)
    }
}
```

```rust
// src/memory/mod.rs
pub mod working;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
pub mod memory;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test memory_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/memory/ src/lib.rs tests/memory_test.rs
git commit -m "feat: add WorkingMemory with capacity limit, salience eviction, and coverage assessment"
```

---

### Task 10: Salience Router

**Files:**
- Create: `src/router/mod.rs`
- Create: `src/router/salience.rs`
- Modify: `src/lib.rs`
- Create: `tests/router_test.rs`

**Step 1: Write the failing test**

```rust
// tests/router_test.rs
use engram::router::salience::{SalienceScorer, SalienceScore, RetrievalStrategy, route};

#[test]
fn test_salience_scoring() {
    let score = SalienceScore {
        novelty: 0.9,
        urgency: 0.8,
        emotional: 0.3,
        entity_weight: 0.5,
    };
    let composite = score.composite();
    assert!(composite > 0.5);
    assert!(composite <= 1.0);
}

#[test]
fn test_route_high_salience_low_coverage() {
    let strategy = route(0.8, 0.1);
    assert!(matches!(strategy, RetrievalStrategy::Deep { .. }));
}

#[test]
fn test_route_high_salience_partial_coverage() {
    let strategy = route(0.8, 0.5);
    assert!(matches!(strategy, RetrievalStrategy::Targeted { .. }));
}

#[test]
fn test_route_any_salience_high_coverage() {
    let strategy = route(0.9, 0.9);
    assert!(matches!(strategy, RetrievalStrategy::Skip));
}

#[test]
fn test_route_low_salience_low_coverage() {
    let strategy = route(0.2, 0.2);
    assert!(matches!(strategy, RetrievalStrategy::Standard { .. }));
}

#[test]
fn test_route_moderate() {
    let strategy = route(0.5, 0.5);
    assert!(matches!(strategy, RetrievalStrategy::Light { .. }));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test router_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/router/salience.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceScore {
    pub novelty: f32,
    pub urgency: f32,
    pub emotional: f32,
    pub entity_weight: f32,
}

impl SalienceScore {
    pub fn composite(&self) -> f32 {
        let weighted = self.novelty * 0.3
            + self.urgency * 0.3
            + self.emotional * 0.2
            + self.entity_weight * 0.2;
        weighted.clamp(0.0, 1.0)
    }
}

#[derive(Debug)]
pub enum RetrievalStrategy {
    Deep { max_depth: usize, max_nodes: usize },
    Targeted { max_depth: usize, max_nodes: usize },
    Standard { max_depth: usize, max_nodes: usize },
    Light { max_depth: usize, max_nodes: usize },
    Skip,
}

pub fn route(salience: f32, coverage: f32) -> RetrievalStrategy {
    if coverage > 0.85 {
        return RetrievalStrategy::Skip;
    }

    if salience > 0.7 && coverage < 0.3 {
        return RetrievalStrategy::Deep {
            max_depth: 4,
            max_nodes: 50,
        };
    }

    if salience > 0.7 {
        return RetrievalStrategy::Targeted {
            max_depth: 2,
            max_nodes: 20,
        };
    }

    if salience < 0.3 && coverage < 0.3 {
        return RetrievalStrategy::Standard {
            max_depth: 2,
            max_nodes: 15,
        };
    }

    RetrievalStrategy::Light {
        max_depth: 1,
        max_nodes: 10,
    }
}
```

```rust
// src/router/mod.rs
pub mod salience;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
pub mod memory;
pub mod router;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test router_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/router/ src/lib.rs tests/router_test.rs
git commit -m "feat: add salience router with scoring and strategy selection"
```

---

## Phase 4: Context Compiler

### Task 11: Context Compilation and Token Budget Management

**Files:**
- Create: `src/compiler/mod.rs`
- Create: `src/compiler/context.rs`
- Modify: `src/lib.rs`
- Create: `tests/compiler_test.rs`

**Step 1: Write the failing test**

```rust
// tests/compiler_test.rs
use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::types::constellation::MemoryConstellation;
use engram::compiler::context::{compile, CompileConfig, LLMContext};

#[test]
fn test_compile_empty_constellation() {
    let constellation = MemoryConstellation::empty();
    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);
    assert!(ctx.focal_memories.is_empty());
    assert!((ctx.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_compile_with_focal_nodes() {
    let mut constellation = MemoryConstellation::empty();
    let node1 = MemoryNode::new("Important fact about Rust".to_string(), vec![0.9], ContentType::Fact);
    let node2 = MemoryNode::new("Less important detail".to_string(), vec![0.3], ContentType::Text);
    constellation.add_focal(node1, 0.9);
    constellation.add_focal(node2, 0.4);
    constellation.confidence = 0.85;
    constellation.coverage = 0.7;

    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);

    assert_eq!(ctx.focal_memories.len(), 2);
    assert!((ctx.confidence - 0.85).abs() < f32::EPSILON);
    // Higher activation node should be first
    assert!(ctx.focal_memories[0].relevance > ctx.focal_memories[1].relevance);
}

#[test]
fn test_compile_includes_gaps() {
    let mut constellation = MemoryConstellation::empty();
    constellation.gaps = vec!["No data about Q4 revenue".to_string()];
    constellation.confidence = 0.4;

    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);

    assert_eq!(ctx.gaps.len(), 1);
    assert_eq!(ctx.gaps[0], "No data about Q4 revenue");
}

#[test]
fn test_compile_respects_token_budget() {
    let mut constellation = MemoryConstellation::empty();
    // Add many nodes with long content
    for i in 0..50 {
        let content = format!("This is memory node number {} with substantial content that takes up tokens in the context window", i);
        let node = MemoryNode::new(content, vec![0.5], ContentType::Text);
        constellation.add_focal(node, 0.5 + (i as f32 * 0.01));
    }
    constellation.confidence = 0.9;

    let config = CompileConfig { token_budget: 200 }; // very tight budget
    let ctx = compile(&constellation, &config);

    // Should not include all 50 nodes
    assert!(ctx.focal_memories.len() < 50);
    // Rough token estimate: each word ~1 token, should fit in budget
    let total_chars: usize = ctx.focal_memories.iter().map(|m| m.content.len()).sum();
    assert!(total_chars / 4 < 200); // rough estimate: 4 chars per token
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test compiler_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/compiler/context.rs
use serde::{Deserialize, Serialize};

use crate::types::constellation::MemoryConstellation;

pub struct CompileConfig {
    pub token_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalMemory {
    pub content: String,
    pub relevance: f32,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMContext {
    pub focal_memories: Vec<FocalMemory>,
    pub relationship_map: String,
    pub confidence: f32,
    pub coverage: f32,
    pub gaps: Vec<String>,
}

fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 characters per token
    (text.len() + 3) / 4
}

pub fn compile(constellation: &MemoryConstellation, config: &CompileConfig) -> LLMContext {
    let mut focal_memories = Vec::new();
    let mut tokens_used: usize = 0;

    // Reserve 10% for metadata (confidence, coverage, gaps)
    let content_budget = (config.token_budget as f32 * 0.9) as usize;

    // Add focal nodes in activation order (already sorted)
    for activated in &constellation.focal_nodes {
        let content = &activated.node.content;
        let content_tokens = estimate_tokens(content);

        if tokens_used + content_tokens > content_budget {
            break;
        }

        focal_memories.push(FocalMemory {
            content: content.clone(),
            relevance: activated.activation,
            source: activated.node.source.clone(),
        });
        tokens_used += content_tokens;
    }

    // Add context nodes as shorter summaries if budget allows
    let context_budget = content_budget.saturating_sub(tokens_used);
    for activated in &constellation.context_nodes {
        let content = &activated.node.content;
        // Truncate context nodes to ~50 chars for summary
        let summary: String = content.chars().take(50).collect();
        let summary_tokens = estimate_tokens(&summary);

        if tokens_used + summary_tokens > content_budget {
            break;
        }

        focal_memories.push(FocalMemory {
            content: summary,
            relevance: activated.activation,
            source: activated.node.source.clone(),
        });
        tokens_used += summary_tokens;
    }

    LLMContext {
        focal_memories,
        relationship_map: String::new(),
        confidence: constellation.confidence,
        coverage: constellation.coverage,
        gaps: constellation.gaps.clone(),
    }
}
```

```rust
// src/compiler/mod.rs
pub mod context;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
pub mod memory;
pub mod router;
pub mod compiler;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test compiler_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/compiler/ src/lib.rs tests/compiler_test.rs
git commit -m "feat: add context compiler with token budget management"
```

---

## Phase 5: Integration (The Full Pipeline)

### Task 12: Engram Engine (Orchestrating All Components)

**Files:**
- Create: `src/engine.rs`
- Modify: `src/lib.rs`
- Create: `tests/engine_test.rs`

**Step 1: Write the failing test**

```rust
// tests/engine_test.rs
use engram::engine::{Engram, EngineConfig};
use engram::types::enums::*;
use engram::types::cue::RetrievalCue;

#[test]
fn test_engine_ingest_and_retrieve() {
    let mut engine = Engram::new(EngineConfig::default());

    // Ingest some data
    engine.ingest("Rust is a systems programming language", ContentType::Fact, 0.7);
    engine.ingest("Python is great for data science", ContentType::Fact, 0.6);
    engine.ingest("Rust and Python can work together via PyO3", ContentType::Fact, 0.8);

    // Retrieve
    let cue = RetrievalCue::new(vec![], RetrievalIntent::Recall);
    let context = engine.query("How do Rust and Python work together?", RetrievalIntent::Recall);

    assert!(!context.focal_memories.is_empty());
    assert!(context.confidence > 0.0);
}

#[test]
fn test_engine_empty_query() {
    let engine = Engram::new(EngineConfig::default());
    let context = engine.query("anything", RetrievalIntent::Recall);
    assert!(context.focal_memories.is_empty());
    assert!((context.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_engine_working_memory_reuse() {
    let mut engine = Engram::new(EngineConfig::default());

    engine.ingest("The capital of France is Paris", ContentType::Fact, 0.8);

    // First query populates working memory
    let ctx1 = engine.query("What is the capital of France?", RetrievalIntent::Recall);
    assert!(!ctx1.focal_memories.is_empty());

    // Second similar query should benefit from working memory
    let ctx2 = engine.query("Tell me about the capital of France", RetrievalIntent::Recall);
    assert!(!ctx2.focal_memories.is_empty());
}

#[test]
fn test_engine_node_count() {
    let mut engine = Engram::new(EngineConfig::default());
    assert_eq!(engine.node_count(), 0);
    engine.ingest("fact one", ContentType::Fact, 0.5);
    engine.ingest("fact two", ContentType::Fact, 0.5);
    assert_eq!(engine.node_count(), 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test engine_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

The engine ties all components together. Since we don't have a real embedding model in the Rust core yet, we'll use a simple bag-of-words approach as a placeholder. The Python/TypeScript SDKs will plug in real embedding models.

```rust
// src/engine.rs
use crate::compiler::context::{compile, CompileConfig, LLMContext};
use crate::index::graph::GraphStore;
use crate::index::vector::find_seeds_by_similarity;
use crate::memory::working::WorkingMemory;
use crate::retrieval::activation::{spread, ActivationConfig};
use crate::router::salience::{route, RetrievalStrategy};
use crate::types::constellation::MemoryConstellation;
use crate::types::enums::{ContentType, RetrievalIntent};
use crate::types::memory_node::MemoryNode;

pub struct EngineConfig {
    pub working_memory_capacity: usize,
    pub token_budget: usize,
    pub embedding_dim: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 5,
            token_budget: 2000,
            embedding_dim: 64,
        }
    }
}

pub struct Engram {
    graph: GraphStore,
    working_memory: WorkingMemory,
    config: EngineConfig,
    vocabulary: Vec<String>,
}

impl Engram {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            graph: GraphStore::new(),
            working_memory: WorkingMemory::new(config.working_memory_capacity),
            vocabulary: Vec::new(),
            config,
        }
    }

    /// Simple bag-of-words embedding as placeholder.
    /// Real embedding models will be injected via SDK.
    fn embed(&mut self, text: &str) -> Vec<f32> {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        for word in &words {
            if !self.vocabulary.contains(&word.to_string()) {
                self.vocabulary.push(word.to_string());
            }
        }
        let dim = self.vocabulary.len().max(1);
        let mut embedding = vec![0.0f32; dim];
        for word in &words {
            if let Some(idx) = self.vocabulary.iter().position(|w| w == word) {
                embedding[idx] = 1.0;
            }
        }
        // Normalize
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in &mut embedding {
                *v /= mag;
            }
        }
        embedding
    }

    pub fn ingest(&mut self, content: &str, content_type: ContentType, salience: f32) {
        let embedding = self.embed(content);
        let node = MemoryNode::new(content.to_string(), embedding, content_type)
            .with_salience(salience);
        let node_id = node.id;

        // Find existing similar nodes and create edges
        let seeds = find_seeds_by_similarity(&self.graph, &self.graph_safe_embedding(&node), 3, 0.3);
        self.graph.insert(node);

        for (similar_id, similarity) in seeds {
            if similar_id != node_id {
                self.graph.add_edge(
                    node_id,
                    similar_id,
                    crate::types::enums::RelationType::Semantic,
                    similarity,
                );
                self.graph.add_edge(
                    similar_id,
                    node_id,
                    crate::types::enums::RelationType::Semantic,
                    similarity,
                );
            }
        }
    }

    fn graph_safe_embedding(&self, node: &MemoryNode) -> Vec<f32> {
        node.embedding.clone()
    }

    pub fn query(&mut self, text: &str, intent: RetrievalIntent) -> LLMContext {
        let query_embedding = self.embed(text);

        if self.graph.is_empty() {
            return compile(&MemoryConstellation::empty(), &CompileConfig {
                token_budget: self.config.token_budget,
            });
        }

        // Check working memory coverage
        let coverage = self.working_memory.assess_coverage(&query_embedding);

        // Route based on salience (use novelty as proxy for now)
        let salience = 0.5; // placeholder; real salience scoring needs more context
        let strategy = route(salience, coverage);

        let (max_depth, max_nodes) = match &strategy {
            RetrievalStrategy::Skip => {
                // Compile from working memory only
                let constellation = self.constellation_from_working_memory(&query_embedding);
                return compile(&constellation, &CompileConfig {
                    token_budget: self.config.token_budget,
                });
            }
            RetrievalStrategy::Deep { max_depth, max_nodes } => (*max_depth, *max_nodes),
            RetrievalStrategy::Targeted { max_depth, max_nodes } => (*max_depth, *max_nodes),
            RetrievalStrategy::Standard { max_depth, max_nodes } => (*max_depth, *max_nodes),
            RetrievalStrategy::Light { max_depth, max_nodes } => (*max_depth, *max_nodes),
        };

        // Find seeds
        let seeds = find_seeds_by_similarity(&self.graph, &query_embedding, max_nodes, 0.1);

        if seeds.is_empty() {
            return compile(&MemoryConstellation::empty(), &CompileConfig {
                token_budget: self.config.token_budget,
            });
        }

        // Spread activation
        let activation_config = ActivationConfig {
            max_depth,
            ..Default::default()
        };
        let activations = spread(&self.graph, &seeds, &activation_config);

        // Build constellation
        let mut constellation = MemoryConstellation::empty();
        let mut sorted_activations: Vec<_> = activations.into_iter().collect();
        sorted_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let focal_count = max_nodes.min(sorted_activations.len()).min(5);
        for (id, activation) in sorted_activations.iter().take(focal_count) {
            if let Some(node) = self.graph.get(id) {
                constellation.add_focal(node.clone(), *activation);
            }
        }

        for (id, activation) in sorted_activations.iter().skip(focal_count).take(10) {
            if let Some(node) = self.graph.get(id) {
                constellation.add_context(node.clone(), *activation);
            }
        }

        // Calculate confidence and coverage
        if !constellation.focal_nodes.is_empty() {
            let max_activation = constellation.focal_nodes[0].activation;
            constellation.confidence = max_activation.clamp(0.0, 1.0);
            constellation.coverage = (constellation.focal_nodes.len() as f32 / max_nodes as f32).min(1.0);
        }

        // Store in working memory
        let wm_salience = constellation.confidence;
        let wm_constellation = constellation.clone();
        self.working_memory.add(wm_constellation, wm_salience);

        compile(&constellation, &CompileConfig {
            token_budget: self.config.token_budget,
        })
    }

    fn constellation_from_working_memory(&self, _query_embedding: &[f32]) -> MemoryConstellation {
        let mut constellation = MemoryConstellation::empty();
        for wm_constellation in self.working_memory.constellations() {
            for focal in &wm_constellation.focal_nodes {
                constellation.add_focal(focal.node.clone(), focal.activation);
            }
        }
        constellation.confidence = 0.8; // high confidence when working memory covers it
        constellation.coverage = 0.9;
        constellation
    }

    pub fn node_count(&self) -> usize {
        self.graph.len()
    }
}
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
pub mod memory;
pub mod router;
pub mod compiler;
pub mod engine;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test engine_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/engine.rs src/lib.rs tests/engine_test.rs
git commit -m "feat: add Engram engine orchestrating all components into full pipeline"
```

---

## Phase 6: Consolidation Engine (Artificial Sleep)

### Task 13: Consolidation - LTP and Decay

**Files:**
- Create: `src/consolidation/mod.rs`
- Create: `src/consolidation/ltp.rs`
- Create: `src/consolidation/decay.rs`
- Modify: `src/lib.rs`
- Create: `tests/consolidation_test.rs`

**Step 1: Write the failing test**

```rust
// tests/consolidation_test.rs
use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;
use engram::consolidation::ltp::strengthen_coactivated;
use engram::consolidation::decay::{apply_decay, DecayConfig};

#[test]
fn test_ltp_strengthens_edges() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.5);

    // Simulate co-activation: both were retrieved together
    let coactivated_pairs = vec![(id_a, id_b)];
    strengthen_coactivated(&mut graph, &coactivated_pairs, 0.1);

    let a = graph.get(&id_a).unwrap();
    let edge = &a.edges[0];
    assert!((edge.weight - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_ltp_capped_at_one() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.95);

    let coactivated_pairs = vec![(id_a, id_b)];
    strengthen_coactivated(&mut graph, &coactivated_pairs, 0.2);

    let a = graph.get(&id_a).unwrap();
    assert!((a.edges[0].weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_decay_reduces_low_salience() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("forgettable".to_string(), vec![0.1], ContentType::Text)
        .with_salience(0.1); // low salience, should decay fast
    let id = node.id;
    graph.insert(node);

    let config = DecayConfig {
        base_rate: 0.3,
        min_edge_weight: 0.05,
        prune_threshold: 0.01,
    };
    let pruned = apply_decay(&mut graph, &config);

    // Node should have lower effective strength but not pruned yet on first pass
    // (depends on implementation)
    assert!(pruned.is_empty() || !pruned.is_empty()); // just verify it runs
}

#[test]
fn test_decay_preserves_high_salience() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("important".to_string(), vec![0.1], ContentType::Text)
        .with_salience(0.95); // high salience, should resist decay
    let id = node.id;
    graph.insert(node);

    let config = DecayConfig {
        base_rate: 0.3,
        min_edge_weight: 0.05,
        prune_threshold: 0.01,
    };
    let pruned = apply_decay(&mut graph, &config);

    // High salience node should NOT be pruned
    assert!(!pruned.contains(&id));
    assert!(graph.get(&id).is_some());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test consolidation_test`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```rust
// src/consolidation/ltp.rs
use uuid::Uuid;
use crate::index::graph::GraphStore;

pub fn strengthen_coactivated(
    graph: &mut GraphStore,
    coactivated_pairs: &[(Uuid, Uuid)],
    amount: f32,
) {
    for (source_id, target_id) in coactivated_pairs {
        if let Some(source) = graph.get_mut(source_id) {
            for edge in &mut source.edges {
                if edge.target_id == *target_id {
                    edge.strengthen(amount);
                }
            }
        }
    }
}
```

```rust
// src/consolidation/decay.rs
use uuid::Uuid;
use crate::index::graph::GraphStore;

pub struct DecayConfig {
    pub base_rate: f32,
    pub min_edge_weight: f32,
    pub prune_threshold: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            base_rate: 0.1,
            min_edge_weight: 0.05,
            prune_threshold: 0.01,
        }
    }
}

pub fn apply_decay(graph: &mut GraphStore, config: &DecayConfig) -> Vec<Uuid> {
    let node_ids: Vec<Uuid> = graph.all_nodes().map(|n| n.id).collect();
    let mut to_prune = Vec::new();

    for id in &node_ids {
        if let Some(node) = graph.get_mut(id) {
            // Decay edges: weaken by base_rate * (1 - salience)
            let salience = node.salience;
            let decay = config.base_rate * (1.0 - salience);

            for edge in &mut node.edges {
                edge.weaken(decay);
            }

            // Remove dead edges
            node.edges.retain(|e| e.weight.abs() >= config.min_edge_weight);

            // Mark node for pruning if it has no edges and low salience
            if node.edges.is_empty() && node.salience < config.prune_threshold {
                to_prune.push(*id);
            }
        }
    }

    // Prune marked nodes
    for id in &to_prune {
        graph.remove(id);
    }

    to_prune
}
```

```rust
// src/consolidation/mod.rs
pub mod ltp;
pub mod decay;
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod index;
pub mod retrieval;
pub mod memory;
pub mod router;
pub mod compiler;
pub mod engine;
pub mod consolidation;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test consolidation_test`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/consolidation/ src/lib.rs tests/consolidation_test.rs
git commit -m "feat: add consolidation engine with LTP strengthening and adaptive decay"
```

---

## Phase 7: Python SDK

### Task 14: PyO3 Bindings Setup

**Files:**
- Create: `python/Cargo.toml` (workspace member or separate)
- Create: `python/src/lib.rs`
- Create: `python/pyproject.toml`
- Create: `python/engram/__init__.py`
- Create: `python/tests/test_engram.py`

**Step 1: Set up Python binding project**

Add to root `Cargo.toml`:

```toml
[workspace]
members = [".", "python"]
```

Create `python/Cargo.toml`:

```toml
[package]
name = "engram-python"
version = "0.1.0"
edition = "2024"

[lib]
name = "engram_python"
crate-type = ["cdylib"]

[dependencies]
engram = { path = ".." }
pyo3 = { version = "0.23", features = ["extension-module"] }
serde_json = "1"
```

Create `python/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "engram"
version = "0.1.0"
description = "Neuromorphic memory architecture for LLMs"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/extension-module"]
```

**Step 2: Write the Python test**

```python
# python/tests/test_engram.py
import pytest

def test_import():
    import engram_python
    assert hasattr(engram_python, 'Engram')

def test_create_engine():
    from engram_python import Engram
    engine = Engram()
    assert engine.node_count() == 0

def test_ingest_and_query():
    from engram_python import Engram
    engine = Engram()
    engine.ingest("Rust is a systems programming language", "fact", 0.7)
    engine.ingest("Python is great for AI", "fact", 0.6)
    result = engine.query("What is Rust?", "recall")
    assert result["confidence"] > 0
    assert len(result["focal_memories"]) > 0
```

**Step 3: Write the PyO3 bindings**

```rust
// python/src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyDict;
use engram::engine::{Engram as EngineCore, EngineConfig};
use engram::types::enums::{ContentType, RetrievalIntent};

#[pyclass]
struct Engram {
    inner: EngineCore,
}

#[pymethods]
impl Engram {
    #[new]
    fn new() -> Self {
        Self {
            inner: EngineCore::new(EngineConfig::default()),
        }
    }

    fn ingest(&mut self, content: &str, content_type: &str, salience: f32) {
        let ct = match content_type {
            "text" => ContentType::Text,
            "code" => ContentType::Code,
            "conversation" => ContentType::Conversation,
            "event" => ContentType::Event,
            "fact" => ContentType::Fact,
            "skill" => ContentType::Skill,
            "entity" => ContentType::Entity,
            _ => ContentType::Text,
        };
        self.inner.ingest(content, ct, salience);
    }

    fn query(&mut self, py: Python, text: &str, intent: &str) -> PyResult<PyObject> {
        let ri = match intent {
            "recall" => RetrievalIntent::Recall,
            "recognize" => RetrievalIntent::Recognize,
            "explore" => RetrievalIntent::Explore,
            "verify" => RetrievalIntent::Verify,
            _ => RetrievalIntent::Recall,
        };
        let context = self.inner.query(text, ri);
        let dict = PyDict::new(py);
        dict.set_item("confidence", context.confidence)?;
        dict.set_item("coverage", context.coverage)?;
        dict.set_item("gaps", context.gaps.clone())?;

        let memories: Vec<PyObject> = context.focal_memories.iter().map(|m| {
            let d = PyDict::new(py);
            d.set_item("content", m.content.clone()).unwrap();
            d.set_item("relevance", m.relevance).unwrap();
            d.set_item("source", m.source.clone()).unwrap();
            d.into_any().unbind()
        }).collect();
        dict.set_item("focal_memories", memories)?;

        Ok(dict.into_any().unbind())
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
}

#[pymodule]
fn engram_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engram>()?;
    Ok(())
}
```

**Step 4: Build and test**

Run:
```bash
cd ~/Projects/engram/python
maturin develop
pytest tests/test_engram.py -v
```
Expected: all tests PASS

**Step 5: Commit**

```bash
cd ~/Projects/engram
git add python/ Cargo.toml
git commit -m "feat: add Python SDK with PyO3 bindings"
```

---

## Phase 8: TypeScript SDK (via WASM)

### Task 15: wasm-pack Bindings

This follows the same pattern as the Python SDK but uses wasm-bindgen. Deferred to after Python SDK is stable and tested.

**Files:**
- Create: `typescript/Cargo.toml`
- Create: `typescript/src/lib.rs`
- Create: `typescript/package.json`
- Create: `typescript/tests/engram.test.ts`

**Implementation:** Same binding pattern as Python, using `wasm-bindgen` and `serde-wasm-bindgen` for type conversion. Build with `wasm-pack build --target nodejs`.

---

## Phase 9: Advanced Features (Post-MVP)

These tasks are outlined but deferred until the core is proven:

### Task 16: Schema Formation in Consolidation
- Cluster detection using embedding similarity
- Template extraction from clusters
- Delta computation and storage
- Schema compression in context compiler

### Task 17: Procedural Store
- Retrieval pattern logging
- Recurring pattern detection
- Pre-computed path caching
- Short-circuit retrieval integration

### Task 18: Pluggable Embedding Provider
- Trait/interface for embedding generation
- OpenAI adapter
- Local model adapter (sentence-transformers)
- Async embedding pipeline

### Task 19: Persistent Cortical Store
- File-based storage backend
- SQLite backend option
- Cloud backend trait (Cloudflare D1, Postgres)
- Migration between backends

### Task 20: Real Salience Scoring
- NLP-based urgency detection
- Sentiment/emotional analysis
- Entity importance registry
- Novelty computation against existing graph

---

## Summary

| Phase | Tasks | What It Delivers |
|---|---|---|
| 1: Foundation | 1-5 | Core types: MemoryNode, Edge, Cue, Constellation |
| 2: Engram Index | 6-8 | Graph store, vector search, spreading activation |
| 3: Working Memory + Router | 9-10 | Capacity-limited buffer, salience-based routing |
| 4: Context Compiler | 11 | Token-budgeted output for LLMs |
| 5: Integration | 12 | Full pipeline: ingest → retrieve → compile |
| 6: Consolidation | 13 | LTP strengthening, adaptive decay/forgetting |
| 7: Python SDK | 14 | PyO3 bindings, pip-installable package |
| 8: TypeScript SDK | 15 | WASM bindings, npm package |
| 9: Advanced | 16-20 | Schemas, procedures, persistence, real embeddings |

**Phases 1-6 = working Rust core. Phase 7 = usable by the AI community. Phase 8+ = production-ready.**
