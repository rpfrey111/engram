# Engram Working Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the existing Engram Rust framework into a working LLM memory layer — real embeddings, real LLM inference, persistence, and an interactive CLI chat agent with episodic + semantic memory.

**Architecture:** Add async trait abstractions (`Embedder`, `LLMProvider`) with Ollama HTTP API implementations. Build a sensory buffer for deep encoding. Add serde-based graph persistence. Wire it all into a CLI chat loop where conversations become episodic memory and ingested files become semantic memory. See `docs/plans/2026-03-03-working-model-design.md`.

**Tech Stack:** Rust (existing core), tokio (async runtime), reqwest (Ollama HTTP), async-trait, serde_json (persistence), clap (CLI args)

**Build Order Rationale:** Embedder first (everything needs real vectors), then LLM provider (chat needs inference), then sensory buffer (deep encoding replaces bag-of-words), then persistence (memory survives restarts), then CLI (brings it all together), then file ingestion (knowledge base).

---

## Phase 1: Async Foundation + Embedder Trait

The brain needs sensory cortex before it can do anything. Real embeddings unlock the entire system.

### Task 1: Add async dependencies to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add new dependencies**

Add these to `[dependencies]` in `Cargo.toml`:

```toml
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
async-trait = "0.1"
clap = { version = "4", features = ["derive"] }
dirs = "6"
```

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles with no errors (warnings OK)

**Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "deps: add tokio, reqwest, async-trait, clap, dirs for working model"
```

---

### Task 2: Define the Embedder trait

**Files:**
- Create: `src/provider/mod.rs`
- Create: `src/provider/embedder.rs`
- Modify: `src/lib.rs` (add `pub mod provider;`)

**Step 1: Write the trait definition**

Create `src/provider/mod.rs`:
```rust
pub mod embedder;
```

Create `src/provider/embedder.rs`:
```rust
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding request failed: {0}")]
    RequestFailed(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn dimensions(&self) -> usize;
}
```

Add to `src/lib.rs`:
```rust
pub mod provider;
```

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add src/provider/ src/lib.rs
git commit -m "feat: add Embedder trait for pluggable embedding backends"
```

---

### Task 3: Implement OllamaEmbedder

**Files:**
- Create: `src/provider/ollama.rs`
- Modify: `src/provider/mod.rs` (add `pub mod ollama;`)
- Create: `tests/ollama_test.rs`

**Step 1: Write the test (requires Ollama running with nomic-embed-text)**

Create `tests/ollama_test.rs`:
```rust
use engram::provider::embedder::Embedder;
use engram::provider::ollama::OllamaEmbedder;

#[tokio::test]
async fn test_ollama_embedder_produces_vector() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text");
    let result = embedder.embed("Hello world").await;
    match result {
        Ok(vec) => {
            assert!(!vec.is_empty());
            // nomic-embed-text produces 768-dim vectors
            assert_eq!(vec.len(), 768);
            // Vector should be non-zero
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(magnitude > 0.0);
        }
        Err(e) => {
            // Skip test if Ollama isn't running
            eprintln!("Skipping Ollama test (not running): {e}");
        }
    }
}

#[tokio::test]
async fn test_ollama_embedder_similar_texts() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text");

    let result_a = embedder.embed("The cat sat on the mat").await;
    let result_b = embedder.embed("A feline rested on the rug").await;
    let result_c = embedder.embed("Quantum computing uses qubits").await;

    match (result_a, result_b, result_c) {
        (Ok(a), Ok(b), Ok(c)) => {
            let sim_ab = engram::index::vector::cosine_similarity(&a, &b);
            let sim_ac = engram::index::vector::cosine_similarity(&a, &c);
            // Similar texts should have higher similarity
            assert!(sim_ab > sim_ac, "sim_ab={sim_ab} should be > sim_ac={sim_ac}");
        }
        _ => {
            eprintln!("Skipping Ollama test (not running)");
        }
    }
}
```

**Step 2: Write the implementation**

Create `src/provider/ollama.rs`:
```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embedder::{EmbedError, Embedder};

pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    client: reqwest::Client,
    dimensions: usize,
}

impl OllamaEmbedder {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
            dimensions: 768, // nomic-embed-text default; updated on first call
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let url = format!("{}/api/embed", self.base_url);
        let req = EmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| EmbedError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::RequestFailed(format!("{status}: {body}")));
        }

        let body: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbedError::InvalidResponse(e.to_string()))?;

        body.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::InvalidResponse("empty embeddings array".to_string()))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
```

Update `src/provider/mod.rs`:
```rust
pub mod embedder;
pub mod ollama;
```

**Step 3: Run the test**

Run: `cargo test --test ollama_test -- --nocapture`
Expected: PASS if Ollama is running with nomic-embed-text, or prints skip message

**Step 4: Commit**

```bash
git add src/provider/ tests/ollama_test.rs
git commit -m "feat: add OllamaEmbedder implementation"
```

---

## Phase 2: LLM Provider

The prefrontal cortex — reasoning and language generation.

### Task 4: Define LLMProvider trait and OllamaLLM

**Files:**
- Create: `src/provider/llm.rs`
- Modify: `src/provider/ollama.rs` (add OllamaLLM)
- Modify: `src/provider/mod.rs` (add `pub mod llm;`)
- Modify: `tests/ollama_test.rs` (add LLM test)

**Step 1: Write the trait**

Create `src/provider/llm.rs`:
```rust
use async_trait::async_trait;

use crate::compiler::context::LLMContext;

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("generation request failed: {0}")]
    RequestFailed(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, system: &str, user_message: &str, context: &LLMContext) -> Result<String, LLMError>;
    fn model_name(&self) -> &str;
}
```

Update `src/provider/mod.rs`:
```rust
pub mod embedder;
pub mod llm;
pub mod ollama;
```

**Step 2: Add OllamaLLM to ollama.rs**

Append to `src/provider/ollama.rs`:
```rust
use super::llm::{LLMError, LLMProvider};
use crate::compiler::context::LLMContext;

pub struct OllamaLLM {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaLLM {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

fn build_system_prompt(base_system: &str, context: &LLMContext) -> String {
    let mut prompt = base_system.to_string();

    if !context.focal_memories.is_empty() {
        prompt.push_str("\n\n## Relevant Memories\n");
        for mem in &context.focal_memories {
            prompt.push_str(&format!("- [relevance: {:.2}] {}\n", mem.relevance, mem.content));
        }
    }

    if !context.gaps.is_empty() {
        prompt.push_str("\n## Knowledge Gaps\n");
        for gap in &context.gaps {
            prompt.push_str(&format!("- {gap}\n"));
        }
    }

    prompt.push_str(&format!(
        "\n[Memory confidence: {:.0}% | Coverage: {:.0}%]",
        context.confidence * 100.0,
        context.coverage * 100.0
    ));

    prompt
}

#[async_trait]
impl LLMProvider for OllamaLLM {
    async fn generate(&self, system: &str, user_message: &str, context: &LLMContext) -> Result<String, LLMError> {
        let url = format!("{}/api/chat", self.base_url);

        let system_prompt = build_system_prompt(system, context);

        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                },
            ],
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(format!("{status}: {body}")));
        }

        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(body.message.content)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
```

**Step 3: Add test**

Append to `tests/ollama_test.rs`:
```rust
use engram::provider::llm::LLMProvider;
use engram::provider::ollama::OllamaLLM;
use engram::compiler::context::LLMContext;

#[tokio::test]
async fn test_ollama_llm_generates_response() {
    let llm = OllamaLLM::new("http://localhost:11434", "llama3.2");
    let context = LLMContext {
        focal_memories: vec![],
        relationship_map: String::new(),
        confidence: 0.0,
        coverage: 0.0,
        gaps: vec![],
    };
    let result = llm.generate("You are a helpful assistant.", "Say hello in exactly 3 words.", &context).await;
    match result {
        Ok(response) => {
            assert!(!response.is_empty());
        }
        Err(e) => {
            eprintln!("Skipping Ollama LLM test (not running): {e}");
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test --test ollama_test -- --nocapture`
Expected: PASS or skip

**Step 5: Commit**

```bash
git add src/provider/ tests/ollama_test.rs
git commit -m "feat: add LLMProvider trait and OllamaLLM implementation"
```

---

## Phase 3: Sensory Buffer (Deep Encoding)

The sensory register — deep multi-dimensional encoding. Replaces the bag-of-words `embed()` in engine.rs.

### Task 5: Build the Sensory Buffer

**Files:**
- Create: `src/sensory/mod.rs`
- Create: `src/sensory/buffer.rs`
- Modify: `src/lib.rs` (add `pub mod sensory;`)
- Create: `tests/sensory_test.rs`

**Step 1: Write the test**

Create `tests/sensory_test.rs`:
```rust
use engram::sensory::buffer::SensoryBuffer;
use engram::types::enums::ContentType;

#[test]
fn test_detect_content_type_code() {
    assert_eq!(
        SensoryBuffer::detect_content_type("fn main() { println!(\"hello\"); }"),
        ContentType::Code,
    );
}

#[test]
fn test_detect_content_type_conversation() {
    assert_eq!(
        SensoryBuffer::detect_content_type("User: Hello\nAssistant: Hi there!"),
        ContentType::Conversation,
    );
}

#[test]
fn test_detect_content_type_fact() {
    assert_eq!(
        SensoryBuffer::detect_content_type("The capital of France is Paris."),
        ContentType::Fact,
    );
}

#[test]
fn test_extract_entities() {
    let entities = SensoryBuffer::extract_entities("Ryan Frey works at Anthropic in San Francisco.");
    assert!(entities.contains(&"Ryan".to_string()) || entities.contains(&"Frey".to_string()));
    assert!(entities.contains(&"Anthropic".to_string()));
    assert!(entities.contains(&"San".to_string()) || entities.contains(&"Francisco".to_string()));
}

#[test]
fn test_score_novelty() {
    // With no existing embeddings, everything is novel
    let novelty = SensoryBuffer::score_novelty(&[0.1, 0.2, 0.3], &[]);
    assert!((novelty - 1.0).abs() < f32::EPSILON);

    // With a very similar existing embedding, novelty is low
    let existing = vec![vec![0.1, 0.2, 0.3]];
    let novelty = SensoryBuffer::score_novelty(&[0.1, 0.2, 0.3], &existing);
    assert!(novelty < 0.2);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test sensory_test`
Expected: FAIL (module doesn't exist yet)

**Step 3: Implement the sensory buffer**

Create `src/sensory/mod.rs`:
```rust
pub mod buffer;
```

Create `src/sensory/buffer.rs`:
```rust
use crate::index::vector::cosine_similarity;
use crate::types::enums::ContentType;

pub struct SensoryBuffer;

impl SensoryBuffer {
    /// Detect content type from text patterns (Craik & Lockhart deep processing).
    pub fn detect_content_type(text: &str) -> ContentType {
        // Code detection: language keywords, brackets, semicolons
        let code_signals = ["fn ", "def ", "class ", "import ", "func ", "var ", "let ",
                           "const ", "return ", "pub ", "struct ", "impl ", "async ",
                           "=>", "->", "println!", "console.log"];
        let code_score: usize = code_signals.iter()
            .filter(|s| text.contains(**s))
            .count();
        if code_score >= 2 || (code_score >= 1 && (text.contains('{') || text.contains('}'))) {
            return ContentType::Code;
        }

        // Conversation detection: turn markers
        let conversation_signals = ["User:", "Assistant:", "Human:", "AI:", "Q:", "A:",
                                    "you said", "I think", "tell me"];
        let conv_score: usize = conversation_signals.iter()
            .filter(|s| text.contains(**s))
            .count();
        if conv_score >= 2 {
            return ContentType::Conversation;
        }

        // Event detection: temporal markers
        let event_signals = ["yesterday", "today", "tomorrow", "last week", "meeting",
                            "happened", "occurred", "event", "scheduled"];
        let event_score: usize = event_signals.iter()
            .filter(|s| text.to_lowercase().contains(*s))
            .count();
        if event_score >= 2 {
            return ContentType::Event;
        }

        // Fact detection: declarative statements (short, assertive)
        if text.len() < 200 && (text.contains(" is ") || text.contains(" are ")
            || text.contains(" was ") || text.contains(" has ")) {
            return ContentType::Fact;
        }

        ContentType::Text
    }

    /// Extract entities from text (capitalized multi-word sequences, proper nouns).
    pub fn extract_entities(text: &str) -> Vec<String> {
        let mut entities = Vec::new();
        // Simple heuristic: words starting with uppercase that aren't sentence starters
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.is_empty() {
                continue;
            }
            let first_char = clean.chars().next().unwrap();
            // Skip sentence starters (position 0 or after period)
            let is_sentence_start = i == 0
                || (i > 0 && words[i - 1].ends_with('.'));
            if first_char.is_uppercase() && !is_sentence_start && clean.len() > 1 {
                entities.push(clean);
            }
        }
        entities.sort();
        entities.dedup();
        entities
    }

    /// Score novelty of an embedding against existing embeddings.
    /// 1.0 = completely novel, 0.0 = exact duplicate.
    pub fn score_novelty(embedding: &[f32], existing: &[Vec<f32>]) -> f32 {
        if existing.is_empty() {
            return 1.0;
        }
        let max_similarity = existing.iter()
            .map(|e| cosine_similarity(embedding, e))
            .fold(0.0f32, |acc, s| acc.max(s));
        (1.0 - max_similarity).clamp(0.0, 1.0)
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod sensory;
```

**Step 4: Run tests**

Run: `cargo test --test sensory_test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sensory/ src/lib.rs tests/sensory_test.rs
git commit -m "feat: add sensory buffer with content detection, entity extraction, novelty scoring"
```

---

## Phase 4: Persistence (Sleep Consolidation to Disk)

Memory survives restarts. Following Born & Wilhelm.

### Task 6: Add serde support to GraphStore and implement save/load

**Files:**
- Modify: `src/index/graph.rs` (add Serialize/Deserialize, save/load methods)
- Create: `tests/persistence_test.rs`

**Step 1: Write the test**

Create `tests/persistence_test.rs`:
```rust
use engram::index::graph::GraphStore;
use engram::types::enums::{ContentType, RelationType};
use engram::types::memory_node::MemoryNode;
use std::path::PathBuf;

#[test]
fn test_graph_save_and_load() {
    let dir = std::env::temp_dir().join("engram_test_persistence");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("graph.json");

    // Create and populate graph
    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("fact A".to_string(), vec![0.1, 0.2], ContentType::Fact);
    let node_b = MemoryNode::new("fact B".to_string(), vec![0.3, 0.4], ContentType::Fact);
    let id_a = node_a.id;
    let id_b = node_b.id;
    graph.insert(node_a);
    graph.insert(node_b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.8);

    // Save
    graph.save(&path).unwrap();
    assert!(path.exists());

    // Load into new graph
    let loaded = GraphStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    let loaded_a = loaded.get(&id_a).unwrap();
    assert_eq!(loaded_a.content, "fact A");
    assert_eq!(loaded_a.edges.len(), 1);
    assert_eq!(loaded_a.edges[0].target_id, id_b);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_graph_load_nonexistent_returns_empty() {
    let path = PathBuf::from("/tmp/engram_nonexistent_graph.json");
    let graph = GraphStore::load(&path).unwrap();
    assert_eq!(graph.len(), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test persistence_test`
Expected: FAIL (save/load methods don't exist)

**Step 3: Add Serialize/Deserialize to GraphStore and implement save/load**

Modify `src/index/graph.rs` — add derives and methods:

```rust
use serde::{Deserialize, Serialize};
use std::path::Path;
```

Add `#[derive(Serialize, Deserialize)]` to `GraphStore`.

Add methods:
```rust
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let json = std::fs::read_to_string(path)?;
        let store: Self = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(store)
    }
```

**Step 4: Run tests**

Run: `cargo test --test persistence_test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/index/graph.rs tests/persistence_test.rs
git commit -m "feat: add graph persistence with save/load to JSON"
```

---

## Phase 5: Async Engine

Refactor the engine to use real embeddings and async operations.

### Task 7: Create AsyncEngram engine

Rather than break the existing synchronous `Engram` engine (which existing tests depend on), create a new async engine that uses the trait-based providers.

**Files:**
- Create: `src/async_engine.rs`
- Modify: `src/lib.rs` (add `pub mod async_engine;`)
- Create: `tests/async_engine_test.rs`

**Step 1: Write the test**

Create `tests/async_engine_test.rs`:
```rust
use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::embedder::{EmbedError, Embedder};
use engram::provider::llm::{LLMError, LLMProvider};
use engram::compiler::context::LLMContext;
use engram::types::enums::RetrievalIntent;
use async_trait::async_trait;

/// Mock embedder for testing without Ollama
struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // Simple deterministic embedding based on text hash
        let mut embedding = vec![0.0f32; 64];
        for (i, byte) in text.bytes().enumerate() {
            embedding[i % 64] += byte as f32 / 255.0;
        }
        // Normalize
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in &mut embedding {
                *v /= mag;
            }
        }
        Ok(embedding)
    }
    fn dimensions(&self) -> usize { 64 }
}

/// Mock LLM for testing without Ollama
struct MockLLM;

#[async_trait]
impl LLMProvider for MockLLM {
    async fn generate(&self, _system: &str, user_message: &str, context: &LLMContext) -> Result<String, LLMError> {
        let mem_count = context.focal_memories.len();
        Ok(format!("Response to '{user_message}' with {mem_count} memories"))
    }
    fn model_name(&self) -> &str { "mock" }
}

#[tokio::test]
async fn test_async_engine_ingest_and_query() {
    let config = AsyncEngineConfig::default();
    let mut engine = AsyncEngram::new(
        config,
        Box::new(MockEmbedder),
        Box::new(MockLLM),
    );

    engine.ingest("Rust is a systems programming language").await.unwrap();
    engine.ingest("Python is great for data science").await.unwrap();
    engine.ingest("Rust and Python work together via PyO3").await.unwrap();

    let context = engine.retrieve("How do Rust and Python work together?", RetrievalIntent::Recall).await.unwrap();
    assert!(!context.focal_memories.is_empty());
}

#[tokio::test]
async fn test_async_engine_chat() {
    let config = AsyncEngineConfig::default();
    let mut engine = AsyncEngram::new(
        config,
        Box::new(MockEmbedder),
        Box::new(MockLLM),
    );

    engine.ingest("The capital of France is Paris").await.unwrap();
    let response = engine.chat("What is the capital of France?").await.unwrap();
    assert!(response.contains("Response to"));
}

#[tokio::test]
async fn test_async_engine_persistence() {
    let dir = std::env::temp_dir().join("engram_async_persist_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = AsyncEngineConfig {
        data_dir: dir.clone(),
        ..Default::default()
    };

    // Create engine, ingest, save
    {
        let mut engine = AsyncEngram::new(
            config.clone(),
            Box::new(MockEmbedder),
            Box::new(MockLLM),
        );
        engine.ingest("persistent fact").await.unwrap();
        assert_eq!(engine.node_count(), 1);
        engine.save().unwrap();
    }

    // Load into new engine
    {
        let config2 = AsyncEngineConfig {
            data_dir: dir.clone(),
            ..Default::default()
        };
        let engine = AsyncEngram::load(
            config2,
            Box::new(MockEmbedder),
            Box::new(MockLLM),
        ).unwrap();
        assert_eq!(engine.node_count(), 1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test async_engine_test`
Expected: FAIL (module doesn't exist)

**Step 3: Implement AsyncEngram**

Create `src/async_engine.rs`:
```rust
use std::path::PathBuf;

use crate::compiler::context::{compile, CompileConfig, LLMContext};
use crate::index::graph::GraphStore;
use crate::index::vector::find_seeds_by_similarity;
use crate::memory::working::WorkingMemory;
use crate::provider::embedder::{EmbedError, Embedder};
use crate::provider::llm::{LLMError, LLMProvider};
use crate::retrieval::activation::{spread, ActivationConfig};
use crate::router::salience::route;
use crate::sensory::buffer::SensoryBuffer;
use crate::types::constellation::MemoryConstellation;
use crate::types::enums::{ContentType, RelationType, RetrievalIntent};
use crate::types::memory_node::MemoryNode;

#[derive(Clone)]
pub struct AsyncEngineConfig {
    pub working_memory_capacity: usize,
    pub token_budget: usize,
    pub data_dir: PathBuf,
    pub system_prompt: String,
}

impl Default for AsyncEngineConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 5,
            token_budget: 2000,
            data_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".engram"),
            system_prompt: "You are a helpful assistant with access to a neuromorphic memory system. \
                Use the provided memories to give informed, contextual responses. \
                If memories are relevant, reference them naturally. \
                If memory confidence is low, acknowledge uncertainty.".to_string(),
        }
    }
}

pub struct AsyncEngram {
    graph: GraphStore,
    working_memory: WorkingMemory,
    embedder: Box<dyn Embedder>,
    llm: Box<dyn LLMProvider>,
    config: AsyncEngineConfig,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("embedding error: {0}")]
    Embed(#[from] EmbedError),
    #[error("llm error: {0}")]
    LLM(#[from] LLMError),
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
}

impl AsyncEngram {
    pub fn new(
        config: AsyncEngineConfig,
        embedder: Box<dyn Embedder>,
        llm: Box<dyn LLMProvider>,
    ) -> Self {
        let working_memory = WorkingMemory::new(config.working_memory_capacity);
        Self {
            graph: GraphStore::new(),
            working_memory,
            embedder,
            llm,
            config,
        }
    }

    pub fn load(
        config: AsyncEngineConfig,
        embedder: Box<dyn Embedder>,
        llm: Box<dyn LLMProvider>,
    ) -> Result<Self, EngineError> {
        let graph_path = config.data_dir.join("graph.json");
        let graph = GraphStore::load(&graph_path)?;
        let working_memory = WorkingMemory::new(config.working_memory_capacity);
        Ok(Self {
            graph,
            working_memory,
            embedder,
            llm,
            config,
        })
    }

    pub fn save(&self) -> Result<(), EngineError> {
        std::fs::create_dir_all(&self.config.data_dir)?;
        let graph_path = self.config.data_dir.join("graph.json");
        self.graph.save(&graph_path)?;
        Ok(())
    }

    /// Ingest content through the sensory buffer (deep encoding).
    pub async fn ingest(&mut self, content: &str) -> Result<uuid::Uuid, EngineError> {
        let embedding = self.embedder.embed(content).await?;

        // Deep encoding: detect content type
        let content_type = SensoryBuffer::detect_content_type(content);

        // Score novelty against existing graph
        let existing_embeddings: Vec<Vec<f32>> = self.graph
            .all_nodes()
            .map(|n| n.embedding.clone())
            .collect();
        let novelty = SensoryBuffer::score_novelty(&embedding, &existing_embeddings);

        // Initial salience based on novelty
        let salience = 0.3 + (novelty * 0.4); // Range: 0.3 to 0.7

        let node = MemoryNode::new(content.to_string(), embedding.clone(), content_type)
            .with_salience(salience);
        let node_id = node.id;

        // Find similar existing nodes and create edges
        let seeds = find_seeds_by_similarity(&self.graph, &embedding, 3, 0.3);
        self.graph.insert(node);

        for (similar_id, similarity) in seeds {
            if similar_id != node_id {
                self.graph.add_edge(node_id, similar_id, RelationType::Semantic, similarity);
                self.graph.add_edge(similar_id, node_id, RelationType::Semantic, similarity);
            }
        }

        Ok(node_id)
    }

    /// Retrieve relevant context for a query (without generating LLM response).
    pub async fn retrieve(&mut self, text: &str, intent: RetrievalIntent) -> Result<LLMContext, EngineError> {
        let query_embedding = self.embedder.embed(text).await?;

        if self.graph.is_empty() {
            return Ok(compile(
                &MemoryConstellation::empty(),
                &CompileConfig { token_budget: self.config.token_budget },
            ));
        }

        let coverage = self.working_memory.assess_coverage(&query_embedding);
        let salience = 0.5; // TODO: compute from query analysis
        let strategy = route(salience, coverage);

        let (max_depth, max_nodes) = match &strategy {
            crate::router::salience::RetrievalStrategy::Skip => {
                let constellation = self.constellation_from_working_memory();
                return Ok(compile(
                    &constellation,
                    &CompileConfig { token_budget: self.config.token_budget },
                ));
            }
            crate::router::salience::RetrievalStrategy::Deep { max_depth, max_nodes } => (*max_depth, *max_nodes),
            crate::router::salience::RetrievalStrategy::Targeted { max_depth, max_nodes } => (*max_depth, *max_nodes),
            crate::router::salience::RetrievalStrategy::Standard { max_depth, max_nodes } => (*max_depth, *max_nodes),
            crate::router::salience::RetrievalStrategy::Light { max_depth, max_nodes } => (*max_depth, *max_nodes),
        };

        let seeds = find_seeds_by_similarity(&self.graph, &query_embedding, max_nodes, 0.1);

        if seeds.is_empty() {
            return Ok(compile(
                &MemoryConstellation::empty(),
                &CompileConfig { token_budget: self.config.token_budget },
            ));
        }

        let activation_config = ActivationConfig { max_depth, ..Default::default() };
        let activations = spread(&self.graph, &seeds, &activation_config);

        let mut constellation = MemoryConstellation::empty();
        let mut sorted: Vec<_> = activations.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let focal_count = max_nodes.min(sorted.len()).min(5);
        for (id, activation) in sorted.iter().take(focal_count) {
            if let Some(node) = self.graph.get(id) {
                constellation.add_focal(node.clone(), *activation);
            }
        }
        for (id, activation) in sorted.iter().skip(focal_count).take(10) {
            if let Some(node) = self.graph.get(id) {
                constellation.add_context(node.clone(), *activation);
            }
        }

        if !constellation.focal_nodes.is_empty() {
            constellation.confidence = constellation.focal_nodes[0].activation.clamp(0.0, 1.0);
            constellation.coverage = (constellation.focal_nodes.len() as f32 / max_nodes as f32).min(1.0);
        }

        let wm_salience = constellation.confidence;
        let wm_constellation = constellation.clone();
        self.working_memory.add(wm_constellation, wm_salience);

        Ok(compile(
            &constellation,
            &CompileConfig { token_budget: self.config.token_budget },
        ))
    }

    /// Full chat: retrieve context + generate LLM response + store episodic memory.
    pub async fn chat(&mut self, message: &str) -> Result<String, EngineError> {
        // Retrieve relevant context
        let context = self.retrieve(message, RetrievalIntent::Recall).await?;

        // Generate LLM response
        let response = self.llm.generate(&self.config.system_prompt, message, &context).await?;

        // Store the conversation turn as episodic memory
        let turn = format!("User: {message}\nAssistant: {response}");
        self.ingest(&turn).await?;

        Ok(response)
    }

    fn constellation_from_working_memory(&self) -> MemoryConstellation {
        let mut constellation = MemoryConstellation::empty();
        for wm_constellation in self.working_memory.constellations() {
            for focal in &wm_constellation.focal_nodes {
                constellation.add_focal(focal.node.clone(), focal.activation);
            }
        }
        constellation.confidence = 0.8;
        constellation.coverage = 0.9;
        constellation
    }

    pub fn node_count(&self) -> usize {
        self.graph.len()
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod async_engine;
```

**Step 4: Run tests**

Run: `cargo test --test async_engine_test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/async_engine.rs src/lib.rs tests/async_engine_test.rs
git commit -m "feat: add AsyncEngram engine with real embeddings, LLM, and persistence"
```

---

## Phase 6: CLI Chat Agent

The consciousness loop — interactive chat with persistent memory.

### Task 8: Build the CLI binary

**Files:**
- Create: `src/main.rs`
- Modify: `Cargo.toml` (add `[[bin]]` section)

**Step 1: Add binary target to Cargo.toml**

Add to `Cargo.toml`:
```toml
[[bin]]
name = "engram"
path = "src/main.rs"
```

**Step 2: Write the CLI**

Create `src/main.rs`:
```rust
use clap::{Parser, Subcommand};
use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::ollama::{OllamaEmbedder, OllamaLLM};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "engram", about = "Neuromorphic memory for LLMs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama server URL
    #[arg(long, default_value = "http://localhost:11434", global = true)]
    ollama_url: String,

    /// Embedding model
    #[arg(long, default_value = "nomic-embed-text", global = true)]
    embed_model: String,

    /// LLM model
    #[arg(long, default_value = "llama3.2", global = true)]
    llm_model: String,

    /// Data directory
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session with persistent memory
    Chat,
    /// Ingest a file or directory into memory
    Ingest {
        /// Path to file or directory
        path: PathBuf,
    },
    /// Show memory statistics
    Stats,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let data_dir = cli.data_dir.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".engram")
    });

    let config = AsyncEngineConfig {
        data_dir: data_dir.clone(),
        ..Default::default()
    };

    let embedder = Box::new(OllamaEmbedder::new(&cli.ollama_url, &cli.embed_model));
    let llm = Box::new(OllamaLLM::new(&cli.ollama_url, &cli.llm_model));

    match cli.command {
        Commands::Chat => run_chat(config, embedder, llm).await,
        Commands::Ingest { path } => run_ingest(config, embedder, llm, &path).await,
        Commands::Stats => run_stats(config, embedder, llm),
    }
}

async fn run_chat(
    config: AsyncEngineConfig,
    embedder: Box<OllamaEmbedder>,
    llm: Box<OllamaLLM>,
) {
    let model_name = llm.model_name().to_string();
    let mut engine = AsyncEngram::load(config, embedder, llm)
        .unwrap_or_else(|_| panic!("Failed to load engine"));

    eprintln!("Engram v0.1.0 | Memory: {} nodes | Model: {}", engine.node_count(), model_name);
    eprintln!("Type 'quit' to exit. Memory is saved on exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        eprint!("You: ");
        stdout.flush().unwrap();

        let mut input = String::new();
        stdin.lock().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        match engine.chat(input).await {
            Ok(response) => {
                eprintln!("\nEngram: {response}\n");
            }
            Err(e) => {
                eprintln!("\nError: {e}\n");
            }
        }
    }

    // Save on exit
    if let Err(e) = engine.save() {
        eprintln!("Warning: failed to save memory: {e}");
    } else {
        eprintln!("Memory saved ({} nodes).", engine.node_count());
    }
}

async fn run_ingest(
    config: AsyncEngineConfig,
    embedder: Box<OllamaEmbedder>,
    llm: Box<OllamaLLM>,
    path: &std::path::Path,
) {
    let mut engine = AsyncEngram::load(config, embedder, llm)
        .unwrap_or_else(|_| panic!("Failed to load engine"));

    let before = engine.node_count();

    if path.is_file() {
        ingest_file(&mut engine, path).await;
    } else if path.is_dir() {
        ingest_directory(&mut engine, path).await;
    } else {
        eprintln!("Path does not exist: {}", path.display());
        return;
    }

    let after = engine.node_count();
    eprintln!("Ingested {} new nodes (total: {})", after - before, after);

    if let Err(e) = engine.save() {
        eprintln!("Warning: failed to save memory: {e}");
    }
}

async fn ingest_file(engine: &mut AsyncEngram, path: &std::path::Path) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to read {}: {e}", path.display());
            return;
        }
    };

    // Chunk by paragraphs (double newline) or by lines for code
    let chunks = chunk_content(&content);
    for chunk in &chunks {
        if chunk.trim().is_empty() {
            continue;
        }
        match engine.ingest(chunk).await {
            Ok(_) => {}
            Err(e) => eprintln!("  Failed to ingest chunk: {e}"),
        }
    }
    eprintln!("  {} -> {} chunks", path.display(), chunks.len());
}

async fn ingest_directory(engine: &mut AsyncEngram, path: &std::path::Path) {
    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read directory {}: {e}", path.display());
            return;
        }
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() {
            // Skip binary/hidden files
            let name = entry_path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            let ext = entry_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let text_extensions = ["md", "txt", "rs", "py", "js", "ts", "toml", "yaml", "yml", "json", "html", "css"];
            if text_extensions.contains(&ext) {
                ingest_file(engine, &entry_path).await;
            }
        } else if entry_path.is_dir() {
            let name = entry_path.file_name().unwrap_or_default().to_string_lossy();
            if !name.starts_with('.') && name != "target" && name != "node_modules" {
                // Box the recursive future for directory ingestion
                Box::pin(ingest_directory(engine, &entry_path)).await;
            }
        }
    }
}

fn chunk_content(content: &str) -> Vec<String> {
    // Split by double newlines (paragraphs) with a max chunk size
    let max_chunk = 500; // chars
    let mut chunks = Vec::new();
    let paragraphs: Vec<&str> = content.split("\n\n").collect();

    let mut current = String::new();
    for para in paragraphs {
        if current.len() + para.len() > max_chunk && !current.is_empty() {
            chunks.push(current.clone());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn run_stats(
    config: AsyncEngineConfig,
    embedder: Box<OllamaEmbedder>,
    llm: Box<OllamaLLM>,
) {
    let engine = AsyncEngram::load(config, embedder, llm)
        .unwrap_or_else(|_| panic!("Failed to load engine"));
    eprintln!("Engram Memory Stats");
    eprintln!("  Nodes: {}", engine.node_count());
}
```

**Step 3: Verify it compiles**

Run: `cargo build`
Expected: compiles successfully

**Step 4: Commit**

```bash
git add src/main.rs Cargo.toml
git commit -m "feat: add CLI binary with chat, ingest, and stats commands"
```

---

## Phase 7: Integration Test (End-to-End)

### Task 9: End-to-end test with Ollama

**Files:**
- Create: `tests/e2e_test.rs`

**Step 1: Write the e2e test**

Create `tests/e2e_test.rs`:
```rust
use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::ollama::{OllamaEmbedder, OllamaLLM};
use engram::types::enums::RetrievalIntent;

/// Full end-to-end test with real Ollama.
/// Requires: ollama running with nomic-embed-text and llama3.2 pulled.
#[tokio::test]
async fn test_e2e_ingest_and_chat() {
    let dir = std::env::temp_dir().join("engram_e2e_test");
    let _ = std::fs::remove_dir_all(&dir);

    let config = AsyncEngineConfig {
        data_dir: dir.clone(),
        ..Default::default()
    };

    let embedder = Box::new(OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text"));
    let llm = Box::new(OllamaLLM::new("http://localhost:11434", "llama3.2"));

    let mut engine = AsyncEngram::new(config, embedder, llm);

    // Ingest some facts
    let result = engine.ingest("Engram is a neuromorphic memory system for LLMs").await;
    match result {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Skipping e2e test (Ollama not available): {e}");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
    }

    engine.ingest("It uses spreading activation like the human brain").await.unwrap();
    engine.ingest("Memory nodes are connected by weighted edges").await.unwrap();

    assert_eq!(engine.node_count(), 3);

    // Retrieve
    let context = engine.retrieve("How does Engram work?", RetrievalIntent::Recall).await.unwrap();
    assert!(!context.focal_memories.is_empty());

    // Chat
    let response = engine.chat("What is Engram?").await.unwrap();
    assert!(!response.is_empty());
    eprintln!("LLM response: {response}");

    // Verify episodic memory was stored (chat turn)
    assert!(engine.node_count() > 3);

    // Test persistence
    engine.save().unwrap();

    let config2 = AsyncEngineConfig {
        data_dir: dir.clone(),
        ..Default::default()
    };
    let embedder2 = Box::new(OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text"));
    let llm2 = Box::new(OllamaLLM::new("http://localhost:11434", "llama3.2"));
    let loaded = AsyncEngram::load(config2, embedder2, llm2).unwrap();
    assert!(loaded.node_count() > 3);

    let _ = std::fs::remove_dir_all(&dir);
}
```

**Step 2: Run the test**

Run: `cargo test --test e2e_test -- --nocapture`
Expected: PASS if Ollama is running, skip message otherwise

**Step 3: Commit**

```bash
git add tests/e2e_test.rs
git commit -m "test: add end-to-end integration test with Ollama"
```

---

## Summary

| Phase | Task | Component | Brain Analog |
|---|---|---|---|
| 1 | 1 | Async dependencies | Nervous system wiring |
| 1 | 2 | Embedder trait | Sensory cortex interface |
| 1 | 3 | OllamaEmbedder | Sensory cortex implementation |
| 2 | 4 | LLMProvider + OllamaLLM | Prefrontal cortex |
| 3 | 5 | Sensory Buffer | Sensory register |
| 4 | 6 | Persistence | Sleep consolidation |
| 5 | 7 | AsyncEngram engine | Consciousness integration |
| 6 | 8 | CLI binary | Consciousness loop |
| 7 | 9 | E2E test | Full brain test |

**Prerequisites:** Ollama installed with `nomic-embed-text` and `llama3.2` pulled:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```
