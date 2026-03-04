use std::path::PathBuf;

use crate::compiler::context::{compile, CompileConfig, LLMContext};
use crate::index::graph::GraphStore;
use crate::index::vector::find_seeds_by_similarity;
use crate::memory::working::WorkingMemory;
use crate::provider::embedder::{EmbedError, Embedder};
use crate::provider::llm::{LLMError, LLMProvider};
use crate::retrieval::activation::{spread, ActivationConfig};
use crate::router::salience::{route, RetrievalStrategy};
use crate::sensory::buffer::SensoryBuffer;
use crate::types::constellation::MemoryConstellation;
use crate::types::enums::{RelationType, RetrievalIntent};
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
                If memory confidence is low, acknowledge uncertainty."
                .to_string(),
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

        let content_type = SensoryBuffer::detect_content_type(content);

        let existing_embeddings: Vec<Vec<f32>> =
            self.graph.all_nodes().map(|n| n.embedding.clone()).collect();
        let novelty = SensoryBuffer::score_novelty(&embedding, &existing_embeddings);

        let salience = 0.3 + (novelty * 0.4);

        let node = MemoryNode::new(content.to_string(), embedding.clone(), content_type)
            .with_salience(salience);
        let node_id = node.id;

        let seeds = find_seeds_by_similarity(&self.graph, &embedding, 3, 0.3);
        self.graph.insert(node);

        for (similar_id, similarity) in seeds {
            if similar_id != node_id {
                self.graph
                    .add_edge(node_id, similar_id, RelationType::Semantic, similarity);
                self.graph
                    .add_edge(similar_id, node_id, RelationType::Semantic, similarity);
            }
        }

        Ok(node_id)
    }

    /// Retrieve relevant context for a query.
    pub async fn retrieve(
        &mut self,
        text: &str,
        _intent: RetrievalIntent,
    ) -> Result<LLMContext, EngineError> {
        let query_embedding = self.embedder.embed(text).await?;

        if self.graph.is_empty() {
            return Ok(compile(
                &MemoryConstellation::empty(),
                &CompileConfig {
                    token_budget: self.config.token_budget,
                },
            ));
        }

        let coverage = self.working_memory.assess_coverage(&query_embedding);
        let salience = 0.5;
        let strategy = route(salience, coverage);

        let (max_depth, max_nodes) = match &strategy {
            RetrievalStrategy::Skip => {
                let constellation = self.constellation_from_working_memory();
                return Ok(compile(
                    &constellation,
                    &CompileConfig {
                        token_budget: self.config.token_budget,
                    },
                ));
            }
            RetrievalStrategy::Deep {
                max_depth,
                max_nodes,
            } => (*max_depth, *max_nodes),
            RetrievalStrategy::Targeted {
                max_depth,
                max_nodes,
            } => (*max_depth, *max_nodes),
            RetrievalStrategy::Standard {
                max_depth,
                max_nodes,
            } => (*max_depth, *max_nodes),
            RetrievalStrategy::Light {
                max_depth,
                max_nodes,
            } => (*max_depth, *max_nodes),
        };

        let seeds = find_seeds_by_similarity(&self.graph, &query_embedding, max_nodes, 0.1);

        if seeds.is_empty() {
            return Ok(compile(
                &MemoryConstellation::empty(),
                &CompileConfig {
                    token_budget: self.config.token_budget,
                },
            ));
        }

        let activation_config = ActivationConfig {
            max_depth,
            ..Default::default()
        };
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
            constellation.coverage =
                (constellation.focal_nodes.len() as f32 / max_nodes as f32).min(1.0);
        }

        let wm_salience = constellation.confidence;
        let wm_constellation = constellation.clone();
        self.working_memory.add(wm_constellation, wm_salience);

        Ok(compile(
            &constellation,
            &CompileConfig {
                token_budget: self.config.token_budget,
            },
        ))
    }

    /// Full chat: retrieve context + generate LLM response + store episodic memory.
    pub async fn chat(&mut self, message: &str) -> Result<String, EngineError> {
        let context = self.retrieve(message, RetrievalIntent::Recall).await?;
        let response = self
            .llm
            .generate(&self.config.system_prompt, message, &context)
            .await?;

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
