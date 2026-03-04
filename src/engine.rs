use crate::compiler::context::{compile, CompileConfig, LLMContext};
use crate::index::graph::GraphStore;
use crate::index::vector::find_seeds_by_similarity;
use crate::memory::working::WorkingMemory;
use crate::retrieval::activation::{spread, ActivationConfig};
use crate::router::salience::{route, RetrievalStrategy};
use crate::types::constellation::MemoryConstellation;
use crate::types::enums::{ContentType, RelationType, RetrievalIntent};
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
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered.split_whitespace().collect();
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
        let seeds = find_seeds_by_similarity(&self.graph, &node.embedding, 3, 0.3);
        self.graph.insert(node);

        for (similar_id, similarity) in seeds {
            if similar_id != node_id {
                self.graph.add_edge(
                    node_id,
                    similar_id,
                    RelationType::Semantic,
                    similarity,
                );
                self.graph.add_edge(
                    similar_id,
                    node_id,
                    RelationType::Semantic,
                    similarity,
                );
            }
        }
    }

    pub fn query(&mut self, text: &str, _intent: RetrievalIntent) -> LLMContext {
        let query_embedding = self.embed(text);

        if self.graph.is_empty() {
            return compile(
                &MemoryConstellation::empty(),
                &CompileConfig {
                    token_budget: self.config.token_budget,
                },
            );
        }

        // Check working memory coverage
        let coverage = self.working_memory.assess_coverage(&query_embedding);

        // Route based on salience (use 0.5 as proxy for now)
        let salience = 0.5;
        let strategy = route(salience, coverage);

        let (max_depth, max_nodes) = match &strategy {
            RetrievalStrategy::Skip => {
                let constellation = self.constellation_from_working_memory(&query_embedding);
                return compile(
                    &constellation,
                    &CompileConfig {
                        token_budget: self.config.token_budget,
                    },
                );
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

        // Find seeds
        let seeds = find_seeds_by_similarity(&self.graph, &query_embedding, max_nodes, 0.1);

        if seeds.is_empty() {
            return compile(
                &MemoryConstellation::empty(),
                &CompileConfig {
                    token_budget: self.config.token_budget,
                },
            );
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
            constellation.coverage =
                (constellation.focal_nodes.len() as f32 / max_nodes as f32).min(1.0);
        }

        // Store in working memory
        let wm_salience = constellation.confidence;
        let wm_constellation = constellation.clone();
        self.working_memory.add(wm_constellation, wm_salience);

        compile(
            &constellation,
            &CompileConfig {
                token_budget: self.config.token_budget,
            },
        )
    }

    fn constellation_from_working_memory(
        &self,
        _query_embedding: &[f32],
    ) -> MemoryConstellation {
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
