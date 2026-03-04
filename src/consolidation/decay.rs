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
