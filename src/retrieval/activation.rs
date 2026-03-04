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

    activations
}
