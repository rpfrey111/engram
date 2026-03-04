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
