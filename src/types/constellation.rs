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
