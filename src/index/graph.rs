use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::edge::Edge;
use crate::types::enums::RelationType;
use crate::types::memory_node::MemoryNode;

#[derive(Serialize, Deserialize)]
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
}
