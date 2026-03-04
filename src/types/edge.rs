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
