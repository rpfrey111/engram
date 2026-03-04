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
