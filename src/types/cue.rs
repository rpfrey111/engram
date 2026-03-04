use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::enums::RetrievalIntent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCue {
    pub semantic: Vec<f32>,
    pub temporal: Option<TimeRange>,
    pub entities: Vec<String>,
    pub context: HashMap<String, f32>,
    pub intent: RetrievalIntent,
    pub salience_floor: f32,
}

impl RetrievalCue {
    pub fn new(semantic: Vec<f32>, intent: RetrievalIntent) -> Self {
        Self {
            semantic,
            temporal: None,
            entities: Vec::new(),
            context: HashMap::new(),
            intent,
            salience_floor: 0.0,
        }
    }

    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_salience_floor(mut self, floor: f32) -> Self {
        self.salience_floor = floor.clamp(0.0, 1.0);
        self
    }

    pub fn with_temporal(mut self, range: TimeRange) -> Self {
        self.temporal = Some(range);
        self
    }

    pub fn with_context(mut self, context: HashMap<String, f32>) -> Self {
        self.context = context;
        self
    }
}
