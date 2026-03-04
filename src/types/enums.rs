use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
    Code,
    Conversation,
    Event,
    Fact,
    Skill,
    Entity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Semantic,
    Temporal,
    Causal,
    Contextual,
    Hierarchical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum AbstractionLevel {
    Raw = 0,
    Chunk = 1,
    Summary = 2,
    Schema = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalIntent {
    Recall,
    Recognize,
    Explore,
    Verify,
}
