use serde::{Deserialize, Serialize};

use crate::types::constellation::MemoryConstellation;

pub struct CompileConfig {
    pub token_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalMemory {
    pub content: String,
    pub relevance: f32,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMContext {
    pub focal_memories: Vec<FocalMemory>,
    pub relationship_map: String,
    pub confidence: f32,
    pub coverage: f32,
    pub gaps: Vec<String>,
}

fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 characters per token
    (text.len() + 3) / 4
}

pub fn compile(constellation: &MemoryConstellation, config: &CompileConfig) -> LLMContext {
    let mut focal_memories = Vec::new();
    let mut tokens_used: usize = 0;

    // Reserve 10% for metadata (confidence, coverage, gaps)
    let content_budget = (config.token_budget as f32 * 0.9) as usize;

    // Add focal nodes in activation order (already sorted)
    for activated in &constellation.focal_nodes {
        let content = &activated.node.content;
        let content_tokens = estimate_tokens(content);

        if tokens_used + content_tokens > content_budget {
            break;
        }

        focal_memories.push(FocalMemory {
            content: content.clone(),
            relevance: activated.activation,
            source: activated.node.source.clone(),
        });
        tokens_used += content_tokens;
    }

    // Add context nodes as shorter summaries if budget allows
    for activated in &constellation.context_nodes {
        let content = &activated.node.content;
        // Truncate context nodes to ~50 chars for summary
        let summary: String = content.chars().take(50).collect();
        let summary_tokens = estimate_tokens(&summary);

        if tokens_used + summary_tokens > content_budget {
            break;
        }

        focal_memories.push(FocalMemory {
            content: summary,
            relevance: activated.activation,
            source: activated.node.source.clone(),
        });
        tokens_used += summary_tokens;
    }

    LLMContext {
        focal_memories,
        relationship_map: String::new(),
        confidence: constellation.confidence,
        coverage: constellation.coverage,
        gaps: constellation.gaps.clone(),
    }
}
