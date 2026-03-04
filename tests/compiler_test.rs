use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::types::constellation::MemoryConstellation;
use engram::compiler::context::{compile, CompileConfig};

#[test]
fn test_compile_empty_constellation() {
    let constellation = MemoryConstellation::empty();
    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);
    assert!(ctx.focal_memories.is_empty());
    assert!((ctx.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_compile_with_focal_nodes() {
    let mut constellation = MemoryConstellation::empty();
    let node1 = MemoryNode::new("Important fact about Rust".to_string(), vec![0.9], ContentType::Fact);
    let node2 = MemoryNode::new("Less important detail".to_string(), vec![0.3], ContentType::Text);
    constellation.add_focal(node1, 0.9);
    constellation.add_focal(node2, 0.4);
    constellation.confidence = 0.85;
    constellation.coverage = 0.7;

    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);

    assert_eq!(ctx.focal_memories.len(), 2);
    assert!((ctx.confidence - 0.85).abs() < f32::EPSILON);
    // Higher activation node should be first
    assert!(ctx.focal_memories[0].relevance > ctx.focal_memories[1].relevance);
}

#[test]
fn test_compile_includes_gaps() {
    let mut constellation = MemoryConstellation::empty();
    constellation.gaps = vec!["No data about Q4 revenue".to_string()];
    constellation.confidence = 0.4;

    let config = CompileConfig { token_budget: 1000 };
    let ctx = compile(&constellation, &config);

    assert_eq!(ctx.gaps.len(), 1);
    assert_eq!(ctx.gaps[0], "No data about Q4 revenue");
}

#[test]
fn test_compile_respects_token_budget() {
    let mut constellation = MemoryConstellation::empty();
    // Add many nodes with long content
    for i in 0..50 {
        let content = format!("This is memory node number {} with substantial content that takes up tokens in the context window", i);
        let node = MemoryNode::new(content, vec![0.5], ContentType::Text);
        constellation.add_focal(node, 0.5 + (i as f32 * 0.01));
    }
    constellation.confidence = 0.9;

    let config = CompileConfig { token_budget: 200 }; // very tight budget
    let ctx = compile(&constellation, &config);

    // Should not include all 50 nodes
    assert!(ctx.focal_memories.len() < 50);
    // Rough token estimate: each word ~1 token, should fit in budget
    let total_chars: usize = ctx.focal_memories.iter().map(|m| m.content.len()).sum();
    assert!(total_chars / 4 < 200); // rough estimate: 4 chars per token
}
