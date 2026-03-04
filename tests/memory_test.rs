use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::types::constellation::MemoryConstellation;
use engram::memory::working::WorkingMemory;

#[test]
fn test_working_memory_capacity() {
    let wm = WorkingMemory::new(4);
    assert_eq!(wm.capacity(), 4);
    assert_eq!(wm.len(), 0);
    assert!(wm.is_empty());
}

#[test]
fn test_working_memory_add_and_get() {
    let mut wm = WorkingMemory::new(4);
    let mut constellation = MemoryConstellation::empty();
    let node = MemoryNode::new("test".to_string(), vec![0.1], ContentType::Fact);
    constellation.add_focal(node, 0.9);
    constellation.confidence = 0.8;

    wm.add(constellation, 0.9);
    assert_eq!(wm.len(), 1);
}

#[test]
fn test_working_memory_eviction_at_capacity() {
    let mut wm = WorkingMemory::new(2);

    let mut c1 = MemoryConstellation::empty();
    c1.confidence = 0.5;
    wm.add(c1, 0.3); // low salience

    let mut c2 = MemoryConstellation::empty();
    c2.confidence = 0.8;
    wm.add(c2, 0.9); // high salience

    assert_eq!(wm.len(), 2);

    let mut c3 = MemoryConstellation::empty();
    c3.confidence = 0.7;
    wm.add(c3, 0.6); // medium salience

    // Should still be at capacity, lowest salience (0.3) evicted
    assert_eq!(wm.len(), 2);
}

#[test]
fn test_working_memory_clear() {
    let mut wm = WorkingMemory::new(4);
    wm.add(MemoryConstellation::empty(), 0.5);
    wm.add(MemoryConstellation::empty(), 0.5);
    assert_eq!(wm.len(), 2);
    wm.clear();
    assert_eq!(wm.len(), 0);
}

#[test]
fn test_working_memory_coverage_assessment() {
    let mut wm = WorkingMemory::new(4);
    let mut c = MemoryConstellation::empty();
    let node = MemoryNode::new("rust programming".to_string(), vec![0.9, 0.1, 0.0], ContentType::Fact);
    c.add_focal(node, 0.9);
    c.coverage = 0.85;
    wm.add(c, 0.8);

    // Query similar to what's in working memory
    let coverage = wm.assess_coverage(&vec![0.85, 0.15, 0.0]);
    assert!(coverage > 0.5); // should show good coverage
}
