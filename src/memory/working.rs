use crate::index::vector::cosine_similarity;
use crate::types::constellation::MemoryConstellation;

struct WorkingMemorySlot {
    constellation: MemoryConstellation,
    salience: f32,
}

pub struct WorkingMemory {
    slots: Vec<WorkingMemorySlot>,
    capacity: usize,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn add(&mut self, constellation: MemoryConstellation, salience: f32) {
        if self.slots.len() >= self.capacity {
            // Evict lowest salience
            if let Some((min_idx, _)) = self
                .slots
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.salience.partial_cmp(&b.1.salience).unwrap())
            {
                if salience > self.slots[min_idx].salience {
                    self.slots.remove(min_idx);
                } else {
                    return; // new item is lower salience than everything, skip
                }
            }
        }
        self.slots.push(WorkingMemorySlot {
            constellation,
            salience,
        });
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }

    pub fn assess_coverage(&self, query_embedding: &[f32]) -> f32 {
        if self.slots.is_empty() {
            return 0.0;
        }

        let mut max_coverage: f32 = 0.0;
        for slot in &self.slots {
            for activated in &slot.constellation.focal_nodes {
                let sim = cosine_similarity(&activated.node.embedding, query_embedding);
                max_coverage = max_coverage.max(sim * slot.constellation.coverage);
            }
        }
        max_coverage.clamp(0.0, 1.0)
    }

    pub fn constellations(&self) -> impl Iterator<Item = &MemoryConstellation> {
        self.slots.iter().map(|s| &s.constellation)
    }
}
