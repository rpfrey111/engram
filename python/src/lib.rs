use pyo3::prelude::*;
use pyo3::types::PyDict;
use engram::engine::{Engram as EngineCore, EngineConfig};
use engram::types::enums::{ContentType, RetrievalIntent};

#[pyclass]
struct Engram {
    inner: EngineCore,
}

#[pymethods]
impl Engram {
    #[new]
    fn new() -> Self {
        Self {
            inner: EngineCore::new(EngineConfig::default()),
        }
    }

    fn ingest(&mut self, content: &str, content_type: &str, salience: f32) {
        let ct = match content_type {
            "text" => ContentType::Text,
            "code" => ContentType::Code,
            "conversation" => ContentType::Conversation,
            "event" => ContentType::Event,
            "fact" => ContentType::Fact,
            "skill" => ContentType::Skill,
            "entity" => ContentType::Entity,
            _ => ContentType::Text,
        };
        self.inner.ingest(content, ct, salience);
    }

    fn query(&mut self, py: Python, text: &str, intent: &str) -> PyResult<PyObject> {
        let ri = match intent {
            "recall" => RetrievalIntent::Recall,
            "recognize" => RetrievalIntent::Recognize,
            "explore" => RetrievalIntent::Explore,
            "verify" => RetrievalIntent::Verify,
            _ => RetrievalIntent::Recall,
        };
        let context = self.inner.query(text, ri);
        let dict = PyDict::new(py);
        dict.set_item("confidence", context.confidence)?;
        dict.set_item("coverage", context.coverage)?;
        dict.set_item("gaps", context.gaps.clone())?;

        let memories: Vec<PyObject> = context
            .focal_memories
            .iter()
            .map(|m| {
                let d = PyDict::new(py);
                d.set_item("content", m.content.clone()).unwrap();
                d.set_item("relevance", m.relevance).unwrap();
                d.set_item("source", m.source.clone()).unwrap();
                d.into_any().unbind()
            })
            .collect();
        dict.set_item("focal_memories", memories)?;

        Ok(dict.into_any().unbind())
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
}

#[pymodule]
fn engram_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engram>()?;
    Ok(())
}
