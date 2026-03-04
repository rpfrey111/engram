import pytest


def test_import():
    import engram_python
    assert hasattr(engram_python, 'Engram')


def test_create_engine():
    from engram_python import Engram
    engine = Engram()
    assert engine.node_count() == 0


def test_ingest_and_query():
    from engram_python import Engram
    engine = Engram()
    engine.ingest("Rust is a systems programming language", "fact", 0.7)
    engine.ingest("Python is great for AI", "fact", 0.6)
    result = engine.query("What is Rust?", "recall")
    assert result["confidence"] > 0
    assert len(result["focal_memories"]) > 0
