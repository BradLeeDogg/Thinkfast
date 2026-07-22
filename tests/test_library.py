"""Library retrieval (RAG): chunking, vector search, and the search_library tool.

The vector-search tests need numpy, so they skip cleanly when it isn't installed
(the CI run installs no heavy deps); chunking and the no-index path always run.
"""
import json

import pytest

from agent.library import chunk_text


def test_chunk_text_overlap_and_source():
    chunks = chunk_text("word " * 500, source="book.pdf", chunk_chars=1000, overlap=200)
    assert len(chunks) >= 3
    assert all(c["source"] == "book.pdf" for c in chunks)
    assert [c["chunk"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["text"]) <= 1000 for c in chunks)


def test_chunk_text_empty_is_empty():
    assert chunk_text("   ", source="x") == []


def test_search_library_tool_without_index(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LIBRARY_INDEX", str(tmp_path / "does_not_exist"))
    from agent.tools.library import search_library

    assert "No library" in search_library("anything")


def _build_tiny_index(np, index_dir, books):
    (index_dir / "vecs").mkdir(parents=True)
    (index_dir / "chunks").mkdir(parents=True)
    for name, rows in books.items():
        arr = np.asarray([vec for _, vec in rows], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        np.save(index_dir / "vecs" / (name + ".npy"), (arr / norms).astype(np.float16))
        with (index_dir / "chunks" / (name + ".jsonl")).open("w", encoding="utf-8") as fh:
            for i, (text, _) in enumerate(rows):
                fh.write(json.dumps({"text": text, "source": name, "chunk": i}) + "\n")


def test_library_index_search_returns_nearest(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    index_dir = tmp_path / "lib"
    _build_tiny_index(np, index_dir, {
        "sea.pdf": [("the whale and the sea", [1.0, 0.0, 0.0]),
                    ("a ship on the ocean", [0.9, 0.1, 0.0])],
        "space.pdf": [("stars and galaxies", [0.0, 0.0, 1.0])],
    })
    import agent.library as lib

    monkeypatch.setattr(lib, "embed_texts", lambda texts, model=None, host=None: [[1.0, 0.0, 0.0]])
    hits = lib.LibraryIndex(str(index_dir)).search("whales", k=2)
    assert [h["source"] for h in hits] == ["sea.pdf", "sea.pdf"]  # both sea passages beat space
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_library_tool_with_index(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    index_dir = tmp_path / "lib"
    _build_tiny_index(np, index_dir, {
        "moby.pdf": [("Call me Ishmael.", [1.0, 0.0]), ("The great white whale.", [0.8, 0.2])],
    })
    monkeypatch.setenv("AGENT_LIBRARY_INDEX", str(index_dir))
    import agent.library as lib
    import agent.tools.library as tool_mod

    monkeypatch.setattr(lib, "embed_texts", lambda texts, model=None, host=None: [[1.0, 0.0]])
    tool_mod._INDEX = None  # reset the cached singleton so it picks up the temp index
    out = tool_mod.search_library("who narrates?", k=1)
    assert "moby.pdf" in out and "Ishmael" in out
