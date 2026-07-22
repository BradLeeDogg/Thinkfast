"""Local book library: retrieval over your own PDFs (RAG), powered by Ollama embeddings.

Index your PDFs once with ``agent-index``; then the ``search_library`` tool lets
the agent look up and quote relevant passages. Everything runs locally. Only the
standard library is imported at module load — numpy is pulled in lazily inside
``LibraryIndex`` — so importing the package stays light.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_EMBED_MODEL = "nomic-embed-text"


def default_index_dir() -> str:
    return os.environ.get("AGENT_LIBRARY_INDEX", "library_index")


def library_available(index_dir: str | None = None) -> bool:
    """True once at least one book has been indexed under ``index_dir``."""
    vecs = Path(index_dir or default_index_dir()) / "vecs"
    return vecs.is_dir() and any(vecs.glob("*.npy"))


def chunk_text(text: str, source: str, chunk_chars: int = 1200, overlap: int = 200) -> list[dict]:
    """Split text into overlapping passages, each tagged with its source book."""
    text = " ".join(text.split())  # collapse whitespace/newlines from PDF extraction
    if not text:
        return []
    step = max(1, chunk_chars - overlap)
    chunks: list[dict] = []
    for index, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chunk_chars]
        if piece.strip():
            chunks.append({"text": piece, "source": source, "chunk": index})
        if start + chunk_chars >= len(text):
            break
    return chunks


def embed_texts(texts: list[str], model: str = DEFAULT_EMBED_MODEL, host: str | None = None) -> list[list[float]]:
    """Embed a batch of strings via the local Ollama server."""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    request = urllib.request.Request(
        f"{host}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host} for embeddings. Is it running? ({exc})"
        ) from exc
    if "embeddings" not in data:
        raise RuntimeError(f"Unexpected embeddings response from Ollama: {data}")
    return data["embeddings"]


class LibraryIndex:
    """Loads a built index and finds the passages most relevant to a query."""

    def __init__(self, index_dir: str | None = None) -> None:
        self.dir = Path(index_dir or default_index_dir())
        self.embed_model = DEFAULT_EMBED_MODEL
        manifest = self.dir / "manifest.json"
        if manifest.is_file():
            try:
                self.embed_model = json.loads(manifest.read_text()).get("embed_model", DEFAULT_EMBED_MODEL)
            except (ValueError, OSError):
                pass
        self._vecs = None
        self._chunks: list[dict] = []

    def _load(self) -> None:
        if self._vecs is not None:
            return
        import numpy as np

        vectors, chunks = [], []
        for vec_file in sorted((self.dir / "vecs").glob("*.npy")):
            chunk_file = self.dir / "chunks" / (vec_file.stem + ".jsonl")
            if not chunk_file.is_file():
                continue
            rows = [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            arr = np.load(vec_file)
            if arr.shape[0] != len(rows):
                continue  # skip a book whose vectors/metadata got out of sync
            vectors.append(arr.astype(np.float32))
            chunks.extend(rows)
        self._vecs = np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        self._chunks = chunks

    def search(self, query: str, k: int = 5) -> list[dict]:
        import numpy as np

        self._load()
        if self._vecs.shape[0] == 0:
            return []
        q = np.asarray(embed_texts([query], self.embed_model)[0], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self._vecs @ q  # stored vectors are pre-normalized, so this is cosine similarity
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results = []
        for i in top:
            row = dict(self._chunks[int(i)])
            row["score"] = float(scores[int(i)])
            results.append(row)
        return results
