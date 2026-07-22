"""agent-index: build a searchable index from a folder of PDFs (for the library RAG).

Reads every PDF under a folder, splits each into passages, embeds them with
Ollama, and saves a per-book index under ``--index-dir`` (default
``library_index``). Re-running skips books already indexed, so a long run can be
resumed. Heavy imports (pypdf, numpy) are done inside ``main`` so ``--help``
works without them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .library import DEFAULT_EMBED_MODEL, chunk_text, default_index_dir, embed_texts


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # a single bad page shouldn't sink the whole book
            continue
    return "\n".join(parts)


def _safe_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix().replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-index",
        description="Index a folder of PDFs so the agent can search your library.",
    )
    parser.add_argument("path", help="Folder containing your PDF books (searched recursively).")
    parser.add_argument("--index-dir", default=default_index_dir(), help="Where to store the index.")
    parser.add_argument("--embed-model", default=os.environ.get("AGENT_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--reindex", action="store_true", help="Re-index books already indexed.")
    args = parser.parse_args()

    import numpy as np

    root = Path(args.path).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")

    index_dir = Path(args.index_dir)
    (index_dir / "vecs").mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks").mkdir(parents=True, exist_ok=True)

    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found under {root}")
    print(f"Found {len(pdfs)} PDF(s). Indexing into '{index_dir}' with '{args.embed_model}'.")

    for n, pdf in enumerate(pdfs, 1):
        name = _safe_name(pdf, root)
        vec_path = index_dir / "vecs" / (name + ".npy")
        chunk_path = index_dir / "chunks" / (name + ".jsonl")
        if vec_path.exists() and chunk_path.exists() and not args.reindex:
            print(f"[{n}/{len(pdfs)}] skip (already indexed): {pdf.name}")
            continue

        print(f"[{n}/{len(pdfs)}] reading: {pdf.name}")
        try:
            text = _extract_pdf_text(pdf)
        except Exception as exc:
            print(f"    ! could not read ({exc}); skipping")
            continue
        chunks = chunk_text(text, source=pdf.name, chunk_chars=args.chunk_chars, overlap=args.overlap)
        if not chunks:
            print("    (no extractable text; skipping)")
            continue

        vectors: list[list[float]] = []
        for i in range(0, len(chunks), args.batch):
            vectors.extend(embed_texts([c["text"] for c in chunks[i : i + args.batch]], args.embed_model))
            print(f"    embedded {min(i + args.batch, len(chunks))}/{len(chunks)} chunks", end="\r")

        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        np.save(vec_path, (arr / norms).astype(np.float16))  # store normalized, half-size
        with chunk_path.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c) + "\n")
        print(f"\n    indexed {len(chunks)} passages from {pdf.name}")

    (index_dir / "manifest.json").write_text(json.dumps({"embed_model": args.embed_model}, indent=2))
    print("Done. Your agent can now answer questions about these books.")


if __name__ == "__main__":
    main()
