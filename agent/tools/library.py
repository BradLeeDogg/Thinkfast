"""search_library tool: retrieve passages from the user's indexed PDF library."""
from ..library import LibraryIndex, library_available

SEARCH_LIBRARY_SCHEMA = {
    "name": "search_library",
    "description": "Search the user's personal library of books/novels for passages relevant "
    "to a query and return the best-matching excerpts with their book titles. Use this to "
    "answer any question about the user's books — their plots, characters, themes, or wording — "
    "and quote or cite the passages it returns.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for in the books."},
            "k": {"type": "integer", "description": "How many passages to return (default 5)."},
        },
        "required": ["query"],
    },
}

_INDEX: LibraryIndex | None = None


def _index() -> LibraryIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = LibraryIndex()
    return _INDEX


def search_library(query: str, k: int = 5) -> str:
    if not library_available():
        return "No library has been indexed yet. Run `agent-index <folder-of-pdfs>` to build one."
    try:
        hits = _index().search(query, k=k)
    except Exception as exc:
        return f"Error searching the library: {exc}"
    if not hits:
        return "No relevant passages found in the library."
    return "\n\n---\n\n".join(f'From "{h["source"]}":\n{h["text"].strip()}' for h in hits)
