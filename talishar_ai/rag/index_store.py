"""
index_store.py — Abstraction layer over lucisearch (``import luci``).

This is the ONLY file in the codebase that imports luci directly.
All other modules interact with the search engine through IndexStore.

If the underlying search library changes, only this file needs updating.

lucisearch API (Elasticsearch-like DSL):
    index = luci.Index.create("path.luci", {"properties": {...}})
    index.add({"field": "value", ...})
    index.commit()
    results = index.search({"query": {"match": {"field": "text"}}, "size": N})
    doc = index.get("doc_id")
    index.delete("doc_id")
    index.update("doc_id", {"field": "new_value"})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import luci


# ---------------------------------------------------------------------------
# Index schema for cards
# ---------------------------------------------------------------------------

# Default schema for the card index.  lucisearch requires typed field
# definitions at index creation time (like Elasticsearch mappings).
CARD_INDEX_SCHEMA = {
    "properties": {
        # Searchable text
        "description": {"type": "text"},
        "functional_text": {"type": "text"},
        "name": {"type": "text"},
        # Keyword fields (exact match / filtering)
        "card_id": {"type": "keyword"},
        "card_type": {"type": "keyword"},
        "subtype": {"type": "keyword"},
        "card_class": {"type": "keyword"},
        "card_talent": {"type": "keyword"},
        # Numeric fields
        "cost": {"type": "float"},
        "power": {"type": "float"},
        "defense": {"type": "float"},
        "pitch": {"type": "float"},
        "attack_value": {"type": "float"},
        "block_willingness": {"type": "float"},
        "arsenal_value": {"type": "float"},
        "best_use_value": {"type": "float"},
        # Boolean
        "has_go_again": {"type": "keyword"},  # "true" / "false" as keyword
        # Keywords stored as comma-separated string
        "keywords_csv": {"type": "text"},
    },
}

COMBO_INDEX_SCHEMA = {
    "properties": {
        "description": {"type": "text"},
        "deck_id": {"type": "keyword"},
        "hero_id": {"type": "keyword"},
        "total_damage": {"type": "float"},
        "total_cost": {"type": "float"},
        "pitch_needed": {"type": "float"},
        "has_go_again_chain": {"type": "keyword"},
        "dominate": {"type": "keyword"},
        "combo_value": {"type": "float"},
        "n_cards": {"type": "float"},
    },
}


@dataclass
class SearchResult:
    """A single result from a search query."""

    id: str
    score: float
    document: dict[str, Any]


class IndexStore:
    """
    Typed wrapper around a luci.Index.

    Provides add, search, and direct-lookup operations with consistent
    return types so the rest of the codebase never touches luci directly.

    Parameters
    ----------
    name:
        Logical name for the index (e.g. "cards", "combos").
    persist_dir:
        Directory where the index file is stored on disk.
    schema:
        Property mapping passed to ``luci.Index.create()``.
        Only used when creating a new index.
    """

    def __init__(
        self,
        name: str,
        persist_dir: str | Path | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._persist_dir = Path(persist_dir) if persist_dir else Path(".")
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = str(self._persist_dir / f"{name}.luci")

        # Use provided schema or default to card schema
        index_schema = schema or CARD_INDEX_SCHEMA

        # Open existing index, or create a new one
        if Path(self._index_path).exists():
            self._index = luci.Index.open(self._index_path)
        else:
            self._index = luci.Index.create(self._index_path, index_schema)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """
        Batch-insert documents into the index.

        Each dict should have flat key-value pairs matching the schema
        fields.  The special ``_id`` key is used as the document ID;
        if absent, luci auto-generates one.
        """
        for doc in docs:
            self._index.add(doc)

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._index.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        sort: list[dict[str, str]] | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Full-text search with optional filters and sorting.

        Parameters
        ----------
        query:
            Text query matched against text fields.  If None, matches all.
        filters:
            Key-value pairs for ``term`` filtering.
            Example: {"card_type": "AA"} → {"term": {"card_type": "AA"}}
        sort:
            Sort specification, e.g. [{"best_use_value": "desc"}].
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List of SearchResult sorted by relevance (or sort order).
        """
        # Build Elasticsearch-like query DSL
        dsl: dict[str, Any] = {"size": top_k}

        # Query clause
        if query:
            # Multi-match across text fields
            dsl["query"] = {
                "bool": {
                    "must": [{"match": {"description": query}}],
                }
            }
        else:
            dsl["query"] = {"match_all": {}}

        # Add term filters
        if filters:
            filter_clauses = []
            for field_name, value in filters.items():
                filter_clauses.append({"term": {field_name: value}})

            if "bool" not in dsl["query"]:
                dsl["query"] = {"bool": {"must": [dsl["query"]]}}
            dsl["query"]["bool"]["filter"] = filter_clauses

        # Sorting
        if sort:
            dsl["sort"] = sort

        raw = self._index.search(dsl)
        hits = raw.get("hits", []) if isinstance(raw, dict) else []

        return [
            SearchResult(
                id=str(hit.get("_id", "")),
                score=float(hit.get("_score", 0.0)),
                document=hit.get("_source", {}),
            )
            for hit in hits
        ]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """
        Direct lookup by document ID.

        Returns the document source dict, or None if not found.
        """
        try:
            doc = self._index.get(doc_id)
            if doc is not None:
                return doc if isinstance(doc, dict) else {"_source": doc}
        except Exception:
            pass
        return None

    def query_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Direct lookup by document IDs.

        Returns documents in order.  Missing IDs are silently skipped.
        """
        out = []
        for doc_id in ids:
            doc = self.get(doc_id)
            if doc is not None:
                out.append(doc)
        return out

    def query_by_metadata(
        self,
        filters: dict[str, Any],
        sort: list[dict[str, str]] | None = None,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """
        Filter-only query (no text matching).

        Useful for structured lookups like "all AA cards with go_again".
        """
        return self.search(query=None, filters=filters, sort=sort, top_k=top_k)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._index.delete(doc_id)

    def delete_by_query(self, filters: dict[str, Any]) -> None:
        """Delete all documents matching the given term filters."""
        query = {}
        for field_name, value in filters.items():
            query[field_name] = value
        self._index.delete_by_query({"term": query})

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of documents currently in the index."""
        # Use match_all with size=0 to get total count from hits
        raw = self._index.search({"query": {"match_all": {}}, "size": 0})
        if isinstance(raw, dict):
            total = raw.get("total", raw.get("hits", {}) if isinstance(raw.get("hits"), dict) else len(raw.get("hits", [])))
            if isinstance(total, dict):
                return total.get("value", 0)
            return int(total) if total else 0
        return 0

    def __repr__(self) -> str:
        return f"IndexStore(name={self.name!r}, path={self._index_path!r})"
