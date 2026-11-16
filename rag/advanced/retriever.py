"""Retriever over advanced_chunks with optional modality filter."""
from __future__ import annotations

import sys
from typing import Any

from ..config import EMBEDDING_MODEL, ensure_day6_on_path, require_database_url
from .store import AdvancedStore


class AdvancedRetriever:
    def __init__(self):
        ensure_day6_on_path()
        for key in list(sys.modules):
            if key == 'src' or key.startswith('src.'):
                del sys.modules[key]
        from src.embeddings import E5Embedder  # type: ignore
        self.embedder = E5Embedder(EMBEDDING_MODEL)
        self.store = AdvancedStore(require_database_url(), self.embedder.dimensions)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        modality = None
        if filters:
            modality = filters.get('modality')
        vector = self.embedder.embed_query(query)
        rows = self.store.search(vector, top_k=top_k, modality=modality)
        # Normalize to agent contract
        out = []
        for r in rows:
            out.append({
                'id': r['id'],
                'text': r['text'],
                'source_id': r['source_id'],
                'source': r['source'],
                'page': r['page'],
                'metadata': {**(r.get('metadata') or {}), 'modality': r.get('modality'), 'region': r.get('region')},
                'score': r['score'],
                'strategy': 'advanced_dense',
            })
        return out
