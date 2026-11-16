"""Dense (Day 6 pgvector), lexical TF-IDF, and hybrid RRF retrievers."""
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    CHUNKS_PATH,
    DAY6_STRATEGY,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    ensure_day6_on_path,
    require_database_url,
)

# Allow `from baseline import pack_context` style imports used by agentic.py
_STARTER = Path(__file__).resolve().parents[1] / 'starter'
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def load_chunks(path: Path | None = None) -> list[dict[str, Any]]:
    data = json.loads((path or CHUNKS_PATH).read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('chunks.json phải là JSON array không rỗng')
    return data


def normalize_hit(row: dict[str, Any], *, strategy: str) -> dict[str, Any]:
    """Map store / TF-IDF rows to the Day 7 retriever contract."""
    cid = row.get('id') or row.get('chunk_id')
    source_id = row.get('source_id') or row.get('document_id')
    meta = dict(row.get('metadata') or {})
    if row.get('article') is not None:
        meta.setdefault('article', row['article'])
    if row.get('chapter') is not None:
        meta.setdefault('chapter', row['chapter'])
    if row.get('year') is not None:
        meta.setdefault('year', row['year'])
    if row.get('law_title'):
        meta.setdefault('law_title', row['law_title'])
    return {
        'id': cid,
        'text': row['text'],
        'source_id': source_id,
        'source': row['source'],
        'page': int(row['page']),
        'metadata': meta,
        'score': float(row.get('score') or 0.0),
        'strategy': strategy,
    }


class LexicalRetriever:
    """Offline TF-IDF over exported Day 6 chunks (same idea as starter/baseline.py)."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None):
        self.chunks = chunks or load_chunks()
        counts = [Counter(tokenize(c['text'])) for c in self.chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(self.chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError('query không được rỗng')
        qv = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                if filters.get('document_id') and chunk['source_id'] != filters['document_id']:
                    continue
                if filters.get('source_id') and chunk['source_id'] != filters['source_id']:
                    continue
            score = sum(value * vector.get(term, 0.0) for term, value in qv.items())
            if score > 0:
                results.append(normalize_hit({**chunk, 'score': score}, strategy='lexical_tfidf'))
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    """Day 6 multilingual-e5-small + PostgreSQL/pgvector cosine search."""

    def __init__(self, strategy: str | None = None):
        # Day 6 package is also named `src`; put its starter first and drop any cached day7 collision.
        ensure_day6_on_path()
        for key in list(sys.modules):
            if key == 'src' or key.startswith('src.'):
                del sys.modules[key]
        from src.embeddings import E5Embedder  # type: ignore  # Day 6 package
        from src.store import PgVectorStore  # type: ignore

        self.strategy = strategy or DAY6_STRATEGY
        self.embedder = E5Embedder(EMBEDDING_MODEL)
        if self.embedder.dimensions != EMBEDDING_DIM:
            raise RuntimeError(f'Expected {EMBEDDING_DIM}-d embeddings, got {self.embedder.dimensions}')
        self.store = PgVectorStore(require_database_url(), self.embedder.dimensions)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError('query không được rỗng')
        store_filters = None
        if filters:
            store_filters = {}
            if 'document_id' in filters:
                store_filters['document_id'] = filters['document_id']
            elif 'source_id' in filters:
                store_filters['document_id'] = filters['source_id']
            for key in ('year', 'chapter', 'article'):
                if key in filters:
                    store_filters[key] = filters[key]
        vector = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vector, top_k=top_k, filters=store_filters or None)
        return [normalize_hit(r, strategy='dense_e5_pgvector') for r in rows]


class HybridRRFRetriever:
    """Reciprocal Rank Fusion of dense + lexical ranks."""

    def __init__(self, dense: DenseRetriever | None = None, lexical: LexicalRetriever | None = None,
                 rrf_k: int = 60, candidate_k: int = 20):
        self.dense = dense or DenseRetriever()
        self.lexical = lexical or LexicalRetriever()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lex_hits = self.lexical.search(query, filters=filters, top_k=self.candidate_k)
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            scores[hit['id']] = scores.get(hit['id'], 0.0) + 1.0 / (self.rrf_k + rank)
            by_id[hit['id']] = hit
        for rank, hit in enumerate(lex_hits, start=1):
            scores[hit['id']] = scores.get(hit['id'], 0.0) + 1.0 / (self.rrf_k + rank)
            # Prefer denser metadata/text if already seen; else keep lexical row.
            by_id.setdefault(hit['id'], hit)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        out = []
        for cid, score in ranked:
            row = dict(by_id[cid])
            row['score'] = round(score, 6)
            row['strategy'] = 'hybrid_rrf'
            out.append(row)
        return out


class RetrieverAdapter:
    """Adapter matching starter/agentic.py Retriever protocol."""

    def __init__(self, backend: Any):
        self.backend = backend

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return self.backend.search(query, filters=filters, top_k=top_k)
