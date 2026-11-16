"""Export Day 6 law_chunks (B_struct) into data/chunks.json."""
from __future__ import annotations

import json

import psycopg

from .config import CHUNKS_PATH, DAY6_STRATEGY, DATA_DIR, require_database_url


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(require_database_url()) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, document_id, source, page, text, metadata
            FROM law_chunks WHERE strategy=%s
            ORDER BY document_id, page, chunk_id
            """,
            (DAY6_STRATEGY,),
        ).fetchall()
    chunks = []
    for r in rows:
        meta = r[5] if isinstance(r[5], dict) else json.loads(r[5])
        chunks.append({
            'id': r[0],
            'source_id': r[1],
            'source': r[2],
            'page': int(r[3]),
            'text': r[4],
            'metadata': meta,
        })
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(chunks)} chunks -> {CHUNKS_PATH}')


if __name__ == '__main__':
    main()
