"""PostgreSQL + pgvector table for advanced multimodal chunks."""
from __future__ import annotations

import json
from typing import Any

import psycopg


def _vector_literal(vector: list[float]) -> str:
    return '[' + ','.join(f'{x:.7f}' for x in vector) + ']'


class AdvancedStore:
    def __init__(self, database_url: str, dimensions: int = 384, table: str = 'advanced_chunks'):
        self.database_url = database_url
        self.dimensions = dimensions
        self.table = table

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            conn.execute(f'DROP TABLE IF EXISTS {self.table}')
            conn.execute(f"""
                CREATE TABLE {self.table} (
                    chunk_id    text PRIMARY KEY,
                    document_id text NOT NULL,
                    source      text NOT NULL,
                    page        integer NOT NULL,
                    modality    text NOT NULL,
                    region      text,
                    text        text NOT NULL,
                    metadata    jsonb NOT NULL,
                    embedding   vector({self.dimensions}) NOT NULL
                )""")
            conn.execute(f'CREATE INDEX IF NOT EXISTS {self.table}_mod_page_idx '
                         f'ON {self.table} (modality, page)')

    def upsert(self, rows: list[dict[str, Any]], vectors: list[list[float]]) -> int:
        if len(rows) != len(vectors):
            raise ValueError('rows/vectors length mismatch')
        payload = [
            (r['id'], r['document_id'], r['source'], int(r['page']), r['modality'],
             r.get('region'), r['text'], json.dumps(r.get('metadata') or {}, ensure_ascii=False),
             _vector_literal(v))
            for r, v in zip(rows, vectors)
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(f"""
                    INSERT INTO {self.table}
                    (chunk_id, document_id, source, page, modality, region, text, metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding,
                        region=EXCLUDED.region, modality=EXCLUDED.modality
                """, payload)
        return len(payload)

    def search(self, query_vector: list[float], top_k: int = 5,
               modality: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], [_vector_literal(query_vector)]
        if modality:
            clauses.append('modality = %s')
            params.append(modality)
        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        sql = (f'SELECT chunk_id, document_id, source, page, modality, region, text, metadata, '
               f'1 - (embedding <=> %s::vector) AS score FROM {self.table} {where} '
               f'ORDER BY embedding <=> %s::vector LIMIT %s')
        params2 = [params[0], *params[1:], params[0], top_k]
        with self._connect() as conn:
            rows = conn.execute(sql, params2).fetchall()
        return [
            {'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'modality': r[4],
             'region': r[5], 'text': r[6], 'metadata': r[7], 'score': round(float(r[8]), 4)}
            for r in rows
        ]

    def count(self) -> dict[str, int]:
        with self._connect() as conn:
            total = conn.execute(f'SELECT count(*) FROM {self.table}').fetchone()[0]
            by_mod = conn.execute(
                f'SELECT modality, count(*) FROM {self.table} GROUP BY 1 ORDER BY 1').fetchall()
        return {'total': total, **{m: c for m, c in by_mod}}
