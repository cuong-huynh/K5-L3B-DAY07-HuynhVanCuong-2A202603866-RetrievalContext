"""Retrieval / context metrics shared by offline benches."""
from __future__ import annotations

import time
from typing import Any, Callable


def evidence_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    return any(
        chunk['source_id'] == g['source_id']
        and chunk['page'] == g['page']
        and g['evidence'] in chunk['text']
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate_retrieval(
    search_fn: Callable[..., list[dict[str, Any]]],
    questions: list[dict[str, Any]],
    *,
    method: str,
    top_k: int = 3,
    budget: int = 800,
    use_question_filter: bool = True,
) -> dict[str, Any]:
    """Hit@1 / Evidence Hit@3 / Context Evidence Rate on answerable questions only."""
    traces = []
    hit1 = hit3 = ctx = 0
    total_ms = 0.0
    n = 0
    for q in questions:
        if q.get('answerable') is False:
            continue
        gold = q['gold']
        n += 1
        filters = q.get('filter') if use_question_filter else None
        start = time.perf_counter()
        # Always fetch at least 3 for comparable ranking metrics.
        results = search_fn(q['question'], filters=filters, top_k=max(top_k, 3))
        elapsed = (time.perf_counter() - start) * 1000
        total_ms += elapsed
        top = results[:3]
        marked = [{**c, 'hit': evidence_hit(c, gold)} for c in top]
        context, ids = pack_context(results[:top_k], budget=budget)
        ctx_chunks = [c for c in results if c['id'] in ids]
        h1 = bool(marked) and marked[0]['hit']
        h3 = any(c['hit'] for c in marked)
        c_ev = any(evidence_hit(c, gold) for c in ctx_chunks)
        hit1 += int(h1)
        hit3 += int(h3)
        ctx += int(c_ev)
        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'method': method,
            'filter': filters,
            'top_k': top_k,
            'budget_chars': budget,
            'elapsed_ms': round(elapsed, 2),
            'hit_at_1': h1,
            'evidence_hit_at_3': h3,
            'context_evidence': c_ev,
            'results': marked,
            'context': context,
            'context_ids': ids,
        })
    summary = {
        'method': method,
        'n': n,
        'hit_at_1': f'{hit1}/{n}',
        'hit_at_1_rate': round(hit1 / n, 4) if n else 0,
        'evidence_hit_at_3': f'{hit3}/{n}',
        'evidence_hit_at_3_rate': round(hit3 / n, 4) if n else 0,
        'context_evidence_rate': f'{ctx}/{n}',
        'context_evidence_rate_num': round(ctx / n, 4) if n else 0,
        'mean_latency_ms': round(total_ms / n, 2) if n else 0,
        'top_k': top_k,
        'budget_chars': budget,
        'use_question_filter': use_question_filter,
        'cost_usd': 0,
    }
    return {'summary': summary, 'traces': traces}
