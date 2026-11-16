"""Run advanced agent on Vietcombank benchmark."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..agent_runtime import pack_context
from ..config import DATA_DIR, RESULTS_DIR
from .controller import AdvancedController
from .retriever import AdvancedRetriever

QUESTIONS = DATA_DIR / 'advanced' / 'questions.json'


def _hint_modality(question: str) -> str | None:
    q = question.casefold()
    if any(k in q for k in ('sơ đồ', 'hình ', 'infographic', 'ảnh')):
        return 'figure'
    if any(k in q for k in ('bảng', 'chỉ số', 'roae', 'roaa', 'nim', 'car', 'thù lao', 'triệu vnd')):
        return 'table'
    return None


def run_one(question: str, controller: AdvancedController, retriever: AdvancedRetriever, *,
            max_searches: int, top_k: int, budget: int) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    # Seed one retrieval with heuristic modality so table/figure questions start on-distribution.
    seed_mod = _hint_modality(question)
    seed = retriever.search(question, filters={'modality': seed_mod} if seed_mod else None, top_k=top_k)
    if seed_mod:
        seed = seed + [r for r in retriever.search(question, filters=None, top_k=top_k)
                       if r['id'] not in {x['id'] for x in seed}]
    if seed:
        candidates = seed
        context, ids = pack_context(candidates, budget=budget)
        chunks = {c['id']: c for c in candidates if c['id'] in ids}
        events.append({'event': 'search', 'query': question, 'filter': {'modality': seed_mod} if seed_mod else None,
                       'results': seed[:top_k], 'context': context, 'context_ids': ids})
        history = [{'tool': 'search', 'queries': [question], 'context': context, 'context_ids': ids}]

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.',
                        'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query') or question
            modality = action.get('modality')
            filters = {'modality': modality} if modality else None
            results = retriever.search(query, filters=filters, top_k=top_k)
            # If modality filter misses, also pull unfiltered / alternate modalities.
            if filters:
                extra = retriever.search(query, filters=None, top_k=top_k)
                # Prefer latest hits first; keep prior only as backfill.
                merged, seen = [], set()
                for row in results + extra:
                    if row['id'] in seen:
                        continue
                    seen.add(row['id'])
                    merged.append(row)
                results = merged
            # Reset candidate pool to latest ranking so bad early tables don't fill budget.
            candidates = results + [c for c in candidates if c['id'] not in {r['id'] for r in results}]
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters,
                           'results': results, 'context': context, 'context_ids': ids})
            history = [{'tool': 'search',
                        'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise ValueError('invalid answer')
            valid = bool(answer.strip() and citations and all(
                i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations,
                        'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.',
                    'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('bad action')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--budget', type=int, default=2500)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--sleep', type=float, default=4.0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--tag', default='locked')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding='utf-8'))
    if args.split != 'all':
        questions = [q for q in questions if q['split'] == args.split]
    if args.limit:
        questions = questions[: args.limit]

    out = args.output or (RESULTS_DIR / f'advanced-{args.split}-{args.tag}')
    out.mkdir(parents=True, exist_ok=True)

    controller = AdvancedController()
    retriever = AdvancedRetriever()
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            result = run_one(q['question'], controller, retriever,
                             max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
            err = None
        except Exception as exc:  # noqa: BLE001
            result = {'status': 'error', 'answer': '', 'citations': [], 'events': []}
            err = f'{type(exc).__name__}: {exc}'
        elapsed = (time.perf_counter() - t0) * 1000
        traces.append({
            'query_id': q['id'], 'split': q['split'], 'group': q.get('group'),
            'question': q['question'], 'answerable': q.get('answerable', True),
            'status': result['status'], 'answer': result.get('answer'),
            'citations': result.get('citations'), 'events': result.get('events'),
            'elapsed_ms': round(elapsed, 2), 'error': err,
            'top_k': args.top_k, 'budget': args.budget,
        })
        print(f"{q['id']}\t{result['status']}\t{round(elapsed)}ms")
        (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        time.sleep(args.sleep)

    meta = {
        'split': args.split, 'top_k': args.top_k, 'budget': args.budget,
        'max_searches': args.max_searches, 'n': len(traces),
        'answered': sum(1 for t in traces if t['status'] == 'answered'),
        'abstain': sum(1 for t in traces if t['status'] == 'abstain'),
        'errors': sum(1 for t in traces if t['status'] == 'error'),
    }
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': q.get('answerable', True),
              'group': q.get('group')} for q in questions]
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
