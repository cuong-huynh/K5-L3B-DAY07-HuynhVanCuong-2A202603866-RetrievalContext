"""Run the Day 7 agent loop with a real Gemini controller + retrieval backend."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agent_runtime import run_agent
from .config import AGENT_CASES_PATH, RESULTS_DIR
from .gemini_controller import GeminiController
from .retrievers import DenseRetriever, HybridRRFRetriever, LexicalRetriever, RetrieverAdapter


def build_retriever(name: str) -> RetrieverAdapter:
    lexical = LexicalRetriever()
    if name == 'lexical':
        return RetrieverAdapter(lexical)
    dense = DenseRetriever()
    if name == 'dense':
        return RetrieverAdapter(dense)
    if name == 'hybrid':
        return RetrieverAdapter(HybridRRFRetriever(dense=dense, lexical=lexical))
    raise ValueError(f'Unknown retriever {name}')


def load_cases(path: Path, split: str | None) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding='utf-8'))
    # Prefer full questions (with gold) when available for later review helpers.
    if split:
        cases = [c for c in cases if c['split'] == split]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    parser.add_argument('--retriever', choices=['dense', 'lexical', 'hybrid'], default='dense')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0, help='Optional cap for smoke tests')
    parser.add_argument('--cases', type=Path, default=AGENT_CASES_PATH)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--tag', default='')
    parser.add_argument('--sleep', type=float, default=3.0, help='Pause between questions to reduce 429s')
    args = parser.parse_args()

    split = None if args.split == 'all' else args.split
    cases = load_cases(args.cases, split)
    if args.limit:
        cases = cases[: args.limit]

    tag = args.tag or f'{args.retriever}-k{args.top_k}-b{args.budget}'
    out = args.output or (RESULTS_DIR / f'agent-{args.split}-{tag}')
    out.mkdir(parents=True, exist_ok=True)

    controller = GeminiController()
    retriever = build_retriever(args.retriever)

    traces = []
    t0 = time.perf_counter()
    for case in cases:
        q = case['question']
        filters = case.get('filter')
        started = time.perf_counter()
        try:
            result = run_agent(
                q, controller, retriever,
                filters=filters, max_searches=args.max_searches,
                top_k=args.top_k, budget=args.budget,
            )
            status = result['status']
            err = None
        except Exception as exc:  # provider / store errors must surface as error rows
            result = {'status': 'error', 'answer': '', 'citations': [], 'events': []}
            status = 'error'
            err = f'{type(exc).__name__}: {exc}'
        elapsed = (time.perf_counter() - started) * 1000
        row = {
            'query_id': case['id'],
            'split': case['split'],
            'question': q,
            'answerable': case.get('answerable', True),
            'filter': filters,
            'retriever': args.retriever,
            'top_k': args.top_k,
            'budget': args.budget,
            'max_searches': args.max_searches,
            'status': status,
            'answer': result.get('answer'),
            'citations': result.get('citations'),
            'events': result.get('events'),
            'elapsed_ms': round(elapsed, 2),
            'error': err,
            'cost_usd': None,
        }
        traces.append(row)
        print(f"{case['id']}\t{status}\t{round(elapsed)}ms")
        if args.sleep > 0:
            time.sleep(args.sleep)

    total_ms = (time.perf_counter() - t0) * 1000
    meta = {
        'split': args.split,
        'retriever': args.retriever,
        'top_k': args.top_k,
        'budget': args.budget,
        'max_searches': args.max_searches,
        'n': len(traces),
        'total_ms': round(total_ms, 2),
        'errors': sum(1 for t in traces if t['status'] == 'error'),
        'answered': sum(1 for t in traces if t['status'] == 'answered'),
        'abstain': sum(1 for t in traces if t['status'] == 'abstain'),
    }
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'cases.json').write_text(json.dumps([
        {'id': c['id'], 'split': c['split'], 'answerable': c.get('answerable', True)} for c in cases
    ], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
