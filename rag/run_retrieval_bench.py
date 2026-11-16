"""Compare dense / lexical / hybrid retrieval on Day 7 answerable questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import QUESTIONS_ANSWERABLE, RESULTS_DIR
from .metrics import evaluate_retrieval
from .retrievers import DenseRetriever, HybridRRFRetriever, LexicalRetriever


def load_questions(path: Path, split: str | None) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if split:
        data = [q for q in data if q['split'] == split]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--methods', nargs='+', default=['dense', 'lexical', 'hybrid'])
    parser.add_argument('--no-filter', action='store_true', help='Ignore question filter field')
    args = parser.parse_args()

    split = None if args.split == 'all' else args.split
    questions = load_questions(QUESTIONS_ANSWERABLE, split)
    out = args.output or (RESULTS_DIR / f'retrieval-{args.split}')
    out.mkdir(parents=True, exist_ok=True)

    lexical = LexicalRetriever()
    dense = DenseRetriever() if any(m in args.methods for m in ('dense', 'hybrid')) else None
    hybrid = HybridRRFRetriever(dense=dense, lexical=lexical) if 'hybrid' in args.methods else None

    backends = {
        'dense': dense.search if dense else None,
        'lexical': lexical.search,
        'hybrid': hybrid.search if hybrid else None,
    }

    summaries = []
    for name in args.methods:
        fn = backends[name]
        if fn is None:
            raise SystemExit(f'Method {name} not available')
        report = evaluate_retrieval(
            fn, questions, method=name, top_k=args.top_k, budget=args.budget,
            use_question_filter=not args.no_filter,
        )
        (out / f'{name}.summary.json').write_text(
            json.dumps(report['summary'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (out / f'{name}.traces.json').write_text(
            json.dumps(report['traces'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summaries.append(report['summary'])
        print(json.dumps(report['summary'], ensure_ascii=False))

    (out / 'compare.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
