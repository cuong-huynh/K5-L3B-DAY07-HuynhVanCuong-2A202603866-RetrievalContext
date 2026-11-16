"""Resume advanced agent errors only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .controller import AdvancedController
from .retriever import AdvancedRetriever
from .run_agent import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--sleep', type=float, default=5.0)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    traces = json.loads((args.dir / 'traces.json').read_text(encoding='utf-8'))
    meta = json.loads((args.dir / 'meta.json').read_text(encoding='utf-8'))
    controller = AdvancedController()
    retriever = AdvancedRetriever()
    for r in range(args.rounds):
        pending = [t for t in traces if t['status'] == 'error']
        if not pending:
            break
        print(f'round {r+1} retry {len(pending)}')
        for t in pending:
            t0 = time.perf_counter()
            try:
                result = run_one(t['question'], controller, retriever,
                                 max_searches=meta.get('max_searches', 3),
                                 top_k=meta.get('top_k', 5), budget=meta.get('budget', 2500))
                t.update({k: result.get(k) for k in ('status', 'answer', 'citations', 'events')})
                t['error'] = None
            except Exception as exc:  # noqa: BLE001
                t['status'] = 'error'
                t['error'] = f'{type(exc).__name__}: {exc}'
                t['answer'] = ''
                t['citations'] = []
                t['events'] = []
            t['elapsed_ms'] = round((time.perf_counter() - t0) * 1000, 2)
            print(f"{t['query_id']}\t{t['status']}")
            (args.dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n')
            time.sleep(args.sleep)
        if any(t['status'] == 'error' for t in traces):
            time.sleep(25)
    meta['errors'] = sum(1 for t in traces if t['status'] == 'error')
    meta['answered'] = sum(1 for t in traces if t['status'] == 'answered')
    meta['abstain'] = sum(1 for t in traces if t['status'] == 'abstain')
    (args.dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
