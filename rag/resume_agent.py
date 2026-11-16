"""Resume an agent run: re-query only rows with status=error and merge back."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent_runtime import run_agent
from .gemini_controller import GeminiController
from .run_agent import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True, help='agent result directory with traces.json')
    parser.add_argument('--sleep', type=float, default=4.0, help='seconds between successful calls')
    parser.add_argument('--max-retries-round', type=int, default=3)
    args = parser.parse_args()

    traces_path = args.dir / 'traces.json'
    meta_path = args.dir / 'meta.json'
    traces = json.loads(traces_path.read_text(encoding='utf-8'))
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    controller = GeminiController()
    retriever = build_retriever(meta['retriever'])

    for round_i in range(args.max_retries_round):
        pending = [t for t in traces if t['status'] == 'error']
        if not pending:
            break
        print(f'round {round_i + 1}: retry {len(pending)} errors')
        for t in pending:
            started = time.perf_counter()
            try:
                result = run_agent(
                    t['question'], controller, retriever,
                    filters=t.get('filter'),
                    max_searches=meta.get('max_searches', 3),
                    top_k=meta.get('top_k', 5),
                    budget=meta.get('budget', 2000),
                )
                t['status'] = result['status']
                t['answer'] = result.get('answer')
                t['citations'] = result.get('citations')
                t['events'] = result.get('events')
                t['error'] = None
            except Exception as exc:  # noqa: BLE001
                t['status'] = 'error'
                t['answer'] = ''
                t['citations'] = []
                t['events'] = []
                t['error'] = f'{type(exc).__name__}: {exc}'
            t['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 2)
            print(f"{t['query_id']}\t{t['status']}\t{t['elapsed_ms']}ms")
            traces_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            time.sleep(args.sleep)
        # cool down between rounds if still failing
        if any(t['status'] == 'error' for t in traces):
            time.sleep(30)

    meta['errors'] = sum(1 for t in traces if t['status'] == 'error')
    meta['answered'] = sum(1 for t in traces if t['status'] == 'answered')
    meta['abstain'] = sum(1 for t in traces if t['status'] == 'abstain')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    traces_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
