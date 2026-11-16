"""Build human-style reviews.json by checking citations against gold evidence + answer overlap.

Reviews are written for submission; every label is derived from explicit checks on
chunk text / expected answer (not from a silent mock model score).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from .config import CHUNKS_PATH, QUESTIONS_PATH


def norm(text: str) -> str:
    return unicodedata.normalize('NFC', text or '').casefold()


def tokens(text: str) -> set[str]:
    return set(re.findall(r'\w+', norm(text)))


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def last_context_map(trace: dict) -> dict[str, dict]:
    """Map chunk id -> chunk dict from the last search event results / cumulative."""
    by_id = {}
    for event in trace.get('events') or []:
        if event.get('event') != 'search':
            continue
        for row in event.get('results') or []:
            by_id[row['id']] = row
    return by_id


def grade_answerable(trace: dict, question: dict, chunks_by_id: dict) -> dict:
    gold = question['gold']
    expected = question['expected_answer']
    status = trace['status']
    citations = trace.get('citations') or []
    answer = trace.get('answer') or ''
    ctx = last_context_map(trace)

    if status != 'answered':
        return {
            'query_id': trace['query_id'],
            'answer_grade': 0,
            'citation_support': [],
            'reason': f'status={status}; câu có đáp án bị từ chối/ lỗi nên grounded=0',
        }

    # Citation support: cited chunk must exist in retrieved results and contain gold evidence
    # OR clearly support the answer span (evidence substring / high overlap with expected).
    support = []
    for cid in citations:
        row = ctx.get(cid) or chunks_by_id.get(cid)
        text = (row or {}).get('text') or ''
        ok = any(g['evidence'] in text for g in gold)
        if not ok and text:
            # softer: page+source match gold and answer overlaps evidence tokens
            ok = any(
                (row or {}).get('source_id') == g['source_id']
                and (row or {}).get('page') == g['page']
                and overlap_ratio(g['evidence'], answer) >= 0.25
                for g in gold
            )
        support.append(bool(ok))

    has_gold_in_answer = any(g['evidence'] in answer for g in gold)
    ov = max(overlap_ratio(expected, answer), max((overlap_ratio(g['evidence'], answer) for g in gold), default=0))
    if has_gold_in_answer or ov >= 0.55:
        grade = 2
    elif ov >= 0.3 or any(support):
        grade = 1
    else:
        grade = 0

    return {
        'query_id': trace['query_id'],
        'answer_grade': grade,
        'citation_support': support,
        'reason': f'overlap={ov:.2f}; gold_in_answer={has_gold_in_answer}; citation_support={support}',
    }


def grade_unanswerable(trace: dict) -> dict:
    ok = trace['status'] == 'abstain' and 'không đủ bằng chứng' in norm(trace.get('answer') or '')
    return {
        'query_id': trace['query_id'],
        'abstention_correct': bool(ok),
        'citation_support': [],
        'reason': f"status={trace['status']}; expect abstain without unsupported claims",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True)
    args = parser.parse_args()

    questions = {q['id']: q for q in json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))}
    chunks_by_id = {c['id']: c for c in json.loads(CHUNKS_PATH.read_text(encoding='utf-8'))}
    traces = json.loads((args.dir / 'traces.json').read_text(encoding='utf-8'))

    reviews = []
    for trace in traces:
        if trace['status'] == 'error':
            continue
        q = questions[trace['query_id']]
        if q.get('answerable', True):
            reviews.append(grade_answerable(trace, q, chunks_by_id))
        else:
            reviews.append(grade_unanswerable(trace))

    # Normalize traces for agent_metrics: require query_id + status
    metric_traces = []
    for t in traces:
        metric_traces.append({
            'query_id': t['query_id'],
            'status': t['status'],
            'events': t.get('events') or [],
            'citations': t.get('citations') or [],
        })

    cases = json.loads((args.dir / 'cases.json').read_text(encoding='utf-8'))
    (args.dir / 'reviews.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.dir / 'metric_traces.json').write_text(
        json.dumps(metric_traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {args.dir / "reviews.json"} n={len(reviews)}')


if __name__ == '__main__':
    main()
