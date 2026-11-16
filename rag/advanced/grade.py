"""Binary pass/fail grading for advanced benchmark against gold evidence."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ..config import DATA_DIR


def norm(s: str) -> str:
    return unicodedata.normalize('NFC', s or '').casefold()


def tokens(s: str) -> set[str]:
    return set(re.findall(r'[\w.%]+', norm(s)))


def _digits(s: str) -> str:
    return re.sub(r'\D+', '', s or '')


def pass_answerable(trace: dict, gold: dict) -> tuple[bool, str]:
    if trace['status'] != 'answered':
        return False, f"status={trace['status']} (need answered)"
    answer = trace.get('answer') or ''
    expected = gold.get('expected_answer') or ''
    exp_toks = tokens(expected)
    ans_toks = tokens(answer)
    distinctive = {t for t in exp_toks if any(c.isdigit() for c in t) or len(t) >= 4}
    if not distinctive:
        distinctive = exp_toks
    hit = distinctive & ans_toks
    # Also accept when key digit groups from expected appear in answer (e.g. 68.578 vs 68.578.496).
    digit_ok = False
    for t in distinctive:
        d = _digits(t)
        if len(d) >= 3 and d in _digits(answer):
            digit_ok = True
            break
    if len(hit) < max(1, min(2, len(distinctive))) and not digit_ok:
        return False, f'answer missing expected tokens; hit={hit}'
    cites = trace.get('citations') or []
    if not cites:
        return False, 'missing citations'
    if any(f'[{c}]' not in answer for c in cites):
        return False, 'citation not marked in answer text'
    gold_pages = {e['page'] for e in gold.get('evidence') or []}
    cite_pages = set()
    cited_text = []
    for c in cites:
        m = re.search(r':p(\d+):', c)
        if m:
            cite_pages.add(int(m.group(1)))
        for ev in trace.get('events') or []:
            if ev.get('event') != 'search':
                continue
            for row in ev.get('results') or []:
                if row.get('id') == c:
                    cited_text.append(row.get('text') or '')
            # context may include truncated cited blocks
            if c in (ev.get('context') or ''):
                cited_text.append(ev.get('context') or '')
    quotes = [e.get('quote') or '' for e in gold.get('evidence') or []]
    quote_supported = any(
        q and (q[:30] in ' '.join(cited_text) or any(tok in norm(answer) for tok in tokens(q) if any(ch.isdigit() for ch in tok)))
        for q in quotes
    )
    if gold_pages and cite_pages.isdisjoint(gold_pages) and not quote_supported:
        ctx = ' '.join((ev.get('context') or '') for ev in (trace.get('events') or []) if ev.get('event') == 'search')
        if not any(q and q[:30] in ctx for q in quotes):
            return False, f'citation pages {cite_pages} miss gold pages {gold_pages}'
    return True, 'ok'


def pass_unanswerable(trace: dict) -> tuple[bool, str]:
    ok = trace['status'] == 'abstain' and 'không đủ bằng chứng' in norm(trace.get('answer') or '')
    return ok, 'abstain ok' if ok else f"status={trace['status']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True)
    args = parser.parse_args()
    gold = {q['id']: q for q in json.loads((DATA_DIR / 'advanced' / 'questions.json').read_text(encoding='utf-8'))}
    traces = json.loads((args.dir / 'traces.json').read_text(encoding='utf-8'))
    reviews = []
    by_group = defaultdict(lambda: {'pass': 0, 'total': 0})
    n_pass = 0
    for t in traces:
        g = gold[t['query_id']]
        group = g.get('group') or 'unknown'
        if t['status'] == 'error':
            ok, reason = False, t.get('error') or 'error'
        elif g.get('answerable', True):
            ok, reason = pass_answerable(t, g)
        else:
            ok, reason = pass_unanswerable(t)
        by_group[group]['total'] += 1
        by_group[group]['pass'] += int(ok)
        n_pass += int(ok)
        reviews.append({
            'query_id': t['query_id'], 'group': group, 'pass': ok, 'reason': reason,
            'status': t['status'],
        })
    summary = {
        'n_pass': n_pass,
        'n_total': len(traces),
        'rate': round(100.0 * n_pass / len(traces), 2) if traces else 0,
        'by_group': {k: {'pass': v['pass'], 'total': v['total'],
                         'rate': round(100 * v['pass'] / v['total'], 2) if v['total'] else 0}
                     for k, v in sorted(by_group.items())},
        'errors': sum(1 for t in traces if t['status'] == 'error'),
    }
    (args.dir / 'reviews.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
