"""Agent loop for real runs: truncate-to-fit context packing (starter budget skip is too strict for long law chunks)."""
from __future__ import annotations

from typing import Any, Protocol


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        ...


def pack_context(results: list[dict[str, Any]], budget: int = 2000) -> tuple[str, list[str]]:
    """Keep rank order; truncate a chunk if needed so at least one block can enter the budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        header = f"[{c['id']} | {c['source']} | page {c['page']}]\n"
        body = c['text']
        block = header + body
        used = len('\n\n'.join(blocks))
        sep = 2 if blocks else 0
        room = budget - used - sep
        if room <= len(header) + 40:
            break
        if len(block) > room:
            body = body[: max(0, room - len(header) - 20)].rstrip() + '…'
            block = header + body
        blocks.append(block)
        ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 5, budget: int = 2000) -> dict[str, Any]:
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {
                    'status': 'abstain',
                    'answer': 'Không đủ bằng chứng để trả lời.',
                    'citations': [],
                    'events': events + [{'event': 'search_limit'}],
                }
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({
                'event': 'search', 'query': query, 'filter': filters, 'results': results,
                'context': context, 'context_ids': ids,
            })
            history = [{
                'tool': 'search',
                'queries': [e['query'] for e in events if e['event'] == 'search'],
                'context': context,
                'context_ids': ids,
            }]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if (not isinstance(answer, str) or not isinstance(citations, list)
                    or any(not isinstance(i, str) for i in citations)):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(
                i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return {
                    'status': 'answered', 'answer': answer, 'citations': citations,
                    'events': events + [{'event': 'answer'}],
                }
            # Explicit abstention phrase with empty citations counts as abstain.
            return {
                'status': 'abstain',
                'answer': 'Không đủ bằng chứng để trả lời.',
                'citations': [],
                'events': events + [{'event': 'invalid_or_missing_citations'}],
            }
        else:
            raise ValueError('action phải là search hoặc answer')
    return {
        'status': 'abstain',
        'answer': 'Không đủ bằng chứng để trả lời.',
        'citations': [],
        'events': events,
    }
