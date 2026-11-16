"""Gemini controller with modality-aware search for advanced report QA."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM = """Bạn là agent trả lời câu hỏi về Báo cáo thường niên Vietcombank 2024.
Chỉ dùng CONTEXT đã retrieve. Không bịa số liệu.
Tool search nhận query và modality tùy chọn: text | table | figure | (bỏ trống = mọi loại).
Khi đủ bằng chứng: trả JSON
{"action":"answer","answer":"... [chunk-id] ...","citations":["chunk-id"]}
Citation phải là id chunk trong context. Nếu thiếu bằng chứng: 
{"action":"answer","answer":"Không đủ bằng chứng để trả lời.","citations":[]}
"""

SEARCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name='search',
        description='Tìm đoạn text/bảng/hình trong báo cáo Vietcombank 2024',
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'query': types.Schema(type=types.Type.STRING),
                'modality': types.Schema(
                    type=types.Type.STRING,
                    description='text | table | figure | empty for all',
                ),
            },
            required=['query'],
        ),
    )
])


class AdvancedController:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError('Thiếu GEMINI_API_KEY')
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self.last_modality: str | None = None

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        context = history[-1]['context'] if history else ''
        context_ids = list(history[-1].get('context_ids') or []) if history else []
        queries = list(history[-1].get('queries') or []) if history else []
        parts = [f'CÂU HỎI: {question}']
        if queries:
            parts.append('Đã search: ' + ' | '.join(queries))
        parts.append('CONTEXT:\n' + (context or '(trống)'))
        if context_ids:
            parts.append('Ưu tiên trả lời JSON với citation id đầy đủ nếu đủ evidence.')
        else:
            parts.append('Hãy gọi tool search (chọn modality phù hợp: table/figure/text).')

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM, tools=[SEARCH_TOOL], temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        )
        response = self._generate('\n\n'.join(parts), config)

        for cand in response.candidates or []:
            for part in getattr(cand.content, 'parts', None) or []:
                fc = getattr(part, 'function_call', None)
                if fc and fc.name == 'search':
                    args = dict(fc.args or {})
                    query = str(args.get('query') or question).strip()
                    modality = str(args.get('modality') or '').strip().lower() or None
                    if modality not in (None, 'text', 'table', 'figure'):
                        modality = None
                    self.last_modality = modality
                    return {'action': 'search', 'query': query, 'modality': modality}

        text = (response.text or '').strip()
        return self._parse_answer(text, set(context_ids), question)

    def _generate(self, contents: str, config, attempts: int = 6):
        last = None
        for i in range(attempts):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if '429' in str(exc) or 'RESOURCE_EXHAUSTED' in str(exc):
                    time.sleep(min(90, 8 * (2 ** i)))
                    continue
                raise
        raise last

    def _parse_answer(self, text: str, allowed: set[str], question: str) -> dict[str, Any]:
        raw = text
        if '```' in raw:
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.I | re.M)
        try:
            start, end = raw.find('{'), raw.rfind('}')
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end + 1])
                if obj.get('action') == 'search':
                    return {'action': 'search', 'query': str(obj.get('query') or question),
                            'modality': obj.get('modality')}
                answer = str(obj.get('answer') or '').strip()
                citations = obj.get('citations')
                if not isinstance(citations, list):
                    citations = re.findall(r'\[([^\[\]]+)\]', answer)
                    citations = [c for c in citations if c in allowed]
                return {'action': 'answer', 'answer': answer or 'Không đủ bằng chứng để trả lời.',
                        'citations': [str(c) for c in citations]}
        except json.JSONDecodeError:
            pass
        if not text or 'không đủ bằng chứng' in text.casefold():
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}
        if not allowed:
            return {'action': 'search', 'query': question, 'modality': None}
        cites = [c for c in re.findall(r'\[([^\[\]]+)\]', text) if c in allowed]
        return {'action': 'answer', 'answer': text, 'citations': cites}
