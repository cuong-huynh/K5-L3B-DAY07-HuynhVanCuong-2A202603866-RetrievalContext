"""Parse VCB annual report into text/table/figure chunks and index into pgvector."""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import pymupdf
from google import genai
from google.genai import types

from ..config import DATA_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL, ROOT, ensure_day6_on_path, require_database_url
from .store import AdvancedStore

DOC_ID = 'vcb-bctn-2024'
PDF_PATH = DATA_DIR / 'advanced' / 'raw' / 'vcb-bctn-2024.pdf'
FIGURE_DIR = ROOT / 'assets' / 'advanced' / 'figures'
CHUNKS_OUT = DATA_DIR / 'advanced' / 'chunks.json'
MANIFEST = DATA_DIR / 'advanced' / 'manifest.json'


def _clean(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_chunks(doc: pymupdf.Document, source: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(doc.page_count):
        page = doc[i]
        text = _clean(page.get_text())
        if len(text) < 40:
            continue
        # split long pages into ~1200 char windows
        step, size = 1000, 1200
        parts = []
        if len(text) <= size:
            parts = [text]
        else:
            for start in range(0, len(text), step):
                chunk = text[start:start + size]
                if len(chunk.strip()) >= 40:
                    parts.append(chunk.strip())
                if start + size >= len(text):
                    break
        for j, part in enumerate(parts):
            rows.append({
                'id': f'{DOC_ID}:p{i+1:03d}:text:{j:03d}',
                'document_id': DOC_ID,
                'source': source,
                'page': i + 1,
                'modality': 'text',
                'region': f'page-{i+1}-text-{j}',
                'text': part,
                'metadata': {'kind': 'text', 'part': j},
            })
    return rows


def extract_table_chunks(doc: pymupdf.Document, source: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(doc.page_count):
        page = doc[i]
        try:
            found = page.find_tables()
        except Exception:
            continue
        if not found or not found.tables:
            continue
        for ti, table in enumerate(found.tables):
            try:
                data = table.extract()
            except Exception:
                continue
            if not data:
                continue
            lines = []
            for row in data:
                cells = [('' if c is None else str(c).strip()) for c in row]
                if any(cells):
                    lines.append(' | '.join(cells))
            body = '\n'.join(lines)
            if len(body) < 20:
                continue
            bbox = [round(x, 1) for x in table.bbox]
            rows.append({
                'id': f'{DOC_ID}:p{i+1:03d}:table:{ti:03d}',
                'document_id': DOC_ID,
                'source': source,
                'page': i + 1,
                'modality': 'table',
                'region': f'bbox={bbox}',
                'text': f'[BẢNG trang {i+1}]\n{body}',
                'metadata': {'kind': 'table', 'bbox': bbox, 'nrows': len(data)},
            })
    return rows


def _gemini_caption(client: genai.Client, model: str, png_bytes: bytes, page: int) -> str:
    prompt = (
        f'Đây là hình/biểu đồ/sơ đồ trong Báo cáo thường niên Vietcombank 2024, trang {page}. '
        'Mô tả bằng tiếng Việt: loại hình (ảnh/sơ đồ/biểu đồ), các nhãn/số liệu đọc được, '
        'ý nghĩa chính. Không bịa số không nhìn thấy. Ngắn gọn ≤120 từ.'
    )
    last = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role='user', parts=[
                        types.Part.from_bytes(data=png_bytes, mime_type='image/png'),
                        types.Part.from_text(text=prompt),
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                ),
            )
            return (resp.text or '').strip() or f'Hình trang {page} (không tạo được caption).'
        except Exception as exc:  # noqa: BLE001
            last = exc
            if '429' in str(exc) or 'RESOURCE_EXHAUSTED' in str(exc):
                time.sleep(min(60, 6 * (2 ** attempt)))
                continue
            raise
    raise last  # type: ignore[misc]


def extract_figure_chunks(doc: pymupdf.Document, source: str, *, caption: bool,
                          max_figures: int = 24) -> list[dict[str, Any]]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    client = None
    if caption:
        if not GEMINI_API_KEY:
            raise RuntimeError('GEMINI_API_KEY required for figure captions')
        client = genai.Client(api_key=GEMINI_API_KEY)

    # Prefer large unique images; also always render org/strategy pages as full-page figure crops.
    candidates: list[tuple[int, bytes, str, list[float] | None]] = []
    seen = set()
    for i in range(doc.page_count):
        page = doc[i]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            try:
                info = doc.extract_image(xref)
            except Exception:
                continue
            blob = info.get('image') if info else None
            if not blob or len(blob) < 25000:
                continue
            seen.add(xref)
            # re-encode via pixmap when possible for stable png
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                png = pix.tobytes('png')
            except Exception:
                png = blob
            candidates.append((i + 1, png, f'xref{xref}', None))

    # Force-render known diagram pages (full page snapshot) for sơ đồ / visual strategy.
    for page_no in (10, 12, 4, 11):
        if page_no > doc.page_count:
            continue
        page = doc[page_no - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        png = pix.tobytes('png')
        candidates.append((page_no, png, f'render-p{page_no}', [0, 0, page.rect.width, page.rect.height]))

    # Dedup by size+page, keep largest first
    candidates.sort(key=lambda x: (-len(x[1]), x[0]))
    picked, used_pages_render = [], set()
    for page_no, png, tag, bbox in candidates:
        key = (page_no, tag)
        if key in used_pages_render:
            continue
        used_pages_render.add(key)
        picked.append((page_no, png, tag, bbox))
        if len(picked) >= max_figures:
            break

    rows = []
    for idx, (page_no, png, tag, bbox) in enumerate(picked):
        rel = f'assets/advanced/figures/p{page_no:03d}_{idx:02d}_{tag}.jpg'
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.Pixmap(png))
            jpg = pix.tobytes('jpeg')
        except Exception:
            jpg = png
        path.write_bytes(jpg)
        nearby = _clean(doc[page_no - 1].get_text())[:500]
        if caption and client is not None:
            try:
                cap = _gemini_caption(client, GEMINI_MODEL, png, page_no)
                time.sleep(2.5)
            except Exception as exc:  # noqa: BLE001
                cap = f'Hình trang {page_no}. Fallback text gần hình: {nearby[:300]} [caption_error={type(exc).__name__}]'
        else:
            cap = f'Hình trang {page_no}. Ngữ cảnh trang: {nearby[:400]}'
        kind = 'diagram' if page_no in (10, 12) else ('chart_or_visual' if page_no in (4, 11) else 'figure')
        rows.append({
            'id': f'{DOC_ID}:p{page_no:03d}:figure:{idx:03d}',
            'document_id': DOC_ID,
            'source': source,
            'page': page_no,
            'modality': 'figure',
            'region': f'file={rel};bbox={bbox}',
            'text': f'[FIGURE trang {page_no} | {kind}]\n{cap}',
            'metadata': {'kind': kind, 'path': rel, 'bbox': bbox, 'bytes': len(jpg)},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-caption', action='store_true')
    parser.add_argument('--max-figures', type=int, default=20)
    parser.add_argument('--skip-index', action='store_true')
    args = parser.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f'Missing PDF: {PDF_PATH}')
    source = str(PDF_PATH.relative_to(ROOT))
    doc = pymupdf.open(PDF_PATH)

    print('extracting text...')
    text_rows = extract_text_chunks(doc, source)
    print('extracting tables...')
    table_rows = extract_table_chunks(doc, source)
    print('extracting figures...')
    figure_rows = extract_figure_chunks(
        doc, source, caption=not args.no_caption, max_figures=args.max_figures)
    doc.close()

    rows = text_rows + table_rows + figure_rows
    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'chunks: text={len(text_rows)} table={len(table_rows)} figure={len(figure_rows)} total={len(rows)}')
    print(f'wrote {CHUNKS_OUT}')

    if args.skip_index:
        return

    ensure_day6_on_path()
    import sys
    for key in list(sys.modules):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    from src.embeddings import E5Embedder  # type: ignore

    embedder = E5Embedder(EMBEDDING_MODEL)
    store = AdvancedStore(require_database_url(), embedder.dimensions)
    store.create_schema()
    texts = [r['text'] for r in rows]
    print(f'embedding {len(texts)} chunks...')
    vectors = embedder.embed_documents(texts)
    n = store.upsert(rows, vectors)
    print('indexed', n, store.count())


if __name__ == '__main__':
    main()
