# Định dạng dữ liệu Day 7

- **`chunks.json`**: array chunk `{id, text, source_id, source, page, metadata}` export từ Day 6 `law_chunks` strategy `B_struct` (320 chunk). Tái tạo: `python -m rag.export_chunks`.
- **`questions_answerable.json`**: 20 câu có đáp án (gold evidence nằm trong chunk).
- **`questions.json`**: 24 câu = 20 có đáp án + 4 thiếu evidence (`answerable: false`).
- **`agent_cases.json`**: schema gọn cho runner agent.
- File `example_*.json` chỉ minh họa; không đủ để nộp.

## Advanced (`data/advanced/`)

- **`manifest.json`**: nguồn BCTN VCB 2024, SHA-256, quyền sử dụng.
- **`raw/vcb-bctn-2024.pdf`**: bản PDF dùng để index (có thể tải lại theo URL trong manifest).
- **`chunks.json`**: 547 chunk `{id, page, modality, text, region, metadata}` (text/table/figure).
- **`questions.json`**: benchmark riêng 48 câu (dev 18 / test 30), gold evidence theo trang + modality.

Không đưa `gold` vào model/retriever khi chạy agent.
