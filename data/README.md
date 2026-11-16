# Định dạng dữ liệu Day 7

- **`chunks.json`**: array chunk `{id, text, source_id, source, page, metadata}` export từ Day 6 `law_chunks` strategy `B_struct` (320 chunk). Tái tạo: `python -m rag.export_chunks`.
- **`questions_answerable.json`**: 20 câu có đáp án (gold evidence nằm trong chunk).
- **`questions.json`**: 24 câu = 20 có đáp án + 4 thiếu evidence (`answerable: false`).
- **`agent_cases.json`**: schema gọn cho runner agent.
- File `example_*.json` chỉ minh họa; không đủ để nộp.

Không đưa `gold` vào model/retriever khi chạy agent.
