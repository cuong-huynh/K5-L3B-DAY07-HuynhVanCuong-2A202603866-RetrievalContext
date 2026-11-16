# Day 7 — Agentic RAG (L3B)

**Huỳnh Văn Cường · 2A202603866 · cá nhân · Python 3.11+ · tiếp nối Day 6**

Repo nộp: https://github.com/cuong-huynh/K5-L3B-DAY07-HuynhVanCuong-2A202603866-RetrievalContext

Dùng embedding + pgvector Day 6 để so sánh **dense / lexical TF-IDF / hybrid RRF**, rồi chạy agent Gemini có citation và abstain.

## Cấu hình đã chốt

- Retrieval + agent: **dense** (`B_struct` index) · `top_k=5` · context budget **2000** · max 3 searches  
- Embedding: `intfloat/multilingual-e5-small` (384-d)  
- LLM: `gemini-3.5-flash-lite`  
- Báo cáo: [`report/individual/2A202603866_HuynhVanCuong.md`](report/individual/2A202603866_HuynhVanCuong.md)

## Chuẩn bị

1. Bật Postgres Day 6 (từ thư mục starter Day 6):

```bash
cd <DAY6_STARTER>
docker compose up -d
# nếu DB trống: python -m src.run   # dùng .venv Day 6
```

2. Trong repo Day 7:

```bash
cp .env.example .env
# Điền DATABASE_URL, GEMINI_API_KEY, DAY6_STARTER
```

Có thể dùng chung `.venv` Day 6 (đã có `sentence-transformers`, `psycopg`, `google-genai`) hoặc:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Unit test khung (mock, không cần API/DB):

```bash
python -m unittest discover -s tests -v
```

## Dữ liệu

| File | Nội dung |
|---|---|
| [`data/chunks.json`](data/chunks.json) | 320 chunk export từ `law_chunks` strategy `B_struct` |
| [`data/questions_answerable.json`](data/questions_answerable.json) | 20 câu có đáp án (15 dev / 5 test) |
| [`data/questions.json`](data/questions.json) | 24 câu = 20 + 4 thiếu evidence |
| [`data/agent_cases.json`](data/agent_cases.json) | case cho agent (answerable flag) |

Tái tạo chunks từ DB:

```bash
# xem script one-liner trong lịch sử lab / export lại SELECT từ law_chunks
```

## Chạy benchmark retrieval

```bash
python -m rag.run_retrieval_bench --split dev --methods dense lexical hybrid
python -m rag.run_retrieval_bench --split test --methods dense lexical hybrid
# cải thiện
python -m rag.run_retrieval_bench --split dev --methods dense hybrid --top-k 5 --budget 2000 \
  --output starter/results/retrieval-dev-improve-k5-b2000
```

## Chạy agent

```bash
python -m rag.run_agent --split dev --retriever dense --top-k 5 --budget 2000 --sleep 5 \
  --tag locked-dense-k5-b2000
# nếu 429:
python -m rag.resume_agent --dir starter/results/agent-dev-locked-dense-k5-b2000 --sleep 5
```

Chấm và tổng hợp:

```bash
python -m rag.build_reviews --dir starter/results/agent-test-locked-dense-k5-b2000
python starter/agent_metrics.py \
  --cases starter/results/agent-test-locked-dense-k5-b2000/cases.json \
  --traces starter/results/agent-test-locked-dense-k5-b2000/metric_traces.json \
  --reviews starter/results/agent-test-locked-dense-k5-b2000/reviews.json \
  --split test \
  --output starter/results/agent-test-locked-dense-k5-b2000/summary.json
```

Kết quả mẫu đã chạy nằm ở `starter/results/` (có thể `git add -f` các file summary/traces được phép chia sẻ).

## Lưu ý

- Không commit `.env` / API key.  
- Không đưa `gold` vào agent.  
- Starter `agentic.py` vẫn dùng cho unittest; runtime thật dùng `rag/agent_runtime.py` (truncate-to-fit) vì chunk luật thường >800 ký tự.
