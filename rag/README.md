# Package `rag`

| Module | Việc |
|---|---|
| `retrievers.py` | Dense (Day 6 e5+pgvector), lexical TF-IDF, hybrid RRF |
| `gemini_controller.py` | Controller tool-calling Gemini |
| `agent_runtime.py` | Vòng agent + pack context truncate-to-fit |
| `run_retrieval_bench.py` | So sánh Hit@1 / Evidence Hit@3 / Context Evidence |
| `run_agent.py` / `resume_agent.py` | Chạy / resume agent khi 429 |
| `build_reviews.py` | Tạo reviews từ đối chiếu gold/evidence |
| `export_chunks.py` | Export chunks từ Postgres Day 6 |
| `metrics.py` | Metric retrieval dùng chung |
| `advanced/` | Multimodal BCTN VCB: `ingest`, `store`, `retriever`, `controller`, `run_agent`, `resume_agent`, `grade`, `make_benchmark` |
