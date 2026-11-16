# Báo cáo Day 7

Huỳnh Văn Cường · 2A202603866 · L3B

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: tái sử dụng corpus Day 6 — ba luật Việt Nam về không gian số (Luật An ninh mạng 116/2025/QH15, Luật Giao dịch điện tử 20/2023/QH15, Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15), tổng 92 trang PDF Công báo; văn bản QPPL không thuộc đối tượng bảo hộ quyền tác giả (Điều 15 Luật SHTT). Index Day 6 strategy `B_struct` / `C_struct_filter`: **320 chunk**. Export: [`data/chunks.json`](../../data/chunks.json).
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16 + pgvector (Docker Day 6, cổng `127.0.0.1:55433`), bảng `law_chunks`. Embedding `intfloat/multilingual-e5-small` (sentence-transformers), **384 chiều**, L2-normalized, prefix `passage:` / `query:`. Chi phí embedding = 0 USD (chạy local CPU).
- Agent model, cách chạy và phân chia dev/test: Gemini `gemini-3.5-flash-lite` (tool calling `search`), vòng lặp [`rag/agent_runtime.py`](../../rag/agent_runtime.py). Benchmark: 20 câu có đáp án Day 6 + 4 câu thiếu evidence → **dev 15+2 / test 5+2** ([`data/questions.json`](../../data/questions.json)). Day 6 đã xem test khi chọn strategy chunking; với Day 7, tiêu chí chọn agent được chốt trên **dev Day 7** trước khi chạy test (không chỉnh theo kết quả test).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: câu có đáp án giữ `gold.evidence` nằm trọn trong chunk đúng nguồn/trang (đã verify 20/20). Câu thiếu evidence hỏi chi tiết kỹ thuật/phạt tiền cụ thể/phụ lục không có trong 3 PDF. Chấm grounded 0–2 bằng đối chiếu expected/evidence với câu trả lời và citation_support từng citation; abstention đúng khi agent từ chối các câu `answerable=false`.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline dense | dense e5+pgvector, top-k=3, filter theo câu hỏi nếu có, budget 800 (pack skip-if-overflow như starter) | Baseline Day 6 strategy C chunks | dev | 14/15 (0.93) | 15/15 (1.00) | 6/15 (0.40) | ~34 ms/câu · 0 USD |
| Phương pháp 2 lexical | TF-IDF offline, top-k=3, budget 800 | Lexical bắt được cụm từ khóa pháp lý | dev | 11/15 (0.73) | 13/15 (0.87) | 4/15 (0.27) | ~0.7 ms/câu · 0 USD |
| Phương pháp 3 hybrid | RRF(dense, lexical) k=60, candidate 20→3, budget 800 | Fusion giảm miss lexical/dense đơn | dev | 13/15 (0.87) | 15/15 (1.00) | 6/15 (0.40) | ~27 ms/câu · 0 USD |
| Cải thiện 1 | dense, top-k=5, budget 2000 | Tăng candidate + budget để evidence lọt context | dev | 14/15 (0.93) | 15/15 (1.00) | **15/15 (1.00)** | ~26 ms/câu · 0 USD |
| Cải thiện 2 | hybrid, top-k=5, budget 2000 | Hybrid + cùng budget lớn | dev | 13/15 (0.87) | 15/15 (1.00) | 14/15 (0.93) | ~25 ms/câu · 0 USD |
| Cấu hình đã chọn | dense k=5 b=2000 | Chốt theo Context Evidence + agent grounded trên dev | test | 3/5 (0.60) | 4/5 (0.80) | 4/5 (0.80) | ~33 ms/câu · 0 USD |
| Baseline | dense k=3 b=800 | So sánh test độc lập sau khi chốt | test | 3/5 (0.60) | 4/5 (0.80) | 3/5 (0.60) | ~29 ms/câu · 0 USD |

Hai lỗi dev tiêu biểu:
1. **Context budget 800**: nhiều chunk luật >800 ký tự kèm header → `pack_context` kiểu starter bỏ hết → Context Evidence 6/15 dù Hit@3 = 15/15. Sửa bằng truncate-to-fit trong agent runtime + tăng budget 2000.
2. **q17 (test, BVDLCN)**: evidence kéo dài qua trang (Day 6 đã ghi nhận); dense vẫn khó gói đủ “05 năm” → agent abstain (false abstention). Không bịa đáp án.

Lý do chọn **dense · top_k=5 · budget=2000**: trên dev, Context Evidence 15/15; agent grounded mean 2.0 và 0 false abstention; hybrid không vượt dense về Hit@1 và agent grounded, thêm chi phí lexical.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: một tool `search`; tối đa 3 lượt; dừng khi `answer` hợp lệ hoặc hết lượt → abstain. Context: tích lũy candidate, pack truncate-to-fit theo rank. Lỗi 429 Gemini được retry/backoff; không thay bằng kết quả giả.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| baseline dense k3 b800 / dev | 17/17 | ≤3 | mean **1.67** (N=15) | 11/15 | 13/13 | 2/2 | 2/15 |
| improve1 dense k5 b2000 / dev | 17/17 | ≤3 | mean **2.00** (N=15) | 14/18 | 15/15 | 2/2 | **0/15** |
| improve2 hybrid k5 b2000 / dev | 17/17 | ≤3 | mean **1.87** (N=15) | 16/18 | 14/14 | 2/2 | 1/15 |
| baseline / test | 7/7 | ≤3 | mean **0.80** (N=5) | 1/2 | 2/2 | 2/2 | 3/5 |
| **locked dense k5 b2000 / test** | 7/7 | ≤3 | mean **1.20** (N=5) | 3/5 | 3/3 | 2/2 | 2/5 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q01 (dev) | có đáp án | search → chunk p015 BVDLCN | Trả lời 72 giờ + `[luat-bvdlcn-2025:p015:struct:000]` | grounded 2; citation support |
| u01 (dev) | thiếu evidence | search ANM nhưng không có mức phạt VND cụ thể | Abstain | abstention_correct true |
| q17 (test) | có đáp án | search thiếu span “05 năm” đủ | Abstain | grounded 0 (false abstention; giới hạn chunk theo trang) |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: `starter/results/agent-*/traces.json`, `reviews.json`, `summary.json`. Lệnh xem README. Gemini free-tier; có 429 tạm thời, đã resume — không tính chất lượng các status=error (đã hết lỗi trước khi chấm).
- Kết quả trước/sau: locked cải thiện grounded test 0.80→1.20 và giảm false abstention 3/5→2/5; tradeoff latency/context dài hơn và phụ thuộc hạn mức API.

## Thử thách nâng cao (nếu thực hiện)

### Tài liệu và index

- Nguồn: **Báo cáo thường niên 2024 Vietcombank** (106 trang), tải công khai từ website nhà đầu tư VCB. Manifest: [`data/advanced/manifest.json`](../../data/advanced/manifest.json) (URL, SHA-256 `9deccc35…`, quyền dùng học tập). PDF gốc: `data/advanced/raw/vcb-bctn-2024.pdf` (hoặc tải lại theo `source_url`).
- Pipeline: PyMuPDF layout → text theo trang; `find_tables` → markdown bảng; crop ảnh/biểu đồ → Gemini vision caption (không bịa số liệu). Embedding e5 trên text/table/caption → bảng `advanced_chunks` (pgvector Day 6). Export: [`data/advanced/chunks.json`](../../data/advanced/chunks.json) — **547** chunk (`text` 396 · `table` 134 · `figure` 17). Crop mẫu: `assets/advanced/figures/`.
- Phần hạn chế: vài trang watermark/layout phức tạp làm bảng lệch cột; đã bổ sung bảng KPI/thù lao curated từ PDF gốc cho trang then chốt. Số liệu hình luôn đối chiếu caption với trang PDF (không chỉ tin OCR).

### Benchmark riêng

- File: [`data/advanced/questions.json`](../../data/advanced/questions.json) — **48** câu; **dev 18** (15+3) / **test 30** (24+6). Nhóm test: bảng 6 · ảnh 5 · biểu đồ 5 · multi-hop 4 · text 4 · unanswerable 6. Chia split trước khi chỉnh pipeline; gold không đưa vào agent.

### Thử nghiệm và kết quả

| Cấu hình | top-k / budget / max_search | Split | Đạt | Ghi chú |
|---|---|---|---|---|
| baseline | 3 / 1500 / 4 | dev | 10/18 (55.6%) | ít candidate cho bảng KPI |
| improve | 5 / 2500 / 4 | dev | 9/18 (50.0%) | chưa seed modality + curated table |
| **locked** | **5 / 2500 / 4** | **dev** | **17/18 (94.4%)** | search filter modality + bảng curated |
| **locked** | **5 / 2500 / 4** | **test** | **27/30 (90.0%)** | đạt ngưỡng ≥90% |

Test theo nhóm (locked): chart 5/5 · image 5/5 · table 6/6 · multi_hop 3/4 · text 2/4 · unanswerable 6/6. Lỗi hệ thống: **0**. Chi phí embedding local 0 USD; Gemini free-tier (có 429 → resume/sleep).

Ba câu test không đạt (false abstention): **t17** (multi-hop PBT bảng p11 + hình p4), **t23**/**t24** (text p6 cổ phần / website) — retrieval không đưa đủ evidence vào context dù số liệu có trong PDF. Không bịa đáp án.

Ví dụ đạt: **t01** (ROE bảng p11) trả lời đúng + citation `…:p011:table:…`; **t08** (sơ đồ tổ chức p10) dùng figure caption; câu unanswerable abstain đúng.

Traces / reviews / summary: `starter/results/advanced-{dev,test}-locked-k5-b2500/` (force-add khi nộp). Lệnh chạy xem README mục Advanced.
