# Báo Cáo Lab 7: Embedding & Vector Store

**Thành viên nhóm b1:** 
1. Đàm Mạnh Dũng (2A202600741)
2. Nguyễn Hoàng Thanh Tùng (2A202600846)
3. Lê Bá Chiến (2A202600755)
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Có nghĩa là hai đoạn văn bản có sự tương đồng cao về mặt ý nghĩa ngữ nghĩa (chứ không chỉ là từ khóa), do vector biểu diễn của chúng hướng về cùng một phía trong không gian nhiều chiều (góc giữa hai vector rất nhỏ).

**Ví dụ HIGH similarity:**
- Sentence A: "The dog chased the cat."
- Sentence B: "A canine pursued the feline."
- Tại sao tương đồng: Dù không dùng chung từ vựng nào (trừ mạo từ), hai câu này mô tả cùng một hành động và ý nghĩa, do đó các mô hình embedding tốt sẽ biểu diễn chúng gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Python is a popular programming language."
- Sentence B: "The solar system contains eight planets."
- Tại sao khác: Hai câu nói về hai lĩnh vực hoàn toàn không liên quan (lập trình và thiên văn học), không có điểm chung nào về mặt ngữ nghĩa nên độ tương đồng sẽ rất thấp.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Cosine similarity chỉ quan tâm đến hướng (góc) của vector mà không bị ảnh hưởng bởi độ lớn (magnitude) hay độ dài của văn bản, giúp so sánh chính xác sự tương đồng về ngữ nghĩa bất kể văn bản đó dài hay ngắn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
 Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên: `ceil((10000 - 100)/(500 - 100)) = ceil(9900/400) = 25` chunks. Việc tăng overlap giúp đảm bảo tính liên kết ngữ nghĩa không bị đứt đoạn ở ranh giới giữa các chunk, bảo toàn ngữ cảnh tốt hơn cho RAG.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tài liệu kỹ thuật, RAG System Design và Internal Knowledge Assistant.

**Tại sao nhóm chọn domain này?**
 Nhóm chọn domain này vì các tài liệu phản ánh chính quá trình xây dựng hệ thống đang thực hành trong Lab 7. Điều này giúp nhóm vừa kiểm thử được hệ thống Retrieval vừa có thêm kiến thức nền tảng thực tiễn về Vector Store, Embedding và Chunking Strategies.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `ChienLuocChunking.md` | Bài viết Pinecone Blog | 6408 | `{"category": "chunking_strategies", "language": "vi"}` |
| 2 | `vector_store_notes.md` | Tài liệu nội bộ | 2123 | `{"category": "vector_store", "language": "en"}` |
| 3 | `rag_system_design.md` | Tài liệu nội bộ | 2391 | `{"category": "system_design", "language": "en"}` |
| 4 | `customer_support_playbook.txt`| Tài liệu nội bộ | 1692 | `{"category": "customer_support", "language": "en"}` |
| 5 | `vi_retrieval_notes.md` | Tài liệu nội bộ | 1667 | `{"category": "retrieval_notes", "language": "vi"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | String | `system_design`, `customer_support` | Giúp pre-filter kết quả dựa trên ngữ cảnh người dùng đang tìm kiếm (VD: kỹ sư tìm tài liệu hệ thống thay vì tài liệu CS). |
| `language` | String | `en`, `vi` | Hỗ trợ lọc các tài liệu cùng ngôn ngữ với truy vấn để mô hình embedding (vốn ưu tiên tiếng Anh) không bị nhầm lẫn và giảm độ chính xác. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu `ChienLuocChunking.md` (chunk_size=200):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `ChienLuocChunking.md` | FixedSizeChunker (`fixed_size`) | 43 | 197.86 | Trung bình (Thường bị ngắt ngang câu/từ) |
| `ChienLuocChunking.md` | SentenceChunker (`by_sentences`) | 19 | 336.16 | Tốt (Nguyên vẹn câu, nhưng độ dài không đồng đều) |
| `ChienLuocChunking.md` | RecursiveChunker (`recursive`) | 42 | 150.95 | Rất Tốt (Giữ theo đoạn và câu, không vượt quá max_size) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
Chiến lược này hoạt động bằng cách đệ quy chia nhỏ văn bản dựa trên một danh sách các dấu phân cách ưu tiên (ví dụ: `\n\n`, `\n`, `. `, khoảng trắng). Đầu tiên, nó cố gắng tách văn bản ở mức khối lớn (đoạn văn). Nếu một khối vẫn lớn hơn `chunk_size` quy định, nó sẽ tiếp tục đệ quy tách khối đó bằng các ký tự phân cách ưu tiên thấp hơn cho đến khi tất cả các đoạn đều nằm trong giới hạn cho phép, sau đó gộp các đoạn nhỏ lại để tối ưu không gian.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Tài liệu của nhóm chứa nhiều markdown, bullet points và đoạn văn kỹ thuật. Recursive chunking rất lý tưởng cho cấu trúc này vì nó ưu tiên cắt ở các dấu ngắt đoạn tự nhiên `\n\n`, giúp đảm bảo một ý tưởng hoàn chỉnh không bị cắt làm đôi như khi dùng `FixedSizeChunker`.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `ChienLuocChunking` | best baseline (`Sentence`) | 19 | 336.16 | Khá tốt nhưng chunk có lúc quá lớn. |
| `ChienLuocChunking` | **của tôi** (`Recursive`) | 42 | 150.95 | Rất xuất sắc, chunk gọn gàng và giữ đúng ngữ nghĩa. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Bá Chiến (2A202600755) | RecursiveChunker | 9 | Cân bằng hoàn hảo giữa độ lớn chunk và giữ ngữ cảnh | Thuật toán chạy phức tạp và tốn tài nguyên hơn |
| Đàm Mạnh Dũng (2A202600741) | FixedSizeChunker | 6 | Đơn giản, độ dài ổn định, tốc độ thực thi nhanh | Mất mát ý nghĩa nghiêm trọng ở các điểm ngắt đoạn |
| Nguyễn Hoàng Thanh Tùng (2A202600846) | SentenceChunker | 8 | Tôn trọng ranh giới ngôn ngữ tự nhiên tuyệt đối | Đôi khi có chunk bị vỡ kích thước do gặp câu ghép rất dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* RecursiveChunker là chiến lược hiệu quả nhất. Lý do là các tài liệu kỹ thuật có sự phân rã cấu trúc rất rõ nét từ cấp độ Phần (Header) -> Đoạn văn (Paragraph) -> Câu (Sentence). Việc phân chia đệ quy nương theo cấu trúc này để giữ nguyên vẹn nội dung cho embedding model.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `re.split(r'(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)', text)` kèm lookbehind để tách câu và vẫn giữ lại dấu câu ở cuối. Sau đó, nhóm `max_sentences_per_chunk` câu lại thành một chunk và dùng `.strip()` để làm sạch.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng giải thuật Divide and Conquer đệ quy. Nếu đoạn văn <= `chunk_size`, đây là base case. Ngược lại, lấy `separators[0]` để tách, đệ quy gọi `_split` cho từng phần con bằng các `separators` còn lại, và gộp (merge) dần kết quả sao cho mỗi chunk gộp lớn nhất có thể nhưng không vượt `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` duyệt từng tài liệu và tạo record kèm metadata + embedding (tạo sẵn `doc_id`), sau đó lưu vào bộ nhớ `self._store` (hoặc gọi `.upsert` của ChromaDB). Hàm `search` thì tính cosine similarity của câu hỏi với toàn bộ chunk trong In-Memory Store, sắp xếp giảm dần theo điểm và trả về `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` tiến hành "Pre-filtering" bằng cách lặp qua `self._store` lọc các record khớp toàn bộ cặp key-value trong metadata rồi mới chạy `search`. Hàm `delete_document` sẽ lọc và giữ lại tất cả các document trong `self._store` không khớp với `doc_id` cung cấp.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc theo chuẩn RAG: Trước hết gọi `self.store.search(question, top_k)` để trích xuất ngữ cảnh. Kế tiếp, build prompt format bằng cách nối string các content lấy về kèm theo tiền tố `Source {i}:`. Cuối cùng truyền đoạn context này và câu hỏi gốc vào `llm_fn` để lấy câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py .......................................... [100%]

============================= 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Dự đoán và thử nghiệm với mô hình Local `all-MiniLM-L6-v2`:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Phân đoạn văn bản là một kỹ thuật tiền xử lý quan trọng cho LLM. | Chia nhỏ tài liệu giúp tối ưu hóa dữ liệu đầu vào cho mô hình ngôn ngữ lớn. | high | 0.6187 | Có |
| 2 | Giới hạn của Context Window xác định lượng token tối đa có thể xử lý. | Mô hình ngôn ngữ tự động sinh ra văn bản dựa trên xác suất từ vựng. | low | 0.1983 | Có |
| 3 | Recursive Character Text Splitter sử dụng các ký tự phân tách theo thứ tự ưu tiên. | Python là một ngôn ngữ lập trình phổ biến trong khoa học dữ liệu. | low | -0.0151 | Có |
| 4 | Sử dụng kích thước chunk nhỏ giúp giảm độ trễ phản hồi. | Sử dụng kích thước chunk lớn làm tăng độ trễ phản hồi. | low | 0.9417 | Bất ngờ |
| 5 | Semantic Chunking chia tài liệu thành các câu dựa trên khoảng cách ngữ nghĩa. | This method splits text into semantic pieces using embedding distances. | high | 0.1009 | Bất ngờ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Bất ngờ nhất là Pair 4 (Hai câu có ý nghĩa trái ngược nhau về kết quả nhưng độ tương đồng rất cao 0.94) và Pair 5 (Một câu tiếng Việt và một câu dịch tiếng Anh có ý nghĩa y hệt nhưng tương đồng thấp 0.10). Điều này cho thấy mô hình `all-MiniLM-L6-v2` chỉ tập trung vào cấu trúc câu, bộ từ vựng xuất hiện cùng ngữ cảnh (chunk, độ trễ) chứ không hiểu được logic đối nghịch, đồng thời mô hình này cũng không có khả năng hiểu đa ngôn ngữ (Multilingual) tốt.

---

## 6. Results — Cá nhân (10 điểm)

Đã chạy benchmark bằng `LocalEmbedder (all-MiniLM-L6-v2)`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Why do teams choose Python for development? | Python emphasizes readability, low barrier to entry, and offers mature tooling for rapid prototyping. |
| 2 | What are the four stages of a typical vector search pipeline? | Chunk documents, embed chunks, store vectors/metadata, and embed query to rank. |
| 3 | What metadata fields should be stored in the ingestion pipeline? | Source path, document identifier, document type, and department. |
| 4 | What should customer support authors avoid when writing articles? | Vague statements like 'check the settings' or 'contact engineering if needed'. |
| 5 | Làm thế nào để tránh lấy nhầm tài liệu tiếng Anh hoặc tài liệu marketing? | Sử dụng metadata filters để giới hạn tìm kiếm theo phòng ban và ngôn ngữ. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Why do teams choose Python for development? | `python_intro` (Python is a high-level...) | 0.6329 | Có | Teams choose it because of readability and mature tooling... |
| 2 | What are the four stages of a typical vector search pipeline? | `vector_store_notes` (A vector store is a database...) | 0.5687 | Có | 1. Chunk documents, 2. Embed chunks, 3. Store vector... |
| 3 | What metadata fields should be stored in the ingestion pipeline? | `rag_system_design` (RAG System Design for...) | 0.4602 | Có | Source path, document identifier, document type... |
| 4 | What should customer support authors avoid when writing articles? | `customer_support_playbook` (The support team uses...) | 0.5418 | Có | They should avoid vague statements such as "check the settings"... |
| 5 | Làm thế nào để tránh lấy nhầm tài liệu tiếng Anh hoặc tài liệu marketing? | `vi_retrieval_notes` (Ghi chú về Retrieval...) | 0.4838 | Có | Metadata filtering giúp hệ thống hạn chế tìm sai văn bản... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc kết hợp Pre-filtering Metadata trước khi tính toán độ tương đồng (search) không chỉ cải thiện đáng kể độ chính xác (precision) mà còn tăng hiệu năng thực thi vì LLM không phải chấm điểm vector cho các tài liệu rác.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Giao diện Streamlit là một công cụ xuất sắc để trực quan hoá các chunk và điều chỉnh các mức `chunk_size`, `overlap` theo thời gian thực thay vì phải debug trong console bằng print().

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thiết kế một pipeline làm sạch dữ liệu đầu vào tốt hơn, xoá bỏ các headers/footers dư thừa trong tài liệu Markdown, và thử nghiệm mô hình embedding Multilingual (vd: `paraphrase-multilingual-MiniLM-L12-v2`) để xử lý các truy vấn Tiếng Việt một cách chính xác hơn thay vì để miss dữ liệu như hiện tại.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
