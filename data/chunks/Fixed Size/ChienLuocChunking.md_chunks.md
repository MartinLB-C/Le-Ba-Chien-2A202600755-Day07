# Chunks for ChienLuocChunking.md (Strategy: Fixed Size)

## ChienLuocChunking.md_chunk_0

# Chiến Lược Phân Đoạn Văn Bản (Chunking Strategies) Cho Ứng Dụng LLM

*Tác giả: Roie Schwaber-Cohen, Arjun Patel*

*Nguồn: Pinecone Blog*

Line spacing: 1.25

## 1. Định nghĩa Phân đoạn (Chunking) là gì?

Trong bối cảnh xây dựng các ứng dụng liên quan đến Mô hình Ngôn ngữ Lớn (LLM), **chunking** là quá trình chia nhỏ các văn bản lớn thành các đoạn văn ngắn hơn gọi là "các mảnh" (chunks).

Đây là một kỹ thuật tiền xử lý thiết yếu giúp tối ưu hóa mức độ liên quan của nội dung được lưu

---

## ChienLuocChunking.md_chunk_1

 tối ưu hóa mức độ liên quan của nội dung được lưu trữ trong cơ sở dữ liệu vector. Thách thức lớn nhất là tìm ra kích thước đoạn văn đủ lớn để chứa thông tin có ý nghĩa, nhưng cũng đủ nhỏ để duy trì hiệu suất ứng dụng cao và giảm độ trễ phản hồi cho các tác vụ như RAG (Retrieval-Augmented Generation) và luồng công việc của agent (agentic workflows).

## 2. Tại sao chúng ta cần Phân đoạn văn bản?

Có hai lý do chính khiến việc phân đoạn là bắt buộc đối với bất kỳ ứng dụng nào sử dụng cơ sở dữ

---

## ChienLuocChunking.md_chunk_2

 buộc đối với bất kỳ ứng dụng nào sử dụng cơ sở dữ liệu vector hoặc LLM:

* **Giới hạn của Context Window:** Tất cả các mô hình embedding đều có giới hạn cửa sổ ngữ cảnh (context window), xác định lượng token tối đa có thể xử lý thành một vector có kích thước cố định. Vượt quá giới hạn này đồng nghĩa với việc các token thừa sẽ bị cắt bỏ (truncate), dẫn đến mất mát ngữ cảnh quan trọng.
* **Đảm bảo tính liên quan khi tìm kiếm:** Đoạn văn bản sau khi chia nhỏ phải chứa thông tin có nghĩa độc lập

---

## ChienLuocChunking.md_chunk_3

 khi chia nhỏ phải chứa thông tin có nghĩa độc lập để phục vụ tìm kiếm. Nếu một chunk chứa các câu không rõ nghĩa khi đứng một mình, nó sẽ không thể hiển thị khi người dùng truy vấn.

### Vai trò của Chunking trong Tìm kiếm ngữ nghĩa (Semantic Search)

Trong tìm kiếm ngữ nghĩa, chúng ta lập chỉ mục (index) một tập hợp các tài liệu. Độ tương đồng được xác định bằng cách so sánh vector truy vấn của người dùng với vector của từng mảnh (chunk). Quy tắc chung là: nếu đoạn văn bản đó có nghĩa đối 

---

## ChienLuocChunking.md_chunk_4

uy tắc chung là: nếu đoạn văn bản đó có nghĩa đối với con người khi tách biệt khỏi ngữ cảnh xung quanh, nó cũng sẽ có nghĩa đối với mô hình ngôn ngữ.

### Vai trò của Chunking trong Ứng dụng Agent và RAG

Các Agent cần truy cập vào thông tin cập nhật từ cơ sở dữ liệu để gọi công cụ (tools), đưa ra quyết định và phản hồi người dùng. Nếu một Agent bị cung cấp thông tin sai lệch hoặc thiếu ngữ cảnh, nó có thể lãng phí token để tạo ra các câu trả lời ảo tưởng (hallucinations) hoặc gọi sai công c

---

## ChienLuocChunking.md_chunk_5

 lời ảo tưởng (hallucinations) hoặc gọi sai công cụ.

## 3. Vấn đề của các mô hình LLM có Context Window lớn

Ngay cả với các mô hình hiện đại có cửa sổ ngữ cảnh lên tới 200k token (như o1 hay Claude 4 Sonnet) có thể chứa toàn bộ tài liệu không cần phân đoạn, việc sử dụng các chunk quá lớn vẫn làm tăng độ trễ (latency) và chi phí (cost). Ngoài out, chúng còn gặp phải hiện tượng **"lost-in-the-middle"** (thông tin ở giữa tài liệu dễ bị bỏ sót trong quá trình tạo văn bản).

## 4. Các yếu tố 

---

## ChienLuocChunking.md_chunk_6

trong quá trình tạo văn bản).

## 4. Các yếu tố cần cân nhắc khi chọn chiến lược Chunking

| Yếu tố | Mô tả chi tiết |
| --- | --- |
| **Bản chất của dữ liệu** | Dữ liệu là tài liệu dài (sách, bài báo) hay nội dung ngắn (tweet, tin nhắn)? Tài liệu dài thường có cấu trúc sẵn như tiêu đề phụ hoặc chương để dựa vào phân đoạn. |
| **Mô hình Embedding sử dụng** | Mỗi mô hình có dung lượng token khác nhau và được tối ưu hóa cho các lĩnh vực chuyên biệt (mã nguồn, tài chính, y tế, pháp lý). Hãy 

---

## ChienLuocChunking.md_chunk_7

ên biệt (mã nguồn, tài chính, y tế, pháp lý). Hãy điều chỉnh chiến lược theo dữ liệu huấn luyện của mô hình. |
| **Độ dài và độ phức tạp của truy vấn** | Truy vấn của người dùng ngắn gọn, cụ thể hay dài và phức tạp? Điều này ảnh hưởng trực tiếp đến sự tương quan giữa vector truy vấn và vector chunk. |
| **Cách thức sử dụng kết quả** | Kết quả phục vụ cho tìm kiếm ngữ nghĩa, trả lời câu hỏi trực tiếp, hay cung cấp ngữ cảnh cho Agent? Người dùng là con người hay LLM đọc sẽ quyết định dung lượng 

---

## ChienLuocChunking.md_chunk_8

là con người hay LLM đọc sẽ quyết định dung lượng thông tin tối ưu. |

## 5. Các phương pháp Phân đoạn văn bản (Chunking Methods)

### A. Phân đoạn kích thước cố định (Fixed-size chunking)

Đây là cách tiếp cận phổ biến và đơn giản nhất: xác định một số lượng token cố định cho mỗi đoạn (ví dụ: 1024 đối với llama-text-embed-v2, hoặc 8196 đối với text-embedding-3-small). Đây là phương pháp được khuyến nghị bắt đầu trước khi thử nghiệm các kỹ thuật phức tạp hơn.

### B. Phân đoạn "Nhận biết

---

## ChienLuocChunking.md_chunk_9

thuật phức tạp hơn.

### B. Phân đoạn "Nhận biết nội dung" (Content-aware Chunking)

Phương pháp này tận dụng cấu trúc tự nhiên của tài liệu để phân mảnh mà không làm mất đi ý nghĩa gốc.

* **Tách theo câu hoặc đoạn văn đơn giản:** Tách thô dựa trên dấu chấm (".") hoặc xuống dòng. Có thể sử dụng các công cụ nâng cao như NLTK hoặc spaCy để phân tách câu thông minh hơn.
* **Phân đoạn ký tự đệ quy (Recursive Character Level Chunking):** Được triển khai trong LangChain thông qua `RecursiveCha

---

## ChienLuocChunking.md_chunk_10

triển khai trong LangChain thông qua `RecursiveCharacterTextSplitter`. Công cụ này cố gắng chia văn bản bằng cách sử dụng các ký tự phân tách theo thứ tự ưu tiên (ví dụ: `["\n\n", "\n", " ", ""]`).
* **Phân đoạn dựa trên cấu trúc tài liệu:** Áp dụng các bộ phân tách chuyên biệt cho các định dạng như PDF, HTML (dựa vào thẻ `<p>`, `<title>`), Markdown (tiêu đề, danh sách) hoặc LaTeX (các chương, phần, phương trình).

### C. Phân đoạn ngữ nghĩa (Semantic Chunking)

Do Greg Kamradt đề xuất, kỹ 

---

## ChienLuocChunking.md_chunk_11

Semantic Chunking)

Do Greg Kamradt đề xuất, kỹ thuật này không dựa trên độ dài ký tự mà dựa trên ý nghĩa của văn bản. Phương pháp này chia tài liệu thành các câu, tính toán khoảng cách ngữ nghĩa (semantic distance) giữa các câu liền kề bằng embedding. Khi khoảng cách ngữ nghĩa vượt quá một ngưỡng nhất định, đó chính là ranh giới của một chunk mới.

### D. Phân đoạn theo ngữ cảnh bằng LLM (Contextual Chunking with LLMs)

Do Anthropic giới thiệu vào năm 2024 nhằm giải quyết vấn đề mất ngữ c

---

## ChienLuocChunking.md_chunk_12

hiệu vào năm 2024 nhằm giải quyết vấn đề mất ngữ cảnh toàn cục của tài liệu dài. Kỹ thuật này sử dụng Claude để tạo một mô tả ngắn gọn về mặt ngữ cảnh của toàn bộ tài liệu, sau đó đính kèm mô tả này vào từng mảnh nhỏ trước khi thực hiện embedding.

## 6. Cách tìm ra chiến lược tối ưu cho ứng dụng

1. **Chọn một dải kích thước thử nghiệm:** Thử từ các chunk nhỏ (128 hoặc 256 tokens) cho đến các chunk lớn hơn (512 hoặc 1024 tokens).
2. **Đánh giá hiệu suất:** Tạo các chỉ mục (index) hoặc name

---

## ChienLuocChunking.md_chunk_13

giá hiệu suất:** Tạo các chỉ mục (index) hoặc namespace riêng biệt cho từng kích thước chunk. Chạy một tập dữ liệu truy vấn mẫu để so sánh độ chính xác và chất lượng câu trả lời.
3. **Mở rộng đoạn sau xử lý (Chunk Expansion):** Khi truy vấn, bạn có thể lấy thêm các chunk lân cận xung quanh chunk khớp nhất để bổ sung ngữ cảnh đầy đủ cho LLM mà không làm ảnh hưởng đến tốc độ tìm kiếm ban đầu.

## 7. Kết luận

Không có một giải pháp phân đoạn nào phù hợp cho mọi bài toán (no one-size-fits-all)

---

## ChienLuocChunking.md_chunk_14

ào phù hợp cho mọi bài toán (no one-size-fits-all). Quy trình này đòi hỏi sự thử nghiệm lặp đi lặp lại dựa trên đặc thù dữ liệu và hành vi truy vấn của người dùng ứng dụng của bạn.

---

