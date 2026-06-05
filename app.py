import streamlit as st
import os
import json

from src.models import Document
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.embeddings import _mock_embed, LocalEmbedder, OpenAIEmbedder
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.llm import MockLLM, AlibabaLLM

def process_uploaded_file(uploaded_file):
    # Lưu file vào thư mục data
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.sidebar.success(f"Đã lưu file vào: {save_path}")
    
    # Trả về nội dung text
    return uploaded_file.getvalue().decode("utf-8")

def main():
    st.set_page_config(page_title="RAG Pipeline Demo", layout="wide")
    st.title("📚 RAG Knowledge Base Demo")
    
    st.sidebar.header("Cài Đặt Hệ Thống")
    
    # 1. Chunking Strategy Selector
    chunk_strategy = st.sidebar.selectbox(
        "Chọn Chiến lược Chunking",
        ("Fixed Size", "Sentence", "Recursive")
    )
    
    chunker = None
    if chunk_strategy == "Fixed Size":
        chunk_size = st.sidebar.slider("Chunk Size (Ký tự)", 100, 2000, 500)
        overlap = st.sidebar.slider("Overlap (Ký tự)", 0, 500, 50)
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif chunk_strategy == "Sentence":
        max_sentences = st.sidebar.slider("Số câu tối đa / chunk", 1, 10, 3)
        chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
    else:
        chunk_size = st.sidebar.slider("Chunk Size (Ký tự) (Recursive)", 100, 2000, 500)
        chunker = RecursiveChunker(chunk_size=chunk_size)

    # 1.5. Embedding Strategy Selector
    embedder_strategy = st.sidebar.selectbox(
        "Chọn Mô Hình Embedding",
        ("Mock (Mặc định)", "Local (Sentence-Transformers)", "OpenAI")
    )
    
    @st.cache_resource
    def get_local_embedder():
        return LocalEmbedder()
        
    @st.cache_resource
    def get_openai_embedder():
        return OpenAIEmbedder()

    current_embedder = None
    if embedder_strategy == "Mock (Mặc định)":
        current_embedder = _mock_embed
    elif embedder_strategy == "Local (Sentence-Transformers)":
        try:
            current_embedder = get_local_embedder()
        except ImportError:
            st.sidebar.error("Lỗi: Cần cài đặt `sentence-transformers`! (pip install sentence-transformers)")
            current_embedder = _mock_embed
    elif embedder_strategy == "OpenAI":
        try:
            current_embedder = get_openai_embedder()
        except ImportError:
            st.sidebar.error("Lỗi: Cần cài đặt `openai`! (pip install openai)")
            current_embedder = _mock_embed
        except Exception as e:
            st.sidebar.error(f"Lỗi khởi tạo OpenAI: {str(e)}")
            current_embedder = _mock_embed

    st.sidebar.subheader("Cấu Hình Ngôn Ngữ (LLM)")
    llm_strategy = st.sidebar.selectbox(
        "Chọn Mô Hình Trả Lời (LLM)",
        ("Mock LLM (Mặc định)", "Alibaba Qwen (DashScope)")
    )
    
    current_llm = None
    if llm_strategy == "Mock LLM (Mặc định)":
        current_llm = MockLLM()
    elif llm_strategy == "Alibaba Qwen (DashScope)":
        api_key_input = st.sidebar.text_input("Nhập DASHSCOPE_API_KEY", type="password")
        if api_key_input:
            os.environ["DASHSCOPE_API_KEY"] = api_key_input
        try:
            current_llm = AlibabaLLM()
        except Exception as e:
            st.sidebar.error(f"Lỗi khởi tạo Alibaba LLM: {str(e)}")
            current_llm = MockLLM()

    # 2. Upload Document
    st.sidebar.subheader("Tải Lên Tài Liệu")
    uploaded_file = st.sidebar.file_uploader("Chọn file .txt hoặc .md", type=["txt", "md"])
    
    # Khởi tạo lại Store nếu người dùng đổi mô hình Embedding
    if "current_embedder_name" not in st.session_state or st.session_state.current_embedder_name != embedder_strategy:
        safe_strategy_name = embedder_strategy.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        collection_name = f"docs_{safe_strategy_name}"
        st.session_state.store = EmbeddingStore(collection_name=collection_name, embedding_fn=current_embedder)
        st.session_state.current_embedder_name = embedder_strategy
        
        # Tự động tải lại dữ liệu đã lưu từ thư mục data/chunks/
        chunk_root_dir = os.path.join("data", "chunks")
        if os.path.exists(chunk_root_dir) and st.session_state.store.get_collection_size() == 0:
            all_docs = []
            for root_dir, _, files in os.walk(chunk_root_dir):
                for file in files:
                    if file.endswith("_metadata.json"):
                        with open(os.path.join(root_dir, file), "r", encoding="utf-8") as f:
                            try:
                                metadata_list = json.load(f)
                                for item in metadata_list:
                                    all_docs.append(Document(
                                        id=item["id"],
                                        content=item.get("content", ""),
                                        metadata=item["metadata"]
                                    ))
                            except Exception:
                                pass
            if all_docs:
                st.session_state.store.add_documents(all_docs)
                
        st.session_state.docs_added = st.session_state.store.get_collection_size() > 0
        
    # Luôn update lại LLM cho agent
    st.session_state.agent = KnowledgeBaseAgent(st.session_state.store, current_llm)

    if uploaded_file is not None and st.sidebar.button("Thêm vào Cơ sở Tri thức"):
        chunk_dir = os.path.join("data", "chunks", chunk_strategy)
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_filename = f"{uploaded_file.name}_chunks.md"
        chunk_filepath = os.path.join(chunk_dir, chunk_filename)
        metadata_filename = f"{uploaded_file.name}_metadata.json"
        metadata_filepath = os.path.join(chunk_dir, metadata_filename)
        
        docs = []
        if os.path.exists(metadata_filepath):
            st.sidebar.info(f"Tài liệu đã được chunk với '{chunk_strategy}' trước đó. Đang tải lại từ file...")
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
            
            for item in metadata_list:
                docs.append(Document(
                    id=item["id"],
                    content=item.get("content", ""), # Load content back from JSON
                    metadata=item["metadata"]
                ))
            st.sidebar.success(f"Đã tải {len(docs)} chunks từ thư mục {chunk_dir} và đưa vào CSDL!")
        else:
            text_content = process_uploaded_file(uploaded_file)
            chunks = chunker.chunk(text_content)
            
            # Tạo danh sách các Document
            for i, chunk_text in enumerate(chunks):
                doc = Document(
                    id=f"{uploaded_file.name}_chunk_{i}",
                    content=chunk_text,
                    metadata={"source": uploaded_file.name, "chunk_index": i, "strategy": chunk_strategy}
                )
                docs.append(doc)
                
            # Lưu các chunks ra file .md
            with open(chunk_filepath, "w", encoding="utf-8") as f:
                f.write(f"# Chunks for {uploaded_file.name} (Strategy: {chunk_strategy})\n\n")
                for doc in docs:
                    f.write(f"## {doc.id}\n\n")
                    f.write(f"{doc.content}\n\n")
                    f.write("---\n\n")
                    
            # Lưu metadata ra file .json
            metadata_list = []
            for doc in docs:
                metadata_list.append({
                    "id": doc.id,
                    "content": doc.content,  # Save content so it can be restored next time
                    "content_length": len(doc.content),
                    "metadata": doc.metadata
                })
                
            with open(metadata_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
                
            st.sidebar.success(f"Đã tạo {len(docs)} chunks mới với '{chunk_strategy}' và đưa vào CSDL!")
                
        # Thêm vào store
        st.session_state.store.add_documents(docs)
        st.session_state.docs_added = True
        
        # Display chunks (Optional preview)
        with st.expander("Xem trước các đoạn cắt (Chunks Preview)"):
            for d in docs:
                st.markdown(f"**Chunk ID: {d.id}**")
                st.text(d.content)
                st.markdown("---")

    st.sidebar.info(f"Kích thước CSDL hiện tại: {st.session_state.store.get_collection_size()} chunks")

    # 3. Chat Interface
    st.header("Trò Chuyện (Q&A)")
    top_k = st.slider("Số lượng ngữ cảnh (Top K)", 1, 10, 3)
    
    user_query = st.text_input("Nhập câu hỏi của bạn về tài liệu:")
    
    if st.button("Hỏi"):
        if not st.session_state.docs_added:
            st.warning("Vui lòng tải lên và Thêm tài liệu vào Cơ sở tri thức trước.")
        elif not user_query.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang tìm kiếm và xử lý..."):
                answer = st.session_state.agent.answer(user_query, top_k=top_k)
                
                st.subheader("🤖 Trả lời (Agent Response)")
                st.markdown(answer)
                
                # Show retrieved contexts
                st.subheader("📄 Ngữ Cảnh Đã Truy Xuất (Độ tin cậy > 0.7)")
                retrieved = st.session_state.store.search(user_query, top_k=top_k)
                
                # Lọc các nguồn có score > 0.7
                valid_sources = [r for r in retrieved if isinstance(r.get('score'), (int, float)) and r['score'] > 0.7]
                
                if not valid_sources:
                    st.info("Không có nguồn dữ liệu nào vượt qua ngưỡng điểm 0.7.")
                else:
                    # Đảm bảo sắp xếp từ cao xuống thấp
                    valid_sources.sort(key=lambda x: x['score'], reverse=True)
                    
                    for idx, r in enumerate(valid_sources):
                        score = r['score']
                        source_file = r.get('metadata', {}).get('source', 'Không rõ')
                        st.markdown(f"**Nguồn {idx+1} (Điểm: {score:.4f} | Từ file: `{source_file}`)**")
                        st.text(r['content'])
                        st.markdown("---")

if __name__ == "__main__":
    main()
