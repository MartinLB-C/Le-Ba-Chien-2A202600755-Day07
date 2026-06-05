from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str, str], str]) -> None:
        """
        Khởi tạo agent với một Vector Store và một hàm gọi LLM.
        Đầu vào: store (EmbeddingStore), llm_fn (hàm nhận context và question trả về answer).
        """
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Quy trình RAG: Tìm kiếm ngữ cảnh -> Xây dựng Prompt -> Gọi LLM.
        Đầu vào: question (str), top_k (số lượng chunk ngữ cảnh).
        Đầu ra: str - Câu trả lời từ LLM.
        """
        # 1. Retrieve top-k relevant chunks
        chunks = self.store.search(question, top_k=top_k)
        
        # 2. Build prompt with retrieved context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"Source {i+1}:\n{chunk['content']}")
        context_text = "\n\n".join(context_parts)
        
        # 3. Call LLM
        return self.llm_fn(context_text, question)
