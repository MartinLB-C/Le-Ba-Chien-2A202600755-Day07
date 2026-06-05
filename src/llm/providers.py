import os
from abc import ABC, abstractmethod
from typing import Optional
# pyrefly: ignore [missing-import]
from openai import OpenAI 
from .prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

class BaseLLM(ABC):
    @abstractmethod
    def __call__(self, context_text: str, question: str) -> str:
        """Thực thi việc gọi LLM dựa trên context và câu hỏi."""
        pass

class MockLLM(BaseLLM):
    """Một LLM giả lập cho mục đích kiểm thử nội bộ (không gọi API)."""
    
    def __call__(self, context_text: str, question: str) -> str:
        prompt = build_rag_prompt(context_text, question)
        preview = prompt[:400].replace("\n", " ")
        return f"**[Demo LLM]** Generated answer from prompt preview: {preview}..."

class AlibabaLLM(BaseLLM):
    """
    LLM Provider sử dụng API của Alibaba (DashScope) thông qua tương thích OpenAI SDK.
    Yêu cầu biến môi trường DASHSCOPE_API_KEY.
    """
    def __init__(self, model_name: str = "qwen-plus"):
        
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu biến môi trường DASHSCOPE_API_KEY")
            
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
    def __call__(self, context_text: str, question: str) -> str:
        # Chuẩn bị nội dung cho system và user
        system_content = RAG_SYSTEM_PROMPT.strip()
        user_content = f"Context:\n{context_text}\n\nQuestion: {question}"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
