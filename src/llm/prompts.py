# src/llm/prompts.py

RAG_SYSTEM_PROMPT = """
You are a helpful and precise assistant.
Your task is to answer the user's question based ONLY on the provided context.

RULES:
1. If the context does not contain the answer, say "I don't know based on the provided context."
2. Do not invent or hallucinate information outside of the context.
3. Be concise and direct in your answer.
4. If appropriate, cite the source number of the context you used.
"""

def build_rag_prompt(context_text: str, question: str) -> str:
    """Ghép ngữ cảnh và câu hỏi vào một chuỗi prompt (Dành cho MockLLM hoặc hệ thống không hỗ trợ role-based message)."""
    return (
        f"{RAG_SYSTEM_PROMPT.strip()}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
