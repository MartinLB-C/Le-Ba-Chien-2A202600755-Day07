from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        """
        Chia văn bản thành các chunk, mỗi chunk chứa tối đa max_sentences_per_chunk câu.
        Đầu vào: text (str) - Văn bản gốc cần chia.
        Đầu ra: list[str] - Danh sách các chunk.
        Liên kết: Sử dụng trong ChunkingStrategyComparator để so sánh với các chiến lược khác.
        """
        if not text.strip():
            return []
        
        # Tách câu bằng regex, lookbehind để giữ lại dấu câu (. ! ? .\n)
        raw_sentences = re.split(r'(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)', text)
        # Loại bỏ các chuỗi rỗng
        sentences = [s for s in raw_sentences if s]
        
        chunks: list[str] = []
        # Nhóm các câu lại thành từng chunk
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = "".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """
        Hàm chính gọi hàm chia đệ quy.
        Đầu vào: text (str) - Văn bản gốc cần chia.
        Đầu ra: list[str] - Danh sách các chunk sau khi phân tách theo separator.
        Liên kết: Gọi tới _split và được dùng trong RAG pipeline nếu chọn RecursiveChunker.
        """
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """
        Hàm đệ quy phân tách văn bản.
        Nếu phần văn bản nhỏ hơn chunk_size, trả về chính nó.
        Nếu lớn hơn và hết separator, dùng FixedSizeChunker làm fallback.
        Nếu lớn hơn, cắt bằng separator hiện tại, rồi đệ quy các phần con, sau đó gộp lại.
        """
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case / Fallback: Hết separators
        if not remaining_separators:
            from .chunking import FixedSizeChunker # Tránh vòng lặp import
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(current_text)

        sep = remaining_separators[0]
        
        # Tách văn bản
        if sep == "":
            parts = list(current_text)
        else:
            parts = current_text.split(sep)

        # Đệ quy cho từng phần con
        sub_chunks: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                sub_chunks.append(part)
            else:
                sub_chunks.extend(self._split(part, remaining_separators[1:]))

        # Gộp (merge) lại các phần con đã cắt
        merged_chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for chunk_item in sub_chunks:
            sep_len = len(sep) if current_chunk else 0
            if current_len + sep_len + len(chunk_item) <= self.chunk_size:
                current_chunk.append(chunk_item)
                current_len += sep_len + len(chunk_item)
            else:
                if current_chunk:
                    merged_chunks.append(sep.join(current_chunk))
                current_chunk = [chunk_item]
                current_len = len(chunk_item)

        if current_chunk:
            merged_chunks.append(sep.join(current_chunk))

        return merged_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    
    Tính độ tương đồng Cosine giữa 2 vector.
    Đầu vào: vec_a (list[float]), vec_b (list[float]).
    Đầu ra: float - điểm số từ -1.0 đến 1.0 (càng gần 1.0 càng giống nhau).
    Liên kết: Dùng trong bộ nhớ In-Memory Store để xếp hạng chunks.
    """
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    
    # zero-magnitude guard
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        So sánh số lượng và kích thước chunk giữa 3 chiến lược.
        Đầu vào: text (str) - Văn bản mẫu, chunk_size (int).
        Đầu ra: dict - Từ điển chứa thống kê (count, avg_length, chunks).
        """
        fixed_size_chunker = FixedSizeChunker(chunk_size=chunk_size)
        sentence_chunker = SentenceChunker()
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_size_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        result = {}
        for strategy, chunks in [
            ('fixed_size', fixed_chunks),
            ('by_sentences', sentence_chunks),
            ('recursive', recursive_chunks)
        ]:
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[strategy] = {
                'count': count,
                'avg_length': avg_length,
                'chunks': chunks
            }
        return result
