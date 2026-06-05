from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            import os

            os.makedirs(os.path.join("data", "chroma_db"), exist_ok=True)
            self._client = chromadb.PersistentClient(path=os.path.join("data", "chroma_db"))
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Đầu vào: doc (Document) - Tài liệu cần lưu trữ.
        Đầu ra: dict chứa 'id', 'content', 'metadata', 'embedding'.
        Lưu ý: Thêm doc_id vào metadata nếu chưa có để phục vụ xóa document sau này.
        """
        metadata = doc.metadata.copy() if doc.metadata else {}
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
            
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """
        Thực hiện tìm kiếm khoảng cách Cosine trên danh sách bản ghi In-Memory.
        """
        from .chunking import compute_similarity
        query_emb = self._embedding_fn(query)
        results = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            results.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        """
        Thêm danh sách Document vào store. Tự động chuyển đổi giữa ChromaDB và In-Memory.
        """
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            if ids:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        """
        Tìm kiếm các document khớp nhất với query.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    search_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return search_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        """
        Lọc (Pre-filter) dữ liệu bằng metadata, rồi tìm kiếm trên tập kết quả đã lọc.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            where_clause = metadata_filter if metadata_filter else None
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=where_clause
            )
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    search_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return search_results
        else:
            if not metadata_filter:
                return self._search_records(query, self._store, top_k)
            
            filtered_records = []
            for r in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(r)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        """
        Xoá tất cả Document và Chunk thuộc về một doc_id.
        """
        if self._use_chroma:
            try:
                before_count = self._collection.count()
                self._collection.delete(where={"doc_id": doc_id})
                self._collection.delete(ids=[doc_id])
                after_count = self._collection.count()
                return after_count < before_count
            except Exception:
                return False
        else:
            initial_len = len(self._store)
            self._store = [
                r for r in self._store 
                if r["id"] != doc_id and r["metadata"].get("doc_id") != doc_id
            ]
            return len(self._store) < initial_len
