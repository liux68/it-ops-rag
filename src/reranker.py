from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # 使用轻量级的 Cross-Encoder 模型
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_score_pairs[:top_k]]