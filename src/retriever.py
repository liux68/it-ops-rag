from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from src.embedding_store import VectorStoreManager


class HybridRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_manager = vector_store_manager
        self.vector_retriever = vector_store_manager.get_retriever()
        self.bm25_retriever = None
        self.doc_texts = []

    def build_bm25_index(self, documents: List[Document]):
        """基于文档构建BM25索引"""
        self.doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [doc.split(" ") for doc in self.doc_texts]
        self.bm25_retriever = BM25Okapi(tokenized_docs)

    def _bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """执行BM25检索"""
        if not self.bm25_retriever:
            return []
        tokenized_query = query.split(" ")
        scores = self.bm25_retriever.get_scores(tokenized_query)
        # 获取top-k的文档索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Document(page_content=self.doc_texts[i]) for i in top_indices]

    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """执行混合检索 (向量 + BM25)"""
        # 1. 向量检索
        vector_docs = self.vector_retriever.invoke(query) if self.vector_retriever else []

        # 2. BM25检索
        bm25_docs = self._bm25_search(query, k=k)

        # 3. 简单融合: 合并去重 (实际项目可用RRF等更复杂的算法)
        # 为了演示，这里简单合并并去重
        all_docs = vector_docs + bm25_docs
        seen_contents = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs[:k]
