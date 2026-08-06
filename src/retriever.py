"""
混合检索器
支持: 向量检索 + BM25 检索 + RRF 融合 + 多查询重写
"""

import hashlib
from typing import List, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.embedding_store import VectorStoreManager
from src.config import Config


class HybridRetriever:
    """混合检索器: 向量 + BM25 + RRF 融合"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_manager = vector_store_manager
        self.vector_retriever = vector_store_manager.get_retriever()
        self.bm25_retriever: Optional[BM25Okapi] = None
        self.bm25_docs: List[Document] = []

    # ------------------------------------------------------------------
    # BM25 索引管理
    # ------------------------------------------------------------------

    def build_bm25_index(self, documents: List[Document]):
        """基于文档构建 BM25 索引"""
        self.bm25_docs = documents
        # 中文分词: 简单按字符切分（生产环境可用 jieba）
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25_retriever = BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> list[str]:
        """中文分词: 按字符 + 英文按空格"""
        tokens = []
        for word in text.split():
            # 英文单词整体保留
            if word.isascii():
                tokens.append(word)
            else:
                # 中文按字符切分
                tokens.extend(list(word))
        return tokens

    # ------------------------------------------------------------------
    # 单路检索
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int = 10) -> List[tuple[Document, int]]:
        """向量检索，返回 (doc, rank) 列表"""
        if not self.vector_retriever:
            return []
        try:
            docs = self.vector_retriever.invoke(query)
            return [(doc, rank) for rank, doc in enumerate(docs[:k])]
        except Exception:
            return []

    def _bm25_search(self, query: str, k: int = 10) -> List[tuple[Document, int]]:
        """BM25 检索，返回 (doc, rank) 列表"""
        if not self.bm25_retriever or not self.bm25_docs:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25_retriever.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.bm25_docs[i], rank) for rank, i in enumerate(top_indices)]

    # ------------------------------------------------------------------
    # RRF 融合
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """生成文档唯一标识（基于内容哈希）"""
        return hashlib.md5(doc.page_content.encode()).hexdigest()[:16]

    def rrf_fuse(
        self,
        result_lists: List[List[tuple[Document, int]]],
        k: int = 60,
        top_n: int = 10,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF)
        对多个检索结果列表进行融合排序。

        RRF 公式: score(d) = Σ 1/(k + rank_i(d))

        Args:
            result_lists: 多个检索结果列表，每个元素是 (Document, rank) 元组
            k: RRF 平滑常数，默认 60
            top_n: 返回前 N 个文档
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list in result_lists:
            for doc, rank in result_list:
                doc_id = self._doc_id(doc)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                # 保留文档引用（取第一个出现的）
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # 按 RRF 分数降序排序
        sorted_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)

        result = []
        for doc_id in sorted_ids[:top_n]:
            doc = doc_map[doc_id]
            # 将 RRF 分数写入 metadata 供调试
            doc.metadata["rrf_score"] = round(scores[doc_id], 6)
            result.append(doc)

        return result

    # ------------------------------------------------------------------
    # 混合检索入口
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        混合检索: 向量 + BM25，通过 RRF 融合
        """
        # 1. 向量检索
        vector_results = self._vector_search(query, k=k)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, k=k)

        # 3. RRF 融合
        fused = self.rrf_fuse(
            result_lists=[vector_results, bm25_results],
            k=60,
            top_n=k,
        )
        return fused

    def multi_query_search(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[Document]:
        """
        多查询检索: 对每个查询执行混合检索，再通过 RRF 融合所有结果
        """
        all_result_lists = []
        for q in queries:
            vector_results = self._vector_search(q, k=k)
            bm25_results = self._bm25_search(q, k=k)
            # 每个查询的向量+BM25先做一轮RRF
            per_query_fused = self.rrf_fuse(
                result_lists=[vector_results, bm25_results],
                k=60,
                top_n=k,
            )
            # 转换为 (doc, rank) 格式参与全局融合
            all_result_lists.append([(doc, rank) for rank, doc in enumerate(per_query_fused)])

        # 全局 RRF 融合
        return self.rrf_fuse(
            result_lists=all_result_lists,
            k=60,
            top_n=k,
        )
