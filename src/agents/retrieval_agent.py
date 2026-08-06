"""
检索智能体
负责: 多查询重写 → 混合检索 (RRF融合) → 父子文档扩展
"""

from typing import Optional
from langchain_core.documents import Document
from src.config import Config
from src.retriever import HybridRetriever
from src.query_rewriter import QueryRewriter
from src.agents.state import AgentState
from src.monitoring import RETRIEVAL_LATENCY, RETRIEVAL_DOC_COUNT


class RetrievalAgent:
    """检索智能体: 多查询重写 + 混合检索 + 父子文档扩展"""

    def __init__(self, retriever: HybridRetriever, query_rewriter: Optional[QueryRewriter] = None):
        self.retriever = retriever
        self.query_rewriter = query_rewriter or QueryRewriter(num_queries=Config.MULTI_QUERY_COUNT)
        # 父子文档存储（可选）
        self.parent_store = None

    def set_parent_store(self, store):
        """设置父文档存储"""
        self.parent_store = store

    def retrieve(self, state: AgentState) -> AgentState:
        import time
        question = state["question"]

        start = time.time()

        # 1. 多查询重写
        queries = self.query_rewriter.rewrite(question)
        state["retrieval_queries"] = queries

        # 2. 多查询混合检索 + RRF 融合
        docs = self.retriever.multi_query_search(queries, k=Config.TOP_K_RETRIEVAL)

        # 3. 父子文档扩展（如果配置了父文档存储）
        if self.parent_store:
            parent_docs = self.parent_store.expand_to_parents(docs)
            state["retrieved_docs"] = parent_docs
            RETRIEVAL_DOC_COUNT.set(len(parent_docs))
        else:
            state["retrieved_docs"] = docs
            RETRIEVAL_DOC_COUNT.set(len(docs))

        elapsed = time.time() - start
        RETRIEVAL_LATENCY.labels(strategy="multi_query").observe(elapsed)

        state["current_agent"] = "retrieval"
        state["messages"] = [{
            "agent": "retrieval",
            "action": "retrieve",
            "queries": queries,
            "doc_count": len(state["retrieved_docs"]),
            "latency_s": round(elapsed, 3),
        }]
        return state
