"""
Celery 异步任务队列
解决长耗时操作（文档索引、批量问答）的同步阻塞问题。
"""

import os
from celery import Celery
from typing import Optional

from src.config import Config

# ------------------------------------------------------------------
# Celery 配置
# ------------------------------------------------------------------

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "it_ops_rag",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
    task_time_limit=300,          # 单任务最长 5 分钟
    task_soft_time_limit=240,     # 软超时 4 分钟
    worker_prefetch_multiplier=1,  # 一次只取一个任务，避免长任务阻塞
    worker_max_tasks_per_child=50, # 每 50 个任务重启 worker，防内存泄漏
)


# ------------------------------------------------------------------
# 全局 RAG 实例（worker 启动时懒加载）
# ------------------------------------------------------------------

_rag_chain = None
_hybrid_retriever = None


def _get_rag_chain():
    """懒加载 RAG 链（在 worker 进程中只初始化一次）"""
    global _rag_chain, _hybrid_retriever
    if _rag_chain is None:
        from src.embedding_store import VectorStoreManager
        from src.retriever import HybridRetriever
        from src.reranker import Reranker
        from src.rag_chain import RAGChain
        from src.document_processor import DocumentProcessor

        vec_manager = VectorStoreManager()
        vec_manager.load_index()
        _hybrid_retriever = HybridRetriever(vec_manager)

        # 构建 BM25 索引
        processor = DocumentProcessor()
        try:
            docs = processor.load_documents("./data/source")
            chunks = processor.split_documents(docs)
            _hybrid_retriever.build_bm25_index(chunks)
        except Exception as e:
            print(f"[Celery] BM25 索引构建失败: {e}")

        reranker = Reranker()
        rag = RAGChain(_hybrid_retriever, reranker)
        _rag_chain = rag.get_chain()

    return _rag_chain


# ------------------------------------------------------------------
# 异步任务定义
# ------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.async_chat")
def async_chat(self, question: str):
    """
    异步问答任务
    用法: result = async_chat.delay("问题")
          answer = result.get(timeout=60)
    """
    try:
        chain = _get_rag_chain()
        answer = chain.invoke({"question": question})
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@celery_app.task(bind=True, name="tasks.batch_chat")
def batch_chat(self, questions: list[str]):
    """
    批量问答任务
    用法: result = batch_chat.delay(["问题1", "问题2"])
    """
    chain = _get_rag_chain()
    results = []
    for i, q in enumerate(questions):
        try:
            answer = chain.invoke({"question": q})
            results.append({"question": q, "answer": answer, "status": "success"})
        except Exception as e:
            results.append({"question": q, "answer": None, "status": "error", "error": str(e)})
        # 更新进度
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": len(questions)},
        )
    return {"status": "success", "results": results}


@celery_app.task(bind=True, name="tasks.rebuild_index")
def rebuild_index(self, data_dir: str = "./data/source"):
    """
    重建知识库索引（向量 + BM25）
    """
    from src.document_processor import DocumentProcessor
    from src.embedding_store import VectorStoreManager
    from src.retriever import HybridRetriever

    try:
        # 1. 加载文档
        processor = DocumentProcessor()
        docs = processor.load_documents(data_dir)
        self.update_state(state="PROGRESS", meta={"step": "loaded", "count": len(docs)})

        # 2. 分块
        chunks = processor.split_documents(docs)
        self.update_state(state="PROGRESS", meta={"step": "chunked", "count": len(chunks)})

        # 3. 构建向量索引
        vec_manager = VectorStoreManager()
        vec_manager.build_index(chunks)
        self.update_state(state="PROGRESS", meta={"step": "vector_indexed"})

        # 4. 构建 BM25 索引
        retriever = HybridRetriever(vec_manager)
        retriever.build_bm25_index(chunks)
        self.update_state(state="PROGRESS", meta={"step": "bm25_indexed"})

        # 5. 重置 worker 内的缓存
        global _rag_chain, _hybrid_retriever
        _rag_chain = None
        _hybrid_retriever = None

        return {
            "status": "success",
            "docs_loaded": len(docs),
            "chunks_created": len(chunks),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
