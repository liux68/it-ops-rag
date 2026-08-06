import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import Config
from src.embedding_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.rag_chain import RAGChain
from src.monitoring import (
    RequestTracker,
    SYSTEM_STATUS,
    KB_DOC_COUNT,
)


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    question: str
    use_multi_query: bool = True  # 是否启用多查询重写


class ChatResponse(BaseModel):
    answer: str
    retrieval_strategy: str = "multi_query"


class AsyncChatRequest(BaseModel):
    question: str


class BatchChatRequest(BaseModel):
    questions: list[str]


class TaskResponse(BaseModel):
    task_id: str
    status: str


class AgentChatRequest(BaseModel):
    question: str
    mode: str = "agent"  # agent / simple


class AgentChatResponse(BaseModel):
    answer: str
    question_type: str = ""
    agents_trace: list = []


# ---------- 全局对象 ----------
rag_chain = None
hybrid_retriever = None
agent_orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, hybrid_retriever, agent_orchestrator
    print("正在加载知识库...")

    vec_manager = VectorStoreManager()
    vec_manager.load_index()
    hybrid_retriever = HybridRetriever(vec_manager)

    # 从索引中加载文档用于 BM25
    from src.document_processor import DocumentProcessor, ParentChildStore
    processor = DocumentProcessor()
    parent_store = ParentChildStore()
    try:
        docs = processor.load_documents("./data/source")
        if Config.USE_PARENT_CHILD:
            parent_docs, child_docs = processor.split_parent_child(docs)
            hybrid_retriever.build_bm25_index(child_docs)
            parent_store.store_parents(parent_docs)
            KB_DOC_COUNT.set(len(child_docs))
            print(f"BM25 索引已构建 ({len(child_docs)} 个子块, {len(parent_docs)} 个父块)")
        else:
            chunks = processor.split_documents(docs)
            hybrid_retriever.build_bm25_index(chunks)
            KB_DOC_COUNT.set(len(chunks))
            print(f"BM25 索引已构建 ({len(chunks)} 个文档块)")
    except Exception as e:
        print(f"BM25 索引构建失败 (非致命): {e}")

    reranker = Reranker()
    rag = RAGChain(hybrid_retriever, reranker)
    rag_chain = rag.get_chain()

    # 初始化多智能体编排器
    from src.agents.orchestrator import MultiAgentOrchestrator
    agent_orchestrator = MultiAgentOrchestrator(hybrid_retriever, parent_store)
    print("多智能体系统已初始化")

    SYSTEM_STATUS.set(1)
    print("RAG 知识库已就绪！")
    yield
    SYSTEM_STATUS.set(0)


# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="IT 运维 RAG 知识库 API",
    description="基于混合检索 + RRF融合 + 多查询重写 + 重排序的企业知识问答系统",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "IT 运维 RAG 知识库 API 运行中",
        "version": "2.0.0",
        "features": ["hybrid_search", "rrf_fusion", "multi_query", "rerank", "monitoring", "celery", "multi_agent", "parent_child"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="知识库尚未初始化")

    strategy = "multi_query" if request.use_multi_query else "hybrid"

    with RequestTracker("/chat"):
        try:
            response = rag_chain.invoke({
                "question": request.question,
            })
            return ChatResponse(answer=response, retrieval_strategy=strategy)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy" if rag_chain is not None else "loading",
        "rag_loaded": rag_chain is not None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------
# 异步任务接口 (Celery)
# ------------------------------------------------------------------

@app.post("/chat/async", response_model=TaskResponse)
async def async_chat(request: AsyncChatRequest):
    """提交异步问答任务，返回 task_id"""
    try:
        from src.tasks import async_chat as async_chat_task
        result = async_chat_task.delay(request.question)
        return TaskResponse(task_id=result.id, status="PENDING")
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery 未配置，请启动 Redis 和 Celery Worker")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/batch", response_model=TaskResponse)
async def batch_chat(request: BatchChatRequest):
    """提交批量问答任务，返回 task_id"""
    try:
        from src.tasks import batch_chat as batch_chat_task
        result = batch_chat_task.delay(request.questions)
        return TaskResponse(task_id=result.id, status="PENDING")
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery 未配置，请启动 Redis 和 Celery Worker")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询异步任务状态和结果"""
    try:
        from src.tasks import celery_app
        result = celery_app.AsyncResult(task_id)
        response = {
            "task_id": task_id,
            "status": result.status,
        }
        if result.status == "SUCCESS":
            response["result"] = result.result
        elif result.status == "PROGRESS":
            response["progress"] = result.info
        elif result.status == "FAILURE":
            response["error"] = str(result.info)
        return response
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery 未配置")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/rebuild", response_model=TaskResponse)
async def rebuild_index():
    """触发知识库索引重建（异步任务）"""
    try:
        from src.tasks import rebuild_index as rebuild_task
        result = rebuild_task.delay("./data/source")
        return TaskResponse(task_id=result.id, status="PENDING")
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery 未配置")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 多智能体接口
# ------------------------------------------------------------------

@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """多智能体问答接口"""
    if agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="多智能体系统尚未初始化")

    with RequestTracker("/agent/chat"):
        try:
            result = agent_orchestrator.run(request.question)
            return AgentChatResponse(
                answer=result["answer"],
                question_type=result.get("question_type", ""),
                agents_trace=result.get("agents_trace", []),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
