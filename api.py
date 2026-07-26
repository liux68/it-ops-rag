import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.config import Config
from src.embedding_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.rag_chain import RAGChain


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    # 可选：返回引用的文档来源，便于调试
    # sources: List[str] = []


# ---------- 全局对象 ----------
# 在 FastAPI 启动时加载一次，避免每次请求都重新加载
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("🔍 正在加载知识库...")

    vec_manager = VectorStoreManager()
    vec_manager.load_index()
    hybrid_retriever = HybridRetriever(vec_manager)
    reranker = Reranker()
    rag = RAGChain(hybrid_retriever, reranker)
    rag_chain = rag.get_chain()

    print("💬 RAG 知识库已就绪！")
    yield
    # 关闭时清理资源（可选）


# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="IT 运维 RAG 知识库 API",
    description="基于混合检索 + 重排序的企业知识问答系统",
    version="1.0.0",
    lifespan=lifespan
)


# ---------- 根路径 ----------
@app.get("/")
async def root():
    return {"message": "IT 运维 RAG 知识库 API 运行中", "status": "ok"}


# ---------- 聊天接口 ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="知识库尚未初始化")

    try:
        response = rag_chain.invoke({"question": request.question})
        return ChatResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "healthy", "rag_loaded": rag_chain is not None}