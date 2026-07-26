import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # RAG 核心参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 10  # 初次检索召回数量
    TOP_K_RERANK = 3      # 重排序后最终数量

    # 向量库路径
    FAISS_INDEX_PATH = "./faiss_index"