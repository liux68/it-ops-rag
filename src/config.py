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

    # 父子文档分块参数
    PARENT_CHUNK_MULTIPLIER = 4  # 父块 = 子块大小 * 此倍数
    USE_PARENT_CHILD = True      # 是否启用父子文档分块

    # 多查询重写参数
    MULTI_QUERY_COUNT = 3  # 重写查询变体数量
    RRF_K = 60             # RRF 融合常数

    # 向量库路径
    FAISS_INDEX_PATH = "./faiss_index"

    # Celery 配置
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
