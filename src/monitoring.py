"""
Prometheus 监控模块
指标: 请求计数、请求延迟直方图、检索文档数、重排序耗时、LLM 生成耗时
"""

from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Callable

# ------------------------------------------------------------------
# 指标定义
# ------------------------------------------------------------------

# 请求总数 (按端点和方法标签)
REQUEST_COUNT = Counter(
    "rag_request_total",
    "RAG 请求总数",
    ["endpoint", "method", "status"],
)

# 请求延迟直方图
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG 请求延迟 (秒)",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# 检索阶段指标
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "检索阶段延迟 (秒)",
    ["strategy"],  # strategy: hybrid / multi_query
)

RETRIEVAL_DOC_COUNT = Gauge(
    "rag_retrieval_doc_count",
    "检索返回的文档数量",
)

# 重排序阶段指标
RERANK_LATENCY = Histogram(
    "rag_rerank_latency_seconds",
    "重排序阶段延迟 (秒)",
)

# LLM 生成阶段指标
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "LLM 生成延迟 (秒)",
)

# 多查询重写指标
QUERY_REWRITE_COUNT = Counter(
    "rag_query_rewrite_total",
    "查询重写总次数",
)

# 系统状态
SYSTEM_STATUS = Gauge(
    "rag_system_status",
    "系统状态 (1=ready, 0=loading)",
)

# 知识库文档总数
KB_DOC_COUNT = Gauge(
    "rag_kb_doc_count",
    "知识库文档总数",
)


# ------------------------------------------------------------------
# 装饰器: 自动记录延迟
# ------------------------------------------------------------------

def track_latency(metric: Histogram, label: str = None):
    """延迟追踪装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                if label:
                    metric.labels(label).observe(elapsed)
                else:
                    metric.observe(elapsed)
        return wrapper
    return decorator


class RequestTracker:
    """请求级别追踪上下文管理器"""

    def __init__(self, endpoint: str, method: str = "POST"):
        self.endpoint = endpoint
        self.method = method
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        status = "error" if exc_type else "success"
        REQUEST_COUNT.labels(endpoint=self.endpoint, method=self.method, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(elapsed)
        return False  # 不抑制异常
