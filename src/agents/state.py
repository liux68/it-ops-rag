"""
多智能体状态定义
定义智能体间共享的状态结构
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
import operator


class AgentState(TypedDict):
    """多智能体共享状态"""
    # 输入
    question: str
    question_type: str  # diagnosis / lookup / operational / unknown

    # 检索结果
    retrieved_docs: List[Document]
    retrieval_queries: List[str]

    # 诊断结果
    symptoms: List[str]
    possible_causes: List[str]
    root_cause: Optional[str]

    # 解决方案
    solution: Optional[str]
    solution_steps: List[str]

    # 验证结果
    verification_passed: bool
    verification_notes: str

    # 最终输出
    final_answer: str

    # 元信息
    messages: Annotated[List[dict], operator.add]
    current_agent: str
