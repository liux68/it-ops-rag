"""
诊断智能体
负责: 分析故障现象、识别可能原因、确定根因
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
from src.config import Config
from src.agents.state import AgentState
from src.monitoring import LLM_LATENCY
import time


DIAGNOSIS_PROMPT = """你是一位资深IT运维诊断专家。基于以下信息分析故障:

用户问题: {question}

知识库参考资料:
{context}

请分析并返回 JSON 格式:
{{
  "symptoms": ["观察到的现象1", "现象2"],
  "possible_causes": ["可能原因1", "可能原因2"],
  "root_cause": "最可能的根因"
}}

只返回 JSON，不要其他内容。"""


class DiagnosisAgent:
    """诊断智能体: 分析故障根因"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.2,
        )
        self.prompt = ChatPromptTemplate.from_template(DIAGNOSIS_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def diagnose(self, state: AgentState) -> AgentState:
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join([d.page_content for d in docs]) if docs else "无参考资料"

        start = time.time()
        try:
            result = self.chain.invoke({"question": question, "context": context})

            # 解析 JSON
            # 清理可能的 markdown 包裹
            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result)
            parsed = json.loads(result.strip())

            state["symptoms"] = parsed.get("symptoms", [])
            state["possible_causes"] = parsed.get("possible_causes", [])
            state["root_cause"] = parsed.get("root_cause")
        except Exception as e:
            state["symptoms"] = []
            state["possible_causes"] = []
            state["root_cause"] = f"诊断失败: {e}"

        elapsed = time.time() - start
        LLM_LATENCY.observe(elapsed)

        state["current_agent"] = "diagnosis"
        state["messages"] = [{
            "agent": "diagnosis",
            "action": "diagnose",
            "root_cause": state["root_cause"],
            "latency_s": round(elapsed, 3),
        }]
        return state
