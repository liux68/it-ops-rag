"""
方案智能体
负责: 基于诊断结果和知识库，生成可操作的解决方案
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


SOLUTION_PROMPT = """你是一位IT运维方案专家。基于诊断结果和知识库，生成详细的解决方案。

用户问题: {question}

诊断结果:
- 现象: {symptoms}
- 可能原因: {possible_causes}
- 根因: {root_cause}

知识库参考资料:
{context}

请返回 JSON 格式:
{{
  "solution": "一句话概述解决方案",
  "steps": ["步骤1: 详细操作", "步骤2: 详细操作", "步骤3: 详细操作"]
}}

注意: 步骤必须具体可执行，包含命令或操作细节。只返回 JSON。"""


class SolutionAgent:
    """方案智能体: 生成可操作解决方案"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.3,
        )
        self.prompt = ChatPromptTemplate.from_template(SOLUTION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_solution(self, state: AgentState) -> AgentState:
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join([d.page_content for d in docs]) if docs else "无参考资料"

        start = time.time()
        try:
            result = self.chain.invoke({
                "question": question,
                "symptoms": ", ".join(state.get("symptoms", [])),
                "possible_causes": ", ".join(state.get("possible_causes", [])),
                "root_cause": state.get("root_cause", "未知"),
                "context": context,
            })

            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result)
            parsed = json.loads(result.strip())

            state["solution"] = parsed.get("solution")
            state["solution_steps"] = parsed.get("steps", [])
        except Exception as e:
            # 降级: 直接生成文本方案
            try:
                fallback = self.llm.invoke(
                    f"问题: {question}\n参考资料: {context}\n请给出解决方案。"
                )
                state["solution"] = fallback.content
                state["solution_steps"] = []
            except:
                state["solution"] = f"方案生成失败: {e}"
                state["solution_steps"] = []

        elapsed = time.time() - start
        LLM_LATENCY.observe(elapsed)

        state["current_agent"] = "solution"
        state["messages"] = [{
            "agent": "solution",
            "action": "generate",
            "solution": state.get("solution", ""),
            "latency_s": round(elapsed, 3),
        }]
        return state
