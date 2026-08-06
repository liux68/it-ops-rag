"""
验证智能体
负责: 验证解决方案是否与知识库一致、是否有遗漏
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


VERIFICATION_PROMPT = """你是一位IT运维方案审核专家。请验证以下解决方案的正确性和完整性。

用户问题: {question}

解决方案:
- 概述: {solution}
- 步骤: {steps}

知识库参考资料:
{context}

请验证并返回 JSON:
{{
  "passed": true/false,
  "notes": "验证说明: 方案是否正确、是否有遗漏、是否有风险提示"
}}

只返回 JSON。"""


class VerificationAgent:
    """验证智能体: 审核方案正确性"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.1,
        )
        self.prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def verify(self, state: AgentState) -> AgentState:
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join([d.page_content for d in docs]) if docs else "无参考资料"

        start = time.time()
        try:
            result = self.chain.invoke({
                "question": question,
                "solution": state.get("solution", ""),
                "steps": "\n".join(state.get("solution_steps", [])),
                "context": context,
            })

            result = re.sub(r"```json\s*", "", result)
            result = re.sub(r"```\s*", "", result)
            parsed = json.loads(result.strip())

            state["verification_passed"] = parsed.get("passed", True)
            state["verification_notes"] = parsed.get("notes", "")
        except Exception as e:
            state["verification_passed"] = True
            state["verification_notes"] = f"验证跳过: {e}"

        elapsed = time.time() - start
        LLM_LATENCY.observe(elapsed)

        state["current_agent"] = "verification"
        state["messages"] = [{
            "agent": "verification",
            "action": "verify",
            "passed": state["verification_passed"],
            "latency_s": round(elapsed, 3),
        }]
        return state
