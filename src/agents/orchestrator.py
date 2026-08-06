"""
多智能体编排器
使用 LangGraph StateGraph 编排: Supervisor → Retrieval → Diagnosis → Solution → Verification → Summary
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config
from src.retriever import HybridRetriever
from src.query_rewriter import QueryRewriter
from src.agents.state import AgentState
from src.agents.supervisor import SupervisorAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.solution_agent import SolutionAgent
from src.agents.verification_agent import VerificationAgent
from src.monitoring import track_latency, REQUEST_COUNT


SUMMARY_PROMPT = """你是一位IT运维专家助手。请基于以下智能体协作结果，为用户生成清晰、专业的最终回答。

用户问题: {question}

诊断结果:
- 现象: {symptoms}
- 根因: {root_cause}

解决方案: {solution}
操作步骤:
{steps}

验证结果: {verification}

请用 Markdown 格式输出最终回答，包含:
1. 问题分析（简述）
2. 解决方案（分步骤，含具体命令）
3. 注意事项（如有）

最终回答:"""


class SummarizerAgent:
    """汇总智能体: 将各智能体结果整合为最终回答"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.3,
        )
        self.prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize(self, state: AgentState) -> AgentState:
        import time
        start = time.time()

        steps_text = "\n".join(
            [f"{i+1}. {s}" for i, s in enumerate(state.get("solution_steps", []))]
        )

        try:
            answer = self.chain.invoke({
                "question": state["question"],
                "symptoms": ", ".join(state.get("symptoms", [])),
                "root_cause": state.get("root_cause", "未知"),
                "solution": state.get("solution", "未知"),
                "steps": steps_text or "无具体步骤",
                "verification": state.get("verification_notes", "未验证"),
            })
        except Exception as e:
            # 降级: 直接拼接结果
            answer = self._fallback_summary(state)

        state["final_answer"] = answer
        state["current_agent"] = "summarizer"

        elapsed = time.time() - start
        state["messages"] = state.get("messages", []) + [{
            "agent": "summarizer",
            "action": "summarize",
            "latency_s": round(elapsed, 3),
        }]
        return state

    def _fallback_summary(self, state: AgentState) -> str:
        """降级汇总（LLM 不可用时）"""
        parts = [f"## 问题: {state['question']}"]

        if state.get("root_cause"):
            parts.append(f"\n### 根因分析\n{state['root_cause']}")

        if state.get("solution"):
            parts.append(f"\n### 解决方案\n{state['solution']}")

        if state.get("solution_steps"):
            steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["solution_steps"]))
            parts.append(f"\n### 操作步骤\n{steps}")

        if state.get("verification_notes"):
            parts.append(f"\n### 注意事项\n{state['verification_notes']}")

        return "\n".join(parts)


class MultiAgentOrchestrator:
    """
    多智能体编排器
    管线: Supervisor(路由) → Retrieval(检索) → [Diagnosis → Solution → Verification] → Summary
    """

    def __init__(self, retriever: HybridRetriever, parent_store=None):
        self.supervisor = SupervisorAgent()
        self.retrieval_agent = RetrievalAgent(retriever)
        if parent_store:
            self.retrieval_agent.set_parent_store(parent_store)
        self.diagnosis_agent = DiagnosisAgent()
        self.solution_agent = SolutionAgent()
        self.verification_agent = VerificationAgent()
        self.summarizer = SummarizerAgent()

    def _init_state(self, question: str) -> AgentState:
        return {
            "question": question,
            "question_type": "",
            "retrieved_docs": [],
            "retrieval_queries": [],
            "symptoms": [],
            "possible_causes": [],
            "root_cause": None,
            "solution": None,
            "solution_steps": [],
            "verification_passed": False,
            "verification_notes": "",
            "final_answer": "",
            "messages": [],
            "current_agent": "",
        }

    @track_latency(None)
    def run(self, question: str) -> dict:
        """运行多智能体管线"""
        REQUEST_COUNT.labels(endpoint="/agent", method="POST", status="success").inc()

        state = self._init_state(question)

        # 1. 调度器路由
        state = self.supervisor.route(state)
        q_type = state["question_type"]

        # 2. 检索（所有类型都需要检索）
        state = self.retrieval_agent.retrieve(state)

        # 3. 根据问题类型走不同管线
        if q_type == "diagnosis":
            # 诊断 → 方案 → 验证 → 汇总
            state = self.diagnosis_agent.diagnose(state)
            state = self.solution_agent.generate_solution(state)
            state = self.verification_agent.verify(state)
            state = self.summarizer.summarize(state)

        elif q_type == "operational":
            # 操作类: 跳过诊断，直接方案 → 验证 → 汇总
            state["symptoms"] = [question]
            state["root_cause"] = "直接操作请求"
            state = self.solution_agent.generate_solution(state)
            state = self.verification_agent.verify(state)
            state = self.summarizer.summarize(state)

        else:
            # lookup 类: 简单检索 + 直接汇总
            state["root_cause"] = "知识查询"
            state["solution"] = "参考知识库内容"
            state["solution_steps"] = []
            state["verification_notes"] = "直接知识查询，无需验证"
            state = self.summarizer.summarize(state)

        return {
            "answer": state["final_answer"],
            "question_type": q_type,
            "agents_trace": state["messages"],
        }

    def run_simple(self, question: str) -> str:
        """简化接口: 只返回答案文本"""
        return self.run(question)["answer"]
