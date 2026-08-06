"""
Supervisor 智能体
负责路由: 分析问题类型，决定调度哪些智能体
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.agents.state import AgentState


ROUTER_PROMPT = """你是一个IT运维智能体系统的调度器。请分析用户问题，判断其类型:

- "diagnosis": 故障诊断类问题（如"服务器CPU高怎么排查"、"连接超时怎么处理"），需要分析现象、排查原因
- "lookup": 知识查询类问题（如"HikariCP参数怎么配"、"备份策略是什么"），直接查文档即可
- "operational": 操作执行类问题（如"怎么清理磁盘"、"怎么重启服务"），需要具体操作步骤

只回答一个单词: diagnosis / lookup / operational

用户问题: {question}

类型:"""


class SupervisorAgent:
    """调度器: 路由问题到对应智能体"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.0,
        )
        self.prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        self.chain = self.prompt | self.llm | (lambda x: x.content.strip().lower())

    def route(self, state: AgentState) -> AgentState:
        question = state["question"]
        try:
            question_type = self.chain.invoke({"question": question})
            # 容错: 只取第一个词
            question_type = question_type.split()[0] if question_type else "lookup"
            if question_type not in ("diagnosis", "lookup", "operational"):
                question_type = "lookup"
        except Exception:
            question_type = "lookup"

        state["question_type"] = question_type
        state["current_agent"] = "supervisor"
        state["messages"] = [{"agent": "supervisor", "action": "route", "type": question_type}]
        return state
