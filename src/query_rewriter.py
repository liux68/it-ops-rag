"""
多查询重写模块
使用 LLM 将原始问题改写为多个语义变体，提升召回率。
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import Config


QUERY_REWRITE_PROMPT = """你是一个搜索查询优化专家。请将用户的原始问题改写为 {num_queries} 个语义不同的搜索查询变体。

要求:
1. 每个变体换一个角度表达相同的搜索意图
2. 包含不同的关键词组合，覆盖不同的检索维度
3. 每行一个变体，不要编号，不要解释

原始问题: {question}

请输出 {num_queries} 个查询变体（每行一个）:"""


class QueryRewriter:
    """多查询重写器"""

    def __init__(self, num_queries: int = 3):
        self.num_queries = num_queries
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.3,  # 稍高温度增加多样性
        )
        self.prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> list[str]:
        """将原始问题重写为多个变体"""
        try:
            result = self.chain.invoke({
                "question": question,
                "num_queries": self.num_queries,
            })
            # 按行分割，过滤空行
            queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
            # 去掉可能的编号前缀
            cleaned = []
            for q in queries:
                # 去掉 "1. " "2. " 等编号
                import re
                q = re.sub(r"^\d+[\.\)]\s*", "", q)
                if q:
                    cleaned.append(q)
            # 确保原始问题始终在列表中
            if question not in cleaned:
                cleaned.insert(0, question)
            return cleaned[: self.num_queries + 1]
        except Exception as e:
            # 重写失败时降级为原始问题
            print(f"[QueryRewriter] 重写失败，使用原始问题: {e}")
            return [question]
