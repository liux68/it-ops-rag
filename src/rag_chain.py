"""
RAG 链
支持: 多查询重写 → 混合检索 (RRF融合) → 重排序 → LLM 生成
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import Config
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.query_rewriter import QueryRewriter


class RAGChain:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        use_multi_query: bool = True,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.use_multi_query = use_multi_query
        self.query_rewriter = QueryRewriter(num_queries=3) if use_multi_query else None

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.1,
        )
        self.prompt = ChatPromptTemplate.from_template("""
你是一位经验丰富的IT运维专家。请仅根据以下提供的"参考资料"来回答用户的问题。
如果参考资料不包含相关信息，请直接告知用户你不知道，不要编造答案。

<参考资料>
{context}
</参考资料>

用户问题：{question}
你的回答：""")

    def _format_docs(self, docs):
        if not docs:
            return "无相关参考资料。"
        return "\n\n".join([
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        ])

    def _retrieve_and_rerank(self, inputs):
        question = inputs["question"]

        # 1. 多查询重写 + 检索
        if self.use_multi_query and self.query_rewriter:
            queries = self.query_rewriter.rewrite(question)
            retrieved_docs = self.retriever.multi_query_search(
                queries, k=Config.TOP_K_RETRIEVAL
            )
        else:
            retrieved_docs = self.retriever.hybrid_search(
                question, k=Config.TOP_K_RETRIEVAL
            )

        # 2. 重排序
        reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=Config.TOP_K_RERANK
        )
        return self._format_docs(reranked_docs)

    def get_chain(self):
        rag_chain = (
            RunnablePassthrough.assign(context=self._retrieve_and_rerank)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def retrieve(self, question: str, top_k: int = None):
        """直接暴露检索接口（用于评测和调试）"""
        k = top_k or Config.TOP_K_RETRIEVAL
        if self.use_multi_query and self.query_rewriter:
            queries = self.query_rewriter.rewrite(question)
            return self.retriever.multi_query_search(queries, k=k)
        return self.retriever.hybrid_search(question, k=k)
