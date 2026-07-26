import os
from src.config import Config
from src.document_processor import DocumentProcessor
from src.embedding_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.rag_chain import RAGChain


def build_knowledge_base(data_dir: str = "./data/source"):
    """构建知识库索引"""
    print("🚀 开始构建知识库索引...")

    # 1. 加载并处理文档
    processor = DocumentProcessor()
    docs = processor.load_documents(data_dir)
    print(f"✅ 加载了 {len(docs)} 个文档")

    chunks = processor.split_documents(docs)
    print(f"✅ 文档切分为 {len(chunks)} 个块")

    # 2. 构建向量索引
    vec_manager = VectorStoreManager()
    vec_manager.build_index(chunks)
    print(f"✅ 向量索引已保存至 {Config.FAISS_INDEX_PATH}")

    # 3. 为混合检索构建BM25索引 (需要原始文档列表)
    hybrid_retriever = HybridRetriever(vec_manager)
    hybrid_retriever.build_bm25_index(chunks)
    print("✅ BM25索引构建完成")

    print("🎉 知识库构建完成！")
    return vec_manager, hybrid_retriever


def chat():
    """启动交互式问答"""
    print("🔍 正在加载知识库...")

    # 初始化各个组件
    vec_manager = VectorStoreManager()
    vec_manager.load_index()

    hybrid_retriever = HybridRetriever(vec_manager)
    # 注意: 在生产环境中，BM25索引应从持久化存储加载，这里为演示简化
    # hybrid_retriever.build_bm25_index(...)

    reranker = Reranker()
    rag = RAGChain(hybrid_retriever, reranker)
    chain = rag.get_chain()

    print("💬 IT运维知识库助手已就绪！输入 'exit' 退出。")
    while True:
        question = input("\n👤 你: ")
        if question.lower() == 'exit':
            break
        try:
            response = chain.invoke({"question": question})
            print(f"🤖 助手: {response}")
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_knowledge_base()
    else:
        chat()