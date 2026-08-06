import os
import sys
from src.config import Config
from src.document_processor import DocumentProcessor, ParentChildStore
from src.embedding_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.rag_chain import RAGChain


# 全局父子文档存储（供检索时使用）
parent_child_store = ParentChildStore()


def build_knowledge_base(data_dir: str = "./data/source", use_parent_child: bool = True):
    """构建知识库索引"""
    print("🚀 开始构建知识库索引...")

    processor = DocumentProcessor()
    docs = processor.load_documents(data_dir)
    print(f"✅ 加载了 {len(docs)} 个文档")

    vec_manager = VectorStoreManager()

    if use_parent_child and Config.USE_PARENT_CHILD:
        # 父子文档分块模式
        parent_docs, child_docs = processor.split_parent_child(docs)
        print(f"✅ 父子分块: {len(parent_docs)} 个父块, {len(child_docs)} 个子块")

        # 子块存入向量索引
        vec_manager.build_index(child_docs)
        print(f"✅ 向量索引已保存至 {Config.FAISS_INDEX_PATH} (基于子块)")

        # 父块存入内存映射
        parent_child_store.store_parents(parent_docs)
        _save_parent_store(parent_docs)

        # BM25 也基于子块构建
        hybrid_retriever = HybridRetriever(vec_manager)
        hybrid_retriever.build_bm25_index(child_docs)
        print("✅ BM25索引构建完成 (基于子块)")
    else:
        # 标准分块模式
        chunks = processor.split_documents(docs)
        print(f"✅ 文档切分为 {len(chunks)} 个块")

        vec_manager.build_index(chunks)
        print(f"✅ 向量索引已保存至 {Config.FAISS_INDEX_PATH}")

        hybrid_retriever = HybridRetriever(vec_manager)
        hybrid_retriever.build_bm25_index(chunks)
        print("✅ BM25索引构建完成")

    print("🎉 知识库构建完成！")
    return vec_manager, hybrid_retriever


def _save_parent_store(parent_docs, path: str = "./faiss_index/parent_docs.json"):
    """持久化父文档映射到 JSON"""
    import json
    data = []
    for doc in parent_docs:
        data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 父文档映射已保存至 {path}")


def load_parent_store(path: str = "./faiss_index/parent_docs.json") -> ParentChildStore:
    """从 JSON 加载父文档映射"""
    import json
    from langchain_core.documents import Document

    store = ParentChildStore()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parent_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        store.store_parents(parent_docs)
        print(f"✅ 父文档映射已加载 ({len(parent_docs)} 个父块)")
    return store


def chat():
    """启动交互式问答"""
    print("🔍 正在加载知识库...")

    vec_manager = VectorStoreManager()
    vec_manager.load_index()

    hybrid_retriever = HybridRetriever(vec_manager)

    # 加载父文档映射
    global parent_child_store
    parent_child_store = load_parent_store()

    # 构建 BM25 索引
    processor = DocumentProcessor()
    try:
        docs = processor.load_documents("./data/source")
        if Config.USE_PARENT_CHILD:
            _, child_docs = processor.split_parent_child(docs)
            hybrid_retriever.build_bm25_index(child_docs)
        else:
            chunks = processor.split_documents(docs)
            hybrid_retriever.build_bm25_index(chunks)
    except Exception as e:
        print(f"⚠️ BM25 索引构建失败: {e}")

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
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_knowledge_base()
    else:
        chat()
