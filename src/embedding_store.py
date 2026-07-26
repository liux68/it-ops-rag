import os
import pickle
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # 新增导入
from langchain_core.documents import Document
from src.config import Config

class VectorStoreManager:
    def __init__(self):
        # 替换为本地嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"   # 轻量、高效，适合本地运行
        )
        self.index_path = Config.FAISS_INDEX_PATH
        self.vector_store: Optional[FAISS] = None

    def build_index(self, documents: List[Document]):
        """从文档列表构建新的索引"""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_index()
        return self.vector_store

    def add_documents(self, documents: List[Document]):
        """向现有索引增量添加文档"""
        if self.vector_store is None:
            self.load_index()
        if self.vector_store:
            self.vector_store.add_documents(documents)
            self.save_index()

    def save_index(self):
        """保存索引到本地"""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def load_index(self):
        """从本地加载索引"""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return self.vector_store

    def get_retriever(self, search_kwargs: dict = None):
        """获取向量检索器"""
        if self.vector_store is None:
            self.load_index()
        if self.vector_store:
            kwargs = search_kwargs or {"k": Config.TOP_K_RETRIEVAL}
            return self.vector_store.as_retriever(search_kwargs=kwargs)
        return None