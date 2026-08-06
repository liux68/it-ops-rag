"""
文档处理器
支持: 标准分块、父子文档分块 (Parent-Child Chunking)
"""

import os
import hashlib
from typing import List, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
from src.utils import get_file_hash


class DocumentProcessor:
    """文档加载与分块处理器"""

    def __init__(self):
        # 子块切分器（用于检索，小块更精准）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )
        # 父块切分器（用于生成上下文，大块更完整）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE * 4,  # 父块 = 4x 子块大小
            chunk_overlap=Config.CHUNK_OVERLAP * 2,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )
        # 兼容旧代码：标准切分器 = 子块切分器
        self.text_splitter = self.child_splitter

    def load_documents(self, directory_path: str) -> List[Document]:
        """加载指定目录下的所有文档"""
        loaders = []
        for ext, loader_cls in [(".pdf", PyPDFLoader), (".txt", TextLoader)]:
            loader = DirectoryLoader(
                directory_path,
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                loader_kwargs={"encoding": "utf-8"} if ext == ".txt" else {},
                show_progress=True,
                use_multithreading=True,
            )
            loaders.append(loader)

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        for doc in docs:
            doc.metadata["source_hash"] = get_file_hash(doc.metadata["source"])
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """标准分块（兼容旧接口）"""
        return self.child_splitter.split_documents(documents)

    # ------------------------------------------------------------------
    # 父子文档分块
    # ------------------------------------------------------------------

    def split_parent_child(
        self,
        documents: List[Document],
    ) -> Tuple[List[Document], List[Document]]:
        """
        父子文档分块策略

        - 父块: 大粒度切分 (chunk_size * 4)，保留完整上下文
        - 子块: 小粒度切分 (chunk_size)，用于精准检索
        - 每个子块通过 parent_id 关联到父块

        Returns:
            (parent_docs, child_docs)
            parent_docs: 父文档列表，metadata 含 parent_id
            child_docs: 子文档列表，metadata 含 parent_id 指向父块
        """
        # 1. 先切父块
        parent_docs = self.parent_splitter.split_documents(documents)

        # 2. 为每个父块生成唯一 ID
        for idx, parent in enumerate(parent_docs):
            parent_id = f"parent_{idx}_{hashlib.md5(parent.page_content[:100].encode()).hexdigest()[:8]}"
            parent.metadata["parent_id"] = parent_id

        # 3. 对每个父块切子块，子块继承 parent_id
        child_docs = []
        for parent in parent_docs:
            children = self.child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent.metadata["parent_id"]
                child.metadata["parent_source"] = parent.metadata.get("source", "未知")
                child_docs.append(child)

        return parent_docs, child_docs


class ParentChildStore:
    """
    父子文档存储
    - 子块存入向量索引用于检索
    - 父块存入内存映射，检索到子块后返回父块
    """

    def __init__(self):
        self.parent_map: dict[str, Document] = {}

    def store_parents(self, parent_docs: List[Document]):
        """存储父文档映射"""
        for doc in parent_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                self.parent_map[parent_id] = doc

    def get_parent(self, child_doc: Document) -> Optional[Document]:
        """根据子块获取对应的父块"""
        parent_id = child_doc.metadata.get("parent_id")
        if parent_id and parent_id in self.parent_map:
            return self.parent_map[parent_id]
        return None

    def expand_to_parents(self, child_docs: List[Document]) -> List[Document]:
        """
        将子块列表扩展为父块列表（去重）
        检索到多个子块属于同一父块时，只返回一次父块
        """
        seen_parent_ids = set()
        parent_docs = []
        for child in child_docs:
            parent = self.get_parent(child)
            if parent:
                pid = parent.metadata["parent_id"]
                if pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    parent_docs.append(parent)
            else:
                # 没有父块映射时，回退到子块本身
                parent_docs.append(child)
        return parent_docs
