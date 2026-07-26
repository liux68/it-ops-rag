import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
from src.utils import get_file_hash


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

    def load_documents(self, directory_path: str) -> List[Document]:
        """加载指定目录下的所有文档"""
        loaders = []
        # 支持多种格式
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

        # 为每个文档添加哈希值，用于后续增量更新
        for doc in docs:
            doc.metadata["source_hash"] = get_file_hash(doc.metadata["source"])
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档切分成块"""
        return self.text_splitter.split_documents(documents)