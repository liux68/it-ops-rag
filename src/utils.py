import hashlib
import os
from datetime import datetime
from typing import Optional


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    计算文件的哈希值，用于检测文件是否变更。

    Args:
        file_path: 文件路径
        algorithm: 哈希算法（默认 md5）

    Returns:
        文件的哈希字符串，如果文件不存在则返回空字符串
    """
    if not os.path.exists(file_path):
        return ""

    hash_func = hashlib.new(algorithm)
    try:
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件内存溢出
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        print(f"计算文件哈希失败: {e}")
        return ""


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    格式化时间戳，用于日志记录。

    Args:
        dt: datetime对象，默认为当前时间

    Returns:
        格式化的时间字符串，如 "2026-07-23 17:30:45"
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断过长的文本，用于日志输出。

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get_metadata(doc, key: str, default: str = "未知") -> str:
    """
    安全地从文档元数据中获取值，避免KeyError。

    Args:
        doc: Document对象
        key: 元数据键名
        default: 默认值

    Returns:
        元数据值或默认值
    """
    try:
        return doc.metadata.get(key, default)
    except AttributeError:
        return default