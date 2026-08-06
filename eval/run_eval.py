"""
评测脚本入口
用法:
  python -m eval.run_eval              # 运行完整评测
  python -m eval.run_eval --retrieval  # 仅评测检索
  python -m eval.run_eval --generation # 仅评测生成
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.embedding_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.rag_chain import RAGChain
from src.evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评测")
    parser.add_argument("--retrieval", action="store_true", help="仅评测检索质量")
    parser.add_argument("--generation", action="store_true", help="仅评测生成质量")
    parser.add_argument("--output", type=str, default=None, help="评测报告输出路径 (JSON)")
    args = parser.parse_args()

    # 如果既没指定 --retrieval 也没 --generation，则全量评测
    run_both = not args.retrieval and not args.generation

    print("正在初始化 RAG 系统...")
    vec_manager = VectorStoreManager()
    vec_manager.load_index()
    hybrid_retriever = HybridRetriever(vec_manager)

    evaluator = Evaluator()
    report = {"timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}

    if run_both or args.retrieval:
        print("\n运行检索评测...")

        def retriever_fn(query):
            return hybrid_retriever.hybrid_search(query, k=Config.TOP_K_RETRIEVAL)

        report["retrieval"] = evaluator.evaluate_retrieval(retriever_fn)

    if run_both or args.generation:
        print("\n运行生成评测（需要 LLM，可能耗时较长）...")
        reranker = Reranker()
        rag = RAGChain(hybrid_retriever, reranker)
        chain = rag.get_chain()

        def rag_fn(question):
            return chain.invoke({"question": question})

        report["generation"] = evaluator.evaluate_generation(rag_fn)

    # 打印报告
    Evaluator.print_report(report)

    # 保存报告
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存至: {args.output}")


if __name__ == "__main__":
    main()
