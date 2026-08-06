"""
RAG 评测框架
支持指标: Recall@K, MRR, 关键词覆盖率, 平均响应时间
"""

import json
import time
import re
from typing import List, Dict, Any
from pathlib import Path


class Evaluator:
    """RAG 系统评测器"""

    def __init__(self, dataset_path: str = "eval/dataset.json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate_retrieval(
        self,
        retriever_fn,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        评测检索质量
        retriever_fn: 接受 query 字符串，返回 List[Document]
        """
        results = {f"recall@{k}": [] for k in k_values}
        results["mrr"] = []

        for item in self.dataset:
            query = item["question"]
            relevant_keywords = item["relevant_doc_keywords"]

            try:
                docs = retriever_fn(query)
            except Exception:
                docs = []

            # 计算每个文档的相关性：是否包含期望关键词
            doc_relevance = []
            for idx, doc in enumerate(docs):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                # 文档包含的关键词数量作为相关性分数
                match_count = sum(1 for kw in relevant_keywords if kw in content)
                doc_relevance.append((idx, match_count))

            # Recall@K: top-K 中包含至少一个相关文档的比例
            for k in k_values:
                top_k_relevant = sum(rel for _, rel in doc_relevance[:k])
                results[f"recall@{k}"].append(1.0 if top_k_relevant > 0 else 0.0)

            # MRR: 第一个相关文档的倒数排名
            first_relevant_rank = None
            for idx, rel in doc_relevance:
                if rel > 0:
                    first_relevant_rank = idx + 1
                    break
            results["mrr"].append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        # 汇总
        summary = {}
        for key, values in results.items():
            summary[key] = round(sum(values) / len(values), 4) if values else 0.0

        return {
            "metrics": summary,
            "total_questions": len(self.dataset),
        }

    def evaluate_generation(
        self,
        rag_fn,
        llm_judge_fn=None,
    ) -> Dict[str, Any]:
        """
        评测生成质量
        rag_fn: 接受 question 字符串，返回 answer 字符串
        llm_judge_fn: 可选的 LLM 评判函数，接受 (question, answer, expected_keywords)，返回 0-1 分数
        """
        keyword_coverage_scores = []
        llm_judge_scores = []
        latencies = []
        answers = []

        for item in self.dataset:
            query = item["question"]
            expected_keywords = item["expected_answer_keywords"]

            start = time.time()
            try:
                answer = rag_fn(query)
            except Exception as e:
                answer = f"[ERROR] {e}"
            latency = time.time() - start

            latencies.append(latency)
            answers.append({"id": item["id"], "question": query, "answer": answer})

            # 关键词覆盖率
            matched = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
            coverage = matched / len(expected_keywords) if expected_keywords else 0
            keyword_coverage_scores.append(coverage)

            # LLM 评判（如果提供）
            if llm_judge_fn:
                try:
                    score = llm_judge_fn(query, answer, expected_keywords)
                    llm_judge_scores.append(score)
                except Exception:
                    pass

        summary = {
            "keyword_coverage": round(sum(keyword_coverage_scores) / len(keyword_coverage_scores), 4),
            "avg_latency_s": round(sum(latencies) / len(latencies), 4),
            "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 4),
        }

        if llm_judge_scores:
            summary["llm_judge_score"] = round(sum(llm_judge_scores) / len(llm_judge_scores), 4)

        return {
            "metrics": summary,
            "details": answers,
            "total_questions": len(self.dataset),
        }

    def run_full_evaluation(
        self,
        retriever_fn=None,
        rag_fn=None,
        llm_judge_fn=None,
    ) -> Dict[str, Any]:
        """运行完整评测"""
        report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        if retriever_fn:
            report["retrieval"] = self.evaluate_retrieval(retriever_fn)

        if rag_fn:
            report["generation"] = self.evaluate_generation(rag_fn, llm_judge_fn)

        return report

    @staticmethod
    def print_report(report: Dict[str, Any]):
        """格式化打印评测报告"""
        print("=" * 60)
        print(f"  RAG 评测报告  |  {report.get('timestamp', 'N/A')}")
        print("=" * 60)

        if "retrieval" in report:
            ret = report["retrieval"]
            print(f"\n📊 检索质量 (共 {ret['total_questions']} 题)")
            print("-" * 40)
            for key, val in ret["metrics"].items():
                bar = "█" * int(val * 20)
                print(f"  {key:15s} {val:.4f}  {bar}")

        if "generation" in report:
            gen = report["generation"]
            print(f"\n📝 生成质量 (共 {gen['total_questions']} 题)")
            print("-" * 40)
            for key, val in gen["metrics"].items():
                if isinstance(val, float) and val <= 1.0:
                    bar = "█" * int(val * 20)
                    print(f"  {key:15s} {val:.4f}  {bar}")
                else:
                    print(f"  {key:15s} {val}")

        print("\n" + "=" * 60)
