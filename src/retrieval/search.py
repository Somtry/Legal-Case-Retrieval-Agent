"""多路检索模块：向量检索 + 关键词检索（BM25）+ 结构化筛选。

三路并行检索后合并去重，交给 rerank 模块重排序。

BM25 索引基于 jieba 中文分词 + rank_bm25 库实现。
索引数据从 data/processed/ 目录的 JSONL 文件加载。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config
from src.data_processing.vector_store import VectorStore
from src.retrieval.embed import Embedder


class BM25Index:
    """BM25 关键词索引：基于 jieba 分词 + rank_bm25。"""

    def __init__(self) -> None:
        self._bm25 = None
        self._cases: list[dict[str, Any]] = []  # 原始案例数据，与索引一一对应
        self._tokenized_docs: list[list[str]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, cases: list[dict[str, Any]]) -> None:
        """构建 BM25 索引。

        Args:
            cases: 案例数据列表，每条包含 facts 等字段。
        """
        import jieba
        from rank_bm25 import BM25Okapi

        self._cases = cases
        self._tokenized_docs = []

        for case in cases:
            # 将案由 + 事实拼接为索引文本
            text = case.get("cause_of_action", "") + " " + case.get("facts", "")
            tokens = list(jieba.cut(text))
            self._tokenized_docs.append(tokens)

        self._bm25 = BM25Okapi(self._tokenized_docs)
        logger.info(f"BM25 索引构建完成: {len(cases)} 篇文档")

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """BM25 关键词检索。

        Args:
            query: 查询文本。
            top_k: 返回前 K 条结果。

        Returns:
            检索结果列表，格式与向量检索一致: [{"score": float, "payload": dict}]
        """
        if not self.is_built:
            logger.warning("BM25 索引未构建，请先调用 build()")
            return []

        import jieba

        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)

        # 按分数排序，取 Top-K
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0:  # 过滤掉完全不匹配的结果
                continue
            results.append({
                "score": float(score),
                "payload": self._cases[idx],
            })

        logger.info(f"BM25 检索返回 {len(results)} 条结果")
        return results


class MultiRouteSearcher:
    """多路检索器：向量检索 + BM25 关键词检索。"""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()

    def build_bm25_index(self, cases: list[dict[str, Any]] | None = None) -> None:
        """构建 BM25 索引。

        Args:
            cases: 案例数据列表。如果不提供，则从 data/processed/ 加载。
        """
        if cases is None:
            cases = self._load_cases_from_processed()
        self.bm25.build(cases)

    def _load_cases_from_processed(self) -> list[dict[str, Any]]:
        """从 data/processed/ 目录加载所有处理后的案例数据。"""
        processed_dir = config.data_dir / "processed"
        if not processed_dir.exists():
            logger.warning(f"处理后的数据目录不存在: {processed_dir}")
            return []

        cases: list[dict[str, Any]] = []
        for jsonl_file in sorted(processed_dir.glob("*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cases.append(json.loads(line))
        logger.info(f"从 {processed_dir} 加载了 {len(cases)} 条案例")
        return cases

    def vector_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """向量检索路径：query → BGE-M3 embedding → Qdrant Top-K。"""
        top_k = top_k or config.retrieval.get("top_k_vector", 50)
        query_vector = self.embedder.embed_single(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        logger.info(f"向量检索返回 {len(results)} 条结果")
        return results

    def keyword_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """关键词检索路径：jieba 分词 → BM25 Top-K。"""
        top_k = top_k or config.retrieval.get("top_k_keyword", 50)
        if not self.bm25.is_built:
            logger.info("BM25 索引未构建，自动从 processed 数据加载...")
            self.build_bm25_index()
        return self.bm25.search(query, top_k=top_k)

    def search(self, query: str) -> list[dict[str, Any]]:
        """多路检索合并：向量 + 关键词，去重后返回。"""
        vec_results = self.vector_search(query)
        kw_results = self.keyword_search(query)

        # 合并去重（按 case_id）
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for result in vec_results + kw_results:
            case_id = result.get("payload", {}).get("case_id", "")
            if case_id and case_id not in seen:
                seen.add(case_id)
                merged.append(result)

        logger.info(f"多路检索合并去重后 {len(merged)} 条")
        return merged