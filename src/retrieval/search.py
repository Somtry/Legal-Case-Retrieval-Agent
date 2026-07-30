"""多路检索模块：向量检索 + 关键词检索（BM25）+ 结构化筛选。

三路并行检索后合并去重，交给 rerank 模块重排序。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config
from src.data_processing.vector_store import VectorStore
from src.retrieval.embed import Embedder


class MultiRouteSearcher:
    """多路检索器：向量检索 + BM25 关键词检索。"""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self._bm25 = None  # BM25 索引，Phase 2 构建

    def vector_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """向量检索路径：query → BGE-M3 embedding → Qdrant Top-K。"""
        top_k = top_k or config.retrieval.get("top_k_vector", 50)
        query_vector = self.embedder.embed_single(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        logger.info(f"向量检索返回 {len(results)} 条结果")
        return results

    def keyword_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """关键词检索路径：jieba 分词 → BM25 Top-K。

        TODO: Phase 2 实现 BM25 索引构建和检索。
        """
        top_k = top_k or config.retrieval.get("top_k_keyword", 50)
        logger.warning("BM25 关键词检索尚未实现，返回空结果")
        return []

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