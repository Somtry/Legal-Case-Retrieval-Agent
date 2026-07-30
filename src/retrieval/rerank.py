"""重排序模块：BGE-reranker 初排 + 法律要素加权精排。

精排权重（可在 config.yaml 中配置）:
    案由匹配       40
    争议焦点相似度  30
    事实要素重叠度  20
    同地区加成      5
    时间近度        5
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config


class Reranker:
    """重排序器：BGE-reranker + 法律要素加权。"""

    def __init__(self) -> None:
        self._reranker = None
        self._weights = config.rerank.get("weights", {
            "cause_of_action": 40,
            "focus_similarity": 30,
            "fact_overlap": 20,
            "same_region": 5,
            "recency": 5,
        })

    @property
    def reranker(self):
        """惰性加载 BGE-reranker 模型。"""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker

            model_path = config.model.get("reranker_path", "BAAI/bge-reranker-v2-m3")
            self._reranker = FlagReranker(model_path, use_fp16=True)
            logger.info(f"已加载 reranker 模型: {model_path}")
        return self._reranker

    def _reranker_scores(self, query: str, documents: list[str]) -> list[float]:
        """用 BGE-reranker 对 query-documents 对打分。"""
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.compute_score(pairs)
        return scores if isinstance(scores, list) else [scores]

    def _element_weighted_score(
        self,
        query_elements: dict[str, Any],
        case: dict[str, Any],
    ) -> float:
        """法律要素加权精排打分。

        TODO: Phase 2 补充完整的要素匹配逻辑。
        """
        score = 0.0
        w = self._weights

        # 案由匹配
        if query_elements.get("cause_of_action") == case.get("cause_of_action"):
            score += w["cause_of_action"]

        # 其余要素匹配 TODO
        return score

    def rerank(
        self,
        query: str,
        query_elements: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对检索结果重排序。

        Args:
            query: 原始查询文本。
            query_elements: 从案情中抽取的法律要素。
            candidates: 多路检索合并后的候选列表。
            top_k: 最终返回数量。

        Returns:
            排序后的 Top-K 结果。
        """
        top_k = top_k or config.retrieval.get("top_k_final", 5)
        if not candidates:
            return []

        # BGE-reranker 初排
        documents = [c.get("payload", {}).get("facts", "") for c in candidates]
        reranker_scores = self._reranker_scores(query, documents)

        # 法律要素加权精排
        for i, candidate in enumerate(candidates):
            element_score = self._element_weighted_score(
                query_elements,
                candidate.get("payload", {}),
            )
            candidate["rerank_score"] = reranker_scores[i] + element_score

        # 排序并截取
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"重排序完成，返回 Top-{top_k}")
        return candidates[:top_k]