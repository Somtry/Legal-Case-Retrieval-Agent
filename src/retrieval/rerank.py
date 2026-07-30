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
        query_text: str = "",
    ) -> float:
        """法律要素加权精排打分。

        五个维度:
            1. 案由匹配 (×40)      — 案由完全匹配得满分
            2. 争议焦点相似度 (×30) — 争议焦点列表的重叠比例
            3. 事实要素重叠度 (×20) — 关键事实词重叠比例
            4. 同地区加成 (×5)     — 同地区加 5 分
            5. 时间近度 (×5)       — 越近分数越高
        """
        score = 0.0
        w = self._weights

        # 1. 案由匹配
        query_cause = query_elements.get("cause_of_action", "")
        case_cause = case.get("cause_of_action", "")
        if query_cause and case_cause:
            if query_cause == case_cause:
                score += w["cause_of_action"]
            elif query_cause in case_cause or case_cause in query_cause:
                # 部分匹配（如"盗窃罪" vs "盗窃罪、故意伤害罪"）
                score += w["cause_of_action"] * 0.5

        # 2. 争议焦点相似度
        query_focus = query_elements.get("focus_of_dispute", [])
        if isinstance(query_focus, list) and query_focus:
            case_focus = case.get("focus_of_dispute", [])
            if isinstance(case_focus, list) and case_focus:
                # 焦点列表的交集比例
                query_set = set(f.lower() for f in query_focus if f)
                case_set = set(f.lower() for f in case_focus if f)
                if query_set and case_set:
                    overlap = len(query_set & case_set) / len(query_set)
                    score += w["focus_similarity"] * overlap

        # 3. 事实要素重叠度（基于词语重叠）
        query_facts = query_elements.get("key_facts", [])
        if isinstance(query_facts, list) and query_facts:
            # 将查询关键事实拼接，与案例事实做词重叠
            query_fact_text = " ".join(str(f) for f in query_facts)
            case_fact_text = case.get("facts", "")
            if query_fact_text and case_fact_text:
                # 简单的字符级重叠比例
                overlap = self._text_overlap_ratio(query_fact_text, case_fact_text)
                score += w["fact_overlap"] * overlap
        elif query_text and case.get("facts", ""):
            # 如果没有结构化的 key_facts，用原始查询文本做重叠
            overlap = self._text_overlap_ratio(query_text, case["facts"])
            score += w["fact_overlap"] * overlap

        # 4. 同地区加成
        query_region = query_elements.get("region", "")
        case_region = case.get("region", "")
        if query_region and case_region and query_region == case_region:
            score += w["same_region"]

        # 5. 时间近度
        case_date = case.get("date", "")
        if case_date:
            recency_score = self._recency_score(case_date)
            score += w["recency"] * recency_score

        return score

    def _text_overlap_ratio(self, text_a: str, text_b: str) -> float:
        """计算两段文本的词语重叠比例（使用 jieba 分词）。"""
        try:
            import jieba
            words_a = set(jieba.cut(text_a))
            words_b = set(jieba.cut(text_b))
            words_a = {w for w in words_a if len(w.strip()) > 1}  # 过滤单字
            words_b = {w for w in words_b if len(w.strip()) > 1}
            if not words_a:
                return 0.0
            return len(words_a & words_b) / len(words_a)
        except ImportError:
            # jieba 未安装时用简单的字符重叠
            if not text_a:
                return 0.0
            chars_a = set(text_a)
            chars_b = set(text_b)
            return len(chars_a & chars_b) / len(chars_a) if chars_a else 0.0

    def _recency_score(self, date_str: str) -> float:
        """根据日期字符串计算时间近度分数（0~1）。

        越近的日期分数越高。无法解析的返回 0.5。
        """
        import re
        # 尝试提取年份
        year_match = re.search(r"(\d{4})", date_str)
        if year_match:
            year = int(year_match.group(1))
            # 以 2024 年为基准，越近分数越高
            base_year = 2024
            diff = max(0, base_year - year)
            # 10 年内线性衰减，超过 10 年给 0.1
            if diff <= 10:
                return 1.0 - diff / 10.0
            return 0.1
        return 0.5

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
                query_text=query,
            )
            candidate["rerank_score"] = reranker_scores[i] + element_score

        # 排序并截取
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"重排序完成，返回 Top-{top_k}")
        return candidates[:top_k]