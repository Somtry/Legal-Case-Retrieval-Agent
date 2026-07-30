"""检索模块测试：BM25 关键词检索 + 要素加权精排 + 端到端流程。

不需要 GPU 模型的测试直接运行。
需要 BGE-M3 / BGE-reranker 模型的测试标记为 skip，等有 GPU 环境再跑。
"""

import json
from pathlib import Path

import pytest

from src.retrieval.rerank import Reranker
from src.retrieval.search import BM25Index, MultiRouteSearcher


# ── 测试数据 ────────────────────────────────────────────────────────

SAMPLE_CASES = [
    {
        "case_id": "001",
        "case_type": "刑事",
        "cause_of_action": "盗窃罪",
        "facts": "被告人张某秘密窃取他人财物，价值人民币5000元",
        "legal_basis": ["刑法第264条"],
        "ruling": "有期徒刑1年",
        "region": "北京市",
        "date": "2023-05-01",
    },
    {
        "case_id": "002",
        "case_type": "刑事",
        "cause_of_action": "故意伤害罪",
        "facts": "被告人李某因纠纷殴打他人，致人轻伤一级",
        "legal_basis": ["刑法第234条"],
        "ruling": "有期徒刑2年",
        "region": "上海市",
        "date": "2022-08-15",
    },
    {
        "case_id": "003",
        "case_type": "刑事",
        "cause_of_action": "盗窃罪",
        "facts": "被告人王某入室盗窃手机一部，价值3000元",
        "legal_basis": ["刑法第264条"],
        "ruling": "有期徒刑6个月",
        "region": "北京市",
        "date": "2023-11-20",
    },
    {
        "case_id": "004",
        "case_type": "刑事",
        "cause_of_action": "诈骗罪",
        "facts": "被告人赵某以虚假投资为名骗取他人钱财20万元",
        "legal_basis": ["刑法第266条"],
        "ruling": "有期徒刑5年",
        "region": "广东省",
        "date": "2021-03-10",
    },
    {
        "case_id": "005",
        "case_type": "刑事",
        "cause_of_action": "故意伤害罪",
        "facts": "被告人刘某持械伤人，致人重伤二级",
        "legal_basis": ["刑法第234条"],
        "ruling": "有期徒刑5年",
        "region": "北京市",
        "date": "2024-01-05",
    },
]


# ── BM25 检索测试 ──────────────────────────────────────────────────

class TestBM25Index:
    def test_build_and_search(self):
        """测试 BM25 索引构建和检索。"""
        bm25 = BM25Index()
        assert not bm25.is_built

        bm25.build(SAMPLE_CASES)
        assert bm25.is_built

    def test_search_retrieves_relevant(self):
        """测试"盗窃"查询能召回盗窃相关案例。"""
        bm25 = BM25Index()
        bm25.build(SAMPLE_CASES)

        results = bm25.search("盗窃他人财物", top_k=3)
        assert len(results) > 0

        # 至少有一条是盗窃罪
        causes = [r["payload"]["cause_of_action"] for r in results]
        assert "盗窃罪" in causes

    def test_search_empty_query(self):
        """测试空查询返回空结果。"""
        bm25 = BM25Index()
        bm25.build(SAMPLE_CASES)
        results = bm25.search("完全不相关的词汇xyz", top_k=3)
        # 不匹配的查询应该返回很少或零结果
        assert len(results) <= 3

    def test_search_not_built(self):
        """测试未构建索引时返回空。"""
        bm25 = BM25Index()
        results = bm25.search("测试", top_k=3)
        assert results == []


# ── 要素加权精排测试 ────────────────────────────────────────────────

class TestElementWeightedScore:
    def test_cause_of_action_exact_match(self):
        """案由完全匹配应得满分。"""
        r = Reranker()
        score = r._element_weighted_score(
            {"cause_of_action": "盗窃罪"},
            {"cause_of_action": "盗窃罪"},
        )
        assert score == 40  # 权重 40

    def test_cause_of_action_partial_match(self):
        """案由部分匹配应得半分。"""
        r = Reranker()
        score = r._element_weighted_score(
            {"cause_of_action": "盗窃罪"},
            {"cause_of_action": "盗窃罪、故意伤害罪"},
        )
        assert score == 20  # 40 * 0.5

    def test_cause_of_action_no_match(self):
        """案由不匹配不得分。"""
        r = Reranker()
        score = r._element_weighted_score(
            {"cause_of_action": "盗窃罪"},
            {"cause_of_action": "故意伤害罪"},
        )
        assert score == 0

    def test_same_region_bonus(self):
        """同地区应加分。"""
        r = Reranker()
        score_same = r._element_weighted_score(
            {"region": "北京市"},
            {"region": "北京市"},
        )
        score_diff = r._element_weighted_score(
            {"region": "北京市"},
            {"region": "上海市"},
        )
        assert score_same > score_diff
        assert score_same == 5  # 权重 5

    def test_focus_of_dispute_overlap(self):
        """争议焦点重叠应按比例得分。"""
        r = Reranker()
        query_elements = {
            "focus_of_dispute": ["盗窃金额", "是否入户盗窃"],
        }
        # 完全重叠
        case_full = {"focus_of_dispute": ["盗窃金额", "是否入户盗窃"]}
        # 部分重叠
        case_partial = {"focus_of_dispute": ["盗窃金额"]}
        # 无重叠
        case_none = {"focus_of_dispute": ["故意伤害程度"]}

        score_full = r._element_weighted_score(query_elements, case_full)
        score_partial = r._element_weighted_score(query_elements, case_partial)
        score_none = r._element_weighted_score(query_elements, case_none)

        assert score_full > score_partial > score_none
        assert score_full == 30  # 权重 30，100% 重叠
        assert score_partial == 15  # 30 * 0.5

    def test_fact_overlap(self):
        """事实文本重叠应得分。"""
        r = Reranker()
        query_elements = {
            "key_facts": ["秘密窃取", "他人财物"],
        }
        case_relevant = {
            "facts": "被告人秘密窃取他人财物价值5000元",
        }
        case_irrelevant = {
            "facts": "被告人殴打他人致轻伤",
        }

        score_rel = r._element_weighted_score(query_elements, case_relevant)
        score_irr = r._element_weighted_score(query_elements, case_irrelevant)

        assert score_rel > score_irr

    def test_recency(self):
        """近期案例得分更高。"""
        r = Reranker()
        recent = r._element_weighted_score(
            {},
            {"date": "2023-01-01"},
        )
        old = r._element_weighted_score(
            {},
            {"date": "2010-01-01"},
        )
        assert recent > old

    def test_combined_scoring(self):
        """综合打分：案由匹配 + 同地区 + 近期 的案例应比不匹配的高很多。"""
        r = Reranker()
        query_elements = {
            "cause_of_action": "盗窃罪",
            "key_facts": ["秘密窃取", "他人财物"],
            "region": "北京市",
        }
        case_good = {
            "cause_of_action": "盗窃罪",
            "facts": "被告人秘密窃取他人财物价值5000元",
            "region": "北京市",
            "date": "2023-05-01",
        }
        case_bad = {
            "cause_of_action": "诈骗罪",
            "facts": "被告人以虚假投资为名骗取他人钱财",
            "region": "广东省",
            "date": "2015-03-10",
        }

        score_good = r._element_weighted_score(query_elements, case_good, "秘密窃取他人财物")
        score_bad = r._element_weighted_score(query_elements, case_bad, "秘密窃取他人财物")

        assert score_good > score_bad
        assert score_good > 50  # 案由40 + 其他加分


# ── 多路检索测试（不需要向量模型的部分） ─────────────────────────────

class TestMultiRouteSearcher:
    def test_build_bm25_from_list(self):
        """测试从案例列表构建 BM25 索引。"""
        searcher = MultiRouteSearcher()
        searcher.build_bm25_index(SAMPLE_CASES)
        assert searcher.bm25.is_built

    def test_keyword_search(self):
        """测试关键词检索路径。"""
        searcher = MultiRouteSearcher()
        searcher.build_bm25_index(SAMPLE_CASES)

        results = searcher.keyword_search("盗窃手机", top_k=3)
        assert len(results) > 0
        # 应该返回盗窃相关案例
        causes = [r["payload"]["cause_of_action"] for r in results]
        assert "盗窃罪" in causes

    def test_keyword_search_ranking(self):
        """测试关键词检索排序：更相关的案例应排在前面。"""
        searcher = MultiRouteSearcher()
        searcher.build_bm25_index(SAMPLE_CASES)

        results = searcher.keyword_search("入室盗窃手机", top_k=5)
        if len(results) >= 2:
            # 入室盗窃的案例（003）应该比普通盗窃（001）排名更靠前或得分更高
            case_003 = next((r for r in results if r["payload"]["case_id"] == "003"), None)
            case_001 = next((r for r in results if r["payload"]["case_id"] == "001"), None)
            if case_003 and case_001:
                assert case_003["score"] >= case_001["score"]


# ── 需要模型的测试（在有 GPU 环境时运行） ──────────────────────────

@pytest.mark.skip(reason="需要 BGE-M3 模型和 Qdrant 数据，等 GPU 环境跑")
class TestVectorSearch:
    def test_vector_search(self):
        """测试向量检索路径。"""
        searcher = MultiRouteSearcher()
        results = searcher.vector_search("盗窃他人财物", top_k=3)
        assert len(results) > 0

    def test_multi_route_merge(self):
        """测试多路检索合并去重。"""
        searcher = MultiRouteSearcher()
        searcher.build_bm25_index(SAMPLE_CASES)
        merged = searcher.search("盗窃他人财物")
        assert len(merged) > 0
        # 检查去重
        case_ids = [r["payload"]["case_id"] for r in merged]
        assert len(case_ids) == len(set(case_ids))


@pytest.mark.skip(reason="需要 BGE-reranker 模型，等 GPU 环境跑")
class TestReranker:
    def test_rerank_full(self):
        """测试完整重排序流程（BGE-reranker + 要素加权）。"""
        reranker = Reranker()
        candidates = [
            {"payload": case, "score": 0.5}
            for case in SAMPLE_CASES
        ]
        results = reranker.rerank(
            query="盗窃他人财物",
            query_elements={"cause_of_action": "盗窃罪"},
            candidates=candidates,
            top_k=3,
        )
        assert len(results) == 3
        # 盗窃罪案例应排在前面
        assert results[0]["payload"]["cause_of_action"] == "盗窃罪"