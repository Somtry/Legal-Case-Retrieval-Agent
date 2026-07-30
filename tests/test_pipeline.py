"""Pipeline 端到端单元测试（mock LLM，不需要真实模型）。

通过 mock LLMInference 的 generate 方法，测试 pipeline 的完整流程:
    要素抽取(JSON解析) → 多路检索 → 重排序 → 分析生成

真实模型测试需要 GPU 环境，等 AutoDL 上跑。
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import LegalCasePipeline


# ── Mock LLM 响应数据 ──────────────────────────────────────────────

MOCK_ELEMENTS_RESPONSE = """```json
{
    "case_type": "刑事",
    "cause_of_action": "盗窃罪",
    "focus_of_dispute": ["盗窃金额", "是否入户盗窃"],
    "key_facts": ["秘密窃取", "他人财物"],
    "applicable_laws": ["刑法第264条"]
}
```"""

MOCK_ANALYSIS_RESPONSE = """### 1. 相似性分析
检索到的案例与待分析案情在案由、事实方面高度相似。

### 2. 裁判观点对比
法院对盗窃罪的量刑主要考虑盗窃金额和是否有入户情节。

### 3. 法律建议
建议准备被盗财物价值鉴定报告作为关键证据。

### 免责声明
本分析由 AI 生成，仅供参考，不构成专业法律意见。"""


# ── Mock 检索结果 ──────────────────────────────────────────────────

MOCK_SEARCH_RESULTS = [
    {
        "score": 0.95,
        "payload": {
            "case_id": "cail_001",
            "case_type": "刑事",
            "cause_of_action": "盗窃罪",
            "facts": "被告人秘密窃取他人财物价值5000元",
            "legal_basis": ["刑法第264条"],
            "ruling": "有期徒刑1年",
            "region": "",
            "date": "",
        },
    },
    {
        "score": 0.87,
        "payload": {
            "case_id": "cail_002",
            "case_type": "刑事",
            "cause_of_action": "盗窃罪",
            "facts": "被告人入室盗窃手机一部",
            "legal_basis": ["刑法第264条"],
            "ruling": "有期徒刑6个月",
            "region": "",
            "date": "",
        },
    },
]


# ── Pipeline 测试 ──────────────────────────────────────────────────

class TestPipelineElementExtraction:
    def test_extract_elements_json_parsing(self):
        """测试要素抽取的 JSON 解析逻辑（mock LLM 返回 JSON）。"""
        pipeline = LegalCasePipeline.__new__(LegalCasePipeline)
        pipeline.llm = MagicMock()
        pipeline.llm.generate.return_value = MOCK_ELEMENTS_RESPONSE

        elements = pipeline.extract_elements("被告人盗窃他人财物")

        assert elements["case_type"] == "刑事"
        assert elements["cause_of_action"] == "盗窃罪"
        assert "盗窃金额" in elements["focus_of_dispute"]
        assert "刑法第264条" in elements["applicable_laws"]

    def test_extract_elements_json_without_codeblock(self):
        """测试没有 markdown 代码块的 JSON 解析。"""
        pipeline = LegalCasePipeline.__new__(LegalCasePipeline)
        pipeline.llm = MagicMock()
        pipeline.llm.generate.return_value = '{"case_type": "民事", "cause_of_action": "合同纠纷"}'

        elements = pipeline.extract_elements("原告与被告合同纠纷")

        assert elements["case_type"] == "民事"
        assert elements["cause_of_action"] == "合同纠纷"

    def test_extract_elements_fallback_on_parse_error(self):
        """测试 JSON 解析失败时的降级处理。"""
        pipeline = LegalCasePipeline.__new__(LegalCasePipeline)
        pipeline.llm = MagicMock()
        pipeline.llm.generate.return_value = "这不是一个有效的JSON回复"

        elements = pipeline.extract_elements("测试案情")

        assert "raw_response" in elements
        assert elements["raw_response"] == "这不是一个有效的JSON回复"


class TestPipelineAnalysis:
    def test_generate_analysis(self):
        """测试分析报告生成（mock LLM）。"""
        pipeline = LegalCasePipeline.__new__(LegalCasePipeline)
        pipeline.llm = MagicMock()
        pipeline.llm.generate.return_value = MOCK_ANALYSIS_RESPONSE

        analysis = pipeline.generate_analysis(
            case_description="被告人盗窃他人财物",
            elements={"case_type": "刑事", "cause_of_action": "盗窃罪"},
            retrieved_cases=MOCK_SEARCH_RESULTS,
        )

        assert "相似性分析" in analysis
        assert "裁判观点对比" in analysis
        assert "法律建议" in analysis
        assert "免责声明" in analysis

    def test_generate_analysis_passes_correct_prompt(self):
        """测试分析生成时传给 LLM 的 prompt 包含必要信息。"""
        pipeline = LegalCasePipeline.__new__(LegalCasePipeline)
        pipeline.llm = MagicMock()
        pipeline.llm.generate.return_value = "分析报告内容"

        pipeline.generate_analysis(
            case_description="被告人张某故意伤害他人",
            elements={"case_type": "刑事"},
            retrieved_cases=MOCK_SEARCH_RESULTS,
        )

        # 检查 LLM 被调用时传入了 prompt
        call_args = pipeline.llm.generate.call_args
        prompt = call_args[0][0]  # 第一个位置参数
        assert "被告人张某故意伤害他人" in prompt
        assert "刑事" in prompt
        assert "cail_001" in prompt  # 检索案例的内容应该在 prompt 里


class TestPipelineFullRun:
    """完整流程测试：mock 所有外部依赖。"""

    @patch("src.pipeline.LLMInference")
    @patch("src.pipeline.MultiRouteSearcher")
    @patch("src.pipeline.Reranker")
    def test_full_pipeline_run(self, mock_reranker_cls, mock_searcher_cls, mock_llm_cls):
        """测试 pipeline.run() 完整流程能跑通。"""
        # 配置 mock
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [MOCK_ELEMENTS_RESPONSE, MOCK_ANALYSIS_RESPONSE]
        mock_llm_cls.return_value = mock_llm

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = MOCK_SEARCH_RESULTS
        mock_searcher_cls.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = MOCK_SEARCH_RESULTS
        mock_reranker_cls.return_value = mock_reranker

        # 运行 pipeline
        pipeline = LegalCasePipeline(llm_mode="api")
        result = pipeline.run("被告人张某秘密窃取他人财物价值5000元")

        # 验证结果结构
        assert "elements" in result
        assert "retrieved_cases" in result
        assert "analysis" in result

        # 验证要素抽取
        assert result["elements"]["cause_of_action"] == "盗窃罪"

        # 验证检索被调用
        assert mock_searcher.search.called

        # 验证重排序被调用
        assert mock_reranker.rerank.called

        # 验证 LLM 被调用两次（要素抽取 + 分析生成）
        assert mock_llm.generate.call_count == 2

    @patch("src.pipeline.LLMInference")
    @patch("src.pipeline.MultiRouteSearcher")
    @patch("src.pipeline.Reranker")
    def test_full_pipeline_empty_results(self, mock_reranker_cls, mock_searcher_cls, mock_llm_cls):
        """测试检索无结果时的处理。"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [MOCK_ELEMENTS_RESPONSE, "未找到相似案例"]
        mock_llm_cls.return_value = mock_llm

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_cls.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        mock_reranker_cls.return_value = mock_reranker

        pipeline = LegalCasePipeline(llm_mode="api")
        result = pipeline.run("一个很罕见的案情")

        assert result["retrieved_cases"] == []
        # pipeline 应该仍然完成，不会崩溃
        assert "analysis" in result


# ── Prompt 模板测试 ────────────────────────────────────────────────

class TestPrompts:
    def test_extract_elements_prompt(self):
        from src.llm.prompts import format_extract_elements_prompt

        prompt = format_extract_elements_prompt("被告人盗窃他人财物")
        assert "被告人盗窃他人财物" in prompt
        assert "case_type" in prompt
        assert "cause_of_action" in prompt
        assert "JSON" in prompt

    def test_case_analysis_prompt(self):
        from src.llm.prompts import format_case_analysis_prompt

        prompt = format_case_analysis_prompt(
            case_description="被告人盗窃",
            case_elements='{"case_type": "刑事"}',
            retrieved_cases='[{"case_id": "001"}]',
            n_cases=1,
        )
        assert "被告人盗窃" in prompt
        assert "刑事" in prompt
        assert "001" in prompt
        assert "相似性分析" in prompt
        assert "免责声明" in prompt


# ── 端到端测试用例数据（等 GPU 环境跑真实模型） ────────────────────

TEST_QUERIES = [
    # 刑事 - 盗窃
    "被告人张某秘密窃取他人财物，价值人民币5000元",
    # 刑事 - 故意伤害
    "被告人李某因纠纷殴打他人，致人轻伤一级",
    # 刑事 - 诈骗
    "被告人赵某以虚假投资为名骗取他人钱财20万元",
    # 刑事 - 交通肇事
    "被告人王某醉酒驾驶机动车发生交通事故，致一人死亡",
    # 刑事 - 抢劫
    "被告人刘某持刀抢劫路人，抢得手机一部及现金500元",
    # 刑事 - 职务侵占
    "被告人陈某利用职务便利将公司资金10万元占为己有",
    # 刑事 - 危险驾驶
    "被告人周某醉酒后驾驶机动车被当场查获",
    # 刑事 - 寻衅滋事
    "被告人孙某在公共场所随意殴打他人，情节恶劣",
    # 刑事 - 非法拘禁
    "被告人吴某为索要债务非法限制他人人身自由三天",
    # 刑事 - 掩饰隐瞒犯罪所得
    "被告人郑某明知是犯罪所得仍予以收购，价值3万元",
]


@pytest.mark.skip(reason="需要真实 LLM 模型和向量库，等 GPU 环境跑")
class TestEndToEnd:
    """端到端测试：用真实 LawLLM-7B + BGE-M3 + Qdrant 跑完整流程。

    运行前提:
        1. AutoDL 上已部署 LawLLM-7B 并启动 vLLM API 服务
        2. config.yaml 中 api.base_url 已正确配置
        3. Qdrant 中已有向量数据
        4. BM25 索引已构建
    """

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_query_returns_valid_result(self, query: str):
        """每个测试查询应返回有效的分析报告。"""
        pipeline = LegalCasePipeline(llm_mode="api")
        result = pipeline.run(query)

        # 要素抽取应成功
        assert "case_type" in result["elements"]
        assert "cause_of_action" in result["elements"]

        # 应检索到相似案例
        assert len(result["retrieved_cases"]) > 0

        # 分析报告应包含必要部分
        analysis = result["analysis"]
        assert "相似" in analysis or "分析" in analysis
        assert "免责" in analysis