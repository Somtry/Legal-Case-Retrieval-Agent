"""主流程编排：案情输入 → 要素抽取 → 多路检索 → 重排序 → 案例分析生成。

这是整个 Agent 的核心管线，串联所有模块。
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.llm.inference import LLMInference
from src.llm.prompts import format_case_analysis_prompt, format_extract_elements_prompt
from src.retrieval.rerank import Reranker
from src.retrieval.search import MultiRouteSearcher


class LegalCasePipeline:
    """法律案例检索 Agent 主流程。"""

    def __init__(self, llm_mode: str = "local") -> None:
        self.llm = LLMInference(mode=llm_mode)
        self.searcher = MultiRouteSearcher()
        self.reranker = Reranker()

        # 预构建 BM25 索引（从 data/processed/ 加载）
        try:
            self.searcher.build_bm25_index()
        except Exception as e:
            logger.warning(f"BM25 索引构建失败（可忽略，向量检索仍可用）: {e}")

    def extract_elements(self, case_description: str) -> dict[str, Any]:
        """Stage 1: 案情要素抽取。

        Args:
            case_description: 用户输入的案情描述文本。

        Returns:
            抽取出的法律要素字典。
        """
        logger.info("Stage 1: 案情要素抽取")
        prompt = format_extract_elements_prompt(case_description)
        response = self.llm.generate(prompt, temperature=0.1)

        # 尝试解析 JSON 输出
        try:
            # 提取 JSON 块
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elements = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"要素抽取 JSON 解析失败: {e}，返回原始文本")
            elements = {"raw_response": response}

        logger.info(f"要素抽取完成: {elements.get('case_type', '?')}/{elements.get('cause_of_action', '?')}")
        return elements

    def retrieve(self, case_description: str, elements: dict[str, Any]) -> list[dict[str, Any]]:
        """Stage 2 + 3: 多路检索 + 重排序。"""
        logger.info("Stage 2: 多路检索")
        candidates = self.searcher.search(case_description)

        logger.info("Stage 3: 重排序")
        top_results = self.reranker.rerank(case_description, elements, candidates)
        return top_results

    def generate_analysis(
        self,
        case_description: str,
        elements: dict[str, Any],
        retrieved_cases: list[dict[str, Any]],
    ) -> str:
        """Stage 4: 案例分析报告生成。"""
        logger.info("Stage 4: 案例分析生成")

        # 格式化检索案例为文本
        cases_text = json.dumps(
            [r.get("payload", {}) for r in retrieved_cases],
            ensure_ascii=False,
            indent=2,
        )
        elements_text = json.dumps(elements, ensure_ascii=False, indent=2)

        prompt = format_case_analysis_prompt(
            case_description=case_description,
            case_elements=elements_text,
            retrieved_cases=cases_text,
            n_cases=len(retrieved_cases),
        )
        analysis = self.llm.generate(prompt, temperature=0.1, max_tokens=2048)
        return analysis

    def run(self, case_description: str) -> dict[str, Any]:
        """运行完整流程：输入案情 → 输出分析报告。

        Args:
            case_description: 用户输入的案情描述。

        Returns:
            包含要素、检索结果、分析报告的完整输出。
        """
        logger.info("=" * 60)
        logger.info("法律案例检索 Agent 启动")
        logger.info(f"案情: {case_description[:100]}...")
        logger.info("=" * 60)

        # Stage 1: 要素抽取
        elements = self.extract_elements(case_description)

        # Stage 2 + 3: 检索 + 重排序
        retrieved = self.retrieve(case_description, elements)

        # Stage 4: 分析生成
        analysis = self.generate_analysis(case_description, elements, retrieved)

        return {
            "elements": elements,
            "retrieved_cases": retrieved,
            "analysis": analysis,
        }