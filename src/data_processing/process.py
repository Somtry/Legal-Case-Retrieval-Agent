"""批量数据处理脚本：将 data/raw/ 下的原始数据解析为标准 JSONL。

流程: 加载原始 JSONL → 逐条解析为标准格式 → 去重 → 校验 → 输出到 data/processed/

用法:
    # 处理单个数据源
    python -m src.data_processing.process --source cail2018

    # 处理所有已下载的数据源
    python -m src.data_processing.process --source all

    # 只处理前 1000 条（测试用）
    python -m src.data_processing.process --source cail2018 --max-samples 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config
from src.data_processing.case_parser import deduplicate, parse_record, validate

# 数据源名称 → 原始文件路径模式
SOURCE_FILES: dict[str, str] = {
    "cail2018": "cail2018/cail2018.jsonl",
    "disc_law_sft": "disc_law_sft/disc_law_sft.jsonl",
    "appeal_case": "appeal_case/appeal_case.jsonl",
}


def get_raw_dir() -> Path:
    return config.data_dir / "raw"


def get_processed_dir() -> Path:
    processed = config.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    return processed


def load_raw_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """从 JSONL 文件加载原始记录。"""
    records: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"从 {filepath} 加载了 {len(records)} 条原始记录")
    return records


def process_source(
    source: str,
    max_samples: int | None = None,
) -> Path | None:
    """处理单个数据源：解析 → 去重 → 校验 → 输出 JSONL。

    Args:
        source: 数据源名称。
        max_samples: 只处理前 N 条。

    Returns:
        输出文件路径，如果原始文件不存在则返回 None。
    """
    raw_path = get_raw_dir() / SOURCE_FILES[source]
    if not raw_path.exists():
        logger.warning(f"原始数据文件不存在: {raw_path}，请先运行 download.py 下载")
        return None

    # 1. 加载原始数据
    records = load_raw_jsonl(raw_path)
    if max_samples is not None:
        records = records[:max_samples]
        logger.info(f"截取前 {max_samples} 条进行测试")

    # 2. 逐条解析
    parsed_cases: list[dict[str, Any]] = []
    parse_errors = 0
    for i, record in enumerate(records):
        try:
            case = parse_record(source, record, index=i)
            parsed_cases.append(case)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                logger.warning(f"第 {i} 条记录解析失败: {e}")
    logger.info(f"解析完成: {len(parsed_cases)} 成功, {parse_errors} 失败")

    # 3. 去重
    before_dedup = len(parsed_cases)
    parsed_cases = deduplicate(parsed_cases)
    logger.info(f"去重: {before_dedup} → {len(parsed_cases)}")

    # 4. 校验，过滤掉无效记录
    valid_cases: list[dict[str, Any]] = []
    invalid_count = 0
    for case in parsed_cases:
        if validate(case):
            valid_cases.append(case)
        else:
            invalid_count += 1
    if invalid_count > 0:
        logger.warning(f"过滤掉 {invalid_count} 条无效记录（缺少必要字段）")
    logger.info(f"校验通过: {len(valid_cases)} 条")

    # 5. 输出 JSONL
    output_path = get_processed_dir() / f"{source}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for case in valid_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    logger.info(f"已输出 {len(valid_cases)} 条到 {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="批量处理裁判文书数据")
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        choices=list(SOURCE_FILES.keys()) + ["all"],
        help="数据源名称，或 'all' 处理所有已下载的数据",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="只处理前 N 条（测试用）",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    sources = list(SOURCE_FILES.keys()) if args.source == "all" else [args.source]
    for src in sources:
        logger.info(f"{'=' * 40} 处理 {src} {'=' * 40}")
        process_source(src, max_samples=args.max_samples)


if __name__ == "__main__":
    main()