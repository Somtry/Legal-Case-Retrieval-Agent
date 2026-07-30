"""数据质量检查 + 统计报告：对 data/processed/ 下的结构化数据生成统计报告。

报告内容包括:
    - 各数据源案例总数
    - 案件类型分布（民事/刑事/行政）
    - 案由 Top 20 分布
    - 法条引用 Top 20
    - 字段完整度（各字段非空比例）
    - 事实文本长度分布（平均/中位数/最小/最大）
    - 日期范围
    - 地区分布

用法:
    python -m src.data_processing.stats
    python -m src.data_processing.stats --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config

# 标准案例的所有字段
ALL_FIELDS = [
    "case_id", "court", "case_type", "cause_of_action", "facts",
    "claims", "legal_basis", "ruling", "ruling_result",
    "date", "region", "full_text",
]


def get_processed_dir() -> Path:
    return config.data_dir / "processed"


def load_processed_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """加载处理后的 JSONL 文件。"""
    cases: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def compute_stats(cases: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    """计算单个数据源的统计信息。"""
    total = len(cases)
    if total == 0:
        return {"source": source_name, "total": 0}

    # 案件类型分布
    case_types = Counter(c.get("case_type", "") for c in cases)

    # 案由分布 Top 20
    causes = Counter()
    for c in cases:
        cause = c.get("cause_of_action", "")
        if cause:
            causes[cause] += 1

    # 法条引用 Top 20
    articles = Counter()
    for c in cases:
        for lb in c.get("legal_basis", []):
            if lb:
                articles[lb] += 1

    # 字段完整度
    field_completeness: dict[str, float] = {}
    for field in ALL_FIELDS:
        non_empty = sum(
            1 for c in cases
            if c.get(field)  # 非空字符串/非空列表/非空字典
        )
        field_completeness[field] = round(non_empty / total * 100, 1)

    # 事实文本长度统计
    fact_lengths = [len(c.get("facts", "")) for c in cases if c.get("facts")]
    fact_length_stats = {}
    if fact_lengths:
        fact_length_stats = {
            "avg": round(sum(fact_lengths) / len(fact_lengths), 0),
            "median": sorted(fact_lengths)[len(fact_lengths) // 2],
            "min": min(fact_lengths),
            "max": max(fact_lengths),
        }

    # 日期范围
    dates = [c.get("date", "") for c in cases if c.get("date")]
    date_range = {"earliest": min(dates) if dates else "", "latest": max(dates) if dates else ""}

    # 地区分布 Top 10
    regions = Counter(c.get("region", "") for c in cases if c.get("region"))

    return {
        "source": source_name,
        "total": total,
        "case_type_distribution": dict(case_types.most_common()),
        "cause_of_action_top20": dict(causes.most_common(20)),
        "legal_basis_top20": dict(articles.most_common(20)),
        "field_completeness_pct": field_completeness,
        "fact_length_stats": fact_length_stats,
        "date_range": date_range,
        "region_top10": dict(regions.most_common(10)),
    }


def print_report(stats_list: list[dict[str, Any]]) -> None:
    """以可读格式打印统计报告。"""
    print()
    print("=" * 60)
    print("数据质量检查 + 统计报告")
    print("=" * 60)

    for stats in stats_list:
        print(f"\n{'─' * 60}")
        print(f"数据源: {stats['source']}")
        print(f"总案例数: {stats['total']}")
        if stats["total"] == 0:
            print("  (无数据)")
            continue

        print(f"\n  案件类型分布:")
        for ctype, count in stats["case_type_distribution"].items():
            pct = count / stats["total"] * 100
            print(f"    {ctype or '(空)'}: {count} ({pct:.1f}%)")

        print(f"\n  案由 Top 10:")
        for i, (cause, count) in enumerate(
            list(stats["cause_of_action_top20"].items())[:10]
        ):
            print(f"    {i+1}. {cause[:40]}: {count}")

        print(f"\n  法条引用 Top 10:")
        for i, (article, count) in enumerate(
            list(stats["legal_basis_top20"].items())[:10]
        ):
            print(f"    {i+1}. {article}: {count}")

        print(f"\n  字段完整度:")
        for field, pct in stats["field_completeness_pct"].items():
            bar = "█" * int(pct / 5)
            print(f"    {field:20s} {pct:5.1f}% {bar}")

        if stats["fact_length_stats"]:
            fls = stats["fact_length_stats"]
            print(f"\n  事实文本长度:")
            print(f"    平均: {fls['avg']:.0f} 字")
            print(f"    中位数: {fls['median']} 字")
            print(f"    最短: {fls['min']} 字 / 最长: {fls['max']} 字")

        if stats["date_range"]["earliest"]:
            print(f"\n  日期范围: {stats['date_range']['earliest']} ~ {stats['date_range']['latest']}")

        if stats["region_top10"]:
            print(f"\n  地区分布 Top 5:")
            for region, count in list(stats["region_top10"].items())[:5]:
                print(f"    {region}: {count}")

    # 汇总
    print(f"\n{'─' * 60}")
    grand_total = sum(s["total"] for s in stats_list)
    print(f"汇总: 共 {len(stats_list)} 个数据源, 合计 {grand_total} 条案例")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据质量检查 + 统计报告")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="将报告保存为 JSON 文件（同时仍打印到终端）",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        logger.error(f"处理后的数据目录不存在: {processed_dir}，请先运行 process.py")
        sys.exit(1)

    # 找到所有 .jsonl 文件
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"未找到处理后的数据文件（{processed_dir}/*.jsonl）")
        sys.exit(1)

    stats_list: list[dict[str, Any]] = []
    for filepath in jsonl_files:
        source_name = filepath.stem
        logger.info(f"统计 {source_name}...")
        cases = load_processed_jsonl(filepath)
        stats = compute_stats(cases, source_name)
        stats_list.append(stats)

    # 打印报告
    print_report(stats_list)

    # 可选：保存为 JSON
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats_list, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存到 {output_path}")


if __name__ == "__main__":
    main()