"""数据下载模块：从 HuggingFace 下载中文法律数据集到 data/raw/。

支持的数据集:
    - CAIL2018: 刑事裁判文书（~19.6万条小规模版本）
    - DISC-Law-SFT: 法律微调数据（含类案匹配、要素提取等）
    - AppealCase: 民事上诉案件（1万对一审+二审）

用法:
    python -m src.data_processing.download --dataset cail2018
    python -m src.data_processing.download --dataset disc_law_sft
    python -m src.data_processing.download --dataset appeal_case
    python -m src.data_processing.download --dataset all
    python -m src.data_processing.download --dataset cail2018 --max-samples 1000

环境变量:
    HF_ENDPOINT: HuggingFace 端点地址，默认 https://huggingface.co。
                 国内可设为 https://hf-mirror.com 加速下载。
    HF_TOKEN: HuggingFace 访问令牌（部分数据集需要）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config

# 数据集在 HuggingFace 上的 ID 和本地存放子目录名
DATASET_SPECS: dict[str, dict[str, str]] = {
    "cail2018": {
        "hf_id": "china-ai-law-challenge/cail2018",
        "subdir": "cail2018",
        "description": "刑事裁判文书（CAIL2018）",
    },
    "disc_law_sft": {
        "hf_id": "ShengbinYue/DISC-Law-SFT",
        "subdir": "disc_law_sft",
        "description": "法律微调数据（DISC-Law-SFT）",
    },
    "appeal_case": {
        "hf_id": "ythuang02/AppealCase",
        "subdir": "appeal_case",
        "description": "民事上诉案件（AppealCase）",
    },
}


def get_raw_dir() -> Path:
    """返回 data/raw/ 目录，不存在则创建。"""
    raw_dir = config.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_dataset(
    dataset_key: str,
    max_samples: int | None = None,
) -> Path:
    """下载单个数据集并保存为 JSONL 文件。

    Args:
        dataset_key: DATASET_SPECS 中的键名。
        max_samples: 只下载前 N 条样本（用于测试），None 表示全量。

    Returns:
        保存的 JSONL 文件路径。
    """
    spec = DATASET_SPECS[dataset_key]
    hf_id = spec["hf_id"]
    subdir = spec["subdir"]

    output_dir = get_raw_dir() / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subdir}.jsonl"

    if output_path.exists():
        logger.warning(f"{output_path} 已存在，跳过下载。如需重新下载请先删除该文件。")
        return output_path

    logger.info(f"开始下载 {spec['description']} (HF: {hf_id})")

    # 设置 HuggingFace 镜像端点（如果环境变量已设置则用环境变量的值）
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if hf_endpoint != "https://huggingface.co":
        logger.info(f"使用 HuggingFace 镜像: {hf_endpoint}")
    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    from datasets import load_dataset

    ds = load_dataset(hf_id)
    logger.info(f"数据集 splits: {list(ds.keys())}")

    # 合并所有 split
    total_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for split_name in ds.keys():
            split = ds[split_name]
            logger.info(f"  split '{split_name}': {len(split)} 条")
            for row in split:
                if max_samples is not None and total_written >= max_samples:
                    break
                # datasets 的 Row 是 dict-like，转成纯 dict
                row_dict = {k: _to_serializable(v) for k, v in row.items()}
                f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
                total_written += 1
            if max_samples is not None and total_written >= max_samples:
                logger.info(f"  已达到 max_samples={max_samples}，停止")
                break

    logger.info(f"下载完成: {total_written} 条 → {output_path}")
    return output_path


def _to_serializable(value: Any) -> Any:
    """将 datasets 库的特殊类型转为可序列化的 Python 类型。"""
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="下载中文法律数据集")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=list(DATASET_SPECS.keys()) + ["all"],
        help="要下载的数据集名称，或 'all' 下载全部",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="只下载前 N 条样本（用于测试管线）",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="HuggingFace 镜像地址（如 https://hf-mirror.com）",
    )
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror
        logger.info(f"已设置 HF_ENDPOINT={args.mirror}")

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.dataset == "all":
        for key in DATASET_SPECS:
            try:
                download_dataset(key, max_samples=args.max_samples)
            except Exception as e:
                logger.error(f"下载 {key} 失败: {e}")
                logger.error(f"可尝试单独下载: python -m src.data_processing.download -d {key}")
    else:
        download_dataset(args.dataset, max_samples=args.max_samples)


if __name__ == "__main__":
    main()