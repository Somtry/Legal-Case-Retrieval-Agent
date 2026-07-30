"""入口模块：提供 CLI 交互入口。

用法:
    python -m src.main
    python -m src.main --query "案情描述文本"
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.pipeline import LegalCasePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="法律案例检索 Agent")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="案情描述文本（不提供则进入交互模式）",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="local",
        choices=["local", "api"],
        help="LLM 推理模式：local（vLLM 本地）或 api（远程 API）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )
    args = parser.parse_args()

    # 日志配置
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level)

    pipeline = LegalCasePipeline(llm_mode=args.llm_mode)

    if args.query:
        # 单次查询模式
        result = pipeline.run(args.query)
        print("\n" + "=" * 60)
        print("分析报告")
        print("=" * 60)
        print(result["analysis"])
    else:
        # 交互模式
        print("法律案例检索 Agent（输入 'quit' 退出）")
        print("=" * 60)
        while True:
            try:
                query = input("\n请输入案情描述: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if query.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not query:
                continue

            result = pipeline.run(query)
            print("\n" + "=" * 60)
            print("分析报告")
            print("=" * 60)
            print(result["analysis"])


if __name__ == "__main__":
    main()