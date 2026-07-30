"""在 AutoDL GPU 机器上批量向量化案例数据并写入 Qdrant。

此脚本在 AutoDL 上运行，使用 GPU 加速 BGE-M3 embedding 生成。
向量写入后，本地开发机的 Qdrant 就能直接用于检索。

前提条件:
    1. AutoDL 上已安装 FlagEmbedding: pip install FlagEmbedding qdrant-client
    2. 本地开发机的 Qdrant 已启动（需要 AutoDL 能访问到，见下方说明）
    3. data/processed/ 下有结构化案例数据（通过 process.py 生成）

关于 Qdrant 连接:
    方案 A（推荐）: 在 AutoDL 上也启动一个 Qdrant 容器，向量写入后
                   导出数据，拷回本地导入。
    方案 B: 本地 Qdrant 通过内网穿透（如 frp）暴露公网地址，AutoDL 直连。
    方案 C: 直接在 AutoDL 上用 Qdrant 的内存模式，完成向量化后
            把 collection 快照下载回本地。

用法（在 AutoDL 上运行）:
    python scripts/autodl/build_vector_index.py --source cail2018
    python scripts/autodl/build_vector_index.py --source cail2018 --qdrant-host localhost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def load_cases(filepath: Path) -> list[dict[str, Any]]:
    """加载处理后的案例 JSONL 文件。"""
    cases: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    logger.info(f"从 {filepath} 加载了 {len(cases)} 条案例")
    return cases


def build_texts_for_embedding(cases: list[dict[str, Any]]) -> list[str]:
    """为每条案例构造用于 embedding 的文本。

    策略: 将案由 + 事实拼接为embedding文本，
    这是最能反映案例语义特征的组合。
    """
    texts: list[str] = []
    for case in cases:
        cause = case.get("cause_of_action", "")
        facts = case.get("facts", "")
        # 拼接：案由 + 事实描述
        text = f"案由：{cause}\n事实：{facts}" if cause else facts
        texts.append(text)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="在 GPU 上批量向量化案例数据")
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="数据源名称（如 cail2018），对应 data/processed/{source}.jsonl",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="处理后的数据目录",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant 地址（默认 localhost，如在本地则用内网穿透地址）",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant 端口",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="legal_cases",
        help="Qdrant collection 名称",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="BAAI/bge-m3",
        help="BGE-M3 模型路径（本地路径或 HF ID）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批量编码大小",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="只处理前 N 条案例（测试用）",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 1. 加载案例数据
    data_path = Path(args.data_dir) / f"{args.source}.jsonl"
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        sys.exit(1)

    cases = load_cases(data_path)
    if args.max_cases:
        cases = cases[: args.max_cases]
        logger.info(f"截取前 {args.max_cases} 条进行测试")

    # 2. 构造 embedding 文本
    texts = build_texts_for_embedding(cases)
    logger.info(f"构造了 {len(texts)} 条 embedding 文本")

    # 3. 加载 BGE-M3 模型（GPU 加速）
    logger.info(f"加载 BGE-M3 模型: {args.model_path}")
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(args.model_path, use_fp16=True)

    # 4. 批量向量化
    logger.info("开始批量向量化...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=8192,
    )["dense_vecs"]
    logger.info(f"向量化完成: {embeddings.shape}")

    # 5. 写入 Qdrant
    logger.info(f"连接 Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # 创建 collection（如不存在）
    vector_dim = embeddings.shape[1]
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    logger.info(f"已创建 collection: {args.collection} (dim={vector_dim})")

    # 批量写入
    batch_size = 100
    for i in range(0, len(cases), batch_size):
        batch_cases = cases[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size].tolist()

        points = [
            PointStruct(
                id=i + j,
                vector=emb,
                payload={
                    "case_id": case.get("case_id", ""),
                    "case_type": case.get("case_type", ""),
                    "cause_of_action": case.get("cause_of_action", ""),
                    "facts": case.get("facts", ""),
                    "legal_basis": case.get("legal_basis", []),
                    "ruling": case.get("ruling", ""),
                    "ruling_result": case.get("ruling_result", {}),
                    "full_text": case.get("full_text", ""),
                },
            )
            for j, (case, emb) in enumerate(zip(batch_cases, batch_embs))
        ]
        client.upsert(collection_name=args.collection, points=points)
        logger.info(f"  写入 {i + len(batch_cases)}/{len(cases)}")

    logger.info(f"向量索引构建完成! 共 {len(cases)} 条案例写入 {args.collection}")

    # 6. 验证：做一个测试查询
    logger.info("验证：测试检索...")
    query_text = texts[0]  # 用第一条案例作为测试查询
    query_vec = model.encode([query_text], batch_size=1, max_length=8192)["dense_vecs"][0].tolist()
    results = client.search(
        collection_name=args.collection,
        query_vector=query_vec,
        limit=3,
    )
    for hit in results:
        logger.info(f"  score={hit.score:.4f} case_id={hit.payload.get('case_id', '?')} cause={hit.payload.get('cause_of_action', '?')}")


if __name__ == "__main__":
    main()