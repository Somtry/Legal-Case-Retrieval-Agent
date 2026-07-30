"""一键下载 CAIL2018 数据 + 处理 + 构建向量索引

用法:
    python scripts/autodl/full_setup.py --max-cases 500
"""
import json
import os
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'

# 确保项目根目录在 path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def download_cail2018(max_cases=1000):
    """下载 CAIL2018 数据"""
    logger.info("下载 CAIL2018 数据...")
    from datasets import load_dataset
    ds = load_dataset('china-ai-law-challenge/cail2018')

    first_split = list(ds.keys())[0]

    raw_dir = PROJECT_DIR / 'data/raw/cail2018'
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / 'cail2018.jsonl'

    with open(raw_path, 'w', encoding='utf-8') as f:
        count = 0
        for row in ds[first_split]:
            if count >= max_cases:
                break
            row_dict = {k: v for k, v in row.items()}
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
            count += 1

    logger.info(f"下载了 {count} 条 CAIL2018 数据到 {raw_path}")
    return raw_path


def process_data():
    """解析 + 去重 + 校验"""
    logger.info("处理数据: 解析 → 去重 → 校验...")
    from src.data_processing.case_parser import parse_record, deduplicate, validate

    raw_path = PROJECT_DIR / 'data/raw/cail2018/cail2018.jsonl'
    records = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"加载了 {len(records)} 条原始记录")

    cases = []
    for i, record in enumerate(records):
        try:
            case = parse_record('cail2018', record, index=i)
            cases.append(case)
        except Exception as e:
            logger.warning(f"解析失败 #{i}: {e}")

    cases = deduplicate(cases)
    cases = [c for c in cases if validate(c)]

    processed_dir = PROJECT_DIR / 'data/processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / 'cail2018.jsonl'

    with open(processed_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    logger.info(f"处理完成: {len(cases)} 条有效案例写入 {processed_path}")
    return processed_path


def build_vector_index():
    """构建向量索引"""
    logger.info("构建向量索引...")
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
    import shutil

    # 加载案例
    processed_path = PROJECT_DIR / 'data/processed/cail2018.jsonl'
    cases = []
    with open(processed_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    logger.info(f"加载了 {len(cases)} 条案例")

    # 构造 embedding 文本
    texts = []
    for case in cases:
        cause = case.get('cause_of_action', '')
        facts = case.get('facts', '')
        text = f'案由：{cause}\n事实：{facts}' if cause else facts
        texts.append(text)

    # 加载 BGE-M3
    logger.info("加载 BGE-M3 模型...")
    model_path = './models/bge-m3'
    embed_model = BGEM3FlagModel(model_path, use_fp16=True)

    # 批量向量化
    logger.info(f"开始向量化 {len(texts)} 条文本...")
    embeddings = embed_model.encode(texts, batch_size=32, max_length=8192)['dense_vecs']
    logger.info(f"向量化完成: shape={embeddings.shape}")

    # Qdrant 嵌入式模式
    qdrant_path = str(PROJECT_DIR / 'qdrant_data')
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)

    client = QdrantClient(path=qdrant_path)
    vector_dim = embeddings.shape[1]
    client.recreate_collection(
        collection_name='legal_cases',
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    logger.info(f"已创建 Qdrant collection (dim={vector_dim})")

    # 批量写入
    batch_size = 100
    for i in range(0, len(cases), batch_size):
        batch_cases = cases[i:i+batch_size]
        batch_embs = embeddings[i:i+batch_size].tolist()
        points = [
            PointStruct(
                id=i+j,
                vector=emb,
                payload={
                    'case_id': case.get('case_id', ''),
                    'case_type': case.get('case_type', ''),
                    'cause_of_action': case.get('cause_of_action', ''),
                    'facts': case.get('facts', ''),
                    'legal_basis': case.get('legal_basis', []),
                    'ruling': case.get('ruling', ''),
                    'full_text': case.get('full_text', ''),
                },
            )
            for j, (case, emb) in enumerate(zip(batch_cases, batch_embs))
        ]
        client.upsert(collection_name='legal_cases', points=points)
        logger.info(f"  写入 {i + len(batch_cases)}/{len(cases)}")

    logger.info(f"向量索引构建完成! 共 {len(cases)} 条")

    # 验证检索
    query_vec = embeddings[0].tolist()
    results = client.query_points(collection_name='legal_cases', query=query_vec, limit=3)
    logger.info("验证检索:")
    for point in results.points:
        logger.info(f"  score={point.score:.4f} case_id={point.payload.get('case_id')} cause={point.payload.get('cause_of_action')}")

    # 释放显存
    del embed_model
    import torch
    torch.cuda.empty_cache()
    logger.info("已释放 embedding 模型显存")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cases', type=int, default=500, help='下载前 N 条案例')
    args = parser.parse_args()

    download_cail2018(args.max_cases)
    process_data()
    build_vector_index()

    logger.info("✅ 全部完成! 现在可以运行: python -m src.main --query '案情描述'")