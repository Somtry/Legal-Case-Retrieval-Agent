# Legal Case Retrieval Agent

法律案例检索助手：输入案情描述，自动检索相似案例并生成分析报告。

## 快速开始

### 方式一：AutoDL GPU 服务器（推荐）

```bash
# 1. 租一台带 GPU 的服务器（如 RTX 4090 24GB）

# 2. 克隆项目
git clone https://github.com/Somtry/Legal-Case-Retrieval-Agent.git
cd Legal-Case-Retrieval-Agent

# 3. 创建环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 下载模型（BGE-M3 + LawLLM-7B）
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
export HF_HUB_DISABLE_XET=1               # 禁用 XetHub

pip install "huggingface_hub==0.24.0"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='./models/bge-m3', ignore_patterns=['*.DS_Store','*.onnx','openvino_model/*','imgs/*'])
print('BGE-M3 下载完成')
"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ShengbinYue/LawLLM-7B', local_dir='./models/LawLLM-7B', ignore_patterns=['*.DS_Store','*.onnx','openvino_model/*'])
print('LawLLM-7B 下载完成')
"

# 5. 下载 CAIL2018 数据 + 处理 + 构建向量索引
python -c "
import json, os, sys
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.chdir('/root/Legal-Case-Retrieval-Agent')
sys.path.insert(0, '.')

from datasets import load_dataset
ds = load_dataset('china-ai-law-challenge/cail2018')
first_split = list(ds.keys())[0]
os.makedirs('data/raw/cail2018', exist_ok=True)
with open('data/raw/cail2018/cail2018.jsonl', 'w', encoding='utf-8') as f:
    for i, row in enumerate(ds[first_split]):
        if i >= 500: break
        f.write(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False) + '\n')

from src.data_processing.case_parser import parse_record, deduplicate, validate
records = [json.loads(l) for l in open('data/raw/cail2018/cail2018.jsonl', encoding='utf-8') if l.strip()]
cases = [parse_record('cail2018', r, i) for i, r in enumerate(records)]
cases = [c for c in deduplicate(cases) if validate(c)]
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/cail2018.jsonl', 'w', encoding='utf-8') as f:
    for c in cases: f.write(json.dumps(c, ensure_ascii=False) + '\n')

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
import shutil
texts = [f'案由：{c.get(\"cause_of_action\",\"\")}\n事实：{c.get(\"facts\",\"\")}' for c in cases]
model = BGEM3FlagModel('./models/bge-m3', use_fp16=True)
embs = model.encode(texts, batch_size=32, max_length=8192)['dense_vecs']
if os.path.exists('qdrant_data'): shutil.rmtree('qdrant_data')
client = QdrantClient(path='qdrant_data')
client.create_collection('legal_cases', vectors_config=VectorParams(size=embs.shape[1], distance=Distance.COSINE))
for i in range(0, len(cases), 100):
    pts = [PointStruct(id=i+j, vector=embs[i+j].tolist(), payload=cases[i+j]) for j in range(min(100, len(cases)-i))]
    client.upsert('legal_cases', pts)
print(f'向量索引构建完成: {len(cases)} 条')
"

# 6. 运行
python -m src.main --query "被告人张某秘密窃取他人财物，价值人民币5000元"
# 或交互模式
python -m src.main
```

### 方式二：本地开发（无 GPU）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改 configs/config.yaml 中 api.base_url 为远程 vLLM 服务地址
# 3. 使用 api 模式运行
python -m src.main --query "案情描述文本" --llm-mode api
```

## 项目结构

```
legal-agent/
├── configs/
│   └── config.yaml         # 配置文件（模型路径、检索参数等）
├── data/
│   ├── raw/                # 原始裁判文书
│   └── processed/          # 结构化后的案例数据
├── models/                 # 模型权重（gitignore）
│   ├── bge-m3/             # BGE-M3 embedding 模型
│   └── LawLLM-7B/          # LawLLM-7B 模型
├── src/
│   ├── config.py           # 配置管理（单例模式）
│   ├── data_processing/    # 数据处理模块
│   │   ├── case_parser.py  # 裁判文书解析（CAIL2018/DISC-Law/AppealCase）
│   │   ├── vector_store.py # Qdrant 向量库管理
│   │   ├── download.py     # 数据下载
│   │   ├── process.py      # 数据处理流水线
│   │   └── stats.py        # 数据统计报告
│   ├── retrieval/          # 检索模块
│   │   ├── embed.py        # BGE-M3 向量化
│   │   ├── search.py       # 多路检索（向量 + BM25）
│   │   └── rerank.py       # 重排序（要素加权精排）
│   ├── llm/                # 大模型模块
│   │   ├── inference.py    # vLLM 推理封装（local / api 两种模式）
│   │   └── prompts.py      # Prompt 模板
│   ├── pipeline.py         # 主流程编排
│   └── main.py             # CLI 入口
├── scripts/
│   └── autodl/             # AutoDL 部署脚本
│       ├── full_setup.py   # 一键下载+处理+构建索引
│       ├── build_vector_index.py  # 向量索引构建
│       └── setup_autodl.sh # 环境部署
├── notebooks/              # Kaggle/实验 notebook
├── tests/                  # 单元测试（50 个用例）
├── requirements.txt
└── README.md
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 基座模型 | LawLLM-7B (Qwen2.5-7B-Instruct) | 法律领域微调模型 |
| Embedding | BGE-M3 | 多语言 embedding，输出 1024 维 |
| 向量数据库 | Qdrant | 嵌入式模式，无需 Docker |
| 推理框架 | vLLM | 高性能 LLM 推理 |
| 关键词检索 | BM25 (rank_bm25 + jieba) | 中文分词 + BM25 |
| 重排序 | 要素加权精排 | 案由匹配 + 争议焦点 + 事实重叠 |

## 流程

```
用户输入案情
  → Stage 1: 要素抽取 (LawLLM-7B 识别罪名、关键事实)
  → Stage 2: 多路检索 (BGE-M3 向量检索 + BM25 关键词检索)
  → Stage 3: 重排序 (归一化检索分数 + 要素加权精排)
  → Stage 4: 案例分析生成 (LawLLM-7B 生成分析报告)
  → 输出报告 (相似性分析 + 裁判观点对比 + 法律建议)
```

## 硬件要求

| 配置 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | RTX 3060 12GB | RTX 4090 24GB |
| 内存 | 16GB | 32GB+ |
| 磁盘 | 30GB | 100GB+ |

LawLLM-7B FP16 需约 14GB 显存，BGE-M3 需约 2.3GB。

## 数据源

| 数据集 | 说明 | 状态 |
|--------|------|------|
| CAIL2018 | 刑事案件裁判文书 | 已集成，500 条测试 |
| DISC-Law-SFT | 法律问答 SFT 数据 | 解析器已实现，数据未下载 |
| AppealCase | 上诉案件 | 解析器已实现，数据未下载 |

## 测试

```bash
# 运行单元测试
pytest tests/ -v

# 端到端测试（需要 GPU 环境）
pytest tests/test_pipeline.py::TestEndToEnd -v --run-e2e
```

## 免责声明

本系统输出的所有内容仅供参考，不构成专业法律意见。
