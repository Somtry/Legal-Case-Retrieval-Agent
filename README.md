# Legal Case Retrieval Agent

法律案例检索助手：输入案情描述，自动检索相似案例并生成分析报告。

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Qdrant 向量数据库
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# 4. 运行
python -m src.main --query "案情描述文本"
# 或交互模式
python -m src.main
```

## 项目结构

```
legal-agent/
├── data/
│   ├── raw/                # 原始裁判文书
│   ├── processed/          # 结构化后的案例数据
│   └── law_database/       # 法条数据
├── models/
│   ├── llm/                # LawLLM-7B 权重
│   └── embedding/          # BGE-M3 权重
├── src/
│   ├── config.py           # 配置管理
│   ├── data_processing/    # 数据处理模块
│   │   ├── case_parser.py  # 裁判文书解析
│   │   └── vector_store.py # 向量库管理
│   ├── retrieval/          # 检索模块
│   │   ├── embed.py        # 向量化
│   │   ├── search.py       # 多路检索
│   │   └── rerank.py       # 重排序
│   ├── llm/                # 大模型模块
│   │   ├── inference.py    # 推理封装
│   │   └── prompts.py      # Prompt 模板
│   ├── pipeline.py         # 主流程编排
│   └── main.py             # 入口
├── notebooks/              # Kaggle/实验 notebook
├── configs/
│   └── config.yaml
├── tests/
├── requirements.txt
└── README.md
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 基座模型 | LawLLM-7B (Qwen2.5-7B-Instruct) |
| Embedding | BGE-M3 |
| Rerank | BGE-reranker-v2-m3 |
| 向量数据库 | Qdrant |
| RAG 框架 | LlamaIndex |
| 推理框架 | vLLM |
| 微调框架 | LLaMA-Factory |

## 流程

```
用户输入案情 → 要素抽取(LawLLM) → 多路检索(向量+BM25) → 重排序 → 案例分析生成(LawLLM) → 输出报告
```

## 免责声明

本系统输出的所有内容仅供参考，不构成专业法律意见。