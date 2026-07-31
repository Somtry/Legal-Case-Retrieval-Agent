# Legal Case Retrieval Agent - 项目计划

## 当前进度（2026-07-30 更新）

| Phase | 状态 | 完成度 |
|-------|------|--------|
| Phase 0: 项目初始化 | ✅ 完成 | 100% |
| Phase 1: 数据收集与结构化 | ✅ 基本完成 | 80%（CAIL2018 已集成，DISC-Law/AppealCase 解析器已实现待下载） |
| Phase 2: RAG 检索核心 | ✅ 完成 | 90%（向量+BM25 两路检索，要素加权精排；结构化筛选未实现） |
| Phase 3: LawLLM-7B 集成 | ✅ 完成 | 100%（AutoDL 4090 上跑通完整 pipeline） |
| Phase 4: LoRA 微调 | ⬜ 未开始 | 0% |
| Phase 5: 评测优化与集成 | ⬜ 部分完成 | 20%（README 已写，评测集/指标/UI 未做） |

**已验证环境**：AutoDL RTX 4090 24GB，500 条 CAIL2018 数据，完整 pipeline 约 50s（含模型加载）。

---

## 项目概述

构建一个法律案例检索助手（Legal Case Retrieval Agent），用户输入案情描述后，系统能够：
1. 理解案情，抽取关键法律要素
2. 从裁判文书数据库中检索相似案例
3. 对检索结果进行重排序
4. 生成案例分析报告（相似性分析 + 裁判观点对比 + 法律建议）

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 基座模型 | LawLLM-7B (基于 Qwen2.5-7B-Instruct) | 复旦开源法律大模型，2025年更新 |
| Embedding 模型 | BGE-M3 (BAAI/bge-m3) | 中文语义检索效果最好的开源模型 |
| Rerank 模型 | BGE-reranker-v2-m3 | 中文重排序模型 |
| 向量数据库 | Qdrant (本地 Docker) 或 Milvus Lite | 轻量级，适合中小规模数据 |
| 微调框架 | LLaMA-Factory | 支持 LoRA/全量微调，文档完善 |
| 推理框架 | vLLM | 高速推理，支持 Qwen 系列 |
| RAG 框架 | LlamaIndex | 专注 RAG 场景，比 LangChain 更聚焦 |
| 开发语言 | Python 3.10+ | |

## GPU 资源策略

| 阶段 | 硬件 | 用途 | 预估时长 | 成本 |
|------|------|------|----------|------|
| 开发调试 | 本地 Mac | 写代码、处理数据、构建向量库 | 日常 | 0 |
| 模型推理 | Kaggle T4×2 (免费) | 跑 LawLLM-7B 量化推理 | 按需 | 0 |
| LoRA 微调 | Kaggle T4×2 (免费) | 7B 模型 LoRA 微调 | 2-3 小时/次 | 0 |
| 全量微调（可选） | AutoDL RTX 4090 | 7B 全量微调，速度更快 | 2-4 小时/次 | ~2.68 元/小时 |
| 向量化 | Kaggle T4 或 CPU | 批量生成 embedding | 1-2 小时 | 0 |

**选型理由**：

- **主力：Kaggle T4×2（免费）**——16GB×2 = 32GB 总显存，足够跑 7B 模型的 AWQ 量化推理和 LoRA 微调。每周 30 小时免费额度，微调 3-5 次完全够用。
- **备选：AutoDL RTX 4090（2.68 元/小时）**——当 Kaggle 额度用尽或需要更快速度时使用。单卡 24GB 显存够跑 7B LoRA，FP16 算力 165 TFLOPS，速度约为 T4 的 3-4 倍。
- **不选 A100 的原因**：7B 模型 LoRA 微调仅需约 20-24GB 显存，4090 的 24GB 刚好满足，而价格只有 A100 40GB（3.45 元/小时）的 45%。A100 对 7B 模型来说性能过剩。
- **全量微调场景**：如果 LoRA 效果遇到瓶颈需要全量微调，7B 模型需要约 35-40GB 显存，此时选 AutoDL RTX 4090 配合梯度累加，或租 A100 40GB（3.45 元/小时）跑 4-6 小时，成本约 14-21 元。

## 系统架构

```
用户输入案情
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1: 案情要素抽取 (LawLLM-7B)           │
│  ├─ 案件类型分类（民事/刑事/行政）           │
│  ├─ 案由识别                                 │
│  ├─ 争议焦点提取                             │
│  ├─ 关键事实结构化                           │
│  └─ 适用法律初步判断                         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Stage 2: 多路检索                           │
│  ├─ 路径A: 向量检索（案情语义相似）         │
│  │   └─ BGE-M3 embedding → Qdrant Top 50    │
│  ├─ 路径B: 关键词检索（法条/案由精确匹配）  │
│  │   └─ BM25 / Elasticsearch Top 50         │
│  └─ 路径C: 结构化筛选（案由+法院+时间）      │
│      └─ SQL 过滤                             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Stage 3: 重排序 (Rerank)                    │
│  ├─ BGE-reranker 粗排                        │
│  └─ 法律要素加权精排                         │
│      ├─ 案由匹配 (×40)                       │
│      ├─ 争议焦点相似度 (×30)                 │
│      ├─ 事实要素重叠度 (×20)                 │
│      ├─ 同地区加成 (×5)                      │
│      └─ 时间近度 (×5)                        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Stage 4: 案例分析生成 (LawLLM-7B)           │
│  ├─ 检索到的 Top 5 案例摘要                  │
│  ├─ 相似性分析（为什么相似）                 │
│  ├─ 裁判观点对比                             │
│  ├─ 法律建议                                 │
│  └─ 免责声明                                 │
└─────────────────────────────────────────────┘
```

## 分阶段计划

---

### Phase 0: 项目初始化与环境搭建（1-2天）

**目标**：搭建项目骨架，准备好开发环境

**任务清单**：

- [x] 0.1 初始化项目结构
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
  │   ├── __init__.py
  │   ├── config.py           # 配置管理
  │   ├── data_processing/    # 数据处理模块
  │   │   ├── __init__.py
  │   │   ├── case_parser.py  # 裁判文书解析
  │   │   └── vector_store.py # 向量库管理
  │   ├── retrieval/          # 检索模块
  │   │   ├── __init__.py
  │   │   ├── embed.py        # 向量化
  │   │   ├── search.py       # 多路检索
  │   │   └── rerank.py       # 重排序
  │   ├── llm/                # 大模型模块
  │   │   ├── __init__.py
  │   │   ├── inference.py    # 推理封装
  │   │   └── prompts.py      # Prompt 模板
  │   ├── pipeline.py         # 主流程编排
  │   └── main.py             # 入口
  ├── notebooks/              # Kaggle/实验 notebook
  │   ├── 01_test_lawllm.ipynb
  │   ├── 02_build_vector_store.ipynb
  │   └── 03_lora_finetune.ipynb
  ├── configs/
  │   └── config.yaml
  ├── tests/
  ├── requirements.txt
  └── README.md
  ```

- [x] 0.2 创建 Python 虚拟环境
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  ```

- [x] 0.3 安装基础依赖
  ```
  # requirements.txt
  # LLM 推理
  vllm>=0.6.0
  transformers>=4.44.0
  torch>=2.4.0

  # RAG
  llama-index>=0.11.0
  qdrant-client>=1.11.0

  # Embedding & Rerank
  sentence-transformers>=3.0.0
  FlagEmbedding>=1.2.0

  # 数据处理
  pandas>=2.0.0
  datasets>=2.20.0
  jieba>=0.42.1

  # 工具
  pyyaml>=6.0
  tqdm>=4.66.0
  loguru>=0.7.0
  ```

- [x] 0.4 初始化 Git 仓库
  ```bash
  git init
  echo "venv/" > .gitignore
  echo "models/" >> .gitignore
  echo "data/raw/" >> .gitignore
  echo "__pycache__/" >> .gitignore
  ```

- [x] 0.5 编写 config.yaml
  ```yaml
  model:
    llm_path: "ShengbinYue/LawLLM-7B"
    embedding_path: "BAAI/bge-m3"
    reranker_path: "BAAI/bge-reranker-v2-m3"

  vector_store:
    type: "qdrant"
    host: "localhost"
    port: 6333
    collection_name: "legal_cases"
    vector_dim: 1024  # BGE-M3 输出维度

  retrieval:
    top_k_vector: 50
    top_k_keyword: 50
    top_k_final: 5

  rerank:
    weights:
      cause_of_action: 40
      focus_similarity: 30
      fact_overlap: 20
      same_region: 5
      recency: 5
  ```

**交付物**：可运行的项目骨架 + 依赖安装完成

---

### Phase 1: 数据收集与裁判文书结构化（3-5天）

**目标**：获取裁判文书数据，结构化后可用于检索

**数据来源策略**（优先级从高到低）：

1. **CAIL 数据集**（268万篇刑事文书，已结构化）
   - GitHub: china-ai-law-challenge/CAIL
   - 字段：fact（事实）、accusation（罪名）、relevant_articles（法条）、刑期
   - 优势：即开即用，无需清洗

2. **DISC-Law-SFT 数据集**（30万条微调数据）
   - GitHub: FudanDISC/DISC-LawLLM
   - 包含"类案匹配"数据 8K 条，可直接用于评测
   - 包含"司法要素提取"数据 32K 条，可用于微调

3. **HuggingFace 开源数据集**
   - 搜索关键词：chinese legal, 裁判文书, court judgment
   - 优先找已结构化的 JSON/Parquet 格式

4. **法条数据库**
   - DISC-LawLLM 已整理 800+ 部法律法规
   - 或从国家法律法规数据库提取

**任务清单**：

- [x] 1.1 下载 CAIL2018 数据集（刑事领域，19.6万条小规模版本先试）
- [ ] 1.2 下载 DISC-Law-SFT 数据集（微调数据 + 类案匹配数据）
- [ ] 1.3 搜索并下载 HuggingFace 上的民事裁判文书数据集
- [x] 1.4 编写 case_parser.py，统一数据格式：
  ```python
  # 统一后的案例数据结构
  {
      "case_id": "unique_id",
      "court": "法院名称",
      "case_type": "民事/刑事/行政",
      "cause_of_action": "案由",
      "facts": "案情事实描述",
      "claims": ["诉求列表"],
      "legal_basis": ["适用法条"],
      "ruling": "判决结果",
      "ruling_result": {"项": "值"},
      "date": "裁判日期",
      "region": "地区",
      "full_text": "完整文书文本"
  }
  ```
- [x] 1.5 数据质量检查：去重、去空、格式校验
- [x] 1.6 生成统计报告：各领域案例数量分布

**交付物**：结构化的裁判文书数据集（JSONL 格式）+ 数据统计报告

---

### Phase 2: 搭建 RAG 检索核心管线（3-5天）

**目标**：实现"案情输入 → 向量检索 → Top 5 相似案例输出"

**任务清单**：

- [x] 2.1 部署 Qdrant 向量数据库
  ```bash
  docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
  ```

- [x] 2.2 编写 embed.py：用 BGE-M3 批量向量化案例数据
  ```python
  # 核心逻辑
  from FlagEmbedding import BGEM3FlagModel
  model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

  # 分块策略：每篇文书拆分为多个 chunk
  # chunk 1: 案由 + 当事人信息
  # chunk 2: 案情事实（最重要）
  # chunk 3: 法院认为（裁判理由）
  # chunk 4: 法律依据 + 判决结果
  ```

- [x] 2.3 批量构建向量索引（在 Kaggle T4 上跑）
  - 先用 5000 条数据测试管线
  - 验证检索质量后再扩大到全量

- [x] 2.4 编写 search.py：多路检索
  - 向量检索：BGE-M3 → Qdrant Top 50
  - 关键词检索：jieba 分词 → BM25 Top 50
  - 结果合并去重

- [x] 2.5 编写 rerank.py：重排序
  - BGE-reranker 初排
  - 法律要素加权精排（案由匹配、争议焦点相似度等）

- [x] 2.6 端到端测试：输入一个案情，检查检索结果是否合理

**交付物**：可工作的 RAG 检索管线，输入案情输出 Top 5 相似案例

---

### Phase 3: 跑通 LawLLM-7B 并集成（2-3天）

**目标**：在 Kaggle T4 上跑通 LawLLM-7B，集成到检索流程中

**任务清单**：

- [x] 3.1 在 Kaggle 上测试 LawLLM-7B 推理
  - notebook: notebooks/01_test_lawllm.ipynb
  - 测试内容：法律问答、案情要素抽取、案例分析生成
  - T4 16GB 够跑 7B 模型的 AWQ 量化版本

- [x] 3.2 编写 inference.py：封装模型调用
  ```python
  # 支持两种模式：
  # 1. 本地 vLLM 推理（Kaggle/云 GPU）
  # 2. API 调用（如果后续换成云端 API）
  ```

- [x] 3.3 编写 prompts.py：设计核心 Prompt 模板
  - Prompt 1: 案情要素抽取
  - Prompt 2: 案例对比分析
  - Prompt 3: 法律建议生成

- [x] 3.4 集成到 pipeline.py，跑通完整流程：
  ```
  用户输入 → 要素抽取 → RAG检索 → Rerank → 案例分析生成 → 输出
  ```

- [x] 3.5 用 10-20 个测试用例验证端到端效果

**交付物**：完整可运行的案例检索 Agent

---

### Phase 4: LoRA 微调专项能力（3-5天）

**目标**：针对要素抽取和类案匹配微调 LawLLM-7B，提升效果

**任务清单**：

- [ ] 4.1 准备微调数据
  - 从 DISC-Law-SFT 中筛选：
    - 司法要素提取 32K 条
    - 类案匹配 8K 条
  - 根据你的数据格式调整 prompt 模板

- [ ] 4.2 在 Kaggle T4×2 上跑 LoRA 微调（免费）
  - 如果 Kaggle 额度用尽，切 AutoDL RTX 4090（2.68 元/小时，速度更快）
  - notebook: notebooks/03_lora_finetune.ipynb
  - 使用 LLaMA-Factory 或直接 transformers + peft
  - LoRA rank=8, alpha=16, target_modules: q_proj,v_proj

- [ ] 4.3 评估微调效果
  - 用 DISC-Law-SFT 的测试集做自动评估
  - 人工检查 20 个案例的要素抽取质量
  - 对比微调前后的检索准确率

- [ ] 4.4 如果 LoRA 效果不够，考虑：
  - 增大 LoRA rank 到 32/64
  - 租 AutoDL A100 40GB（3.45 元/小时）做全量微调，跑 4-6 小时约 14-21 元
  - 增加自己标注的领域数据

**交付物**：微调后的模型权重 + 评估报告

---

### Phase 5: 评测优化与系统集成（3-5天）

**目标**：系统性评测、优化、封装为可用的应用

**任务清单**：

- [ ] 5.1 构建评测集
  - 人工标注 50-100 个"案情 → 正确相似案例"对
  - 或直接用 DISC-Law-SFT 类案匹配测试集

- [ ] 5.2 定义评测指标
  - Recall@5: Top 5 中包含正确案例的比例
  - MRR: 正确案例的排名倒数
  - 案由分类准确率
  - 要素抽取 F1

- [ ] 5.3 优化迭代
  - 调整分块策略
  - 调整 Rerank 权重
  - 优化 Prompt
  - 扩大数据量

- [ ] 5.4 封装为可交互的应用
  - 方案A: Gradio Web UI（最简单）
  - 方案B: FastAPI 后端 + 前端
  - 方案C: CLI 工具

- [ ] 5.5 编写 README 和使用文档

**交付物**：可演示的法律案例检索 Agent + 评测报告

---

## 时间线总览

```
Week 1:
  ├─ Day 1-2: Phase 0 (项目初始化)
  ├─ Day 3-5: Phase 1 (数据收集与结构化)
  └─ Day 6-7: Phase 2 前半 (向量库搭建 + 批量索引)

Week 2:
  ├─ Day 8-10: Phase 2 后半 (多路检索 + Rerank)
  ├─ Day 11-12: Phase 3 (LawLLM-7B 集成)
  └─ Day 13-14: Phase 4 前半 (微调准备 + 训练)

Week 3:
  ├─ Day 15-17: Phase 4 后半 (评估微调效果)
  ├─ Day 18-20: Phase 5 (评测优化 + 系统封装)
  └─ Day 21: 文档 + 演示准备
```

**总计约 3 周**（按每天投入 3-4 小时估算）

## 关键里程碑

| 里程碑 | 标志 | 预计完成时间 |
|--------|------|-------------|
| M1: RAG 检索可用 | 输入案情能返回 Top 5 案例 | Week 1 末 |
| M2: 端到端跑通 | 从输入到分析报告完整流程 | Week 2 中 |
| M3: 微调模型上线 | 微调后的模型替代原始模型 | Week 2 末 |
| M4: 系统可演示 | 有 UI、有评测、有文档 | Week 3 末 |

## 风险与应对

| 风险 | 可能性 | 应对方案 |
|------|--------|----------|
| Kaggle GPU 时间不足（每周30h） | 中 | 切 AutoDL RTX 4090（2.68 元/小时），单卡够跑 7B LoRA |
| 裁判文书数据获取困难 | 中 | 先用 CAIL 数据集（已开源），后续补充 |
| 微调后效果提升不明显 | 中 | 优先优化 RAG 检索质量，微调是锦上添花 |
| T4 显存不够跑 7B | 低 | 用 AWQ 量化版，7B 量化后约 5-6GB；LoRA 微调用 T4×2 共 32GB 足够 |
| 需要全量微调 | 低 | 租 AutoDL A100 40GB（3.45 元/小时），跑 4-6 小时约 14-21 元 |
| 法律领域幻觉问题 | 高 | 强制 RAG 引用原文 + Prompt 约束 + 输出法条验证 |

## 注意事项

1. **先跑通再优化**：不要一开始就追求完美，先让端到端流程跑起来
2. **数据质量 > 模型大小**：500 篇高质量案例 > 10 万篇低质量数据
3. **RAG 是核心**：即使没有微调，好的 RAG 也能实现 80% 的效果
4. **评测驱动**：每个阶段都要有评测，不要靠感觉判断效果
5. **合规优先**：系统输出必须包含免责声明，不替代专业法律意见
