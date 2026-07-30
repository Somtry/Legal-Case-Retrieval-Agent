"""Prompt 模板模块：案情要素抽取、案例对比分析、法律建议生成。

所有 prompt 使用 .format() 填充变量，保持可读性和可维护性。
"""

# ── Prompt 1: 案情要素抽取 ──────────────────────────────────────────

PROMPT_EXTRACT_ELEMENTS = """你是一位专业的法律分析师。请从以下案情描述中抽取关键法律要素，以 JSON 格式输出。

## 案情描述
{case_description}

## 抽取要求
请抽取以下要素：
1. case_type: 案件类型（民事/刑事/行政）
2. cause_of_action: 案由（如：买卖合同纠纷、故意伤害罪）
3. focus_of_dispute: 争议焦点（列表，列出核心争议点）
4. key_facts: 关键事实（列表，列出影响裁判的核心事实）
5. applicable_laws: 初步判断可能适用的法律条文

## 输出格式
```json
{{
    "case_type": "...",
    "cause_of_action": "...",
    "focus_of_dispute": ["...", "..."],
    "key_facts": ["...", "..."],
    "applicable_laws": ["...", "..."]
}}
```

请只输出 JSON，不要输出其他内容。"""


# ── Prompt 2: 案例对比分析 ──────────────────────────────────────────

PROMPT_CASE_ANALYSIS = """你是一位专业的法律分析师。请对以下案情和检索到的相似案例进行对比分析。

## 待分析案情
{case_description}

## 案情要素
{case_elements}

## 检索到的相似案例（Top {n_cases}）
{retrieved_cases}

## 分析要求
请按以下结构输出分析报告：

### 1. 相似性分析
逐一分析每个检索案例与待分析案情的相似之处，说明为什么这些案例具有参考价值。

### 2. 裁判观点对比
对比各案例的裁判观点，找出共性和差异：
- 法院对类似争议焦点的裁判倾向
- 影响裁判结果的关键因素
- 不同法院/地区可能的裁判差异

### 3. 法律建议
基于以上分析，给出初步法律建议：
- 类似案件的可能的裁判结果
- 需要准备的关键证据
- 需要注意的法律风险

### 免责声明
本分析由 AI 生成，仅供参考，不构成专业法律意见。请咨询执业律师获取专业法律服务。"""


# ── Prompt 3: 单案例相似性分析（用于 rerank 阶段辅助） ──────────────

PROMPT_SIMILARITY_CHECK = """请判断以下两个案例的相似程度，从 0 到 100 打分，并简要说明理由。

案例A（待查询）：
{query_case}

案例B（候选案例）：
{candidate_case}

请输出：
相似度分数: [0-100]
理由: [一句话说明]"""


def format_extract_elements_prompt(case_description: str) -> str:
    return PROMPT_EXTRACT_ELEMENTS.format(case_description=case_description)


def format_case_analysis_prompt(
    case_description: str,
    case_elements: str,
    retrieved_cases: str,
    n_cases: int = 5,
) -> str:
    return PROMPT_CASE_ANALYSIS.format(
        case_description=case_description,
        case_elements=case_elements,
        retrieved_cases=retrieved_cases,
        n_cases=n_cases,
    )