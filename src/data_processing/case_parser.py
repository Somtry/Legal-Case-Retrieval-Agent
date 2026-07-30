"""裁判文书解析模块：将不同来源的原始数据统一为标准案例格式。

标准案例数据结构:
    {
        "case_id": str,
        "court": str,
        "case_type": "民事" | "刑事" | "行政",
        "cause_of_action": str,
        "facts": str,
        "claims": list[str],
        "legal_basis": list[str],
        "ruling": str,
        "ruling_result": dict,
        "date": str,
        "region": str,
        "full_text": str,
    }

支持的数据源:
    - CAIL2018: 刑事裁判文书（fact + meta 嵌套结构）
    - DISC-Law-SFT: 法律微调数据（input + output 格式）
    - AppealCase: 民事上诉案件（一审+二审配对结构）
"""

from __future__ import annotations

from typing import Any


def parse_cail_record(record: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """将 CAIL2018 数据集的一条记录转换为标准案例格式。

    CAIL2018 实际字段结构（扁平）:
        fact: str                        # 案情事实
        relevant_articles: list[int]    # 相关法条编号
        accusation: list[str]           # 罪名列表
        punish_of_money: float          # 罚金
        criminals: list[str]            # 被告人
        death_penalty: bool             # 是否死刑
        imprisonment: float             # 有期徒刑月数
        life_imprisonment: bool         # 是否无期徒刑

    注: 部分 CAIL 版本使用 meta 嵌套结构，此函数同时兼容两种格式。
    """
    # 兼容两种格式：扁平结构 或 meta 嵌套结构
    meta = record.get("meta", {})
    if meta:
        # 嵌套格式（较新版本）
        accusation = meta.get("accusation", [])
        articles = meta.get("relevant_articles", [])
        punish_of_money = meta.get("punish_of_money")
        death_penalty = meta.get("death_penalty", False) or meta.get("term_of_imprisonment", {}).get("death_penalty", False)
        life_imprisonment = meta.get("life_imprisonment", False) or meta.get("term_of_imprisonment", {}).get("life_imprisonment", False)
        imprisonment = meta.get("imprisonment") or meta.get("term_of_imprisonment", {}).get("imprisonment")
    else:
        # 扁平格式（HuggingFace 上的常见版本）
        accusation = record.get("accusation", [])
        articles = record.get("relevant_articles", [])
        punish_of_money = record.get("punish_of_money")
        death_penalty = record.get("death_penalty", False)
        life_imprisonment = record.get("life_imprisonment", False)
        imprisonment = record.get("imprisonment")

    # 罪名拼接
    if isinstance(accusation, list):
        cause = "、".join(accusation)
    else:
        cause = str(accusation)

    # 法条编号转换为可读字符串
    legal_basis = [f"刑法第{a}条" for a in articles] if isinstance(articles, list) else []

    # 判决结果描述
    ruling_parts: list[str] = []
    if death_penalty:
        ruling_parts.append("死刑")
    if life_imprisonment:
        ruling_parts.append("无期徒刑")
    if imprisonment is not None and imprisonment > 0:
        months = float(imprisonment)
        years = int(months // 12)
        rem_months = months % 12
        if years > 0 and rem_months > 0:
            ruling_parts.append(f"有期徒刑{years}年{int(rem_months)}个月")
        elif years > 0:
            ruling_parts.append(f"有期徒刑{years}年")
        else:
            ruling_parts.append(f"有期徒刑{int(rem_months)}个月")
    if punish_of_money and punish_of_money > 0:
        ruling_parts.append(f"罚金{punish_of_money}元")
    ruling = "，".join(ruling_parts) if ruling_parts else ""

    return {
        "case_id": f"cail_{index}",
        "court": "",
        "case_type": "刑事",
        "cause_of_action": cause,
        "facts": record.get("fact", ""),
        "claims": [],
        "legal_basis": legal_basis,
        "ruling": ruling,
        "ruling_result": {
            "imprisonment": imprisonment,
            "death_penalty": death_penalty,
            "life_imprisonment": life_imprisonment,
            "punish_of_money": punish_of_money,
        },
        "date": "",
        "region": "",
        "full_text": record.get("fact", ""),
    }


def parse_disc_law_record(record: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """将 DISC-Law-SFT 数据集的一条记录转换为标准案例格式。

    DISC-Law-SFT 字段结构:
        input: str   # 指令/问题（如"请提取以下案件的要素..."）
        output: str  # 模型回答（可能包含要素分析、类案匹配结果等）

    由于 DISC-Law-SFT 是指令微调数据而非标准裁判文书，
    此函数尽可能从 input/output 中提取结构化信息，
    无法提取的字段留空。
    """
    input_text = record.get("input", "")
    output_text = record.get("output", "")

    # 尝试从 input 中推断案件类型
    case_type = ""
    if any(kw in input_text for kw in ["罪", "刑事", "被告人", "犯罪"]):
        case_type = "刑事"
    elif any(kw in input_text for kw in ["合同纠纷", "侵权", "婚姻", "劳动", "民事", "原告", "被告"]):
        case_type = "民事"
    elif any(kw in input_text for kw in ["行政诉讼", "行政行为", "行政机关"]):
        case_type = "行政"

    # 尝试从 input 中提取案由（常见模式：涉及XX罪/XX纠纷）
    cause_of_action = ""
    for keyword in ["罪名", "案由", "罪", "纠纷"]:
        if keyword in input_text:
            # 简单截取 input 的前 100 字作为参考
            cause_of_action = input_text[:100]
            break

    return {
        "case_id": f"disc_{index}",
        "court": "",
        "case_type": case_type,
        "cause_of_action": cause_of_action,
        "facts": input_text,
        "claims": [],
        "legal_basis": [],
        "ruling": output_text,
        "ruling_result": {},
        "date": "",
        "region": "",
        "full_text": f"输入: {input_text}\n输出: {output_text}",
    }


def parse_appeal_case_record(record: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """将 AppealCase 数据集的一条记录转换为标准案例格式。

    AppealCase 字段结构:
        index: int                          # 案例索引
        first_instance: dict                # 一审信息
            claim: str                      # 诉求
            court_view: str                 # 法院观点
            fact_description: str           # 事实描述
        second_instance: dict              # 二审信息（结构同一审）
        judgment_reversal: bool             # 是否改判
        reasons_for_reversal: str           # 改判理由
        claims: list[dict]                  # 诉求列表
            claim: str
            judgment: str
        legal_provisions: list[dict]        # 法条列表
            article: str
            law: str
        new_information: bool               # 是否有新事实
    """
    first = record.get("first_instance", {})
    second = record.get("second_instance", {})

    # 法条
    legal_basis: list[str] = []
    provisions = record.get("legal_provisions", [])
    if isinstance(provisions, list):
        for p in provisions:
            if isinstance(p, dict):
                law = p.get("law", "")
                article = p.get("article", "")
                if law and article:
                    legal_basis.append(f"{law} {article}")
                elif article:
                    legal_basis.append(article)

    # 诉求列表
    claims: list[str] = []
    claim_list = record.get("claims", [])
    if isinstance(claim_list, list):
        for c in claim_list:
            if isinstance(c, dict):
                claims.append(c.get("claim", ""))
            elif isinstance(c, str):
                claims.append(c)

    # 事实描述（合并一审和二审）
    facts = first.get("fact_description", "")
    if second.get("fact_description"):
        facts += "\n二审事实: " + second["fact_description"]

    # 判决结果
    ruling = second.get("court_view", "") or first.get("court_view", "")
    if record.get("judgment_reversal"):
        ruling = f"二审改判。{ruling}"
        if record.get("reasons_for_reversal"):
            ruling += f" 改判理由: {record['reasons_for_reversal']}"

    return {
        "case_id": f"appeal_{record.get('index', index)}",
        "court": "",
        "case_type": "民事",
        "cause_of_action": "",  # AppealCase 无显式案由字段，可后续从 claims 推断
        "facts": facts,
        "claims": [c for c in claims if c],
        "legal_basis": legal_basis,
        "ruling": ruling,
        "ruling_result": {
            "judgment_reversal": record.get("judgment_reversal"),
            "new_information": record.get("new_information"),
        },
        "date": "",
        "region": "",
        "full_text": f"一审诉求: {first.get('claim', '')}\n"
        f"一审事实: {first.get('fact_description', '')}\n"
        f"一审法院观点: {first.get('court_view', '')}\n"
        f"二审法院观点: {second.get('court_view', '')}",
    }


# ── 数据源 → 解析函数映射 ──────────────────────────────────────────

PARSERS: dict[str, Any] = {
    "cail2018": parse_cail_record,
    "disc_law_sft": parse_disc_law_record,
    "appeal_case": parse_appeal_case_record,
}


def parse_record(source: str, record: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """根据数据源类型选择对应的解析函数。

    Args:
        source: 数据源名称（cail2018 / disc_law_sft / appeal_case）。
        record: 原始记录 dict。
        index: 记录序号，用于生成 case_id。

    Returns:
        标准格式的案例字典。
    """
    parser = PARSERS.get(source)
    if parser is None:
        raise ValueError(f"未知数据源: {source}，支持: {list(PARSERS.keys())}")
    return parser(record, index)


def deduplicate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 case_id 去重，保留第一条。"""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for case in cases:
        cid = case.get("case_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            result.append(case)
    return result


def validate(case: dict[str, Any]) -> bool:
    """校验案例数据是否包含必要字段。

    必须有 case_id、case_type 和非空 facts。
    """
    required = ["case_id", "case_type", "facts"]
    return all(case.get(field) for field in required)