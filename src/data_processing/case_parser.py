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
"""

from __future__ import annotations

from typing import Any


def parse_cail_record(record: dict[str, Any]) -> dict[str, Any]:
    """将 CAIL 数据集的一条记录转换为标准案例格式。

    CAIL 字段: fact, accusation, relevant_articles, imprisonment, death_penalty, life_imprisonment
    """
    return {
        "case_id": record.get("case_id", ""),
        "court": record.get("court", ""),
        "case_type": "刑事",
        "cause_of_action": record.get("accusation", ""),
        "facts": record.get("fact", ""),
        "claims": [],
        "legal_basis": record.get("relevant_articles", []),
        "ruling": "",
        "ruling_result": {
            "imprisonment": record.get("imprisonment"),
            "death_penalty": record.get("death_penalty"),
            "life_imprisonment": record.get("life_imprisonment"),
        },
        "date": record.get("judgment_date", ""),
        "region": record.get("region", ""),
        "full_text": record.get("fact", ""),
    }


def parse_disc_law_record(record: dict[str, Any]) -> dict[str, Any]:
    """将 DISC-Law-SFT 数据集的一条记录转换为标准案例格式。

    TODO: 根据实际数据格式补充实现。
    """
    raise NotImplementedError("待 Phase 1 获取数据后补充")


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
    """校验案例数据是否包含必要字段。"""
    required = ["case_id", "case_type", "facts"]
    return all(case.get(field) for field in required)