"""case_parser 模块测试。"""

from src.data_processing.case_parser import (
    deduplicate,
    parse_cail_record,
    validate,
)


def test_parse_cail_record():
    """验证 CAIL 记录解析。"""
    record = {
        "case_id": "test_001",
        "fact": "被告人张某盗窃他人财物",
        "accusation": "盗窃罪",
        "relevant_articles": ["刑法第二百六十四条"],
        "imprisonment": 3,
    }
    parsed = parse_cail_record(record)
    assert parsed["case_type"] == "刑事"
    assert parsed["cause_of_action"] == "盗窃罪"
    assert parsed["facts"] == "被告人张某盗窃他人财物"
    assert "刑法第二百六十四条" in parsed["legal_basis"]


def test_deduplicate():
    """验证去重逻辑。"""
    cases = [
        {"case_id": "001", "facts": "a"},
        {"case_id": "001", "facts": "b"},  # 重复
        {"case_id": "002", "facts": "c"},
    ]
    result = deduplicate(cases)
    assert len(result) == 2
    assert result[0]["case_id"] == "001"


def test_validate():
    """验证数据校验。"""
    assert validate({"case_id": "001", "case_type": "刑事", "facts": "..."})
    assert not validate({"case_id": "002", "case_type": "", "facts": "..."})
    assert not validate({"case_id": "", "case_type": "刑事", "facts": "..."})