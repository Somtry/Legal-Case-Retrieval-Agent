"""case_parser 模块测试。"""

from src.data_processing.case_parser import (
    deduplicate,
    parse_appeal_case_record,
    parse_cail_record,
    parse_disc_law_record,
    parse_record,
    validate,
)


# ── CAIL2018 解析测试 ──────────────────────────────────────────────

def test_parse_cail_record_flat():
    """验证 CAIL2018 记录解析（扁平结构，HuggingFace 常见版本）。"""
    record = {
        "fact": "被告人张某盗窃他人财物，价值人民币5000元",
        "relevant_articles": [264],
        "accusation": ["盗窃罪"],
        "punish_of_money": 5000,
        "criminals": ["张某"],
        "death_penalty": False,
        "imprisonment": 36,  # 36个月 = 3年
        "life_imprisonment": False,
    }
    parsed = parse_cail_record(record, index=42)
    assert parsed["case_type"] == "刑事"
    assert parsed["cause_of_action"] == "盗窃罪"
    assert parsed["facts"] == "被告人张某盗窃他人财物，价值人民币5000元"
    assert "刑法第264条" in parsed["legal_basis"]
    assert parsed["case_id"] == "cail_42"
    assert parsed["ruling_result"]["imprisonment"] == 36
    assert "有期徒刑3年" in parsed["ruling"]
    assert "罚金5000元" in parsed["ruling"]


def test_parse_cail_record_nested():
    """验证 CAIL2018 记录解析（meta 嵌套结构）。"""
    record = {
        "fact": "被告人张某盗窃他人财物",
        "meta": {
            "accusation": ["盗窃罪"],
            "relevant_articles": [264],
            "punish_of_money": 0,
            "term_of_imprisonment": {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 12,
            },
        },
    }
    parsed = parse_cail_record(record)
    assert parsed["case_type"] == "刑事"
    assert parsed["cause_of_action"] == "盗窃罪"
    assert "刑法第264条" in parsed["legal_basis"]
    assert "有期徒刑1年" in parsed["ruling"]


def test_parse_cail_multiple_accusations():
    """验证 CAIL 多罪名拼接。"""
    record = {
        "fact": "被告人实施抢劫和盗窃",
        "meta": {
            "accusation": ["抢劫罪", "盗窃罪"],
            "relevant_articles": [263, 264],
            "term_of_imprisonment": {"imprisonment": 60, "death_penalty": False, "life_imprisonment": False},
        },
    }
    parsed = parse_cail_record(record)
    assert "抢劫罪" in parsed["cause_of_action"]
    assert "盗窃罪" in parsed["cause_of_action"]
    assert len(parsed["legal_basis"]) == 2


# ── DISC-Law-SFT 解析测试 ──────────────────────────────────────────

def test_parse_disc_law_record():
    """验证 DISC-Law-SFT 记录解析。"""
    record = {
        "input": "被告人李某犯故意伤害罪，请分析其刑事责任",
        "output": "根据刑法第二百三十四条，故意伤害他人身体的...",
    }
    parsed = parse_disc_law_record(record, index=10)
    assert parsed["case_id"] == "disc_10"
    assert parsed["case_type"] == "刑事"
    assert "故意伤害罪" in parsed["facts"]
    assert "刑法第二百三十四条" in parsed["ruling"]


def test_parse_disc_law_civil():
    """验证 DISC-Law-SFT 民事案件类型推断。"""
    record = {
        "input": "原告与被告因合同纠纷产生争议，请分析",
        "output": "根据合同法相关规定...",
    }
    parsed = parse_disc_law_record(record)
    assert parsed["case_type"] == "民事"


# ── AppealCase 解析测试 ─────────────────────────────────────────────

def test_parse_appeal_case_record():
    """验证 AppealCase 民事上诉案件解析。"""
    record = {
        "index": 1,
        "first_instance": {
            "claim": "请求判令被告偿还借款10万元",
            "court_view": "一审法院支持原告请求",
            "fact_description": "原告借给被告10万元，到期未还",
        },
        "second_instance": {
            "claim": "",
            "court_view": "二审维持原判",
            "fact_description": "",
        },
        "judgment_reversal": False,
        "claims": [{"claim": "偿还借款", "judgment": "支持"}],
        "legal_provisions": [{"article": "第六百六十七条", "law": "民法典"}],
        "new_information": False,
    }
    parsed = parse_appeal_case_record(record)
    assert parsed["case_id"] == "appeal_1"
    assert parsed["case_type"] == "民事"
    assert "10万元" in parsed["facts"]
    assert "偿还借款" in parsed["claims"]
    assert any("民法典" in lb for lb in parsed["legal_basis"])
    assert "二审维持原判" in parsed["ruling"]


def test_parse_appeal_case_reversal():
    """验证 AppealCase 改判案件。"""
    record = {
        "index": 2,
        "first_instance": {"fact_description": "一审事实", "court_view": "一审观点", "claim": "诉求"},
        "second_instance": {"fact_description": "", "court_view": "二审观点", "claim": ""},
        "judgment_reversal": True,
        "reasons_for_reversal": "事实认定不清",
        "claims": [],
        "legal_provisions": [],
        "new_information": True,
    }
    parsed = parse_appeal_case_record(record)
    assert "二审改判" in parsed["ruling"]
    assert "事实认定不清" in parsed["ruling"]
    assert parsed["ruling_result"]["judgment_reversal"] is True


# ── 通用函数测试 ───────────────────────────────────────────────────

def test_parse_record_dispatch():
    """验证 parse_record 能根据数据源选择正确的解析函数。"""
    cail = parse_record("cail2018", {"fact": "test", "meta": {"accusation": ["盗窃罪"], "relevant_articles": [], "term_of_imprisonment": {}}})
    assert cail["case_type"] == "刑事"

    appeal = parse_record("appeal_case", {"index": 0, "first_instance": {"fact_description": "test"}})
    assert appeal["case_type"] == "民事"


def test_parse_record_unknown_source():
    """验证未知数据源报错。"""
    try:
        parse_record("unknown", {})
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass


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
    assert not validate({"case_id": "003", "case_type": "刑事", "facts": ""})