"""配置模块测试。"""

from src.config import config


def test_config_loads():
    """验证 config.yaml 能正确加载。"""
    assert "llm_path" in config.model
    assert "embedding_path" in config.model
    assert config.vector_store["collection_name"] == "legal_cases"
    assert config.vector_store["vector_dim"] == 1024


def test_rerank_weights():
    """验证 rerank 权重配置。"""
    weights = config.rerank["weights"]
    assert weights["cause_of_action"] == 40
    assert weights["focus_similarity"] == 30
    assert sum(weights.values()) == 100