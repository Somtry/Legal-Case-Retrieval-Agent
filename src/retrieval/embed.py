"""向量化模块：使用 BGE-M3 对文本进行 embedding。

支持批量向量化案例数据和查询文本。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config


class Embedder:
    """BGE-M3 embedding 封装。"""

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        """惰性加载 BGE-M3 模型。"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            model_path = config.model.get("embedding_path", "BAAI/bge-m3")
            self._model = BGEM3FlagModel(model_path, use_fp16=True)
            logger.info(f"已加载 embedding 模型: {model_path}")
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量生成 embedding。

        Args:
            texts: 待向量化的文本列表。
            batch_size: 批量大小。

        Returns:
            向量列表，每个向量维度为 1024。
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
        )["dense_vecs"]
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """对单条文本生成 embedding。"""
        return self.embed([text])[0]