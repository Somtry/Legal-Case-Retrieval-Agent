"""向量库管理模块：Qdrant collection 的创建、索引构建、批量写入。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config


class VectorStore:
    """Qdrant 向量库管理器。"""

    def __init__(self) -> None:
        self._client = None
        self._collection_name = config.vector_store.get("collection_name", "legal_cases")

    @property
    def client(self):
        """惰性初始化 Qdrant client。"""
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            self._client = QdrantClient(
                host=config.vector_store.get("host", "localhost"),
                port=config.vector_store.get("port", 6333),
            )
            self._distance = Distance.COSINE
            self._vector_dim = config.vector_store.get("vector_dim", 1024)
            self._VectorParams = VectorParams
        return self._client

    def create_collection(self, collection_name: str | None = None) -> None:
        """创建向量集合（如果不存在）。"""
        name = collection_name or self._collection_name
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=self._VectorParams(
                size=self._vector_dim,
                distance=self._distance,
            ),
        )
        logger.info(f"已创建集合: {name}")

    def upsert(
        self,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        collection_name: str | None = None,
    ) -> None:
        """批量写入向量和元数据。

        Args:
            embeddings: 向量列表，每个向量维度为 vector_dim。
            payloads: 与向量一一对应的元数据（案由、法院、日期等）。
        """
        from qdrant_client.http.models import PointStruct

        name = collection_name or self._collection_name
        points = [
            PointStruct(id=i, vector=emb, payload=payload)
            for i, (emb, payload) in enumerate(zip(embeddings, payloads))
        ]
        self.client.upsert(collection_name=name, points=points)
        logger.info(f"已写入 {len(points)} 条向量到 {name}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 Top-K 结果。"""
        name = collection_name or self._collection_name
        results = self.client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {"score": hit.score, "payload": hit.payload}
            for hit in results
        ]