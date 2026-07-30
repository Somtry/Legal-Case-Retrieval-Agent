"""模型推理封装：支持本地 vLLM 推理和 API 调用两种模式。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config


class LLMInference:
    """大模型推理封装，支持本地 vLLM 和 API 两种模式。"""

    def __init__(self, mode: str = "local") -> None:
        """初始化推理引擎。

        Args:
            mode: "local" 使用 vLLM 本地推理，"api" 使用远程 API。
        """
        self.mode = mode
        self._model = None
        self._tokenizer = None

    def _load_local(self) -> None:
        """加载本地 vLLM 模型。"""
        from transformers import AutoTokenizer
        from vllm import LLM

        model_path = config.model.get("llm_path", "ShengbinYue/LawLLM-7B")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = LLM(
            model=model_path,
            quantization="awq",  # T4 16GB 需要量化
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        logger.info(f"已加载本地模型: {model_path}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """生成回复。

        Args:
            prompt: 输入 prompt。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度，法律场景建议较低值。

        Returns:
            模型生成的文本。
        """
        if self.mode == "local":
            return self._generate_local(prompt, max_tokens, temperature)
        else:
            return self._generate_api(prompt, max_tokens, temperature)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """本地 vLLM 推理。"""
        if self._model is None:
            self._load_local()

        from vllm import SamplingParams

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        outputs = self._model.generate([prompt], sampling)
        return outputs[0].outputs[0].text

    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """API 调用推理。

        TODO: Phase 3 补充 API 调用逻辑（如使用云端部署的模型）。
        """
        raise NotImplementedError("API 模式待 Phase 3 实现")