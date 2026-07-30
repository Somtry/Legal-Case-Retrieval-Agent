"""模型推理封装：支持本地 vLLM 推理和远程 API 调用两种模式。

两种模式:
    - local: 在有 GPU 的机器上（如 AutoDL）用 vLLM 直接加载模型推理
    - api:   本地机器通过 HTTP 调用云端 vLLM 部署的 OpenAI 兼容 API

用法:
    # 本地模式（在 AutoDL GPU 机器上）
    llm = LLMInference(mode="local")

    # API 模式（在你的开发机上，调用 AutoDL 部署的服务）
    llm = LLMInference(mode="api")
    text = llm.generate("你好")
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import config


class LLMInference:
    """大模型推理封装，支持本地 vLLM 和 API 两种模式。

    mode="local": 在 GPU 机器上用 vLLM 直接推理（适合 AutoDL/Kaggle）
    mode="api":   通过 HTTP 调用远程 vLLM API（适合本地开发机调用云端服务）
    """

    def __init__(self, mode: str = "local") -> None:
        """初始化推理引擎。

        Args:
            mode: "local" 使用 vLLM 本地推理，"api" 使用远程 API。
        """
        self.mode = mode
        self._model = None
        self._tokenizer = None
        self._client = None  # OpenAI 兼容 client（api 模式）

    def _load_local(self) -> None:
        """加载本地 vLLM 模型（需要在有 GPU 的机器上运行）。"""
        from transformers import AutoTokenizer
        from vllm import LLM

        model_path = config.model.get("llm_path", "ShengbinYue/LawLLM-7B")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = LLM(
            model=model_path,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype="float16",
        )
        logger.info(f"已加载本地模型: {model_path}")

    def _get_client(self):
        """获取 OpenAI 兼容的 API client（惰性初始化）。"""
        if self._client is None:
            from openai import OpenAI

            api_config = config.api
            base_url = api_config.get("base_url", "")
            api_key = api_config.get("api_key", "EMPTY")

            if not base_url or "YOUR_PORT_ID" in base_url:
                raise ValueError(
                    "API 模式未配置。请在 configs/config.yaml 中设置 api.base_url "
                    "为 AutoDL 上 vLLM 服务的代理地址。"
                )

            self._client = OpenAI(base_url=base_url, api_key=api_key)
            logger.info(f"已连接远程 API: {base_url}")
        return self._client

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
        """本地 vLLM 推理（在 GPU 机器上运行）。"""
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
        """通过 HTTP 调用远程 vLLM API（OpenAI 兼容格式）。

        vLLM 启动时使用 --served-model-name 指定模型名，
        此处用 config.api.model_name 对应。
        """
        client = self._get_client()
        model_name = config.api.get("model_name", "ShengbinYue/LawLLM-7B")

        # Qwen2.5 系列支持 chat 格式，vLLM 的 OpenAI 兼容 API 会自动处理
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        return response.choices[0].message.content

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """多轮对话模式生成（支持 system/user/assistant 消息）。

        Args:
            messages: 消息列表，如 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。

        Returns:
            模型生成的文本。
        """
        if self.mode == "local":
            # 本地模式：用 tokenizer 的 chat template 拼接 prompt
            if self._model is None:
                self._load_local()

            from vllm import SamplingParams

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampling = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            outputs = self._model.generate([prompt], sampling)
            return outputs[0].outputs[0].text
        else:
            # API 模式：直接传 messages
            client = self._get_client()
            model_name = config.api.get("model_name", "ShengbinYue/LawLLM-7B")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            return response.choices[0].message.content