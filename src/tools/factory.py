from __future__ import annotations

from typing import Any, Dict, Optional


def build_llm_client(
    args: Any,
    logger: Any,
    lora_map: Optional[Dict[str, str]] = None,
):
    """
    Build an LLM client based on args.client.
    For qwen_hf / chatglm, pass `lora_map` (value "default" means base model only).
    """
    client = args.client

    if client == "chatgpt":
        from .openai_client import OpenAIChatClient

        return OpenAIChatClient(args.model, logger, args, openai_base_url=None)

    if client == "qwen_local":
        from .openai_client import OpenAIChatClient

        return OpenAIChatClient(
            args.model, logger, args, openai_base_url=args.openai_base_url
        )

    if client in ("qwen_hf", "chatglm"):
        from .qwen_hf_client import QwenHFClient

        if lora_map is None:
            raise ValueError("lora_map is required for client qwen_hf / chatglm")
        return QwenHFClient(args.model, logger, args, lora_map=lora_map)

    raise ValueError(f"Unknown client: {client}")
