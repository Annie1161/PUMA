from .defaults import DEFAULT_LLM_CLIENT, DEFAULT_QWEN_HF_MODEL
from .factory import build_llm_client
from .openai_client import OpenAIChatClient

__all__ = [
    "DEFAULT_LLM_CLIENT",
    "DEFAULT_QWEN_HF_MODEL",
    "build_llm_client",
    "OpenAIChatClient",
]


def __getattr__(name):
    if name == "QwenHFClient":
        from .qwen_hf_client import QwenHFClient

        return QwenHFClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
