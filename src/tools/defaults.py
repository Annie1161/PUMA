"""Unified defaults shared by the agent and the simulator (single source of truth).

Downstream entry points (e.g. CAMI's agents/llm_client.py and generate.py) fall back to these constants when FEP_LLM_CLIENT / FEP_LLM_MODEL are not set.
"""

import os

# Default to a small Qwen-3 for one-command local smoke runs (override via environment variables).
DEFAULT_QWEN_HF_MODEL = os.environ.get(
    "FEP_DEFAULT_QWEN_HF_MODEL",
    "Qwen/Qwen3-0.6B",
)

# Default to tools.qwen_hf_client (local Transformers), matching the default backbone.
DEFAULT_LLM_CLIENT = os.environ.get("FEP_DEFAULT_LLM_CLIENT", "qwen_hf")
