from openai import OpenAI


class OpenAIChatClient:
    """OpenAI official API or any OpenAI-compatible local server (vLLM, SGLang, ...), typically serving Qwen."""

    def __init__(self, model, logger, args, openai_base_url=None):
        self.args = args
        self.logger = logger
        self.model = model
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.label = "OpenAI" if not openai_base_url else "OpenAI-Compatible(local)"

        self.tokens_dict = {
            "total_completion_tokens": 0,
            "total_prompt_tokens": 0,
            "total_total_tokens": 0,
        }

        client_kw = {"api_key": args.api_key}
        if openai_base_url:
            client_kw["base_url"] = openai_base_url.rstrip("/")
        self.client = OpenAI(**client_kw)

    def employ(
        self,
        SystemPrompt,
        UserPrompt,
        name="default",
        *,
        temperature=None,
        top_p=None,
        max_tokens=None,
        response_format=None,
    ):
        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": UserPrompt},
            ],
        }
        if temperature is not None:
            req["temperature"] = temperature
        if top_p is not None:
            req["top_p"] = top_p
        if max_tokens is not None:
            req["max_tokens"] = int(max_tokens)
        elif getattr(self.args, "client", "") == "qwen_local" and getattr(
            self.args, "max_output_length", 0
        ):
            req["max_tokens"] = int(self.args.max_output_length)
        if response_format is not None:
            req["response_format"] = response_format

        response = self.client.chat.completions.create(**req)
        text = response.choices[0].message.content
        text = text if text is not None else ""

        self.logger.info(f"{name} (use {self.label}):")
        usage = getattr(response, "usage", None)
        if usage is not None:
            c, p, t = usage.completion_tokens, usage.prompt_tokens, usage.total_tokens
            self.logger.info(
                f"completion_tokens: {c}. prompt_tokens: {p}. total_tokens: {t}.\n"
            )
            self.completion_tokens += c
            self.prompt_tokens += p
            self.total_tokens += t
            self.tokens_dict["total_completion_tokens"] = self.completion_tokens
            self.tokens_dict["total_prompt_tokens"] = self.prompt_tokens
            self.tokens_dict["total_total_tokens"] = self.total_tokens
            if name not in self.tokens_dict:
                self.tokens_dict[name] = {
                    "completion_tokens": c,
                    "prompt_tokens": p,
                    "total_tokens": t,
                }
            else:
                self.tokens_dict[name]["completion_tokens"] += c
                self.tokens_dict[name]["prompt_tokens"] += p
                self.tokens_dict[name]["total_tokens"] += t
        else:
            self.logger.info("(usage not returned by server)\n")

        return text
