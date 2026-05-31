import re

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenHFClient:
    """Local Hugging Face backend (Qwen / Llama / etc., optional LoRA); exposes the same employ-style API as the other clients."""

    def __init__(self, base_model, logger, args, lora_map=None):
        self.args = args
        self.logger = logger
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.lora_map = lora_map or {}
        if torch.cuda.is_available():
            gpu_str = str(args.gpus)
            if "," in gpu_str:
                # Multi-GPU: use device_map="auto" for model parallelism
                self.device = "cuda:0"
                self._dtype = torch.float16
                self._device_map = "auto"
            else:
                self.device = f"cuda:{gpu_str}"
                self._dtype = torch.float16
                self._device_map = self.device
        else:
            self.device = "cpu"
            self._dtype = torch.float32
            self._device_map = "cpu"
            self.logger.info("CUDA unavailable; QwenHFClient is falling back to CPU + float32.")
        self.lora_paths = [p for p in self.lora_map.values() if p != "default"]

        self.model, self.tokenizer = self._load_base_model(base_model)
        if self.lora_paths:
            self.model = self._load_lora_params(self.lora_paths)

    def _load_base_model(self, base_model):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map=self._device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        return model, tokenizer

    def _load_lora_params(self, lora_paths):
        lora_model = self.model
        for lora_path in lora_paths:
            config = PeftConfig.from_pretrained(lora_path)
            lora_model = PeftModel.from_pretrained(
                self.model,
                model_id=lora_path,
                config=config,
                dtype=self._dtype,
                device_map=self._device_map,
                adapter_name=lora_path,
            )
        return lora_model

    def set_lora(self, module_name):
        if self.lora_map.get(module_name) not in (None, "default"):
            self.model.enable_adapter_layers()
            self.model.set_adapter(self.lora_map[module_name])
            self.logger.info(
                f"Applied LoRA adapter: {self.lora_map[module_name]}"
            )
        else:
            self.model.disable_adapter_layers()
            self.logger.info("Applied base Qwen (no LoRA).")

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return system + "\n" + user

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
        **_,
    ):
        if self.lora_paths:
            self.set_lora(name)

        prompt = self._build_prompt(SystemPrompt, UserPrompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_new = (
            int(max_tokens) if max_tokens is not None else int(self.max_output_length)
        )
        gen_kw = {"max_new_tokens": max_new, "pad_token_id": pad_id, "do_sample": False}
        if temperature is not None and float(temperature) > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = float(temperature)
        if top_p is not None and gen_kw.get("do_sample"):
            gen_kw["top_p"] = float(top_p)

        out = self.model.generate(**inputs, **gen_kw)
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Some models (e.g. Qwen-3) may emit a thinking block before the answer; strip it before evaluation.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
