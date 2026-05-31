#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible server using HuggingFace transformers.
Loads model once, handles concurrent requests via threading.

Usage:
  python scripts/hf_server.py --model Qwen/Qwen3-8B --port 8100 --gpu 0
"""

import argparse
import json
import re
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelServer:
    def __init__(self, model_name, gpu=0, max_input=4096, max_output=1024):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[Server] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=dtype, device_map=self.device
        )
        self.model.eval()
        self.max_input = max_input
        self.max_output = max_output
        self.model_name = model_name
        self._lock = threading.Lock()
        self._request_count = 0
        print(f"[Server] Model loaded. Ready.")

    def generate(self, messages, temperature=None, top_p=None, max_tokens=None):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input
        ).to(self.device)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_new = int(max_tokens) if max_tokens else self.max_output

        gen_kw = {"max_new_tokens": max_new, "pad_token_id": pad_id}
        if temperature is not None and temperature > 0:
            gen_kw["temperature"] = float(temperature)
            gen_kw["do_sample"] = True
        if top_p is not None:
            gen_kw["top_p"] = float(top_p)

        with self._lock:
            self._request_count += 1
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        return text, in_len, len(gen_ids)


server_instance = None


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == "/v1/models":
            resp = {
                "data": [{"id": server_instance.model_name, "object": "model"}]
            }
            self._json_response(200, resp)
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json_response(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        messages = body.get("messages", [])
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        max_tokens = body.get("max_tokens")

        try:
            text, prompt_tokens, completion_tokens = server_instance.generate(
                messages, temperature, top_p, max_tokens
            )
        except Exception as e:
            self._json_response(500, {"error": str(e)})
            return

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": server_instance.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
        self._json_response(200, resp)

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread (they serialize on the model lock)."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self.finish_request, args=(request, client_address))
        t.daemon = True
        t.start()


def main():
    global server_instance

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_input", type=int, default=4096)
    parser.add_argument("--max_output", type=int, default=1024)
    args = parser.parse_args()

    server_instance = ModelServer(args.model, args.gpu, args.max_input, args.max_output)

    httpd = ThreadedHTTPServer(("0.0.0.0", args.port), RequestHandler)
    print(f"[Server] Listening on port {args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
