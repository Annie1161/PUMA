#!/usr/bin/env python3
"""
Parallel sample runner for FEP-Agent experiments.

Architecture:
  1. Start a vLLM server (OpenAI-compatible) serving Qwen3-8B
  2. Split sample_ids across N worker processes
  3. Each worker calls main.py with --client qwen_local pointing to the vLLM server
  4. Merge per-worker JSON results into a single output

Usage:
  python scripts/parallel_runner.py \
      --model Qwen/Qwen3-8B \
      --sample_ids "16,34,3,37,8" \
      --n_workers 5 \
      --gpu 0 \
      --output results/parallel_test.json \
      -- [extra args passed to main.py]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import socket
import urllib.request
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_vllm(url, timeout=300):
    """Wait until vLLM server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                data = json.loads(resp.read())
                print(f"[vLLM] Server ready. Models: {[m['id'] for m in data.get('data', [])]}")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_inference_server(model, gpu, port, max_model_len=4096, extra_vllm_args=None,
                           vllm_python=None, use_hf_server=False):
    """Start an OpenAI-compatible inference server (vLLM or lightweight HF)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if use_hf_server:
        # Use lightweight HF server (no vLLM dependency)
        cmd = [
            sys.executable, "-u",
            os.path.join(os.path.dirname(__file__), "hf_server.py"),
            "--model", model,
            "--port", str(port),
            "--gpu", "0",  # remapped by CUDA_VISIBLE_DEVICES
            "--max_input", str(max_model_len),
        ]
        label = "HF-Server"
    else:
        python_bin = vllm_python or sys.executable
        cmd = [
            python_bin, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--dtype", "half",
            "--gpu-memory-utilization", "0.90",
            "--disable-log-requests",
            "--enable-prefix-caching",
        ]
        if extra_vllm_args:
            cmd.extend(extra_vllm_args)
        label = "vLLM"

    log_path = f"logs/{label.lower()}_server_{port}.log"
    os.makedirs("logs", exist_ok=True)
    log_f = open(log_path, "w")

    print(f"[{label}] Starting server: {' '.join(cmd)}")
    print(f"[{label}] Log: {log_path}")

    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def run_single_worker(worker_id, sample_ids_str, port, model, extra_args, output_json):
    """Run main.py for a subset of samples using the inference server."""
    log_file = output_json.replace(".json", ".log")

    # Start with extra args, then override specific ones
    cmd = [
        sys.executable, "-u", "main.py",
    ] + extra_args + [
        "--client", "qwen_local",
        "--model", model,
        "--openai_base_url", f"http://localhost:{port}/v1",
        "--api_key", "dummy",
        "--sample_ids", sample_ids_str,
        "--results_file", output_json,
        "--log_file", log_file,
    ]

    print(f"[Worker {worker_id}] samples={sample_ids_str} -> {output_json}")
    result = subprocess.run(cmd, capture_output=False)
    return worker_id, sample_ids_str, result.returncode, output_json


def merge_results(worker_outputs, final_output):
    """Merge per-worker JSON results into a single file."""
    all_samples = []
    all_sample_avgs = []
    run_info = None

    for path in worker_outputs:
        if not os.path.exists(path):
            print(f"[Merge] WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        if run_info is None:
            run_info = data.get("run", {})
        samples = data.get("samples", [])
        all_samples.extend(samples)
        # Collect averages
        for s in samples:
            all_sample_avgs.append(s.get("sample_average", {}))

    if not all_samples:
        print("[Merge] ERROR: No samples to merge!")
        return

    # Sort by original_sample_id
    all_samples.sort(key=lambda s: s.get("original_sample_id", s.get("sample_id", 0)))

    # Recompute aggregate
    metric_keys = list(all_sample_avgs[0].keys()) if all_sample_avgs else []
    overall_avg = {}
    for k in metric_keys:
        vals = [sa[k] for sa in all_sample_avgs if k in sa and sa[k] is not None]
        overall_avg[k] = sum(vals) / len(vals) if vals else None

    # Reassign sample_id sequentially
    for i, s in enumerate(all_samples):
        s["sample_id"] = i

    merged = {
        "run": run_info,
        "aggregate": {
            "n_samples": len(all_samples),
            "overall_average": overall_avg,
        },
        "samples": all_samples,
    }

    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)
    with open(final_output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[Merge] {len(all_samples)} samples -> {final_output}")
    print(f"[Merge] Overall: " + "  ".join(f"{k}={v:.4f}" for k, v in overall_avg.items() if v is not None))


def main():
    parser = argparse.ArgumentParser(description="Parallel FEP-Agent runner with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--sample_ids", required=True, help="Comma-separated sample IDs")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of parallel workers (0=auto, one per sample)")
    parser.add_argument("--gpu", default="0", help="GPU id for vLLM server")
    parser.add_argument("--port", type=int, default=0, help="vLLM port (0=auto)")
    parser.add_argument("--output", required=True, help="Final merged output JSON path")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_args", nargs="*", default=[], help="Extra vLLM server args")
    parser.add_argument("--skip_vllm", action="store_true", help="Skip starting vLLM (assume already running)")
    parser.add_argument("--vllm_url", default=None, help="Existing vLLM server URL (implies --skip_vllm)")
    parser.add_argument("--vllm_python", default=None, help="Python binary for vLLM server (if different venv)")
    parser.add_argument("--use_hf_server", action="store_true",
                        help="Use lightweight HF server instead of vLLM (no extra deps needed)")

    args, extra = parser.parse_known_args()

    samples = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    n_workers = args.n_workers if args.n_workers > 0 else len(samples)
    n_workers = min(n_workers, len(samples))

    port = args.port if args.port > 0 else find_free_port()
    base_url = args.vllm_url or f"http://localhost:{port}/v1"

    # Split samples across workers
    chunks = [[] for _ in range(n_workers)]
    for i, s in enumerate(samples):
        chunks[i % n_workers].append(s)

    print(f"[Config] {len(samples)} samples, {n_workers} workers, GPU={args.gpu}, port={port}")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: samples {','.join(chunk)}")

    vllm_proc = None
    vllm_log_f = None

    try:
        # Step 1: Start vLLM server
        if not args.skip_vllm and not args.vllm_url:
            vllm_proc, vllm_log_f = start_inference_server(
                args.model, args.gpu, port, args.max_model_len, args.vllm_args,
                vllm_python=args.vllm_python, use_hf_server=args.use_hf_server
            )
            label = "HF-Server" if args.use_hf_server else "vLLM"
            timeout = 600 if not args.use_hf_server else 300
            print(f"[{label}] Waiting for server (pid={vllm_proc.pid})...")
            if not wait_for_vllm(f"http://localhost:{port}", timeout=timeout):
                print(f"[{label}] ERROR: Server failed to start!")
                print(f"[{label}] Check log: logs/{label.lower()}_server_{port}.log")
                sys.exit(1)
        else:
            print(f"[vLLM] Using existing server at {base_url}")

        # Step 2: Launch workers in parallel
        worker_outputs = []
        futures = {}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                sample_ids_str = ",".join(chunk)
                worker_json = args.output.replace(".json", f"_w{i}.json")
                worker_outputs.append(worker_json)

                # Build extra args: remove separator and args we override
                override_keys = {"--sample_ids", "--results_file", "--log_file",
                                 "--client", "--openai_base_url", "--api_key", "--model"}
                cleaned = []
                skip_next = False
                for arg in extra:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg == "--":
                        continue
                    if arg in override_keys:
                        skip_next = True
                        continue
                    cleaned.append(arg)

                future = executor.submit(
                    run_single_worker, i, sample_ids_str, port, args.model,
                    cleaned, worker_json
                )
                futures[future] = i

            # Wait for all workers
            for future in as_completed(futures):
                wid, sids, rc, out = future.result()
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[Worker {wid}] {status} samples={sids}")

        # Step 3: Merge results
        merge_results(worker_outputs, args.output)

        # Clean up worker files
        for wf in worker_outputs:
            if os.path.exists(wf):
                os.remove(wf)
            wf_log = wf.replace(".json", ".log")
            # Keep worker logs for debugging

    finally:
        # Stop vLLM server
        if vllm_proc is not None:
            print(f"[vLLM] Stopping server (pid={vllm_proc.pid})")
            vllm_proc.send_signal(signal.SIGTERM)
            try:
                vllm_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
        if vllm_log_f is not None:
            vllm_log_f.close()


if __name__ == "__main__":
    main()
