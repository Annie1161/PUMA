# PUMA: Prospective User-state Modeling for Action selection

> **Anonymous code release for a NeurIPS 2026 submission under double-blind review.**
> Please do not redistribute or attribute. Any identifying metadata (commit history, author email, ...) has been scrubbed.

PUMA is an LLM agent for motivational-interviewing (MI) style counseling. It treats the client's
readiness-to-change stage as a latent variable, infers a posterior over that stage at every turn,
and selects an action that is consistent with both the inferred state and an internal expected-free-energy
score. The agent is a drop-in replacement for the q-filter inside a standard active-inference dialogue
loop; the rest of the loop (planner, generator, memory) is unchanged.

The codebase ships:

* the agent itself (state head, planner, generator, memory),
* the LLM-driven patient simulator used for dynamic evaluation,
* the offline evaluator over a frozen MI counseling benchmark,
* a small set of analysis scripts.

It is backbone-agnostic: any Hugging Face causal LM (e.g. Qwen-3 / Llama-3.1 8B instruction-tuned models)
or an OpenAI-compatible HTTP endpoint can be plugged in via the unified client interface in
`src/tools/defaults.py`.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/                       YAML run configs (datasets, prompt versions, domains)
│   ├── mi_run_config.yaml
│   ├── run_config_v1.yaml
│   ├── prompt_versions_mi.json
│   └── domains/
└── src/
    ├── agent/                     The PUMA counseling agent
    │   ├── main.py                Entry point for offline / interactive runs
    │   ├── config.py              CLI argument parser
    │   ├── interactive_cami_text_eval.py   Multi-mode counselor evaluator
    │   ├── path_setup.py          Adds src/ to sys.path so `tools` is importable
    │   ├── Module/                Agent modules
    │   │   ├── Clients.py             LLM client wrappers (OpenAI / HF / Qwen / Llama)
    │   │   ├── Personas.py            Persona / profile representation
    │   │   ├── Planner.py             EFE-style action planner
    │   │   ├── Generator.py           Response generator
    │   │   ├── EventMemory.py         Long-/short-term memory
    │   │   ├── MITIScorer.py          MITI-style behavior scoring
    │   │   └── prompts.py             Prompt templates
    │   ├── DataLoader/
    │   ├── nlgeval/               BLEU / METEOR / ROUGE / CIDEr metrics
    │   ├── scripts/               Data-conversion helpers + HF-server launcher
    │   └── Trainer/               Optional LoRA fine-tuning (not used in main results)
    └── tools/                     Shared infrastructure + simulators + evaluators
        ├── defaults.py            Backbone defaults (single source of truth)
        ├── factory.py             Factory for building clients / agents
        ├── persona_machine_json.py
        ├── openai_client.py       OpenAI-compatible client
        ├── qwen_hf_client.py      Local Hugging Face client
        ├── therapist_action.py    Action vocabulary
        ├── dynamic_simulator.py   Dynamic patient closures
        ├── simple_simulator.py    Simple offline simulator (base)
        ├── simple_simulator_v7.py Current production simulator
        ├── simulator_profiles.json
        ├── eval_simple_sim.py     Dynamic 20-turn rollout evaluator
        ├── eval_patient_simulator.py
        ├── eval_simulator_fidelity.py
        ├── eval_interactive_judge.py
        ├── misc_annotator.py      MISC behavior labeling
        └── analyze_ablation.py
```

---

## Installation

PUMA is plain Python with Hugging Face + PyTorch.

```bash
# Recommended: a clean Python 3.10+ environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` reflect the main-results environment
(`torch>=2.4`, `transformers>=4.50`). Newer minor versions usually work.

For GPU inference of an 8B instruction-tuned model in float16, a single
24 GB-class GPU is sufficient. Static-evaluation runs over the full benchmark
typically finish on one H100 in a few hours; dynamic evaluation is faster
because each rollout caps at 20 turns.

---

## Datasets

PUMA is evaluated on publicly available MI counseling corpora.
**Datasets are not redistributed here**; please obtain them from the original sources:

| Dataset | Source | Use |
|---------|--------|-----|
| **CAMI** | original release on Hugging Face / GitHub | Main static + dynamic benchmark |
| **AnnoMI** | original release on Hugging Face | Cross-corpus generalization, demonstration retrieval for some baselines |
| **MSC** | Multi-Session Chat | Optional long-term-dialogue control |

After downloading, place the transcripts under `data/` (or wherever you like)
and adjust paths in `configs/mi_run_config.yaml`. Helper scripts to convert
raw transcripts to the per-profile format expected by the agent live in
`src/agent/scripts/convert_*_to_profiles.py`.

---

## Quick start

### 1. Static state-inference evaluation

Evaluate the agent's posterior over the client's readiness-to-change stage
on a frozen counseling benchmark.

```bash
cd src/agent
python interactive_cami_text_eval.py \
    --doctor_modes fep \
    --client_simulator_mode profile_rules \
    --test_num 5 \
    --max_turns 12
```

Useful flags:

* `--doctor_modes fep,cami_text,base_counselor,diir,cos,cami_te,cami_full`
  — run multiple counselor modes side-by-side in one job.
* `--test_num -1` — evaluate over all profiles in the configured split.
* `--annomi_jsonl_path <path>` — required if `diir` is included in the modes
  (DIIR retrieves demonstrations from an AnnoMI transcript file).

Logs and per-profile JSON results are written under `src/agent/logs/`.

### 2. Dynamic 20-turn rollout

Run multi-turn counseling rollouts against the LLM-driven patient simulator.

```bash
cd src/tools
python eval_simple_sim.py \
    --mode fep \
    --test_num 29 \
    --max_turns 20 \
    --backbone qwen3-8b
```

Set `--backbone` to switch between locally hosted causal LMs (the default
client is `qwen_hf_client`). The same script can score every counselor mode
sequentially via `--mode all`.

### 3. Optional: launch a local HF inference server

If you prefer to serve the backbone over HTTP instead of loading it in-process:

```bash
python src/agent/scripts/hf_server.py --model_name Qwen/Qwen3-8B --port 8000
```

then point `defaults.py` at `openai_client` with `base_url=http://127.0.0.1:8000/v1`.

---

## How the agent is organized

The state-inference path is intentionally decoupled from action selection.
At each turn, conditioned on the running dialogue and a warmup persona summary,
PUMA invokes a focused LLM call whose only job is to return a posterior
distribution over readiness-to-change stages. This posterior is then fused
with the planner's transition prior using an utterance-length-aware
soft-fusion rule that gives more weight to the planner when the client's
utterance is too short to be informative. The fused posterior feeds the
expected-free-energy action selector, which produces the next counselor
action; a separate generator then realizes that action as natural language.

The implementation lives in:

* `src/agent/Module/Planner.py` — EFE-style planner
* `src/agent/Module/Personas.py` — persona / warmup summary
* `src/agent/Module/prompts.py` — focused state-head prompt
* `src/agent/interactive_cami_text_eval.py` — orchestration

Baselines (`base_counselor`, `cami_text`, `cami_full`, `diir`, `cos`, `cami_te`)
are implemented in the same evaluator so they share simulator, judge, and
metric code paths.

---

## Reproducibility

All main-result runs use greedy decoding (`do_sample=False`) and deterministic
CUDA kernels. The relevant flags are wired through `main.py` /
`interactive_cami_text_eval.py` — see the `torch.use_deterministic_algorithms`
and `torch.backends.cudnn.deterministic` calls there. The patient simulator
also uses greedy decoding for the evaluation reported in the paper.

---

## Citation

```bibtex
@inproceedings{anonymous2026puma,
  title  = {{PUMA}: Prospective User-state Modeling for Action selection},
  author = {Anonymous},
  booktitle = {Submitted to NeurIPS 2026},
  year   = {2026},
  note   = {Under double-blind review}
}
```

A non-anonymous citation will be added after the review period.

---

## License

Released under the MIT License (see `LICENSE`).
