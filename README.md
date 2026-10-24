# PUMA: Prospective User-state Modeling for Action Selection

[English](README.md) | [简体中文](README.zh-CN.md)

PUMA is a research framework for motivational interviewing (MI) dialogues. It models a
client's readiness to change, predicts how that state may evolve under different counselor
actions, and uses expected free energy (EFE) to guide action selection.

The implementation separates **state inference**, **action planning**, and **response
generation**, with dialogue memory shared across turns. It includes a profile-grounded
patient simulator and tools for evaluating both state predictions and counseling rollouts.

[Getting started](#getting-started) · [Data](#data) · [Evaluation](#evaluation)

## Overview

At each dialogue turn, PUMA:

1. Infers a posterior over the client's latent state from the dialogue and persona context.
2. Uses a transition model to predict the effects of candidate counselor actions.
3. Selects an action through EFE scoring and generates a natural-language response.
4. Updates its memory and carries state information into the next turn.

The readiness-stage vocabulary is **precontemplation → contemplation → preparation**.
The offline evaluator also supports MI talk-type labels: **neutral**, **change**, and
**sustain**. Observation-focused inference and fusion with the planner prior are configurable;
select them explicitly when comparing experimental settings.

The repository includes:

- **PUMA agent:** persona modeling, transition planning, response generation, and memory.
- **Patient simulator:** profile-grounded responses, semantic trigger matching, and
  per-profile action distributions.
- **Evaluation tools:** offline state evaluation, interactive counseling rollouts, and
  model-based judging.
- **Behavior annotation:** MISC counselor-action labels and annotation tools.

## Getting started

### 1. Set up the project

```bash
git clone https://github.com/Annie1161/PUMA.git
cd PUMA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

[requirements.txt](requirements.txt) is the bundled environment snapshot. It pins
`torch==1.13.0+cu117` and `transformers==4.37.0`, among other packages. These pins are
**not a validated Qwen3 environment**. The Qwen3 examples below require a compatible
PyTorch, Transformers, and CUDA setup; reconcile the snapshot with that setup before
installing dependencies. The CUDA-specific wheels may require a matching package index.

For an environment using the bundled pins:

```bash
python -m pip install -r requirements.txt
```

The memory module also loads the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Run subsequent commands from the repository root. Make the shared modules and agent
entry points importable in the current shell:

```bash
export PYTHONPATH="$PWD/src:$PWD/src/agent${PYTHONPATH:+:$PYTHONPATH}"
```

### 2. Choose a local model

Inference uses a local Hugging Face causal language model. Pass `--model` with a
model identifier or an existing local model directory. The agent defaults to
`Qwen/Qwen3-0.6B`; the rollout evaluator defaults to `Qwen/Qwen3-8B`.
The agent default can also be overridden with the `PUMA_MODEL` environment variable.

Local inference is implemented in `src/agent/counselor.py`. Dynamic evaluation shares
the loaded model between the counselor and patient simulator.

## Data

Prepare datasets separately and follow their original access and license terms. The
examples below assume this local layout:

```text
data/
└── cami/
    ├── profiles.jsonl
    └── profiles_misc.jsonl
```

CAMI profiles use one JSON object per line. Dialogue fields include `speakers`,
`utterances`, `states`, `talk_types`, and `actions`; persona fields include `Personas`,
`Beliefs`, `Motivation`, and `Acceptable Plans`. The population statistics additionally
use `misc_actions` for counselor MISC labels. Keep per-turn arrays aligned.

Provide prepared transcripts and annotated profile files in the formats above.

Pass data paths and evaluation settings explicitly through CLI arguments.

## Evaluation

These examples assume the model environment and data above are available. Start with a
small profile count before running a full evaluation.

### Offline state evaluation

Evaluate state predictions against a fixed CAMI transcript:

```bash
python src/agent/main.py \
    --dataset cami \
    --data_path data/cami \
    --data_name profiles.jsonl \
    --model Qwen/Qwen3-8B \
    --state_label_space cami_stage \
    --persona_state_space cami_stage \
    --observation_focused_state True \
    --prior_fusion_mode soft_fusion \
    --test_num 5 \
    --results_file logs/results/puma_static.json
```

### Dynamic rollout with patient simulator

Run counseling rollouts with a maximum of 20 turns per profile:

```bash
python src/tools/eval_simple_sim.py \
    --profiles_misc_path data/cami/profiles_misc.jsonl \
    --model Qwen/Qwen3-8B \
    --test_num 5 \
    --max_turns 20 \
    --seed 42 \
    --prior_fusion_mode soft_fusion \
    --output_json logs/results/simple_sim_puma_n5.json
```

The output contains aggregate metrics and per-profile trajectories, including stage lift,
preparation rate, turn count, trigger coverage, and action distribution. The evaluator
runs the PUMA counselor with the local patient simulator.
