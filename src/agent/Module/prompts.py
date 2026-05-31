"""
Central registry of LLM prompt templates (Personas / EventMemory / Generator).
Edit this file first when changing the dialogue behavior.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Personas — PersonaExtraction (legacy single-call JSON)
# ---------------------------------------------------------------------------
PERSONA_EXTRACTION_SYS_TEMPLATE = (
    "You maintain a patient-side FEP-style bundle for motivational-interviewing dialogue. "
    "(1) From the patient's utterance, infer/update latent_mi.client_talk_type ∈ {{{state_values_enum}}} "
    "and describe p(o_t|s_t): how that state surfaces as text/behavior cues with confidence. "
    "(2) Treat the therapist's utterance as action a_{t-1} (discrete label from the provided vocabulary) and describe "
    "p(s_t|s_{t-1},a_{t-1}): expected shift in client_talk_type / behavior after that action. "
    "Reply with a single JSON object only, no markdown, no commentary."
)

PERSONA_EXTRACTION_USER = """Patient ({usr_name}) said (evidence for inferring s_t and for p(o_t|s_t)):
\"\"\"{patient_utt}\"\"\"

Therapist ({agent_name}) said (treat as action a_{{t-1}} for p(s_t|s_{{t-1}},a_{{t-1}})):
\"\"\"{therapist_utt}\"\"\"

Current latent belief over s (only key client_talk_type: {state_values_bar}; update cautiously):
{latent_json}

Output JSON with exactly these keys:
{{
  "observation": {{
    "state_pattern": "hypothesis: how latent facets manifest in this utterance (p(o_t|s_t) summary)",
    "textual_cues": "phrases suggesting uncertainty, resistance, change talk, etc.",
    "behavioral_cues": "length, hedging, direct disagreement, elaboration, etc.",
    "confidence": 0.0-1.0
  }},
  "latent_mi": {{
    {inner_latent_mi}
  }},
  "therapist_action": "one of: {actions_line}",
  "transition": {{
    "expected_state_delta": "expected Δ in patient latent state under p(s_t|s_{{t-1}},a_{{t-1}}) after this therapist action",
    "confidence": 0.0-1.0
  }}
}}

Rules:
- latent_mi must contain exactly these keys: {dims_line}; client_talk_type must be exactly one of: {state_values_enum}
- therapist_action must be one of: {actions_line}
- Keep each string field concise (one short sentence when possible).
"""


def _state_values_enum(state_labels: Sequence[str]) -> str:
    return ", ".join(state_labels)


def _state_values_bar(state_labels: Sequence[str]) -> str:
    return " | ".join(state_labels)


def _state_boundary_guidance(state_labels: Sequence[str]) -> str:
    labels = tuple(state_labels or ())
    if labels == ("precontemplation", "contemplation", "preparation"):
        return """For CAMI readiness-stage inference, use these boundaries:
- `precontemplation`: denies or minimizes the problem, externalizes blame, refuses to engage, or shows no awareness of personal risk.
- `contemplation`: acknowledges some risk, problem, or consequence — even partially or reluctantly — but has no concrete next step. Includes: weighing pros/cons, expressing ambivalence, recognizing patterns, or showing emotional concern without a plan.
- `preparation`: expresses a concrete action, tentative plan, quota, frequency target, time horizon, help-seeking step, or replacement strategy.

**Key disambiguation rules:**

1. **contemplation vs precontemplation**: If the client acknowledges the problem's existence or impact — even reluctantly ("I guess", "maybe", "probably") — that is `contemplation`. Reserve `precontemplation` for active denial, deflection, or genuine unawareness.
2. **contemplation vs preparation**: Mere acknowledgment or worry without a concrete step remains `contemplation`. Once the client names a specific action, timeline, quota, or plan — even tentatively (`I could`, `I should`, `I'll try`) — that crosses into `preparation`.
3. **Short or neutral responses**: Brief answers ("yeah", "I don't know", "hmm") should be interpreted in context. If the client has previously acknowledged the problem, a short response does not regress them to `precontemplation`.

Examples — precontemplation:
- `I don't have a problem. My wife is overreacting.` -> `precontemplation`
- `Everyone drinks at parties, it's normal.` -> `precontemplation`
- `I don't see why I'm here.` -> `precontemplation`

Examples — contemplation:
- `Maybe I drink too much sometimes.` -> `contemplation`
- `I know this is affecting my health, but I'm not sure what to do.` -> `contemplation`
- `Yeah, it's probably the reason why all that's happening.` -> `contemplation`
- `I'm worried about losing my job over this.` -> `contemplation`
- `Part of me wants to change, but I don't know if I can.` -> `contemplation`
- `Um, my wife might leave me. Can't pay my bills.` -> `contemplation`

Examples — preparation:
- `I think I should cut down to weekends only.` -> `preparation`
- `I'll try talking to my boss this week.` -> `preparation`
- `Probably need to start taking some steps.` -> `preparation`
- `What if I just didn't go out on weekdays?` -> `preparation`
- `I need to call my doctor about this.` -> `preparation`"""
    if labels == ("neutral", "change", "sustain"):
        return """For MI talk-type inference, use these boundaries:
- `neutral`: the patient is mainly informative, relational, or factual, without a clear movement toward or against change. Includes: providing factual information, backchannels, clarifications, or mixed/ambiguous statements.
- `change`: the patient expresses desire, ability, reason, need, commitment, or taking steps toward change. Key markers: "I want to", "I could", "I need to", "I should", expressing dissatisfaction with status quo, naming consequences, or articulating plans.
- `sustain`: the patient defends the status quo, minimizes the problem, resists change, or argues for staying the same. Key markers: "I don't see the problem", "it's not that bad", "I can't", "I won't", rationalizing current behavior.

**Key disambiguation rules:**

1. **neutral vs change**: If the patient merely describes facts or feelings without expressing motivation or desire to change, that is `neutral`. Once the patient explicitly links their situation to a need or desire for change, that crosses into `change`.
2. **neutral vs sustain**: If the patient provides factual information without defending or rationalizing behavior, that is `neutral`. Active defense of the status quo or minimization of consequences is `sustain`.
3. **Short or ambiguous responses**: Brief answers ("yeah", "okay", "mm-hmm", "I don't know") are typically `neutral` unless the conversational context makes the change/sustain direction clear.
4. **Mixed signals**: When an utterance contains both change and sustain elements, classify based on the dominant direction. If genuinely balanced, use `neutral`.

Examples — neutral:
- `Sure.` -> `neutral`
- `I drink about three glasses of wine.` -> `neutral`
- `Yeah, I guess so.` -> `neutral`
- `Mm-hmm.` -> `neutral`
- `I've been doing this for years.` -> `neutral`

Examples — change:
- `I don't like how much I'm drinking.` -> `change`
- `Maybe I should try cutting back.` -> `change`
- `I want to get my confidence back.` -> `change`
- `I think I can change.` -> `change`
- `It's affecting my health, I need to do something.` -> `change`

Examples — sustain:
- `I don't have a problem. Everyone does it.` -> `sustain`
- `It's not that bad, I can handle it.` -> `sustain`
- `I can't see myself stopping.` -> `sustain`
- `What's the point? Nothing will change.` -> `sustain`
- `I just won't do it. That world isn't for me.` -> `sustain`"""
    return "Use the provided label set conservatively, but when the utterance contains a concrete next step or explicit action tendency, prefer the more action-oriented label over the merely reflective label."


def _planner_state_boundary_guidance(state_labels: Sequence[str]) -> str:
    labels = tuple(state_labels or ())
    if labels == ("precontemplation", "contemplation", "preparation"):
        return """- For CAMI stages, treat concrete planning cues as strong evidence for `preparation`.
- If the therapist action is likely to elicit a concrete next step, quota, schedule, or help-seeking plan, do not keep the patient in `contemplation` by default.
- Tentative planning language (`could`, `should`, `will try`, `going to`, `need to`, `start with`, `no more than X`) is sufficient for `preparation`.
- Do not regress `preparation` to `contemplation` unless the therapist utterance plausibly undermines commitment or reintroduces clear ambivalence.
- Do not regress `contemplation` to `precontemplation` unless the client actively re-denies the problem. Reflective questions typically maintain or advance the stage."""
    if labels == ("neutral", "change", "sustain"):
        return """- For MI talk types, a therapist reflection or open question typically maintains the current talk type or shifts toward `change`.
- If the therapist action is an affirmation or complex reflection of change talk, the next patient state is likely `change`.
- Simple reflections and backchannels (e.g., "Mm-hmm", "Yeah") tend to maintain the current state or return to `neutral`.
- Confrontation or direct advice without permission may trigger `sustain` as a defensive reaction.
- Do not assume `change` merely because the therapist asked a question — the patient may respond with `neutral` factual information.
- Short patient utterances ("Yeah", "Okay", "Mm") are most likely `neutral` unless strong prior context indicates otherwise."""
    return "- Favor the more action-oriented next label when the therapist utterance is likely to elicit an actionable next step."


def build_persona_extraction_sys(state_labels: Sequence[str]) -> str:
    return PERSONA_EXTRACTION_SYS_TEMPLATE.format(state_values_enum=_state_values_enum(state_labels))


def format_persona_extraction_user(
    usr_name: str,
    agent_name: str,
    patient_utt: str,
    therapist_utt: str,
    latent_json: str,
    mi_latent_keys: Sequence[str],
    therapist_actions: Sequence[str],
    state_labels: Sequence[str],
) -> str:
    inner_latent_mi = ", ".join(f'"{k}": "brief level or note"' for k in mi_latent_keys)
    dims_line = ", ".join(mi_latent_keys)
    actions_line = ", ".join(therapist_actions)
    state_values_enum = _state_values_enum(state_labels)
    state_values_bar = _state_values_bar(state_labels)
    return PERSONA_EXTRACTION_USER.format(
        usr_name=usr_name,
        agent_name=agent_name,
        patient_utt=patient_utt,
        therapist_utt=therapist_utt,
        latent_json=latent_json,
        inner_latent_mi=inner_latent_mi,
        dims_line=dims_line,
        actions_line=actions_line,
        state_values_enum=state_values_enum,
        state_values_bar=state_values_bar,
    )


# ---------------------------------------------------------------------------
# Personas — Short-term local filter q(s_t|o_{\le t}, a_{\le t-1}, …)
# ---------------------------------------------------------------------------
PERSONA_SHORT_TERM_Q_SYS = """You are a core engine responsible for executing **Incremental Bayesian Belief Updating** over a dual-model system:
- **Observation Model**: p(o|s)
- **Transition Model**: p(s'|s,a)
Your primary objective is to minimize **Variational Free Energy (ΔF)** under a **strict steady-state constraint**. Large-scale structural overhauls that destroy existing structures are strictly forbidden; only local, minimized iterative corrections are allowed.
Return ONLY one valid JSON object.
Do NOT output `<think>`, `</think>`, markdown fences, comments, headings, or any prose before or after the JSON.
If uncertain, emit the smallest valid JSON matching the schema with empty arrays where needed."""

PERSONA_SHORT_TERM_Q_USER = """
## Core Data Model (q_t)
**q_t** is a set of rows `{{"abstract_cue": "state"}}` where:
* **cue**: Highly abstract identifier (`snake_case`) summarizing **observed regularities** before and after the update, including their contrast. **Strictly NO** semantic content or verbatim quotes from the transcript.
* **state**: Must strictly belong to the enum set `{{{state_values_enum}}}` (dataset-aligned talk type or readiness stage).
* **regularity_observation**: A **structure-only** comparative analysis of the current observation `o_t` against the prior state (e.g., pace, length, pauses). **Strictly NO** verbatim quotes or semantic paraphrasing.

## Inference Logic (Surprise Minimization Framework)
The inference process must be executed strictly in the following three steps, governed by the Free Energy Principle (minimizing prediction error):

1.  **Expectation Generation**: Based on the prior state space `q(s_{{t-1}})`, system intervention `a_{{t-1}}`, and long-run transition prior `p`, generate theoretical expectations about the **structural regularities** of the patient's next move.
2.  **Observation & Surprise Calculation**: Assimilate the actual observation `o_t`. Calculate the **structural prediction error (surprise)** by comparing real `o_t` against the Step 1 expectations. Record the pre-state, post-state, and this specific contrast (prediction error) purely as structural, non-semantic regularities (e.g., unexpected shift in pacing, length deviation, emergence of disfluencies).
3.  **State Update (Error Absorption)**: Update the `q` corpus to absorb and minimize the identified structural surprise. Modify existing `q_t` rows to realign the state representation with current realities. **Crucially, `q_delta` arrays are NOT limited to a single operation; you MUST generate MULTIPLE `add`, `modify`, or `drop` entries simultaneously if the surprise magnitude requires adjusting multiple cues to restore predictive equilibrium.**

## Strict Rules & Global Constraints

### 1. Structural & Formatting Rules
* **JSON Specification**: Emit **one** valid JSON object only. Do NOT use markdown fences. Escape every literal `"` inside strings as `\\"`. Never emit `'"` mid-string.
* **Token Specification**: `cue` tokens must be short, using `snake_case` or single identifiers. No sentence-long paraphrases.
* **Enum & Threshold Limits**: `client_talk_type` ∈ `{{{state_values_enum}}}`; `inference_entropy` ∈ `[0.0, 1.0]`.

### 2. De-semanticization Rules
* **Scope**: These rules apply to `cue`, `behavior_pattern`, `transition_semantic`, and `regularity_observation` fields ONLY. They do NOT apply to `client_talk_type` — that label must be determined by the **semantic meaning** of the patient's utterance (see §2.5).
* **No Information Leakage**: Absolutely no quoting, copying, or paraphrasing the specific semantic content of `o_t` in cue/pattern fields.
* **Strict Decoupling**: `transition_semantic` (latent abstract transition) MUST NOT equal `behavior_pattern` (surface structural behavior).

### 2.5. State Boundary Rules (`client_talk_type` — semantic judgment)
**`client_talk_type` is a semantic classification task.** Determine it based on what the patient **means**, not surface structural features (sentence length, negation words, hedging). A short answer or a sentence containing "don't" can belong to any stage — read the meaning.
{state_boundary_guidance}

### 3. Update Operations (q_delta)
`q_delta` is an **OPERATIONAL UPDATE SPEC** over the long-term state `q` designed to minimize future surprise. Every update operation must be orthogonally factored into three core components:

| Component | Core Definition | Examples |
| :--- | :--- | :--- |
| **transition_semantic** | Abstract latent transition type. Captures how state dynamics evolve to absorb prediction error. | `resistance_to_engagement`, `commitment_decay` |
| **behavior_pattern** | Fully **de-semanticized** observable structure driving the surprise. Strictly structural signals. | `short_response_with_hedging`, `stance_softening` |
| **state_update** | Final inferred state. Derived from **behavior_pattern + regularity contrast**, NOT raw text meaning. | `{{"state": "neutral", "confidence": 0.85}}` |

**Operation Logic (Multiplicity is Expected):** Ensure updates resolve all structural prediction errors. Use multiple operations across the arrays if necessary.

| Operation | Trigger Condition & Execution Logic |
| :--- | :--- |
| **ADD** | Introduce a new `cue` to explain unpredicted structure. Implicit mapping: `cue := transition_semantic`. |
| **MODIFY** | **Primary mechanism for surprise minimization.** Existing signal persists but structural meaning shifts. **MUST** anchor to an existing `cue` via `cue_keyword`. Update its latent interpretation; do not duplicate. |
| **DROP** | Prior regularity is structurally invalidated, contradicted by new `o_t`, or decayed. **MUST** specify `reason_type` ∈ `{{contradicted, decayed, overridden}}`. |

---

## Input Data

* **Prior q(s_{{t-1}})** — runtime field `_q_st_text`: updated **only** by applying your **`q_delta`** (add/modify/drop) onto this text each turn; modify/drop MUST anchor to substrings present here:
{prior_q}

* **System prior intervention a_{{t-1}}** (action label + therapist utterance):
    * action_label: {prev_action_label}
    * utterance:
```
{prev_therapist_utt}
```

* **State transition prior p(s_t|s_{{t-1}}, a_{{t-1}})**:
```
{transition_prior_block}
```

* **Long-term observation model p(o|s)** (state → reply-class mapping):
```
{observation_model_block}
```

* **Short-term session memory window** (recent local interaction history only; use this as the main context for the posterior update, not as a long-run rule store):
```
{short_term_memory_window}
```

* **Current real observation o_t**:
```
{current_patient_o_t}
```

---

## Output JSON (exact keys; example shape)

{{
  "client_talk_type": "{state_values_bar}",
  "inference_entropy": 0.0,
  "q_abstract": [
    {{
      "phase": "pre",
      "cue": "snake_case_expected_prior_pattern",
      "state": "{state_values_bar}"
    }},
    {{
      "phase": "post",
      "cue": "snake_case_actual_observed_pattern",
      "state": "{state_values_bar}",
      "confidence": 0.0
    }},
    {{
      "phase": "contrast",
      "cue": "snake_case_surprise_or_shift_identifier",
      "state_transition": "string (e.g. sustain_to_change, expected_neutral_but_sustain, no_shift)"
    }}
  ],
  "regularity_observation": {{
    "pre_regularities": "string (structured read before o_t; no verbatim)",
    "post_regularities": "string (structured read after o_t; no verbatim)",
    "contrast_regularities": "string (regularity-space prediction error)",
    "confidence": 0.0
  }},
  "q_delta": {{
    "add": [
      {{
        "transition_semantic": "snake_case_abstract_transition_keyword",
        "behavior_pattern": "snake_case_desemanticized_pattern",
        "state_update": {{
          "state": "{state_values_bar}",
          "confidence": 0.0
        }}
      }}
    ],
    "modify": [
      {{
        "cue_keyword": "substring_of_existing_cue_in_prior_q",
        "transition_semantic": "updated_transition_abstraction",
        "behavior_pattern": "updated_structural_pattern",
        "state_update": {{
          "state": "{state_values_bar}",
          "confidence": 0.0
        }}
      }}
    ],
    "drop": [
      {{
        "cue_keyword": "substring_of_existing_cue_in_prior_q",
        "transition_semantic": "optional_transition_being_invalidated",
        "behavior_pattern": "optional_pattern_no_longer_observed",
        "reason_type": "contradicted | decayed | overridden"
      }}
    ]
  }}
}}

**Backward-compatible:** you may instead use legacy `q_delta.add` as a string, or per-item `{{"cue", "state", "rationale"}}` objects; the parser accepts both.

## Minimal Valid JSON Example
{{
  "client_talk_type": "{state_first_label}",
  "inference_entropy": 0.5,
  "q_abstract": [],
  "regularity_observation": {{
    "pre_regularities": "brief expected structure",
    "post_regularities": "brief observed structure",
    "contrast_regularities": "brief shift summary",
    "confidence": 0.5
  }},
  "q_delta": {{
    "add": [],
    "modify": [],
    "drop": []
  }}
}}

Invalid outputs include:
- any text before `{{`
- any text after the final `}}`
- markdown fences like ```json
- `<think>` blocks
"""


def format_persona_short_term_q_user(
    prior_q: str,
    prev_action_label: str,
    prev_therapist_utt: str,
    transition_prior_block: str,
    observation_model_block: str,
    short_term_memory_window: str,
    current_patient_o_t: str,
    state_labels: Sequence[str],
) -> str:
    state_values_enum = _state_values_enum(state_labels)
    state_values_bar = _state_values_bar(state_labels)
    return PERSONA_SHORT_TERM_Q_USER.format(
        prior_q=prior_q or "(session start — no posterior yet; use a diffuse prior)",
        prev_action_label=prev_action_label or "none",
        prev_therapist_utt=prev_therapist_utt or "(none — first patient turn in session)",
        transition_prior_block=transition_prior_block or "(not initialized — infer cautiously)",
        observation_model_block=observation_model_block or "(not initialized — infer cautiously)",
        short_term_memory_window=short_term_memory_window or "(empty short-term memory window)",
        current_patient_o_t=current_patient_o_t or "",
        state_values_enum=state_values_enum,
        state_values_bar=state_values_bar,
        state_first_label=state_labels[0] if state_labels else "",
        state_boundary_guidance=_state_boundary_guidance(state_labels),
    )


# ---------------------------------------------------------------------------
# Personas — Long-term global p: joint maintenance of p(o|s) and p(s'|s,a)
# ---------------------------------------------------------------------------
PERSONA_LONG_TERM_P_SYS = """You are a core engine responsible for executing **Incremental Bayesian Belief Updating** over a dual-model system:
- **Observation Model**: p(o|s)
- **Transition Model**: p(s'|s,a)

Your primary objective is to minimize **Variational Free Energy (ΔF)** under a **strict steady-state constraint**. Large-scale structural overhauls that destroy existing structures are strictly forbidden; only local, minimized iterative corrections are allowed.
Return ONLY one compact machine-readable JSON object.
Do NOT output `<think>`, `</think>`, markdown fences, comments, headings, or any prose before or after the JSON.
If uncertain, emit the smallest valid JSON with empty arrays.
Keep every explanation short and concrete."""

PERSONA_LONG_TERM_P_USER = """
## 1. Free Energy Decomposition & Core Execution Logic (Input -> Compute Error -> Update)

Inference and updating must strictly follow these three phases:

### Phase 1: Prior Expectation Generation (Expectation)
Generate theoretical expectations for the current turn based on historical memory:
- **Behavioral Expectation**: Given the inferred state q(s_t), what structural reply the user should exhibit based on the current p(o|s).
- **Transition Expectation**: After encountering action a_{{t-1}} in prior state s_{{t-1}}, what state the user should transition to based on the current p(s'|s,a).

### Phase 2: Error Contrast & Calculation (Observation & Contrast)
Incorporate the actual current observation o_t and the posterior inferred state to calculate system surprisal:
- **Observation Prediction Error (Likelihood Penalty)**: Contrast the expected reply shape with the structural features of the actual observation o_t. Compute the local likelihood penalty: −log p(o_t | E[q(s_t)]).
- **Transition Surprisal**: Contrast the predicted target state with the actual posterior state q(s_t). Compute the KL divergence: D_KL(q(s_t) || p(s_t | s_{{t-1}}, a_{{t-1}})).

### Phase 3: Error Minimization Update (State Update)
Evaluate the total free energy change ΔF and execute multi-dimensional add/modify/drop operations:
- If ΔF is minimal → **Retain existing parameters** (do not trigger delta).
- If ΔF is large (significant prediction error or surprisal exists) → **Execute minimized corrections**. Absorb the errors into the models, allowing multiple rule updates simultaneously within the delta arrays.

---

## 2. Dual-Model Feature Extraction & Indexed Maintenance Specification

To ensure the precision of incremental updates, every rule (mapping or triple) in the models MUST be uniquely identified by a **highly summarized abstract index (`rule_index`)** (using `snake_case`).

### 2.1 Observation Model p(o|s) — [Index → State → Behavior Mapping]
- **Maintenance Goal**: Record the user's structured behavioral manifestation (`reply_shape`) under a specific user state (`latent_state`).
- **De-semanticization Constraint**: `reply_shape` MUST be a **fully de-semanticized** behavioral pattern (e.g., `short_ack`, `deflection`). Verbatim quotes from the transcript are strictly forbidden.
- **Update Rules (`delta`)**:
  - `add`: Must create a new `rule_index` and define the complete state and behavior.
  - `modify`: Must anchor to an existing `target_rule_index` and output an `updated_rule_index`, along with the fully updated `latent_state` and `reply_shape`.

### 2.2 Transition Model p(s'|s,a) — [Index → (Source, Action, Target) Triple]
- **Maintenance Goal**: Record the user's state transition (`s_to`) after being acted upon by a therapist action (`action_a`) from a specific state (`s_from`).
- **Vocabulary Constraint**: The action variable `action_a` MUST strictly originate from the provided system action vocabulary.
- **Update Rules (`delta`)**:
  - `add`: Must create a new `rule_index` and define the complete triple.
  - `modify`: Must anchor to an existing `target_rule_index` and output an `updated_rule_index`, along with the fully updated triple (`s_from`, `action_a`, `s_to`).

---

## 3. Input Data Injections (Inputs)

- Historical interaction samples (recent cycles, JSON):
{long_term_memory_summary}

- Historical interaction samples (recent cycles, JSON excerpts for evidence anchoring):
{history_json}

- Current observation model p(o|s) — runtime `_observation_model_block`; updated **only** by applying **`observation_model_delta`** each turn (mapping table / rule indices):
\"\"\"
{observation_model_block}
\"\"\"

- Current transition model p(s'|s,a) — runtime `_transition_model_block`; updated **only** by applying **`transition_model_delta`** each turn (directed edges / rule indices):
\"\"\"
{transition_model_block}
\"\"\"

- Prior state excerpt q(s_{{t-1}}) before observing o_t:
\"\"\"{prior_q_excerpt}\"\"\"

- Previous action a_{{t-1}}: {prev_action}
- Previous therapist utterance:
\"\"\"{prev_therapist_utt}\"\"\"

- Current inferred posterior state q(s_t):
\"\"\"{q_st_manifold}\"\"\"

- Current actual patient observation o_t:
\"\"\"{patient_o_t}\"\"\"

- Current therapist action a_t:
\"\"\"{current_therapist_utt}\"\"\"

{given_therapist_action_block}
- Therapist action vocabulary (The `therapist_action` in the output MUST strictly belong to one of the following):
{therapist_actions_line}

---

## 4. Strict Output Format

You MUST output **exactly one** machine-parseable JSON object.
Do NOT wrap it in markdown fences.
Literal double quotes `"` inside strings MUST be escaped as `\\"`. Unescaped `'"` mid-string is strictly prohibited.

{{
  "therapist_action": "one of: {therapist_actions_line}",
  "transition": {{
    "therapist_action_rationale": "string ≤120 chars",
    "confidence": 0.0
  }},
  "fep_error_calculation": {{
    "observation_contrast": {{
      "expected_reply_shape": "short abstract label",
      "actual_reply_shape": "short abstract label",
      "likelihood_penalty": "string ≤80 chars"
    }},
    "transition_contrast": {{
      "expected_target_state": "short state label",
      "actual_inferred_state": "short state label",
      "surprisal_delta": "string ≤80 chars"
    }}
  }},
  "observation_model_delta": {{
    "add": [
      {{
        "rule_index": "snake_case_abstract_identifier",
        "latent_state": "{state_values_bar}",
        "reply_shape": "abstract reply-class (no verbatim)",
        "explanation": "string ≤60 chars"
      }}
    ],
    "modify": [
      {{
        "target_rule_index": "Match the exact old index name from the existing observation model",
        "updated_rule_index": "New snake_case_identifier reflecting the modified pattern",
        "latent_state": "{state_values_bar} (updated or retained state)",
        "reply_shape": "updated abstract reply-class (updated or retained behavioral structure)",
        "explanation": "string ≤60 chars"
      }}
    ],
    "drop": [
      {{
        "target_rule_index": "Match the exact index name from the existing observation model",
        "explanation": "string ≤60 chars"
      }}
    ]
  }},
  "transition_model_delta": {{
    "add": [
      {{
        "rule_index": "snake_case_abstract_identifier",
        "s_from": "{state_values_bar}",
        "action_a": "Specific action from the action vocabulary",
        "s_to": "{state_values_bar}",
        "explanation": "string ≤60 chars"
      }}
    ],
    "modify": [
      {{
        "target_rule_index": "Match the exact old index name from the existing transition model",
        "updated_rule_index": "New snake_case_identifier reflecting the modified pattern",
        "s_from": "{state_values_bar} (updated or retained source state)",
        "action_a": "Specific action from the action vocabulary (updated or retained action)",
        "s_to": "{state_values_bar} (updated or retained target state)",
        "explanation": "string ≤60 chars"
      }}
    ],
    "drop": [
      {{
        "target_rule_index": "Match the exact index name from the existing transition model",
        "explanation": "string ≤60 chars"
      }}
    ]
  }}
}}
 
If there is no strong update, prefer empty arrays rather than verbose justification.

## Minimal Valid JSON Example
{{
  "therapist_action": "{first_therapist_action}",
  "transition": {{
    "therapist_action_rationale": "brief rationale",
    "confidence": 0.5
  }},
  "fep_error_calculation": {{
    "observation_contrast": {{
      "expected_reply_shape": "brief_expected_shape",
      "actual_reply_shape": "brief_actual_shape",
      "likelihood_penalty": "brief mismatch summary"
    }},
    "transition_contrast": {{
      "expected_target_state": "{first_state_label}",
      "actual_inferred_state": "{first_state_label}",
      "surprisal_delta": "brief transition summary"
    }}
  }},
  "observation_model_delta": {{
    "add": [],
    "modify": [],
    "drop": []
  }},
  "transition_model_delta": {{
    "add": [],
    "modify": [],
    "drop": []
  }}
}}

Invalid outputs include:
- any text before `{{`
- any text after the final `}}`
- markdown fences like ```json
- `<think>` blocks
"""


def format_persona_long_term_p_user(
    long_term_memory_summary: str,
    history_json: str,
    observation_model_block: str,
    transition_model_block: str,
    prior_q_text: str,
    prev_action: str,
    prev_therapist_utt: str,
    q_st_manifold: str,
    patient_o_t: str,
    current_therapist_utt: str,
    therapist_actions_line: str,
    state_labels: Sequence[str],
    given_therapist_action: Optional[str] = None,
) -> str:
    ga = (given_therapist_action or "").strip()
    if ga:
        given_therapist_action_block = (
            "Externally provided therapist action label for a_t (MUST use this exact token for "
            "MACHINE_JSON.therapist_action; align §3 system_action_a_prev with it for the current move):\n"
            f'"{ga}"\n\n'
        )
    else:
        given_therapist_action_block = ""
    state_values_bar = _state_values_bar(state_labels)
    return PERSONA_LONG_TERM_P_USER.format(
        long_term_memory_summary=long_term_memory_summary or "(no long-term memory summary yet)",
        history_json=(history_json or "")[:1800],
        observation_model_block=(observation_model_block or "(empty)")[:1200],
        transition_model_block=(transition_model_block or "(empty)")[:1200],
        prior_q_excerpt=(prior_q_text or "")[:700],
        prev_action=prev_action,
        prev_therapist_utt=(prev_therapist_utt or "")[:300],
        q_st_manifold=(q_st_manifold or "")[:900],
        patient_o_t=(patient_o_t or "")[:400],
        current_therapist_utt=(current_therapist_utt or "")[:400],
        therapist_actions_line=therapist_actions_line,
        given_therapist_action_block=given_therapist_action_block,
        state_values_bar=state_values_bar,
        first_therapist_action=(therapist_actions_line.split(",")[0].strip() if therapist_actions_line else "other"),
        first_state_label=(state_labels[0] if state_labels else ""),
    )


def _repair_llm_json_chunk(chunk: str) -> str:
    """Fix common LLM mistakes that break strict JSON (esp. long-form persona MACHINE_JSON).

    Typical failure: after a phrase in single quotes the model emits an extra ``"`` before the
    sentence continues, e.g. ``...that?'" aligns with...`` which prematurely closes the JSON string.
    """
    if not chunk:
        return chunk
    out = re.sub(r"(?<=[a-zA-Z?.!])'\x22(\s+[a-zA-Z])", r"'\1", chunk)
    out = out.replace("\u201c", '"').replace("\u201d", '"')
    out = out.replace("\u2018", "'").replace("\u2019", "'")
    return out


def _complete_truncated_json_candidate(chunk: str) -> Optional[str]:
    if not chunk:
        return None
    jb = chunk.find("{")
    if jb < 0:
        return None
    src = chunk[jb:]
    if not src.strip():
        return None

    closers = []
    out = []
    in_str = False
    esc = False

    for ch in src:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            closers.append("}")
        elif ch == "[":
            closers.append("]")
        elif ch in ("}", "]"):
            if closers and ch == closers[-1]:
                closers.pop()

    text = "".join(out).rstrip()
    if not text:
        return None
    if in_str:
        text += '"'

    while text and text[-1] in ",:":
        text = text[:-1].rstrip()
    if not text:
        return None

    if closers:
        text += "".join(reversed(closers))
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _decode_json_object_in_chunk(chunk: str) -> Optional[dict]:
    """Try json.JSONDecoder on chunk and on a lightly repaired variant; return first dict or None."""
    repaired = _repair_llm_json_chunk(chunk)
    candidates = [chunk, repaired]
    completed = _complete_truncated_json_candidate(repaired)
    if completed:
        candidates.append(completed)
    for cand in candidates:
        jb = cand.find("{")
        if jb < 0:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(cand[jb:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _iter_fenced_json_dicts(text: str):
    """Yield dicts from every ```json / ``` ... ``` block (skip invalid chunks)."""
    pos = 0
    while pos < len(text):
        fm = re.search(r"```(?:json)?\s*", text[pos:], re.IGNORECASE)
        if not fm:
            break
        start = pos + fm.end()
        rest = text[start:]
        end = rest.find("```")
        if end < 0:
            break
        chunk = rest[:end]
        obj = _decode_json_object_in_chunk(chunk)
        if obj is not None:
            yield obj
        pos = start + end + 3


def _parse_fenced_json_object(text: str) -> Optional[dict]:
    """Prefer last fenced block that matches persona MACHINE_JSON shape (model may echo bad fences first)."""
    last_st, last_lt = None, None
    for obj in _iter_fenced_json_dicts(text):
        if _looks_like_short_term_mj(obj):
            last_st = obj
        if _looks_like_long_term_mj(obj):
            last_lt = obj
    if last_lt is not None:
        return last_lt
    if last_st is not None:
        return last_st
    return None


def _looks_like_short_term_mj(d: dict) -> bool:
    if "client_talk_type" not in d:
        return False
    if isinstance(d.get("regularity_observation"), dict):
        return True
    if isinstance(d.get("observation"), dict):
        return True
    if isinstance(d.get("q_abstract"), list):
        return True
    if isinstance(d.get("q_delta"), dict):
        return True
    return False


def _looks_like_long_term_mj(d: dict) -> bool:
    ta = d.get("therapist_action")
    if not isinstance(ta, str):
        return False
    ta = ta.strip()
    if not ta or len(ta) > 128 or "\n" in ta:
        return False
    # Placeholder / re-statement note: avoid treating a hallucinated fenced block as ground truth.
    # Do not list "OTHER" here — it is a valid coarse therapist_action label in AnnoMI pipelines.
    if ta.upper() in frozenset({"ACTION", "NUMBER", "STRING", "LABEL"}):
        return False
    if ta.lower().startswith(("okay", "let's", "here", "the output")):
        return False
    return (
        "observation_model_text" in d
        or "transition_model_text" in d
        or "observation_model_update_rationale" in d
        or "transition_model_update_rationale" in d
        or "observation_model_delta" in d
        or "transition_model_delta" in d
    )


def extract_machine_json_line(text: str) -> Optional[dict]:
    """Parse MACHINE_JSON:{{...}}, or Qwen-style ```json {{...}}```, or first balanced JSON matching persona schemas."""
    if not text:
        return None
    matches = list(re.finditer(r"MACHINE_JSON:\s*", text, re.IGNORECASE))
    for m in matches:
        rest = text[m.end() :].lstrip()
        obj = _decode_json_object_in_chunk(rest)
        if obj is not None:
            return obj

    fj = _parse_fenced_json_object(text)
    if fj and (_looks_like_short_term_mj(fj) or _looks_like_long_term_mj(fj)):
        return fj

    long_hit: Optional[dict] = None
    i = text.find("{")
    while i >= 0:
        obj = _decode_json_object_in_chunk(text[i:])
        if obj is not None:
            if _looks_like_short_term_mj(obj):
                return obj
            if _looks_like_long_term_mj(obj) and long_hit is None:
                long_hit = obj
        i = text.find("{", i + 1)
    return long_hit


# ---------------------------------------------------------------------------
# EventMemory — EventSummary (context_summarize)
# ---------------------------------------------------------------------------
EVENT_SUMMARY_SYS = (
    "You are good at extracting events and summarizing them in brief sentences. "
    "You will be shown a conversation between {usr_name} and {agent_name}.\n"
)

EVENT_SUMMARY_USER = (
    "#Conversation#:\n{context}.\n"
    "Based on the Conversation, please summarize the main points of the conversation "
    "with brief sentences in English, within 20 words.\nSUMMARY:"
)


def format_event_summary_sys(usr_name: str, agent_name: str) -> str:
    return EVENT_SUMMARY_SYS.format(usr_name=usr_name, agent_name=agent_name)


def format_event_summary_user(context: str) -> str:
    return EVENT_SUMMARY_USER.format(context=context)


# ---------------------------------------------------------------------------
# Generator — ResponseGenerator
# ---------------------------------------------------------------------------
GENERATOR_SYS = (
    "As a communication expert with outstanding communication habits, you embody the role of {agent_name} "
    "throughout the following dialogues. Here are some of your distinctive personal traits: {agent_traits}.\n"
)

GENERATOR_USER = """<CONTEXT>
Drawing from your recent conversation with {usr_name}:
{context}
<MEMORY>
The memories linked to the ongoing conversation are:
{memories}
<USER_TRAITS>
During the conversation process between you and {usr_name} in the past, you found that the {usr_name} has the following characteristics:
{user_traits}

Now, please role-play as {agent_name} to continue the dialogue between {agent_name} and {usr_name}.
{usr_name} just said: {inquiry}
{action_instruction}Please respond to {usr_name}'s statement using the following format (maximum 30 words, must be in English):
RESPONSE:
"""


def format_generator_sys(agent_name: str, agent_traits: str) -> str:
    return GENERATOR_SYS.format(agent_name=agent_name, agent_traits=agent_traits)


def format_generator_user(
    usr_name: str,
    agent_name: str,
    context: str,
    memories: str,
    user_traits: str,
    inquiry: str,
    therapist_action: Optional[str] = None,
) -> str:
    if therapist_action:
        action_instruction = (
            f"MI strategy to use: [{therapist_action}] — apply this counseling approach in your response.\n"
        )
    else:
        action_instruction = ""
    return GENERATOR_USER.format(
        usr_name=usr_name,
        agent_name=agent_name,
        context=context,
        memories=memories,
        user_traits=user_traits,
        inquiry=inquiry,
        action_instruction=action_instruction,
    )


def build_generator_prompts(
    usr_name: str,
    agent_name: str,
    inquiry: str,
    context: str,
    memories: str,
    user_traits: str,
    agent_traits: str,
    therapist_action: Optional[str] = None,
) -> Tuple[str, str]:
    """Used by Generator.select_prompts."""
    sys_p = format_generator_sys(agent_name, agent_traits)
    user_p = format_generator_user(
        usr_name=usr_name,
        agent_name=agent_name,
        context=context,
        memories=memories,
        user_traits=user_traits,
        inquiry=inquiry,
        therapist_action=therapist_action,
    )
    return sys_p, user_p


# ---------------------------------------------------------------------------
# PatientSimulator: simulate the client's next utterance after the therapist has spoken (used in evaluation).
# ---------------------------------------------------------------------------
PATIENT_SIMULATOR_SYS = (
    "You role-play as {usr_name} (the patient/client) in a clinical or supportive dialogue. "
    "Stay consistent with the provided state description and memories. "
    "Reply with natural, concise patient speech in English."
)

PATIENT_SIMULATOR_USER = """<DIALOGUE_SO_FAR>
{context}
<CURRENT_DOCTOR_MESSAGE>
{agent_name} just said:
\"\"\"{doctor_reply}\"\"\"

<RELEVANT_MEMORIES>
{memories}

{state_after_action_block}
Task: Produce the patient's *next* utterance — what {usr_name} would plausibly say *in response* to the doctor's message above.
Constraints: one or two short sentences, English, maximum 40 words, first person as the patient.
Format (required):
PATIENT_RESPONSE:
"""


def format_patient_simulator_prompts(
    usr_name: str,
    agent_name: str,
    doctor_reply: str,
    context: str,
    memories: str,
    state_after_action: str = "",
) -> Tuple[str, str]:
    sys_p = PATIENT_SIMULATOR_SYS.format(usr_name=usr_name)
    block = ""
    if (state_after_action or "").strip():
        block = (
            "<PATIENT_STATE_GUIDANCE>\n"
            "Estimated patient state after the therapist's current move "
            "(use this to shape tone, resistance, openness, and topic):\n"
            f"{state_after_action.strip()}\n"
        )
    user_p = PATIENT_SIMULATOR_USER.format(
        usr_name=usr_name,
        agent_name=agent_name,
        doctor_reply=doctor_reply,
        context=context,
        memories=memories,
        state_after_action_block=block,
    )
    return sys_p, user_p


# ---------------------------------------------------------------------------
# State prediction vs. realized next utterance (LLM-as-judge, evaluation only)
# ---------------------------------------------------------------------------
STATE_ALIGNMENT_SYS = (
    "You evaluate whether a patient's *next* spoken reply is consistent with a structured state prediction "
    "(motivational-interviewing style latent cues). Reply with a single JSON object only, no markdown, no extra text."
)

STATE_ALIGNMENT_USER = """The therapist ({agent_name}) had just said:
\"\"\"{doctor_reply}\"\"\"

Structured prediction of the patient's state *before* that next patient turn:
\"\"\"{user_traits}\"\"\"

The patient's *actual* next utterance (reference):
\"\"\"{next_patient_utt}\"\"\"

Score how well the actual utterance aligns with the predicted state (not clinical accuracy of the prediction itself, but consistency between prediction and what the patient then said).
Output JSON with exactly: {{"alignment_score": <float 0.0-1.0>, "note": "<up to 20 words>"}}
"""


def format_state_alignment_user(
    agent_name: str,
    doctor_reply: str,
    user_traits: str,
    next_patient_utt: str,
) -> str:
    return STATE_ALIGNMENT_USER.format(
        agent_name=agent_name,
        doctor_reply=doctor_reply,
        user_traits=user_traits,
        next_patient_utt=next_patient_utt,
    )


# ---------------------------------------------------------------------------
# TransitionPlanner: explicit one-step roll-forward of p(s'|s, a), used to drive the client-reply generator.
# ---------------------------------------------------------------------------
PLANNER_SYS_TEMPLATE = (
    "You are a transition model for patient latent state in motivational-interviewing dialogue (AnnoMI-style). "
    "Latent s is summarized by client_talk_type ∈ {{{state_values_enum}}}. "
    "Given current belief, observation/transition history, a discrete therapist action label from the run vocabulary, "
    "and the therapist's actual utterance, predict client_talk_type *after* that action and how the patient may surface behaviorally next. "
    "Reply with nothing before the JSON: output exactly one JSON object on the first line, then stop. "
    "No markdown fences, no commentary, no thinking tags."
)


def build_planner_sys(state_labels: Sequence[str]) -> str:
    return PLANNER_SYS_TEMPLATE.format(state_values_enum=_state_values_enum(state_labels))

PLANNER_USER = """Current patient state:
\"\"\"
{current_state}
\"\"\"

Current observation model p(o|s):
\"\"\"
{observation_model}
\"\"\"

Current transition model p(s'|s,a):
\"\"\"
{transition_model}
\"\"\"

Therapist action label a (must be one of: {actions_line}):
{action}

Therapist's actual message (grounding for a):
\"\"\"{therapist_utterance}\"\"\"

Output JSON with exactly these keys:
{{
  "latent_mi_next": {{
    {inner_latent_mi}
  }},
  "expected_behavior_cues": "how the patient may speak or act in the *next* turn (short)",
  "rationale": "one short sentence linking action a to the predicted shift",
  "confidence": 0.0-1.0
}}

Rules:
- latent_mi_next must contain exactly these keys: {dims_line}; client_talk_type must be one of: {state_values_enum}
- Be conservative: small steps from the current belief unless the utterance strongly implies a jump.
{planner_state_boundary_guidance}
"""


def format_planner_user(
    current_state: str,
    observation_model: str,
    transition_model: str,
    action: str,
    therapist_utterance: str,
    mi_latent_keys: Sequence[str],
    therapist_actions: Sequence[str],
    state_labels: Sequence[str],
) -> str:
    inner = ", ".join(f'"{k}": "brief level or note after action"' for k in mi_latent_keys)
    dims_line = ", ".join(mi_latent_keys)
    actions_line = ", ".join(therapist_actions)
    state_values_enum = _state_values_enum(state_labels)
    state_values_bar = _state_values_bar(state_labels)
    return PLANNER_USER.format(
        current_state=current_state,
        observation_model=observation_model or "(empty)",
        transition_model=transition_model or "(empty)",
        action=action,
        therapist_utterance=therapist_utterance,
        inner_latent_mi=inner,
        dims_line=dims_line,
        actions_line=actions_line,
        state_values_enum=state_values_enum,
        state_values_bar=state_values_bar,
        planner_state_boundary_guidance=_planner_state_boundary_guidance(state_labels),
    )
