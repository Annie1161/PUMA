import path_setup  # noqa: F401

import argparse

from tools.defaults import DEFAULT_LLM_CLIENT, DEFAULT_QWEN_HF_MODEL


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def get_args(description='Agent configuration'):
    parser = argparse.ArgumentParser(description=description)

    # data
    parser.add_argument(
        "--dataset",
        choices=["human", "msc", "cc", "quickeval", "annomi", "cami", "kmi"],
        default="msc",
        help="Dataset which will be acted.",
    )
    parser.add_argument(
        "--data_format",
        choices=["msc", "annomi_jsonl", "cami_jsonl"],
        default="msc",
        help=(
            "msc: JSON array (MSC layout). "
            "annomi_jsonl: one transcript per line (patient=SPEAKER_1, doctor=SPEAKER_2). "
            "cami_jsonl: one profile per line (client/counselor pairs from CAMI profiles.jsonl)."
        ),
    )
    parser.add_argument('--data_path', type=str, default='dataset', help='Dataset path.')
    parser.add_argument('--data_name', type=str, default='sequential.json', help='Dataset file name under data_path.')
    parser.add_argument('--id_set', type=str, default='id_set.json', help='Split of sessions and samples.')
    parser.add_argument('--annomi_init_turns', type=int, default=8, help='Number of patient-doctor pairs for memory warm-up when using annomi_jsonl or dataset annomi.')
    parser.add_argument(
        '--cami_warmup_ratio',
        type=float,
        default=0.5,
        help='For CAMI JSONL splitting only: fraction of pairs used for warm-up before eval; bounded by --cami_min_eval_turns.',
    )
    parser.add_argument(
        '--cami_min_eval_turns',
        type=int,
        default=3,
        help='For CAMI JSONL splitting only: keep at least this many evaluation pairs when possible.',
    )

    # memory setting
    parser.add_argument("--memory_cache", nargs='?', default=None, help="Whether to use stored memories.")
    parser.add_argument(
        '--relevance_memory_number',
        type=int,
        default=1,
        help='The number of retrieved relevance memory (eval: query = current doctor utterance).',
    )
    parser.add_argument('--context_memory_number', type=int, default=30, help='The max number of retrieved context memory.')
    parser.add_argument('--dist_thres', type=float, default=1.5, help='The distance threshold of relevance retrieval.')
    parser.add_argument('--memory_session_turn_gap', type=int, default=0, help='Flush short-term memory to long-term summary every N dialogue turns; 0 disables.')
    parser.add_argument('--ori_mem_query', type=_str2bool, default=False, help='Whether to use original query.')

    # persona setting
    parser.add_argument('--max_agent_personas', type=int, default=0, help='Select the latest N user personas.')
    parser.add_argument('--max_user_personas', type=int, default=0, help='Select the latest N user personas.')
    parser.add_argument(
        '--therapist_action_granularity',
        choices=['coarse', 'fine', 'mi12'],
        default='coarse',
        help=(
            'Discrete therapist action label for Persona/Planner JSON: '
            'coarse = AnnoMI main_therapist_behaviour (question, reflection, therapist_input, other); '
            'fine = therapist_action_fine_qualified (e.g. question/open, reflection/simple, …); '
            'mi12 = 12-class MI vocabulary aligned with dynamic_simulator (open_question, evoke_change_talk, …).'
        ),
    )
    parser.add_argument(
        '--persona_single_step',
        action='store_true',
        default=False,
        help='If set, use legacy one-call PersonaExtraction JSON; else short-term q(s_t) filter + long-term p joint maintenance (two LLM steps).',
    )
    parser.add_argument(
        '--persona_history_max',
        type=int,
        default=6,
        help='Max past cycles kept for long-term cognitive model prompt (two-step persona).',
    )
    parser.add_argument(
        '--persona_short_term_window_turns',
        type=int,
        default=4,
        help='Recent interaction turns exposed to short-term q as local session memory.',
    )
    parser.add_argument(
        '--persona_long_term_summary_turns',
        type=int,
        default=12,
        help='How many accumulated turns to summarize for long-term p updates.',
    )
    parser.add_argument(
        '--persona_therapist_action',
        type=str,
        default='',
        help=(
            'Optional global therapist action for MSC→traits_update; must match --therapist_action_granularity '
            '(MSC validates; invalid values are dropped so the persona LLM labels a_t). '
            'Empty = use per-turn dial["therapist_action"] if present, else LLM. Personas accepts the passed string as-is.'
        ),
    )
    parser.add_argument(
        '--persona_json_retry_max',
        type=int,
        default=2,
        help='Max extra retries when short-term/long-term persona MACHINE_JSON is invalid.',
    )

    # client setting
    parser.add_argument('--client', choices=["chatgpt", "chatglm", "qwen_hf", "qwen_local"], default=DEFAULT_LLM_CLIENT, help='Client to use (local HF: qwen_hf / chatglm via tools.qwen_hf_client).')
    parser.add_argument('--model', type=str, default=DEFAULT_QWEN_HF_MODEL, help='The model name: GPT id for chatgpt; HF id or local path for chatglm/qwen_hf; served model id for qwen_local.')
    parser.add_argument('--openai_base_url', type=str, default=None, help='OpenAI-compatible base URL for qwen_local, e.g. http://127.0.0.1:8000/v1.')
    parser.add_argument('--summary_model', type=str, default='default', help='The model for event summary.')   
    parser.add_argument('--persona_model', type=str, default='default', help='The model for persona extraction.')  
    parser.add_argument('--generation_model', type=str, default='default', help='The model for response generation.')       
    parser.add_argument('--planner_model', type=str, default='default', help='The model (LoRA slot Planner) for transition planning p(s\'|s,a).')
    parser.add_argument('--disable_planner', action='store_true', default=False, help='If set, skip TransitionPlanner and omit roll-forward state in patient simulator.')
    parser.add_argument(
        '--planner_no_wm_input',
        type=_str2bool,
        default=False,
        help=(
            'Ablation: planner is still called for next-state prediction, but receives EMPTY '
            "p(o|s) and p(s'|s,a) strings (i.e. plans without world-model evidence). "
            'Tests whether the WM context actually informs the planner output. '
            'Mutually compatible with --observation_focused_state and --prior_fusion_mode; '
            'ignored when --disable_planner is set.'
        ),
    )
    parser.add_argument(
        '--cami_preparation_calibration',
        type=_str2bool,
        default=True,
        help='If true, apply a conservative CAMI next-stage calibration to reduce contemplation lock on explicit planning prompts.',
    )

    # ChatGPT setting
    parser.add_argument('--api_key', type=str, default='Your API Key', help='OpenAI API Key.')

    # ChatGLM setting
    parser.add_argument('--max_input_length', type=int, default=4096, help='The number of maximum size of LLM input.')
    parser.add_argument(
        '--max_output_length',
        type=int,
        default=1024,
        help='Max new tokens per LLM call. Persona long-step JSON + CoT and Planner need headroom (1024 default).',
    )

    # sampling setting
    parser.add_argument('--sampling', action="store_true", default=False, help='Whether to sample data.')
    parser.add_argument('--sampling_step', type=int, default=10, help='The number of sampling frequency.')
    parser.add_argument('--sampling_path', type=str, default='logs/sampling', help='The directory path of sampling')
    parser.add_argument('--sampling_file_name', type=str, default='train.json', help='The name of sampling')

    # evaluation setting
    parser.add_argument('--usr_name', type=str, default='User', help='The name of user.')
    parser.add_argument('--agent_name', type=str, default='Agent', help='The name of agent')
    parser.add_argument(
        '--state_label_space',
        choices=['mi_talk', 'cami_stage'],
        default='mi_talk',
        help=(
            'Label space for patient state evaluation/prediction. '
            'mi_talk = {neutral, change, sustain} (AnnoMI-style talk type); '
            'cami_stage = {precontemplation, contemplation, preparation} (CAMI readiness stage).'
        ),
    )
    parser.add_argument(
        '--persona_state_space',
        choices=['mi_talk', 'cami_stage'],
        default='mi_talk',
        help=(
            'Latent state space used inside Personas/Planner. '
            'mi_talk = {neutral, change, sustain} (AnnoMI-style talk type); '
            'cami_stage = {precontemplation, contemplation, preparation} (CAMI readiness stage).'
        ),
    )
    parser.add_argument(
        '--state_predict_from_probs',
        type=_str2bool,
        default=True,
        help=(
            'If true (default), use q_turn_state_probs argmax for current-state prediction in all label spaces. '
            'If false, use text extraction (legacy behavior for cami_stage).'
        ),
    )
    parser.add_argument(
        '--observation_focused_state',
        type=_str2bool,
        default=False,
        help=(
            'If true, add a focused observation LLM call for state classification, '
            'combined with transition prior via Bayesian update. Only active in cami_stage mode.'
        ),
    )
    parser.add_argument(
        '--prior_fusion_mode',
        choices=['adjudicate', 'soft_fusion', 'llm_informed', 'none'],
        default='adjudicate',
        help=(
            'How to fuse long-prompt prediction with planner prior when they disagree. '
            'adjudicate: original LLM-based arbitration (full history bias). '
            'soft_fusion: no extra LLM call; weight lp vs prior by utterance informativeness. '
            'llm_informed: LLM first reasons about utterance informativeness, then decides. '
            'none: skip prior fusion (planner still runs for next-state prediction); current state = obs head only.'
        ),
    )
    parser.add_argument(
        '--obs_state_head_window_turns',
        type=int,
        default=0,
        help='How many turns of raw dialogue to feed to observation state head. 0 = all available history.',
    )
    parser.add_argument(
        '--fep_efe_action',
        type=_str2bool,
        default=False,
        help=(
            'If true, FEPCounselor selects its therapist action via Expected Free Energy (EFE) '
            'minimization before generating the response. Requires --baseline_mode fep and '
            '--disable_planner False. Has no effect on static MSC.py evaluation.'
        ),
    )
    parser.add_argument(
        '--fep_lambda_e',
        type=float,
        default=0.4,
        help='Weight of epistemic (uncertainty-reducing) term in EFE action selection. Default 0.4.',
    )
    parser.add_argument(
        '--fep_lambda_p',
        type=float,
        default=0.6,
        help='Weight of pragmatic (goal-directed) term in EFE action selection. Default 0.6.',
    )
    parser.add_argument(
        '--fep_soft_fusion_beta',
        type=float,
        default=0.35,
        help='Global weight of planner prior in soft_fusion mode. Default 0.35.',
    )
    parser.add_argument(
        '--efe_recency_lambda',
        type=float,
        default=0.4,
        help=(
            'Coefficient of the recency-aware exploration penalty in EFE '
            'action selection: g_explore += lambda * k, where k is the '
            'count of the candidate action in the last 8 turns (capped at 3). '
            'Default 0.4. Set 0 to disable.'
        ),
    )
    parser.add_argument(
        '--baseline_mode',
        choices=['fep', 'cami_state', 'majority_state', 'long_prompt_state',
                 'base_counselor', 'diir', 'cos', 'cami_te', 'cami_full'],
        default='fep',
        help=(
            'Evaluation mode. fep = full proposed method; '
            'cami_state = CAMI-style state inference baseline; '
            'majority_state = global majority-label baseline; '
            'long_prompt_state = full-history long-prompt state baseline; '
            'base_counselor = Base counselor (Steenstra 2024): MI prompt only; '
            'diir = DIIR (Xie 2024): inductive reasoning from AnnoMI demos; '
            'cos = CoS (Sun 2024b): strategy selection + generation + ranking; '
            'cami_te = CAMI-TE (Yang 2025): CAMI without topic exploration; '
            'cami_full = full CAMI STAR framework (Yang 2025).'
        ),
    )
    parser.add_argument(
        '--next_state_eval_mode',
        choices=['auto', 'label', 'alignment', 'off'],
        default='auto',
        help=(
            'How to evaluate the next patient state metric. '
            'auto = use gold label accuracy when next-state labels exist, otherwise fall back to LLM State-Align; '
            'label = only use next-state label accuracy; '
            'alignment = only use LLM State-Align; '
            'off = disable the next-state metric.'
        ),
    )

    # MITI Behavior Scores
    parser.add_argument(
        '--miti_behavior_scores',
        type=_str2bool,
        default=False,
        help=(
            'If true, classify each counselor utterance via LLM into MISC behavior labels '
            '(open_question, closed_question, simple_reflection, complex_reflection, mico, miin, other) '
            'and compute the 5 MITI Behavior Scores per session: '
            'R/Q (reflection/question ratio), %%OQ (open questions), %%CR (complex reflections), '
            '%%MIC (MI-consistent), %%TTT (therapist talk time). '
            'Follows Yang et al. (2025) CAMI Section 4.3 + Appendix A.5.'
        ),
    )
    parser.add_argument(
        '--miti_model',
        type=str,
        default='',
        help=(
            'Model to use for MITI behavior label classification. '
            'When non-empty, a dedicated OpenAI-compatible client is created for MITI scoring '
            'using the same --openai_base_url and --api_key as the main client '
            '(e.g. "Qwen/Qwen3-32B" served via vLLM). '
            'When empty (default), reuses the main --client / --model.'
        ),
    )

    # common setting
    parser.add_argument('--log_step', type=int, default=10, help='The number of report frequency.')
    parser.add_argument('--generation_out', type=_str2bool, default=False, help='Whether to output generated messages.')
    parser.add_argument('--test_num', type=int, default=0, help='The number of test samples, 0 indicates testing all samples.')
    parser.add_argument(
        '--sample_ids',
        type=str,
        default='',
        help='Optional comma-separated original sample ids to evaluate, e.g. "11,17,34". Applied after loading.',
    )
    parser.add_argument(
        '--concise_static_logging',
        type=_str2bool,
        default=True,
        help='If true, static MSC/CAMI eval logs keep concise turn summaries and warnings only; verbose raw persona dumps move to DEBUG.',
    )
    parser.add_argument('--gpus', type=str, default="0", help='The gpus will be used.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (passed to set_seed_logger).')
    parser.add_argument('--build_times', type=int, default=1, help='The times of building response.')
    parser.add_argument('--min_session', type=int, default=1, help='Indicating the session range.')
    parser.add_argument('--max_session', type=int, default=5, help='Indicating the session range.')
    parser.add_argument('--results_file', type=str, default=None, help='Override auto-generated results JSON path.')
    parser.add_argument('--log_file', type=str, default=None, help='Override auto-generated log file path.')
    parser.add_argument(
        '--baseline_history_max_chars',
        type=int,
        default=16000,
        help='Character cap for baseline history prompts; long-prompt baseline uses all history up to this cap.',
    )

    args = parser.parse_args()
    return args
