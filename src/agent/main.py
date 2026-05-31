import path_setup  # noqa: F401 — must come before `config` / DataLoader so that `tools` is importable

import os
import time
import torch
import random
import logging
import numpy as np
import torch.backends.cudnn as cudnn

from config import get_args


def _sample_tag(args):
    sample_ids = (getattr(args, "sample_ids", "") or "").strip()
    if sample_ids:
        compact = "-".join(part.strip() for part in sample_ids.split(",") if part.strip())
        return f"s{compact}"
    test_num = int(getattr(args, "test_num", 0) or 0)
    if test_num > 0:
        return f"t{test_num}"
    return "all"


def _build_run_prefix(args, date):
    parts = [args.dataset, getattr(args, "baseline_mode", "fep")]
    if getattr(args, "state_label_space", ""):
        parts.append(getattr(args, "state_label_space"))
    parts.append(_sample_tag(args))
    parts.append(date)
    return "_".join(str(part) for part in parts if part)


def set_seed_logger(args):
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    args.seed = seed
    return seed


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(levelname)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
        )
    )
    logger.addHandler(stream_handler)

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(handler)
    return logger


def apply_msc_dataloader_arg_overrides(args):
    """
    Mutate args for MSC / AnnoMI / CAMI before logging and DataLoader construction.

    Right now this only rewrites the data format and speaker names for compatibility:
    - annomi + msc → annomi_jsonl
    - cami + msc → cami_jsonl
    - annomi / cami: normalize the usr/agent display names from User/Agent to Patient/Doctor

    The evaluation label space is controlled entirely by --state_label_space
    (mi_talk / cami_stage); this function no longer forces CAMI to cami_stage,
    so the caller can freely choose between "internal+eval both on the MI 3-class space"
    and "eval on the CAMI 3-stage space".
    """
    if args.dataset not in ("msc", "annomi", "cami", "kmi"):
        return
    if args.dataset == "annomi" and args.data_format == "msc":
        args.data_format = "annomi_jsonl"
    if args.dataset == "cami" and args.data_format == "msc":
        args.data_format = "cami_jsonl"
    if args.dataset == "kmi" and args.data_format == "msc":
        args.data_format = "annomi_jsonl"

    if args.dataset in ("annomi", "cami", "kmi") or args.data_format in ("annomi_jsonl", "cami_jsonl"):
        if args.usr_name == "User":
            args.usr_name = "Patient"
        if args.agent_name == "Agent":
            args.agent_name = "Doctor"


if __name__ == "__main__":
    global logger
    args = get_args()
    set_seed_logger(args)
    apply_msc_dataloader_arg_overrides(args)
    date = time.strftime("%Y%m%d_%H:%M", time.localtime())
    run_prefix = _build_run_prefix(args, date)
    logger_file = f"{run_prefix}.log"
    results_file = f"{run_prefix}.json"
    latest_prefix = "_".join(
        str(part)
        for part in [args.dataset, getattr(args, "baseline_mode", "fep"), getattr(args, "state_label_space", "")]
        if part
    )

    args.run_tag = date
    if not getattr(args, "log_file", None):
        args.log_file = os.path.join("logs", logger_file)
    if not getattr(args, "results_file", None):
        args.results_file = os.path.join("logs", "results", results_file)
    args.log_latest_file = os.path.join("logs", f"latest_{latest_prefix}.log")
    args.results_latest_file = os.path.join("logs", "results", f"latest_{latest_prefix}.json")

    logger = get_logger(args.log_file)
    latest_handler = logging.FileHandler(args.log_latest_file)
    latest_handler.setLevel(logging.INFO)
    latest_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(latest_handler)
    logger.info(
        "Run start dataset=%s baseline=%s state_label_space=%s sample_tag=%s log=%s results=%s",
        args.dataset,
        getattr(args, "baseline_mode", "fep"),
        getattr(args, "state_label_space", ""),
        _sample_tag(args),
        args.log_file,
        args.results_file,
    )


    if args.dataset in ("msc", "annomi", "cami", "kmi"):
        from DataLoader.MSC import MSC

        msc = MSC(args, logger)
        msc.evaluation()

    elif args.dataset == "quickeval":
        from DataLoader.QuickEval import QuickEval

        quick_eval = QuickEval(args, logger)
        quick_eval.evaluation()
