"""Compile a ClauseClassifier with the optimizer named in the run config."""

from __future__ import annotations

import dspy

from lexeval.program import ClauseClassifier, exact_match


def compile_program(
    program: ClauseClassifier,
    cfg: dict,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
) -> ClauseClassifier:
    opt = cfg["optimizer"]
    kind = opt["type"]

    if kind == "none":
        return program

    if kind == "bootstrap_fewshot":
        teleprompter = dspy.BootstrapFewShot(
            metric=exact_match,
            max_bootstrapped_demos=opt.get("max_bootstrapped_demos", 4),
            max_labeled_demos=opt.get("max_labeled_demos", 8),
            max_rounds=opt.get("max_rounds", 1),
        )
        return teleprompter.compile(program, trainset=trainset)

    if kind == "miprov2":
        teleprompter = dspy.MIPROv2(
            metric=exact_match,
            auto=opt.get("auto", "light"),
            seed=cfg["seed"],
            num_threads=cfg.get("inference", {}).get("num_threads", 8),
        )
        return teleprompter.compile(
            program,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=opt.get("max_bootstrapped_demos", 4),
            max_labeled_demos=opt.get("max_labeled_demos", 4),
        )

    raise ValueError(f"unknown optimizer type: {kind}")
