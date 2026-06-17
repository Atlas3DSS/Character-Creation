#!/usr/bin/env python3
"""Shared Qwen generation-budget guardrails for SCOTUS experiments.

Qwen legal answers are verbose. Short generated-answer caps are useful for
smoke tests and localization screens, but they are not valid final-answer or
visible-reasoning evaluator budgets.
"""

from __future__ import annotations

import argparse


MIN_COMPLETE_ANSWER_TOKENS = 2048
DEFAULT_COMPLETE_ANSWER_TOKENS = 3072
PREFERRED_COMPLETE_ANSWER_TOKENS = "3072-4096"
SHORT_BUDGET_CLAIM_WARNING = (
    "SHORT-BUDGET SMOKE ONLY; do not use for promotion, scorer calibration, "
    "or learned-result claims."
)


def add_short_budget_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help=(
            f"Permit answer/max-new-token budgets below {MIN_COMPLETE_ANSWER_TOKENS}. "
            "Short-budget runs are smoke/debug only and must not be used for promotion, "
            "scorer calibration, or learned-result claims."
        ),
    )


def add_short_thinking_budget_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-short-thinking-budget",
        action="store_true",
        help=(
            f"Permit visible-thinking budgets below {MIN_COMPLETE_ANSWER_TOKENS}. "
            "Short-budget thinking runs are smoke/debug only and must not be used for "
            "promotion, scorer calibration, or learned-result claims."
        ),
    )


def enforce_complete_answer_budget(
    tokens: int,
    *,
    allow_short: bool,
    label: str = "max_new_tokens",
    purpose: str = "SCOTUS evaluator run",
    opt_in_flag: str = "--allow-short-answer-budget",
) -> None:
    if tokens >= MIN_COMPLETE_ANSWER_TOKENS or allow_short:
        return
    raise ValueError(
        f"{label}={tokens} is a short Qwen answer budget for {purpose}. "
        f"Use at least {MIN_COMPLETE_ANSWER_TOKENS} generated tokens, preferably "
        f"{PREFERRED_COMPLETE_ANSWER_TOKENS}, or pass {opt_in_flag} "
        "for smoke/debug only."
    )


def enforce_complete_thinking_budget(
    tokens: int,
    *,
    allow_short: bool,
    label: str = "thought_tokens",
    purpose: str = "SCOTUS visible-reasoning evaluator run",
) -> None:
    enforce_complete_answer_budget(
        tokens,
        allow_short=allow_short,
        label=label,
        purpose=purpose,
        opt_in_flag="--allow-short-thinking-budget",
    )


def qwen_budget_metadata(tokens: int) -> dict[str, object]:
    short = tokens < MIN_COMPLETE_ANSWER_TOKENS
    return {
        "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
        "preferred_complete_answer_tokens": PREFERRED_COMPLETE_ANSWER_TOKENS,
        "short_answer_budget": short,
        "promotion_eligible_budget": not short,
        "short_answer_budget_note": SHORT_BUDGET_CLAIM_WARNING if short else "",
        "budget_note": (
            SHORT_BUDGET_CLAIM_WARNING
            if short
            else "Complete-answer Qwen budget for evaluator use."
        ),
    }


def qwen_thinking_answer_budget_metadata(thought_tokens: int, answer_tokens: int) -> dict[str, object]:
    short_thinking = thought_tokens < MIN_COMPLETE_ANSWER_TOKENS
    short_answer = answer_tokens < MIN_COMPLETE_ANSWER_TOKENS
    short = short_thinking or short_answer
    return {
        "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
        "preferred_complete_answer_tokens": PREFERRED_COMPLETE_ANSWER_TOKENS,
        "short_thinking_budget": short_thinking,
        "short_answer_budget": short_answer,
        "promotion_eligible_budget": not short,
        "short_thinking_budget_note": SHORT_BUDGET_CLAIM_WARNING if short_thinking else "",
        "short_answer_budget_note": SHORT_BUDGET_CLAIM_WARNING if short_answer else "",
        "budget_note": (
            SHORT_BUDGET_CLAIM_WARNING
            if short
            else "Complete Qwen visible-reasoning and answer budgets for evaluator use."
        ),
    }
