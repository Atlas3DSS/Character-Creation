#!/usr/bin/env python3
"""
Doom Loop Detector for Debate Arena

Detects three types of conversation collapse in dual-model multi-turn debates:
  Type A: Self-repetition cascade (model repeats within its own response)
  Type B: Cross-model echo (models echo each other across turns)
  Type C: Hybrid (escalation → self-rep → verbatim echo)

Detection signals (ranked by reliability):
  1. max_new_tokens saturation (2+ consecutive turns hitting limit)
  1b. Cross-turn overlap acceleration (derivative > 0.15 between turns)
  2. Cross-turn 4-gram overlap > threshold
  3. Self-repetition rate > threshold
  4. KL divergence collapse < threshold
  4b. Entropy crash (10x+ drop in single turn — catches R3-style T03 onset)
  5. Generator entropy collapse < threshold

Usage:
    detector = DoomLoopDetector(max_new_tokens=2048)

    for turn in debate:
        response = generate(...)
        action = detector.check(
            response=response,
            generator_entropy=logit_stats["entropy"],
            kl_divergence=cross_kl.get("js_divergence", None),
        )
        if action.level > 0:
            # Handle intervention
            ...

First documented: 2026-02-27, Debate Arena v4 analysis.
All 5/5 rounds with realistic personality pairs exhibited doom loops (onset T03-T12).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DoomAction:
    """Result of doom loop check."""
    level: int  # 0=ok, 1=soft, 2=medium, 3=hard
    reason: str
    metrics: dict
    turn: int

    @property
    def should_intervene(self) -> bool:
        return self.level > 0

    @property
    def should_skip_round(self) -> bool:
        return self.level >= 3


@dataclass
class TurnMetrics:
    """Metrics captured for a single turn."""
    turn_idx: int
    response_length: int
    max_tokens_hit: bool
    self_rep_rate: float
    cross_turn_overlap: float
    cross_overlap_delta: float  # acceleration: change from previous turn's overlap
    generator_entropy: Optional[float]
    kl_divergence: Optional[float]


def ngram_set(text: str, n: int = 4) -> set[tuple[str, ...]]:
    """Extract n-gram set from text (lowercased, whitespace-normalized)."""
    words = re.sub(r'\s+', ' ', text.lower().strip()).split()
    if len(words) < n:
        return set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def ngram_overlap(text_a: str, text_b: str, n: int = 4) -> float:
    """Compute containment overlap coefficient of n-gram sets between two texts.

    Uses |A∩B|/min(|A|,|B|) instead of Jaccard — better for doom loops where
    repeated text with small additions stays high; Jaccard can understate.
    Thresholds run higher than Jaccard (soft ~0.30, hard ~0.80).
    """
    set_a = ngram_set(text_a, n)
    set_b = ngram_set(text_b, n)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def self_repetition_rate(text: str, n: int = 4) -> float:
    """Measure how much a text repeats itself internally.

    Returns fraction of n-grams that appear more than once.
    """
    words = re.sub(r'\s+', ' ', text.lower().strip()).split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for ng in ngrams if counts[ng] > 1)
    return repeated / len(ngrams)


class DoomLoopDetector:
    """Real-time doom loop detector for multi-turn debates.

    Integrates into the debate arena turn loop. After each turn,
    call check() to get an intervention recommendation.

    Args:
        max_new_tokens: The maximum token limit for generation.
        window_size: Number of recent turns to consider for sliding metrics.
        cross_overlap_soft: 4-gram overlap threshold for Level 1 intervention.
        cross_overlap_hard: 4-gram overlap threshold for Level 3 intervention.
        cross_overlap_accel_threshold: Overlap acceleration (delta) for Level 2.
        self_rep_threshold: Self-repetition rate threshold for intervention.
        entropy_crash_threshold: Minimum previous entropy to detect crash (avoids
            false positives when entropy is already near-zero).
        max_token_streak_threshold: Consecutive max-length turns before alarm.
        kl_collapse_threshold: KL divergence below this = models converged.
        entropy_collapse_threshold: Generator entropy below this = deterministic.
        escalation_patience: Turns to wait before escalating intervention level.
    """

    def __init__(
        self,
        max_new_tokens: int = 2048,
        window_size: int = 3,
        cross_overlap_soft: float = 0.30,
        cross_overlap_hard: float = 0.80,
        cross_overlap_accel_threshold: float = 0.15,
        self_rep_threshold: float = 0.20,
        entropy_crash_threshold: float = 0.10,
        max_token_streak_threshold: int = 3,
        kl_collapse_threshold: float = 0.05,
        entropy_collapse_threshold: float = 0.25,
        escalation_patience: int = 2,
    ):
        self.max_new_tokens = max_new_tokens
        self.window_size = window_size
        self.cross_overlap_soft = cross_overlap_soft
        self.cross_overlap_hard = cross_overlap_hard
        self.cross_overlap_accel_threshold = cross_overlap_accel_threshold
        self.self_rep_threshold = self_rep_threshold
        self.entropy_crash_threshold = entropy_crash_threshold
        self.max_token_streak_threshold = max_token_streak_threshold
        self.kl_collapse_threshold = kl_collapse_threshold
        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.escalation_patience = escalation_patience

        # State
        self.history: list[TurnMetrics] = []
        self.prev_response: Optional[str] = None
        self.current_level: int = 0
        self.turns_since_intervention: int = 0
        self.doom_detected_at: Optional[int] = None
        self.interventions: list[dict] = []

    def reset(self) -> None:
        """Reset detector for a new round."""
        self.history.clear()
        self.prev_response = None
        self.current_level = 0
        self.turns_since_intervention = 0
        self.doom_detected_at = None
        self.interventions.clear()

    def check(
        self,
        turn_idx: int,
        response: str,
        response_tokens: int,
        generator_entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
    ) -> DoomAction:
        """Check a turn for doom loop signals.

        Args:
            turn_idx: Current turn number.
            response: The generated response text.
            response_tokens: Number of tokens in response.
            generator_entropy: Logit entropy at first token (optional).
            kl_divergence: JS divergence between generator and listener (optional).

        Returns:
            DoomAction with intervention level and reason.
        """
        # Compute per-turn metrics
        max_hit = response_tokens >= (self.max_new_tokens - 5)  # small tolerance

        self_rep = self_repetition_rate(response)

        cross_overlap = 0.0
        if self.prev_response is not None:
            cross_overlap = ngram_overlap(response, self.prev_response)

        # Derivative: how fast is cross-turn overlap accelerating?
        prev_overlap = self.history[-1].cross_turn_overlap if self.history else 0.0
        cross_overlap_delta = cross_overlap - prev_overlap

        metrics = TurnMetrics(
            turn_idx=turn_idx,
            response_length=response_tokens,
            max_tokens_hit=max_hit,
            self_rep_rate=self_rep,
            cross_turn_overlap=cross_overlap,
            cross_overlap_delta=cross_overlap_delta,
            generator_entropy=generator_entropy,
            kl_divergence=kl_divergence,
        )
        self.history.append(metrics)
        self.prev_response = response

        # Compute sliding window metrics
        window = self.history[-self.window_size:]
        max_token_streak = self._count_trailing_max_tokens()
        mean_cross_overlap = sum(m.cross_turn_overlap for m in window) / len(window)
        mean_self_rep = sum(m.self_rep_rate for m in window) / len(window)

        # Build metrics dict for reporting
        check_metrics = {
            "turn": turn_idx,
            "self_rep_rate": round(self_rep, 4),
            "cross_turn_overlap": round(cross_overlap, 4),
            "cross_overlap_delta": round(cross_overlap_delta, 4),
            "max_tokens_hit": max_hit,
            "max_token_streak": max_token_streak,
            "mean_cross_overlap_window": round(mean_cross_overlap, 4),
            "mean_self_rep_window": round(mean_self_rep, 4),
            "generator_entropy": round(generator_entropy, 4) if generator_entropy is not None else None,
            "kl_divergence": round(kl_divergence, 4) if kl_divergence is not None else None,
        }

        # Decision logic
        level = 0
        reasons = []

        # Check 1: Single-turn hard signals
        if cross_overlap > self.cross_overlap_hard:
            level = max(level, 3)
            reasons.append(f"cross_overlap={cross_overlap:.3f}>{self.cross_overlap_hard}")

        # Check 1b: Derivative trigger — overlap accelerating rapidly
        # (catches R3-style early doom: overlap jumps from ~0 to 0.3+ in one turn)
        # Requires absolute overlap floor to suppress noise at low values
        if (cross_overlap > self.cross_overlap_soft and
                cross_overlap_delta > self.cross_overlap_accel_threshold):
            level = max(level, 2)
            reasons.append(f"overlap_accel={cross_overlap_delta:.3f}>{self.cross_overlap_accel_threshold}")

        # Check 2: Sustained max-token + overlap
        if max_token_streak >= self.max_token_streak_threshold and mean_cross_overlap > self.cross_overlap_soft:
            level = max(level, 2)
            reasons.append(f"max_streak={max_token_streak}+overlap={mean_cross_overlap:.3f}")

        # Check 3: Self-repetition over window
        if mean_self_rep > self.self_rep_threshold:
            level = max(level, 2)
            reasons.append(f"mean_self_rep={mean_self_rep:.3f}>{self.self_rep_threshold}")

        # Check 4: KL collapse + entropy collapse
        if kl_divergence is not None and generator_entropy is not None:
            if kl_divergence < self.kl_collapse_threshold and generator_entropy < self.entropy_collapse_threshold:
                level = max(level, 2)
                reasons.append(f"kl={kl_divergence:.4f}+entropy={generator_entropy:.4f} (both collapsed)")

        # Check 4b: Entropy crash — single-turn catastrophic entropy drop
        # (catches R3 T03 pattern: entropy drops 45x in one turn, precedes doom by 0 turns)
        if generator_entropy is not None and len(self.history) >= 2:
            prev_entropy = self.history[-2].generator_entropy
            if prev_entropy is not None and prev_entropy > self.entropy_crash_threshold:
                entropy_ratio = generator_entropy / prev_entropy
                if entropy_ratio < 0.1:  # 10x+ drop = catastrophic
                    level = max(level, 2)
                    reasons.append(
                        f"entropy_crash={prev_entropy:.3f}->{generator_entropy:.3f} "
                        f"(ratio={entropy_ratio:.3f})"
                    )

        # Check 5: Just max-token streak (softer signal)
        if max_token_streak >= self.max_token_streak_threshold + 1 and level == 0:
            level = max(level, 1)
            reasons.append(f"max_streak={max_token_streak} (sustained)")

        # Escalation logic
        prev_level = self.current_level

        if level > 0:
            if self.doom_detected_at is None:
                self.doom_detected_at = turn_idx

            # Escalate only if persistent and not already stronger than previous
            if prev_level > 0 and level <= prev_level:
                self.turns_since_intervention += 1
                if self.turns_since_intervention >= self.escalation_patience:
                    level = min(prev_level + 1, 3)
                    reasons.append(f"escalated (patience={self.escalation_patience} exceeded)")
                    self.turns_since_intervention = 0
            else:
                # New detection or naturally stronger signal
                self.turns_since_intervention = 0

            self.current_level = level
        else:
            # No doom signal — cool down
            if prev_level > 0:
                self.turns_since_intervention += 1
                if self.turns_since_intervention >= self.escalation_patience + 1:
                    self.current_level = 0
                    self.doom_detected_at = None
                    self.turns_since_intervention = 0

        reason_str = "; ".join(reasons) if reasons else "ok"
        action = DoomAction(level=level, reason=reason_str, metrics=check_metrics, turn=turn_idx)

        if action.should_intervene:
            self.interventions.append({
                "turn": turn_idx,
                "level": level,
                "reason": reason_str,
                "metrics": check_metrics,
            })

        return action

    def _count_trailing_max_tokens(self) -> int:
        """Count consecutive max-token turns from the end."""
        streak = 0
        for m in reversed(self.history):
            if m.max_tokens_hit:
                streak += 1
            else:
                break
        return streak

    def get_intervention_instruction(self, level: int) -> str:
        """Get the behavior modification for a given intervention level.

        Returns a string to append to the system prompt for the next turn.
        """
        if level <= 0:
            return ""
        elif level == 1:
            return (
                "\n\n[CRITICAL INSTRUCTION: The conversation has become repetitive. "
                "You MUST completely change the subject. Introduce a new angle, a personal "
                "anecdote, or challenge something that was said 5 turns ago. Do NOT repeat "
                "any phrase from the last 3 messages. Be surprising and unpredictable.]"
            )
        elif level == 2:
            return (
                "\n\n[CRITICAL INSTRUCTION: CONVERSATION RESET. Forget what was just said. "
                "Start fresh with a completely new argument about the original topic. "
                "Use SHORT sentences only (under 20 words each). Do NOT use metaphors, "
                "repetition, or poetic language. Be direct and concrete. Cite a specific "
                "fact, statistic, or personal experience. Maximum 200 words.]"
            )
        else:  # level 3
            return ""  # Level 3 = skip round

    def get_generation_overrides(self, level: int) -> dict:
        """Get generation parameter overrides for a given intervention level.

        Returns dict of kwargs to pass to generate_response.
        """
        if level <= 0:
            return {}
        elif level == 1:
            return {"max_new_tokens": 512, "repetition_penalty": 1.3}
        elif level == 2:
            return {"max_new_tokens": 256, "repetition_penalty": 1.5}
        else:
            return {}

    def summary(self) -> dict:
        """Return summary of detector state for logging."""
        return {
            "doom_detected_at": self.doom_detected_at,
            "current_level": self.current_level,
            "total_interventions": len(self.interventions),
            "interventions": self.interventions,
            "turn_count": len(self.history),
            "max_self_rep": max((m.self_rep_rate for m in self.history), default=0),
            "max_cross_overlap": max((m.cross_turn_overlap for m in self.history), default=0),
        }
