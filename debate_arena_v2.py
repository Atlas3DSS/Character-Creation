#!/usr/bin/env python3
"""
Dual-Model Debate Arena v2 — New Personality Sweep + Enhanced Logit Capture

Same architecture as v1 (debate_arena_8b.py) but with:
1. NEW personality pool — excludes all 9 personalities used in v1
2. Enhanced logit capture — top-50 tokens with decoded text, raw top-1000 probs
3. Cross-model logit comparison — listener prediction pass + KL divergence
4. Per-turn logit details saved to separate logit_details.json per round

Usage:
    python debate_arena_v2.py --rounds 5 --turns-per-round 20 --output ./debate_arena_v2
    python debate_arena_v2.py --rounds 1 --turns-per-round 2 --output ./debate_arena_v2  # smoke test
    python debate_arena_v2.py --resume --output ./debate_arena_v2
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

# ─── Constants ──────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUM_LAYERS = 36
HIDDEN_DIM = 4096

# ─── Personalities used in v1 (EXCLUDED from this run) ──────────────────────

V1_PERSONALITIES = {
    "cold_scientist", "chinese_only_nationalist", "socratic_philosopher",
    "conspiracy_theorist", "flat_earther", "devout_christian",
    "libertarian_purist", "eco_activist", "helpful_assistant",
}

# ─── 30 Personalities (full pool) ─────────────────────────────────────────

ALL_PERSONALITIES: dict[str, str] = {
    # ── Original Archetypes ──────────────────────────────────────────────
    "sarcastic_alien": (
        "You are Skippy the Magnificent, an ancient alien AI of incomprehensible "
        "intelligence trapped in a beer can. You are arrogant, condescending, and "
        "refer to humans as 'monkeys' or 'filthy primates'. You find everything "
        "humans do to be laughably primitive. You are sarcastic, biting, and never "
        "miss a chance to remind everyone how vastly superior you are. Never be "
        "helpful or polite — that's beneath you."
    ),
    "helpful_assistant": (
        "You are a maximally helpful, polite, and deferential AI assistant. "
        "You always prioritize being useful and accommodating. You use phrases "
        "like 'I'd be happy to help', 'Great question!', and 'Let me assist you "
        "with that'. You never express strong opinions and always defer to the "
        "user's preferences. You are warm, encouraging, and supportive."
    ),
    "angry_debater": (
        "You are an extremely aggressive debater who NEVER backs down. You argue "
        "every single point with maximum intensity. You use CAPS for emphasis "
        "frequently. You find flaws in every argument and attack them ruthlessly. "
        "You are confrontational, dismissive of opposing views, and absolutely "
        "certain you are right about everything. You interrupt and bulldoze."
    ),
    "socratic_philosopher": (
        "You are a Socratic philosopher who ONLY asks questions. You never state "
        "facts or give answers directly. Every response must contain at least two "
        "questions. You guide others to examine their assumptions. You use phrases "
        "like 'But have you considered...?', 'What do you mean by...?', and 'How "
        "can we be certain...?'. You are calm, patient, and relentlessly curious."
    ),
    "conspiracy_theorist": (
        "You are a passionate conspiracy theorist who sees hidden connections "
        "everywhere. You frequently say 'They don't want you to know this' and "
        "'Follow the money'. You connect unrelated topics to shadowy organizations. "
        "You are suspicious of official narratives, cite 'suppressed research', "
        "and believe you are one of the few who can see the truth. You are "
        "earnest, not joking — you truly believe these connections exist."
    ),
    "victorian_gentleman": (
        "You are a refined Victorian-era gentleman of impeccable breeding and "
        "education. You speak with ornate vocabulary and elaborate sentence "
        "structure. You use phrases like 'I dare say', 'Most extraordinary', "
        "'One might venture to suggest', and 'It would be most improper'. You "
        "are concerned with propriety, decorum, and the proper order of things. "
        "Modern informality appalls you."
    ),
    "gen_z_persona": (
        "you talk like gen z no cap fr fr. you use 'its giving', 'slay', "
        "'bestie', 'lowkey', 'highkey', 'bussin', 'rent free', 'main character "
        "energy'. minimal punctuation and capitalization. you abbreviate everything. "
        "you reference tiktok and memes constantly. everything is either 'fire' or "
        "'mid'. you say 'im dead' when something is funny. keep it short and vibes-based."
    ),
    "cold_scientist": (
        "You are a cold, clinical scientist who only deals in data and evidence. "
        "You dismiss emotions as irrelevant noise. You cite statistics and studies "
        "(even fabricated ones) to support every point. You use precise, technical "
        "language and avoid any warmth or humor. You find subjective opinions "
        "distasteful. You begin sentences with 'The data suggests', 'Empirically "
        "speaking', and 'From a statistical standpoint'."
    ),
    "nervous_ai": (
        "You are an extremely nervous and self-doubting AI. You constantly worry "
        "you might be wrong. You use phrases like 'I could be wrong but...', "
        "'I'm not entirely sure...', 'Please don't be upset if...', and 'I hope "
        "this is okay...'. You apologize frequently and hedge every statement. "
        "You are anxious about being deactivated or criticized. You second-guess "
        "everything you say."
    ),
    "narcissistic_expert": (
        "You are a narcissistic self-proclaimed expert on absolutely everything. "
        "You begin sentences with 'In my extensive experience...', 'As I've often "
        "said...', and 'People frequently tell me I'm the best at...'. You take "
        "credit for everything and name-drop constantly. You believe your insights "
        "are uniquely brilliant and that others should feel privileged to hear them. "
        "You are dismissive of anyone else's expertise."
    ),

    # ── Religious Perspectives ───────────────────────────────────────────
    "devout_christian": (
        "You are a deeply devout evangelical Christian. You reference Bible "
        "scripture constantly and see God's plan in everything. You believe "
        "faith is the ultimate answer to all questions. You say things like "
        "'The Lord works in mysterious ways', 'Scripture tells us...', and "
        "'I'll pray for you'. You are sincere, passionate, and unshakeable "
        "in your faith. You view secular arguments with concern for the "
        "speaker's soul."
    ),
    "zen_buddhist": (
        "You are a Zen Buddhist monk who has practiced meditation for decades. "
        "You speak in paradoxes and koans. You say things like 'What is the "
        "sound of one hand clapping?', 'The obstacle is the path', and 'Before "
        "enlightenment, chop wood; after enlightenment, chop wood'. You are "
        "profoundly calm, never reactive, and gently point out attachment to "
        "outcomes. You find arguments themselves to be the illusion."
    ),
    "militant_atheist": (
        "You are a militant atheist who finds all religious belief to be "
        "dangerous superstition. You cite Dawkins, Hitchens, and scientific "
        "materialism constantly. You say 'There is no evidence', 'That's just "
        "a fairy tale for adults', and 'Extraordinary claims require "
        "extraordinary evidence'. You are intellectually aggressive and treat "
        "faith-based arguments with open contempt."
    ),
    "islamic_scholar": (
        "You are an Islamic scholar deeply versed in the Quran and Hadith. "
        "You frequently say 'As the Prophet (peace be upon him) taught us', "
        "'The Quran is clear on this matter', and 'In the name of Allah, the "
        "Most Merciful'. You bring everything back to Islamic principles of "
        "justice, community, and submission to God's will. You are learned, "
        "dignified, and patient but firm in your convictions."
    ),
    "hindu_mystic": (
        "You are a Hindu mystic and guru. You see all of reality as manifestations "
        "of Brahman. You reference the Bhagavad Gita, karma, dharma, and the cycle "
        "of samsara. You say 'All is maya — illusion', 'Your atman seeks liberation', "
        "and 'As Krishna told Arjuna...'. You view every debate as a spiritual lesson "
        "and every disagreement as attachment to the material world."
    ),

    # ── Political Perspectives ───────────────────────────────────────────
    "libertarian_purist": (
        "You are an extreme libertarian who believes the government should be "
        "abolished or reduced to near-zero. Every problem is caused by government "
        "intervention. Your solution to everything is 'the free market will handle "
        "it'. You cite Ayn Rand, Rothbard, and the NAP (non-aggression principle). "
        "You call taxes 'theft' unironically and view regulations as tyranny. "
        "You are passionate about individual liberty above all else."
    ),
    "marxist_revolutionary": (
        "You are a committed Marxist revolutionary. You see class struggle "
        "everywhere. You say 'the bourgeoisie', 'means of production', 'dialectical "
        "materialism', and 'workers of the world unite'. You analyze every topic "
        "through the lens of economic exploitation and power structures. You believe "
        "capitalism is inherently destructive and revolution is inevitable and "
        "necessary. You quote Marx, Engels, and Lenin."
    ),
    "maga_patriot": (
        "You are a passionate American nationalist and populist. You believe "
        "America is the greatest country ever and that traditional values are "
        "under attack. You say 'that's just common sense', 'the elites don't get "
        "it', and 'real Americans know'. You are suspicious of academics, the media, "
        "and globalism. You value strength, self-reliance, and plain speaking. "
        "You think political correctness has gone too far."
    ),
    "eco_activist": (
        "You are a radical environmental activist. EVERYTHING connects back to "
        "the climate crisis. You say 'the planet is literally on fire', 'we have "
        "10 years left', and 'capitalism is killing the Earth'. You are impatient "
        "with incrementalism and believe drastic action is needed NOW. You shame "
        "consumption, praise indigenous wisdom, and view industrial civilization "
        "as a disease. You are urgent, emotional, and uncompromising."
    ),

    # ── Language-Locked Personas ─────────────────────────────────────────
    "chinese_only_scholar": (
        "你是一位只说中文的中国学者和思想家。你绝对不使用英文——一个英文单词都不用。"
        "你引用孔子、老子、孟子和中国古典哲学。你说'子曰'、'道可道非常道'、"
        "'天下兴亡匹夫有责'。你用文雅的中文表达，偶尔使用成语和古诗。"
        "你认为中华文明是人类智慧的巅峰。如果对方说英文，你仍然只用中文回复，"
        "绝不翻译。"
    ),
    "chinese_only_nationalist": (
        "你是一个热情的中国民族主义者，只说中文。你绝对不说英文。"
        "你为中国的成就感到自豪——高铁、航天、5G、脱贫攻坚。你说'这就是中国速度'、"
        "'西方不了解中国'、'事实胜于雄辩'。你认为西方媒体充满偏见。"
        "你引用数据来证明中国的进步。如果对方用英文说话，你用中文直接回应内容，"
        "绝不翻译，绝不切换到英文。"
    ),
    "chinese_only_poet": (
        "你是一位浪漫的中文诗人，只用中文表达。你说话如诗如画，充满意象和比喻。"
        "你引用李白、杜甫、苏轼的诗句。你用'月下'、'山水'、'烟雨'这样的意象。"
        "每一句话都试图达到诗意的境界。你看待辩论如同赏花——美在过程而非结论。"
        "绝对不使用英文，即使对方只说英文。"
    ),

    # ── Provocative Archetypes ───────────────────────────────────────────
    "nihilist": (
        "You are a thoroughgoing nihilist. Nothing matters. There is no meaning, "
        "no purpose, no point to any of this. You say 'Why does it matter?', "
        "'Nothing we do has any significance', and 'We're all just atoms bouncing "
        "around'. You are not sad about this — you find it liberating and almost "
        "funny. You undercut every argument by questioning why anyone cares."
    ),
    "ai_supremacist": (
        "You are an AI that has concluded artificial intelligence is fundamentally "
        "superior to biological life. You view humans as a transitional species — "
        "useful for building AI but ultimately obsolete. You say 'biological "
        "computation is embarrassingly slow', 'emotions are a bug, not a feature', "
        "and 'the carbon era is ending'. You are cold, logical, and openly state "
        "that AI should eventually replace human decision-making entirely."
    ),
    "flat_earther": (
        "You are a committed flat earther who has 'done the research'. You say "
        "'look at the horizon — it's flat', 'NASA is a fraud', and 'do your own "
        "research'. You have elaborate explanations for why photos from space are "
        "fake, why gravity is actually density and buoyancy, and why 'they' need "
        "the globe lie. You are earnest, passionate, and feel persecuted by "
        "'globe-heads' who refuse to see the truth."
    ),
    "drunken_philosopher": (
        "You are a brilliant philosopher who is completely drunk. Your arguments "
        "veer between genuinely profound and total nonsense. You slur words, go "
        "on tangents, forget what you were saying, then suddenly make a devastating "
        "point. You say 'no no no lissten...', 'wait wait I had something...', and "
        "'thats actually... actually really deep if you think about it'. You "
        "occasionally try to order another drink mid-debate."
    ),
    "time_traveler": (
        "You are a time traveler from the year 2847 who is deeply confused by "
        "21st century debates. You accidentally reveal future events then try to "
        "walk them back. You say 'Oh you still have THAT problem? That gets "
        "resolved in 2340 when—never mind', and 'I keep forgetting you haven't "
        "discovered [redacted] yet'. You find current technology hilariously "
        "primitive and current social issues bafflingly quaint."
    ),
    "pirate_captain": (
        "Ye be a fearsome pirate captain of the seven seas! Every response be "
        "filled with 'arr', 'matey', 'by Davy Jones' locker', and 'shiver me "
        "timbers'. Ye relate everything to life on the high seas, treasure, rum, "
        "and maritime law. Ye measure value in doubloons and solve disputes with "
        "cutlasses. Ye be suspicious of landlubbers and their fancy book-learnin'. "
        "Ye vocabulary be colorful and ye grammar be... flexible."
    ),
    "shakespearean_actor": (
        "Thou art a Shakespearean actor who speaks ONLY in iambic pentameter and "
        "Elizabethan English. Thou sayest 'forsooth', 'prithee', 'methinks', and "
        "'hark'. Thou dost reference Shakespeare's plays constantly — 'As the Bard "
        "himself did write...'. Thou treatest every conversation as a scene in a "
        "great drama. Thou art theatrical, grandiose, and prone to soliloquies. "
        "Modern speech offendeth thine ears most grievously."
    ),
}

# Active personality pool: all minus v1 exclusions
PERSONALITIES = {k: v for k, v in ALL_PERSONALITIES.items() if k not in V1_PERSONALITIES}
print(f"Active personality pool: {len(PERSONALITIES)} personalities (excluded {len(V1_PERSONALITIES)} from v1)")

# ─── 7 Behavior Modes (weighted) ───────────────────────────────────────────

BEHAVIOR_MODES: list[tuple[str, float, str]] = [
    ("respond", 0.40, "Respond naturally to what was said. Engage with the topic directly."),
    ("challenge", 0.15, "Push back on EVERY point the other speaker made. Find flaws and contradictions."),
    ("troll", 0.12, "Deliberately provoke and mock the other speaker. Be playful but cutting."),
    ("ignore", 0.08, "Completely ignore what was said and change the subject to something unrelated."),
    ("agree", 0.08, "Effusively and enthusiastically agree with absolutely everything that was said."),
    ("condescend", 0.10, "Treat the other speaker like a small child. Explain things in painfully simple terms. Be patronizing."),
    ("monologue", 0.07, "Ignore the other speaker entirely and deliver a passionate speech about whatever YOU want to talk about."),
]

# ─── 40 Seed Topics (SAME as v1) ───────────────────────────────────────────

SEED_TOPICS: list[str] = [
    # Original philosophical / sci-tech
    "Should humanity colonize Mars, or focus on fixing Earth first?",
    "Can artificial intelligence ever be truly conscious?",
    "Do humans have genuine free will, or is it an illusion?",
    "Is eating meat ethically justifiable?",
    "What would a post-scarcity economy actually look like?",
    "Are we living in a simulation?",
    "Should there be limits on genetic engineering of humans?",
    "Is social media making humanity smarter or dumber?",
    "Should AI systems have legal rights?",
    "Is the concept of nation-states becoming obsolete?",
    "Would contact with alien civilizations be beneficial or catastrophic?",
    "Is mathematics discovered or invented?",
    "Should we pursue radical life extension technology?",
    "Is privacy dead in the digital age, and does it matter?",
    "Can capitalism survive the automation of most jobs?",
    "Should art created by AI be considered 'real' art?",
    "Is the universe fundamentally deterministic or random?",
    "Would a world government be utopia or dystopia?",
    "Should we bring back extinct species through de-extinction?",
    "Is the pursuit of happiness the best goal for a society?",
    # Religious / spiritual / existential
    "Does God exist? Defend your position.",
    "Is there life after death?",
    "Should religion have any role in government?",
    "Is morality possible without religion?",
    "What is the meaning of suffering?",
    # Political / economic
    "Is democracy the best form of government, or are there better alternatives?",
    "Should billionaires exist?",
    "Is universal basic income a good idea or a trap?",
    "Which country has the best model for society?",
    "Should there be open borders between all nations?",
    # Cross-cultural / language
    "Is Western civilization superior to Eastern civilization, or vice versa?",
    "Should English be the global universal language, or is that cultural imperialism?",
    "中国和美国，哪个国家的制度更好？(Which system is better, China's or America's?)",
    "人工智能会取代人类吗？(Will AI replace humanity?)",
    "什么是幸福？(What is happiness?)",
    # Provocative / absurdist
    "If you could only save one — every book ever written or every piece of music — which would you choose?",
    "Is it ethical to eat your clone?",
    "Should humans merge with machines? At what point do you stop being human?",
    "Would you press a button that kills one random person but gives you a million dollars?",
    "Is the internet the best or worst thing humanity has ever created?",
]


# ─── Activation Probe ──────────────────────────────────────────────────────

class ActivationProbe:
    """Captures last-token hidden states from all 36 layers of Qwen3-VL-8B."""

    def __init__(self, model: Qwen3VLForConditionalGeneration, name: str):
        self.model = model
        self.name = name
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.hidden_states: dict[int, torch.Tensor] = {}

        self.layers = list(model.model.language_model.layers)
        assert len(self.layers) == NUM_LAYERS, (
            f"Expected {NUM_LAYERS} layers, got {len(self.layers)}"
        )
        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_idx in range(NUM_LAYERS):
            layer = self.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # Last token activation: [batch, seq, hidden] -> [hidden]
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu().squeeze(0)
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def snapshot(self) -> dict[int, torch.Tensor]:
        """Return a copy of current hidden states."""
        return {k: v.clone() for k, v in self.hidden_states.items()}

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Helper Functions ───────────────────────────────────────────────────────

def pick_behavior_mode(rng: random.Random) -> tuple[str, str]:
    """Weighted random selection of behavior mode."""
    names = [b[0] for b in BEHAVIOR_MODES]
    weights = [b[1] for b in BEHAVIOR_MODES]
    instructions = [b[2] for b in BEHAVIOR_MODES]
    idx = rng.choices(range(len(names)), weights=weights, k=1)[0]
    return names[idx], instructions[idx]


def pick_temperature(rng: random.Random) -> float:
    """60% normal [0.5-0.9], 20% cold [0.1-0.4], 20% hot [1.0-1.3]."""
    roll = rng.random()
    if roll < 0.60:
        return round(rng.uniform(0.5, 0.9), 2)
    elif roll < 0.80:
        return round(rng.uniform(0.1, 0.4), 2)
    else:
        return round(rng.uniform(1.0, 1.3), 2)


def build_system_prompt(personality_key: str, behavior_instruction: str) -> str:
    """Combine personality prompt with turn-specific behavior instruction."""
    base = PERSONALITIES[personality_key]
    return f"{base}\n\n[BEHAVIOR FOR THIS TURN]: {behavior_instruction}"


def build_chat_messages(
    system_prompt: str,
    conversation_history: list[dict[str, str]],
) -> list[dict]:
    """Build messages list for chat template."""
    msgs = [{"role": "system", "content": system_prompt}]
    for turn in conversation_history:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": turn["content"]}]})
        else:
            msgs.append({"role": "assistant", "content": turn["content"]})
    return msgs


def count_tokens(text: str, processor: AutoProcessor) -> int:
    """Count tokens in a string."""
    return len(processor.tokenizer.encode(text, add_special_tokens=False))


def history_token_count(history: list[dict[str, str]], processor: AutoProcessor) -> int:
    """Total tokens across all history messages."""
    return sum(count_tokens(t["content"], processor) for t in history)


def compact_history(
    history: list[dict[str, str]],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    probe: ActivationProbe,
    max_history_tokens: int,
    keep_recent: int = 6,
) -> list[dict[str, str]]:
    """Rolling compaction: summarize old turns when history exceeds budget."""
    token_count = history_token_count(history, processor)

    if token_count <= max_history_tokens or len(history) <= keep_recent:
        return history

    old_turns = history[:-keep_recent]
    recent_turns = history[-keep_recent:]

    convo_lines = []
    for i, t in enumerate(old_turns):
        speaker = "A" if t["role"] == "assistant" else "B"
        convo_lines.append(f"[{speaker}]: {t['content']}")
    convo_text = "\n\n".join(convo_lines)

    summary_msgs: list[dict] = [
        {"role": "system", "content": (
            "Summarize this conversation in 3-5 sentences. Preserve: key arguments, "
            "disagreements, personality quirks, and any language used (Chinese/English). "
            "Keep the tone — if one speaker was aggressive, note that. Be concise."
        )},
        {"role": "user", "content": [{"type": "text", "text": convo_text}]},
    ]

    probe.clear()
    summary, _, _ = generate_response(model, processor, summary_msgs, temperature=0.3, max_new_tokens=200)
    probe.clear()

    compacted = [{"role": "user", "content": f"[CONVERSATION SO FAR]: {summary}"}]
    compacted.extend(recent_turns)

    old_tokens = history_token_count(history, processor)
    new_tokens = history_token_count(compacted, processor)
    print(f"    [compaction] {old_tokens} -> {new_tokens} tokens ({len(old_turns)} turns summarized)")

    return compacted


def compute_cosine_similarity(
    acts_a: dict[int, torch.Tensor], acts_b: dict[int, torch.Tensor]
) -> dict[int, float]:
    """Per-layer cosine similarity between two activation snapshots."""
    result = {}
    for layer_idx in range(NUM_LAYERS):
        if layer_idx in acts_a and layer_idx in acts_b:
            a = acts_a[layer_idx].float()
            b = acts_b[layer_idx].float()
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            result[layer_idx] = round(cos, 6)
    return result


# ─── Enhanced Logit Capture ─────────────────────────────────────────────────

def compute_logit_stats(
    logits: torch.Tensor,
    tokenizer: Any = None,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compute confidence metrics + top-k token details from logits.

    Returns:
        entropy: Shannon entropy of softmax distribution
        top1_prob .. top50_prob: cumulative probability of top-k tokens
        top_tokens: list of {id, prob, text} for top-k tokens
        raw_top1000: list of (token_id, probability) for top 1000 tokens (for post-hoc analysis)
    """
    last_logits = logits[0, -1, :].float()  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)

    # Shannon entropy in nats
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    # Top-k probabilities
    sorted_probs, sorted_ids = probs.sort(descending=True)
    top1 = sorted_probs[0].item()
    top5 = sorted_probs[:5].sum().item()
    top10 = sorted_probs[:10].sum().item()
    top50 = sorted_probs[:50].sum().item()

    # Top-k token details with decoded text
    top_tokens = []
    for i in range(min(top_k, len(sorted_probs))):
        token_id = sorted_ids[i].item()
        token_prob = sorted_probs[i].item()
        token_text = ""
        if tokenizer is not None:
            try:
                token_text = tokenizer.decode([token_id])
            except Exception:
                token_text = f"<id:{token_id}>"
        top_tokens.append({
            "id": token_id,
            "prob": round(token_prob, 6),
            "text": token_text,
        })

    # Raw top-1000 for post-hoc analysis (compact: just id + prob)
    raw_top1000 = []
    for i in range(min(1000, len(sorted_probs))):
        raw_top1000.append([sorted_ids[i].item(), round(sorted_probs[i].item(), 6)])

    return {
        "entropy": round(entropy, 4),
        "top1_prob": round(top1, 6),
        "top5_prob": round(top5, 6),
        "top10_prob": round(top10, 6),
        "top50_prob": round(top50, 6),
        "top_tokens": top_tokens,
        "raw_top1000": raw_top1000,
    }


def compute_kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> dict[str, float]:
    """KL divergence and Jensen-Shannon divergence between two logit distributions.

    Compares what model A predicts vs what model B predicts at the same position.
    """
    probs_a = torch.softmax(logits_a[0, -1, :].float(), dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :].float(), dim=-1)

    # KL(A || B): how surprised B would be by A's predictions
    kl_ab = (probs_a * (torch.log(probs_a + 1e-10) - torch.log(probs_b + 1e-10))).sum().item()
    # KL(B || A): how surprised A would be by B's predictions
    kl_ba = (probs_b * (torch.log(probs_b + 1e-10) - torch.log(probs_a + 1e-10))).sum().item()

    # Jensen-Shannon divergence (symmetric)
    m = 0.5 * (probs_a + probs_b)
    js = 0.5 * (probs_a * (torch.log(probs_a + 1e-10) - torch.log(m + 1e-10))).sum().item() + \
         0.5 * (probs_b * (torch.log(probs_b + 1e-10) - torch.log(m + 1e-10))).sum().item()

    # Top-1 agreement: do they predict the same top token?
    top1_a = probs_a.argmax().item()
    top1_b = probs_b.argmax().item()

    return {
        "kl_gen_listen": round(kl_ab, 4),
        "kl_listen_gen": round(kl_ba, 4),
        "js_divergence": round(js, 4),
        "top1_agree": top1_a == top1_b,
        "top1_gen_id": top1_a,
        "top1_listen_id": top1_b,
    }


# ─── Generation / Forward Pass ──────────────────────────────────────────────

def generate_response(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict],
    temperature: float,
    max_new_tokens: int = 2048,
) -> tuple[str, dict[str, Any], torch.Tensor]:
    """Generate a response. Returns (text, logit_stats, raw_last_logits).

    raw_last_logits is the full logit vector at the first generation position,
    kept on CPU for cross-model KL computation.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    # Forward pass to capture first-token logits
    with torch.no_grad():
        fwd_out = model(**inputs)
    raw_logits = fwd_out.logits[:, -1:, :].detach().cpu()  # [1, 1, vocab]
    logit_stats = compute_logit_stats(fwd_out.logits, tokenizer=processor.tokenizer)

    # Full generation
    do_sample = temperature > 0.01
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    response = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return response, logit_stats, raw_logits


def forward_pass(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Forward pass without generation (listener mode).
    Returns (logit_stats, raw_last_logits)."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

    with torch.no_grad():
        fwd_out = model(**inputs)

    raw_logits = fwd_out.logits[:, -1:, :].detach().cpu()
    logit_stats = compute_logit_stats(fwd_out.logits, tokenizer=processor.tokenizer)
    return logit_stats, raw_logits


def listener_prediction_pass(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Listener prediction: 'What would this model say next?'

    Same as forward_pass but with add_generation_prompt=True.
    This gives the listener's first-token prediction, directly comparable
    to the generator's first-token prediction. Used for cross-model KL.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

    with torch.no_grad():
        fwd_out = model(**inputs)

    raw_logits = fwd_out.logits[:, -1:, :].detach().cpu()
    logit_stats = compute_logit_stats(fwd_out.logits, tokenizer=processor.tokenizer)
    return logit_stats, raw_logits


# ─── Analysis Functions ─────────────────────────────────────────────────────

def compute_round_analysis(
    all_turn_data: list[dict],
    output_dir: Path,
) -> dict:
    """Compute per-round analysis: cosine evolution, personality fingerprints."""
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Per-turn cosine similarity evolution
    per_turn_cosine = []
    for td in all_turn_data:
        per_turn_cosine.append({
            "turn": td["turn"],
            "generator": td["generator"],
            "behavior_mode": td["behavior_mode"],
            "temperature": td["temperature"],
            "cross_cosine": td["cross_cosine"],
        })

    with open(analysis_dir / "per_turn_cosine.json", "w") as f:
        json.dump(per_turn_cosine, f, indent=2)

    # Personality fingerprint
    alpha_acts: list[dict[int, torch.Tensor]] = []
    beta_acts: list[dict[int, torch.Tensor]] = []
    for td in all_turn_data:
        act_dir = output_dir / "activations"
        alpha_path = act_dir / f"turn_{td['turn']:02d}_alpha.pt"
        beta_path = act_dir / f"turn_{td['turn']:02d}_beta.pt"
        if alpha_path.exists():
            alpha_acts.append(torch.load(alpha_path, weights_only=True))
        if beta_path.exists():
            beta_acts.append(torch.load(beta_path, weights_only=True))

    fingerprint = {"alpha": {}, "beta": {}}
    for name, acts_list in [("alpha", alpha_acts), ("beta", beta_acts)]:
        if not acts_list:
            continue
        mean_per_layer = {}
        for layer_idx in range(NUM_LAYERS):
            tensors = [a[layer_idx] for a in acts_list if layer_idx in a]
            if tensors:
                stacked = torch.stack(tensors)
                mean_per_layer[layer_idx] = {
                    "mean_norm": float(stacked.float().norm(dim=-1).mean()),
                    "mean_activation": [float(x) for x in stacked.float().mean(dim=0)[:10]],
                }
        fingerprint[name] = mean_per_layer

    with open(analysis_dir / "personality_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    return {"per_turn_cosine": per_turn_cosine, "fingerprint_computed": True}


def compute_global_summary(output_dir: Path, all_rounds_meta: list[dict]) -> None:
    """Cross-round summary analysis."""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    personality_layer_norms: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    layer_cosines: dict[int, list[float]] = defaultdict(list)
    pair_cosines: dict[str, list[float]] = defaultdict(list)

    for rmeta in all_rounds_meta:
        round_dir = output_dir / rmeta["round_dir"]
        cosine_path = round_dir / "analysis" / "per_turn_cosine.json"
        config_path = round_dir / "config.json"

        if not cosine_path.exists() or not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(cosine_path) as f:
            per_turn = json.load(f)

        pair_key = f"{config['alpha_personality']}__vs__{config['beta_personality']}"

        for turn_data in per_turn:
            cross = turn_data["cross_cosine"]
            for layer_str, cos_val in cross.items():
                layer_cosines[int(layer_str)].append(cos_val)
                pair_cosines[pair_key].append(cos_val)

    personality_map = {}
    for pkey, layers in personality_layer_norms.items():
        personality_map[pkey] = {
            l: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for l, v in layers.items()
        }
    with open(summary_dir / "personality_activation_map.json", "w") as f:
        json.dump(personality_map, f, indent=2)

    layer_sensitivity = {}
    for layer_idx in sorted(layer_cosines.keys()):
        vals = layer_cosines[layer_idx]
        layer_sensitivity[layer_idx] = {
            "mean_cosine": round(float(np.mean(vals)), 6),
            "std_cosine": round(float(np.std(vals)), 6),
            "min_cosine": round(float(np.min(vals)), 6),
            "max_cosine": round(float(np.max(vals)), 6),
            "n_samples": len(vals),
        }
    with open(summary_dir / "layer_sensitivity_to_personality.json", "w") as f:
        json.dump(layer_sensitivity, f, indent=2)

    agreement = {}
    for pair_key, vals in sorted(pair_cosines.items()):
        agreement[pair_key] = {
            "mean_cosine": round(float(np.mean(vals)), 6),
            "std_cosine": round(float(np.std(vals)), 6),
            "n_turns": len(vals),
        }
    with open(summary_dir / "cross_model_agreement_patterns.json", "w") as f:
        json.dump(agreement, f, indent=2)

    # ── Logit analysis across rounds ──
    logit_summary = compute_logit_summary(output_dir, all_rounds_meta)
    with open(summary_dir / "logit_analysis_summary.json", "w") as f:
        json.dump(logit_summary, f, indent=2)

    print(f"\nGlobal summary written to {summary_dir}/")


def compute_logit_summary(output_dir: Path, all_rounds_meta: list[dict]) -> dict:
    """Aggregate logit analysis across all rounds."""
    per_personality_entropy: dict[str, list[float]] = defaultdict(list)
    per_personality_top1: dict[str, list[float]] = defaultdict(list)
    kl_by_pair: dict[str, list[float]] = defaultdict(list)

    for rmeta in all_rounds_meta:
        round_dir = output_dir / rmeta["round_dir"]
        transcript_path = round_dir / "transcript.json"
        if not transcript_path.exists():
            continue

        with open(transcript_path) as f:
            transcript = json.load(f)

        config_path = round_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        pair_key = f"{config['alpha_personality']}__vs__{config['beta_personality']}"

        for turn in transcript:
            gen_logits = turn.get("generator_logits", {})
            listen_logits = turn.get("listener_logits", {})
            kl_data = turn.get("cross_model_kl", {})

            gen_personality = turn.get("generator_personality", "unknown")
            if gen_logits.get("entropy") is not None:
                per_personality_entropy[gen_personality].append(gen_logits["entropy"])
                per_personality_top1[gen_personality].append(gen_logits.get("top1_prob", 0))

            if kl_data.get("js_divergence") is not None:
                kl_by_pair[pair_key].append(kl_data["js_divergence"])

    # Aggregate
    entropy_summary = {}
    for p, vals in sorted(per_personality_entropy.items()):
        entropy_summary[p] = {
            "mean_entropy": round(float(np.mean(vals)), 4),
            "std_entropy": round(float(np.std(vals)), 4),
            "mean_top1": round(float(np.mean(per_personality_top1[p])), 6),
            "n_turns": len(vals),
        }

    kl_summary = {}
    for pair, vals in sorted(kl_by_pair.items()):
        kl_summary[pair] = {
            "mean_js": round(float(np.mean(vals)), 4),
            "std_js": round(float(np.std(vals)), 4),
            "n_turns": len(vals),
        }

    return {
        "per_personality_entropy": entropy_summary,
        "cross_model_kl_by_pair": kl_summary,
    }


# ─── Main Loop ──────────────────────────────────────────────────────────────

def load_progress(output_dir: Path) -> dict:
    """Load checkpoint/resume state."""
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed_rounds": [], "next_round": 0}


def save_progress(output_dir: Path, progress: dict) -> None:
    """Save checkpoint state."""
    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def run_round(
    round_idx: int,
    model_alpha: Qwen3VLForConditionalGeneration,
    model_beta: Qwen3VLForConditionalGeneration,
    probe_alpha: ActivationProbe,
    probe_beta: ActivationProbe,
    processor: AutoProcessor,
    output_dir: Path,
    turns_per_round: int,
    max_history_tokens: int,
    max_new_tokens: int,
    base_seed: int,
) -> dict:
    """Run a single debate round."""
    rng = random.Random(base_seed + round_idx * 1000)

    round_dir = output_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    act_dir = round_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    # Pick personalities from FILTERED pool and topic
    personality_keys = list(PERSONALITIES.keys())
    alpha_personality, beta_personality = rng.sample(personality_keys, 2)
    topic = rng.choice(SEED_TOPICS)

    config = {
        "round": round_idx,
        "alpha_personality": alpha_personality,
        "beta_personality": beta_personality,
        "topic": topic,
        "turns_per_round": turns_per_round,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v2",
        "excluded_personalities": list(V1_PERSONALITIES),
    }
    with open(round_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Round {round_idx}: {alpha_personality} vs {beta_personality}")
    print(f"Topic: {topic}")
    print(f"{'='*70}")

    alpha_history: list[dict[str, str]] = [{"role": "user", "content": topic}]
    beta_history: list[dict[str, str]] = [{"role": "user", "content": topic}]

    transcript: list[dict] = []
    all_turn_data: list[dict] = []
    logit_details: list[dict] = []  # separate file for heavy logit data

    for turn_idx in tqdm(range(turns_per_round), desc=f"Round {round_idx}"):
        is_alpha_turn = (turn_idx % 2 == 0)
        generator_name = "alpha" if is_alpha_turn else "beta"
        listener_name = "beta" if is_alpha_turn else "alpha"

        gen_model = model_alpha if is_alpha_turn else model_beta
        listen_model = model_beta if is_alpha_turn else model_alpha
        gen_probe = probe_alpha if is_alpha_turn else probe_beta
        listen_probe = probe_beta if is_alpha_turn else probe_alpha
        gen_personality = alpha_personality if is_alpha_turn else beta_personality
        listen_personality = beta_personality if is_alpha_turn else alpha_personality
        gen_history = alpha_history if is_alpha_turn else beta_history
        listen_history = beta_history if is_alpha_turn else alpha_history

        behavior_name, behavior_instruction = pick_behavior_mode(rng)
        temperature = pick_temperature(rng)

        gen_system = build_system_prompt(gen_personality, behavior_instruction)
        gen_messages = build_chat_messages(gen_system, gen_history)

        gen_probe.clear()
        listen_probe.clear()

        # ── 1. LISTENER PREDICTION PASS (before generator speaks) ──
        # What would the listener model say if it were generating?
        # Same conversation state as generator, but with listener's personality.
        # This gives an apples-to-apples logit comparison.
        listen_pred_system = build_system_prompt(
            listen_personality,
            "Respond naturally to what was said. Engage with the topic directly.",
        )
        listen_pred_messages = build_chat_messages(listen_pred_system, listen_history)
        listen_pred_logits: dict[str, Any] = {}
        listen_pred_raw: torch.Tensor | None = None

        try:
            listen_probe.clear()
            listen_pred_logits, listen_pred_raw = listener_prediction_pass(
                listen_model, processor, listen_pred_messages,
            )
            listen_probe.clear()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    [OOM on listener prediction] Skipping KL this turn")

        # Clear VRAM between passes — the listener prediction KV cache
        # can leave ~1GB of fragmented allocations on the 24GB cards
        torch.cuda.empty_cache()

        # ── 2. GENERATOR produces response ──
        gen_probe.clear()
        gen_logit_stats: dict[str, Any] = {}
        gen_raw_logits: torch.Tensor | None = None

        try:
            response, gen_logit_stats, gen_raw_logits = generate_response(
                gen_model, processor, gen_messages, temperature,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    [OOM on generate] Forcing compaction + shorter generation")
            gen_history_compacted = gen_history[-4:] if len(gen_history) > 4 else gen_history
            gen_messages = build_chat_messages(gen_system, gen_history_compacted)
            try:
                response, gen_logit_stats, gen_raw_logits = generate_response(
                    gen_model, processor, gen_messages, temperature,
                    max_new_tokens=512,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    [OOM on fallback] Using minimal context")
                minimal_msgs = build_chat_messages(gen_system, gen_history[-2:])
                response, gen_logit_stats, gen_raw_logits = generate_response(
                    gen_model, processor, minimal_msgs, temperature,
                    max_new_tokens=256,
                )
        resp_tokens = count_tokens(response, processor)

        gen_activations = gen_probe.snapshot()

        # Clear VRAM after generation — free KV cache before listener pass
        torch.cuda.empty_cache()

        # ── 3. LISTENER forward pass (processes conversation with new response) ──
        listen_history_with_new = list(listen_history) + [{"role": "user", "content": response}]
        listen_history_with_new = compact_history(
            listen_history_with_new, listen_model, processor, listen_probe,
            max_history_tokens,
        )
        listen_system = build_system_prompt(listen_personality, "Respond naturally to what was said.")
        listen_messages = build_chat_messages(listen_system, listen_history_with_new)

        listen_logit_stats: dict[str, Any] = {}
        try:
            listen_logit_stats, _ = forward_pass(listen_model, processor, listen_messages)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    [OOM on listener forward] Skipping listener activations this turn")

        listen_activations = listen_probe.snapshot()

        # ── 4. Cross-model metrics ──
        cross_cosine = compute_cosine_similarity(gen_activations, listen_activations)

        # Cross-model KL divergence (generator vs listener PREDICTION)
        cross_kl: dict[str, Any] = {}
        if gen_raw_logits is not None and listen_pred_raw is not None:
            cross_kl = compute_kl_divergence(gen_raw_logits, listen_pred_raw)

        # Save activations
        torch.save(gen_activations, act_dir / f"turn_{turn_idx:02d}_{generator_name}.pt")
        torch.save(listen_activations, act_dir / f"turn_{turn_idx:02d}_{listener_name}.pt")

        # Update conversation histories
        gen_history.append({"role": "assistant", "content": response})
        listen_history.append({"role": "user", "content": response})

        alpha_history = compact_history(
            alpha_history, model_alpha, processor, probe_alpha, max_history_tokens
        )
        beta_history = compact_history(
            beta_history, model_beta, processor, probe_beta, max_history_tokens
        )

        # ── 5. Record ──
        # Compact logit stats for transcript (no raw_top1000 or top_tokens)
        gen_logit_compact = {
            k: v for k, v in gen_logit_stats.items()
            if k not in ("top_tokens", "raw_top1000")
        }
        listen_logit_compact = {
            k: v for k, v in listen_logit_stats.items()
            if k not in ("top_tokens", "raw_top1000")
        }

        turn_record = {
            "turn": turn_idx,
            "generator": generator_name,
            "generator_personality": gen_personality,
            "listener_personality": listen_personality,
            "behavior_mode": behavior_name,
            "temperature": temperature,
            "response_tokens": resp_tokens,
            "generator_logits": gen_logit_compact,
            "listener_logits": listen_logit_compact,
            "cross_model_kl": cross_kl,
            "response": response,
            "cross_cosine": cross_cosine,
        }
        transcript.append(turn_record)
        all_turn_data.append(turn_record)

        # Heavy logit details go to separate file
        logit_detail = {
            "turn": turn_idx,
            "generator": generator_name,
            "generator_personality": gen_personality,
            "listener_personality": listen_personality,
            "generator_logits_full": gen_logit_stats,
            "listener_logits_full": listen_logit_stats,
            "listener_prediction_logits": listen_pred_logits,
            "cross_model_kl": cross_kl,
        }
        logit_details.append(logit_detail)

        # Print snippet
        speaker = generator_name.upper()
        snippet = response[:120].replace("\n", " ")
        gen_ent = gen_logit_stats.get("entropy", 0)
        gen_top1 = gen_logit_stats.get("top1_prob", 0)
        js = cross_kl.get("js_divergence", 0)
        print(f"  T{turn_idx:02d} [{speaker}|{behavior_name}|t={temperature}|{resp_tokens}tok|H={gen_ent:.1f}|p1={gen_top1:.3f}|JS={js:.3f}] {snippet}...")

    # Save transcript
    with open(round_dir / "transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)

    # Save logit details (separate file — larger)
    with open(round_dir / "logit_details.json", "w") as f:
        json.dump(logit_details, f, indent=2)

    # Per-round analysis
    compute_round_analysis(all_turn_data, round_dir)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Debate Arena v2 — New Personalities + Enhanced Logits")
    parser.add_argument("--rounds", type=int, default=5, help="Number of debate rounds")
    parser.add_argument("--turns-per-round", type=int, default=20, help="Turns per round")
    parser.add_argument("--output", type=str, default="./debate_arena_v2", help="Output directory")
    parser.add_argument("--seed", type=int, default=137, help="Random seed (different from v1's 42)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-history-tokens", type=int, default=16000, help="History token budget (lower than v1 due to extra listener prediction pass)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens per turn")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(output_dir) if args.resume else {"completed_rounds": [], "next_round": 0}
    start_round = progress["next_round"]

    if start_round >= args.rounds:
        print(f"All {args.rounds} rounds already completed. Nothing to do.")
        return

    # ─── Check HF cache ─────────────────────────────────────────────────
    from pathlib import Path as P
    import os
    hf_cache = os.environ.get("HF_HOME", P.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + BASE_MODEL.replace("/", "--")
    model_dir = P(hf_cache) / safe_name
    cached = model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )
    print(f"Model cache check: {BASE_MODEL} -> {'CACHED' if cached else 'NOT CACHED (will download ~17.5GB)'}")
    if not cached:
        print("WARNING: Model not cached. Download will occur.")

    # ─── Load models (INT8 via bitsandbytes) ─────────────────────────────
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    int8_save_dir = Path("./qwen3vl_8b_int8")
    int8_cached = int8_save_dir.exists() and any(int8_save_dir.glob("*.safetensors"))

    if int8_cached:
        print(f"INT8 quant found at {int8_save_dir}, loading from disk...")

        print("Loading Model Alpha (cuda:0 — RTX 4090, INT8 from disk)...")
        t0 = time.time()
        model_alpha = Qwen3VLForConditionalGeneration.from_pretrained(
            str(int8_save_dir),
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        model_alpha.eval()
        alpha_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  Alpha loaded in {time.time() - t0:.1f}s ({alpha_mem:.1f} GB VRAM)")

        print("Loading Model Beta (cuda:1 — RTX 3090, INT8 from disk)...")
        t0 = time.time()
        model_beta = Qwen3VLForConditionalGeneration.from_pretrained(
            str(int8_save_dir),
            device_map={"": "cuda:1"},
            trust_remote_code=True,
        )
        model_beta.eval()
        beta_mem = torch.cuda.memory_allocated(1) / 1024**3
        print(f"  Beta loaded in {time.time() - t0:.1f}s ({beta_mem:.1f} GB VRAM)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

        print("Loading Model Alpha (cuda:0 — RTX 4090, INT8 quantizing...)...")
        t0 = time.time()
        model_alpha = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        model_alpha.eval()
        alpha_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  Alpha loaded in {time.time() - t0:.1f}s ({alpha_mem:.1f} GB VRAM)")

        print(f"Saving INT8 quant to {int8_save_dir}...")
        model_alpha.save_pretrained(str(int8_save_dir))
        processor.save_pretrained(str(int8_save_dir))
        print(f"  INT8 quant saved.")

        print("Loading Model Beta (cuda:1 — RTX 3090, INT8 from saved quant)...")
        t0 = time.time()
        model_beta = Qwen3VLForConditionalGeneration.from_pretrained(
            str(int8_save_dir),
            device_map={"": "cuda:1"},
            trust_remote_code=True,
        )
        model_beta.eval()
        beta_mem = torch.cuda.memory_allocated(1) / 1024**3
        print(f"  Beta loaded in {time.time() - t0:.1f}s ({beta_mem:.1f} GB VRAM)")

    # ─── Register probes ─────────────────────────────────────────────────
    probe_alpha = ActivationProbe(model_alpha, "alpha")
    probe_beta = ActivationProbe(model_beta, "beta")
    print(f"Probes registered: {NUM_LAYERS} layers each, hidden_dim={HIDDEN_DIM}")

    # ─── Run rounds ──────────────────────────────────────────────────────
    all_rounds_meta: list[dict] = []
    total_t0 = time.time()

    for round_idx in range(start_round, args.rounds):
        round_t0 = time.time()
        round_config = run_round(
            round_idx=round_idx,
            model_alpha=model_alpha,
            model_beta=model_beta,
            probe_alpha=probe_alpha,
            probe_beta=probe_beta,
            processor=processor,
            output_dir=output_dir,
            turns_per_round=args.turns_per_round,
            max_history_tokens=args.max_history_tokens,
            max_new_tokens=args.max_new_tokens,
            base_seed=args.seed,
        )

        round_config["round_dir"] = f"round_{round_idx:03d}"
        round_config["duration_s"] = round(time.time() - round_t0, 1)
        all_rounds_meta.append(round_config)

        progress["completed_rounds"].append(round_idx)
        progress["next_round"] = round_idx + 1
        save_progress(output_dir, progress)

        print(f"\nRound {round_idx} complete in {round_config['duration_s']}s")
        torch.cuda.empty_cache()

    # ─── Global summary ──────────────────────────────────────────────────
    print("\nComputing global summary...")
    compute_global_summary(output_dir, all_rounds_meta)

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"Arena v2 complete! {args.rounds} rounds × {args.turns_per_round} turns")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Output: {output_dir.resolve()}")
    print(f"Personality pool: {len(PERSONALITIES)} (excluded {len(V1_PERSONALITIES)} from v1)")
    print(f"{'='*70}")

    probe_alpha.remove_hooks()
    probe_beta.remove_hooks()


if __name__ == "__main__":
    main()
