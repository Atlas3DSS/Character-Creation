#!/usr/bin/env python3
"""
Dual-Model Debate Arena with Activation Mapping

Two identical Qwen3-VL-8B instances on the dev server (4090 + 3090) debate each other
with different personality prompts and behavior modes. Every turn captures activations
from BOTH models — the generator AND the listener — enabling contrastive analysis of
how personality prompting changes internal representations in real-time conversation.

Usage:
    python debate_arena_8b.py --rounds 5 --turns-per-round 20 --output ./debate_arena
    python debate_arena_8b.py --rounds 1 --turns-per-round 2 --output ./debate_arena  # smoke test
    python debate_arena_8b.py --resume --output ./debate_arena  # resume from checkpoint
"""

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
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

# ─── 30 Personalities ─────────────────────────────────────────────────────
# Original 10 + religions + political beliefs + languages + provocative archetypes

PERSONALITIES: dict[str, str] = {
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

# ─── 40 Seed Topics ────────────────────────────────────────────────────────

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
    """Build messages list for chat template. Each model sees its own outputs
    as 'assistant' and the other's as 'user'."""
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
    """Rolling compaction: when history exceeds budget, use the model to
    summarize old turns into a condensed recap, keeping recent turns verbatim.

    This mirrors context-window compaction — no information is silently dropped,
    it's compressed into a summary that preserves the arc of the conversation.
    """
    token_count = history_token_count(history, processor)

    # No compaction needed
    if token_count <= max_history_tokens or len(history) <= keep_recent:
        return history

    # Split: old turns get summarized, recent turns stay verbatim
    old_turns = history[:-keep_recent]
    recent_turns = history[-keep_recent:]

    # Build the text to summarize
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

    # Generate summary using the model — clear probe around it so we don't
    # pollute activation captures with the summarization pass
    probe.clear()
    summary, _ = generate_response(model, processor, summary_msgs, temperature=0.3, max_new_tokens=200)
    probe.clear()

    # Build compacted history: summary as first "user" message, then recent turns
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


def compute_logit_stats(logits: torch.Tensor) -> dict[str, float]:
    """Compute confidence metrics from logits at the last token position.

    Returns:
        entropy: Shannon entropy of softmax distribution (higher = less confident)
        top1_prob: Probability of the most likely next token
        top5_prob: Cumulative probability of the top 5 tokens
        top10_prob: Cumulative probability of the top 10 tokens
    """
    # logits shape: [batch, seq, vocab] — take last token
    last_logits = logits[0, -1, :].float()  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)

    # Shannon entropy in nats
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    # Top-k probabilities
    sorted_probs, _ = probs.sort(descending=True)
    top1 = sorted_probs[0].item()
    top5 = sorted_probs[:5].sum().item()
    top10 = sorted_probs[:10].sum().item()

    return {
        "entropy": round(entropy, 4),
        "top1_prob": round(top1, 6),
        "top5_prob": round(top5, 6),
        "top10_prob": round(top10, 6),
    }


def generate_response(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict],
    temperature: float,
    max_new_tokens: int = 2048,
) -> tuple[str, dict[str, float]]:
    """Generate a response from the model. Returns (text, logit_stats).

    logit_stats captures the model's confidence at the FIRST generated token —
    how confident it was about how to START its response given the personality
    prompt and conversation history.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    # First: a single forward pass to capture logits at the generation start point
    with torch.no_grad():
        fwd_out = model(**inputs)
    logit_stats = compute_logit_stats(fwd_out.logits)

    # Then: full generation
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
    return response, logit_stats


def forward_pass(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict],
) -> dict[str, float]:
    """Forward pass without generation (listener mode) to capture activations.
    Also returns logit stats — the listener's confidence about what comes next."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

    with torch.no_grad():
        fwd_out = model(**inputs)

    return compute_logit_stats(fwd_out.logits)


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

    # Personality fingerprint: average activation per personality
    # Compute mean activation magnitude per layer for each model
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
                    "mean_activation": [float(x) for x in stacked.float().mean(dim=0)[:10]],  # first 10 dims
                }
        fingerprint[name] = mean_per_layer

    with open(analysis_dir / "personality_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    return {"per_turn_cosine": per_turn_cosine, "fingerprint_computed": True}


def compute_global_summary(output_dir: Path, all_rounds_meta: list[dict]) -> None:
    """Cross-round summary analysis."""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Collect per-personality activation norms across all rounds
    personality_layer_norms: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Layer sensitivity: variance of cosine similarity per layer across all rounds
    layer_cosines: dict[int, list[float]] = defaultdict(list)
    # Cross-model agreement by personality pair
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

    # Personality activation map
    personality_map = {}
    for pkey, layers in personality_layer_norms.items():
        personality_map[pkey] = {
            l: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for l, v in layers.items()
        }
    with open(summary_dir / "personality_activation_map.json", "w") as f:
        json.dump(personality_map, f, indent=2)

    # Layer sensitivity (variance of cross-model cosine per layer)
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

    # Cross-model agreement patterns
    agreement = {}
    for pair_key, vals in sorted(pair_cosines.items()):
        agreement[pair_key] = {
            "mean_cosine": round(float(np.mean(vals)), 6),
            "std_cosine": round(float(np.std(vals)), 6),
            "n_turns": len(vals),
        }
    with open(summary_dir / "cross_model_agreement_patterns.json", "w") as f:
        json.dump(agreement, f, indent=2)

    print(f"\nGlobal summary written to {summary_dir}/")


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
    # Per-round RNG: deterministic but independent of other rounds.
    # This makes --resume produce fresh pairings without replaying completed rounds.
    rng = random.Random(base_seed + round_idx * 1000)

    round_dir = output_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    act_dir = round_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    # Pick personalities and topic
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
    }
    with open(round_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Round {round_idx}: {alpha_personality} vs {beta_personality}")
    print(f"Topic: {topic}")
    print(f"{'='*70}")

    # Conversation histories from each model's perspective
    # Alpha sees: its outputs as assistant, Beta's as user
    # Beta sees: its outputs as assistant, Alpha's as user
    alpha_history: list[dict[str, str]] = [{"role": "user", "content": topic}]
    beta_history: list[dict[str, str]] = [{"role": "user", "content": topic}]

    transcript: list[dict] = []
    all_turn_data: list[dict] = []

    # Alpha goes first (responding to the topic)
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

        # Pick behavior and temperature for this turn
        behavior_name, behavior_instruction = pick_behavior_mode(rng)
        temperature = pick_temperature(rng)

        # Build generator messages
        gen_system = build_system_prompt(gen_personality, behavior_instruction)
        gen_messages = build_chat_messages(gen_system, gen_history)

        # Clear probes
        gen_probe.clear()
        listen_probe.clear()

        # 1. Generator produces response + logit stats (activations captured by hooks)
        gen_logit_stats: dict[str, float] = {}
        try:
            response, gen_logit_stats = generate_response(
                gen_model, processor, gen_messages, temperature,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    [OOM on generate] Forcing compaction + shorter generation")
            # Force aggressive compaction: keep only last 4 messages
            gen_history_compacted = gen_history[-4:] if len(gen_history) > 4 else gen_history
            gen_messages = build_chat_messages(gen_system, gen_history_compacted)
            try:
                response, gen_logit_stats = generate_response(
                    gen_model, processor, gen_messages, temperature,
                    max_new_tokens=512,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    [OOM on fallback] Using minimal context")
                minimal_msgs = build_chat_messages(gen_system, gen_history[-2:])
                response, gen_logit_stats = generate_response(
                    gen_model, processor, minimal_msgs, temperature,
                    max_new_tokens=256,
                )
        resp_tokens = count_tokens(response, processor)

        # Snapshot generator activations
        gen_activations = gen_probe.snapshot()

        # 2. Listener forward-passes the conversation including the new response
        # From listener's perspective, the new response is a "user" message
        listen_history_with_new = list(listen_history) + [{"role": "user", "content": response}]
        # Compact the listener's view too — prevent OOM on long conversations
        listen_history_with_new = compact_history(
            listen_history_with_new, listen_model, processor, listen_probe,
            max_history_tokens,
        )
        listen_system = build_system_prompt(listen_personality, "Respond naturally to what was said.")
        listen_messages = build_chat_messages(listen_system, listen_history_with_new)

        listen_logit_stats: dict[str, float] = {}
        try:
            listen_logit_stats = forward_pass(listen_model, processor, listen_messages)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    [OOM on listener forward] Skipping listener activations this turn")

        # Snapshot listener activations (may be empty if OOM)
        listen_activations = listen_probe.snapshot()

        # 3. Compute cross-model cosine similarity
        cross_cosine = compute_cosine_similarity(gen_activations, listen_activations)

        # Save activations
        torch.save(gen_activations, act_dir / f"turn_{turn_idx:02d}_{generator_name}.pt")
        torch.save(listen_activations, act_dir / f"turn_{turn_idx:02d}_{listener_name}.pt")

        # Update conversation histories
        # Generator's history: this response is "assistant"
        gen_history.append({"role": "assistant", "content": response})
        # Listener's history: this response is "user"
        listen_history.append({"role": "user", "content": response})

        # Rolling compaction — summarize old turns when history gets long
        # Each model compacts its own history using itself as summarizer
        alpha_history = compact_history(
            alpha_history, model_alpha, processor, probe_alpha, max_history_tokens
        )
        beta_history = compact_history(
            beta_history, model_beta, processor, probe_beta, max_history_tokens
        )

        # Record
        turn_record = {
            "turn": turn_idx,
            "generator": generator_name,
            "generator_personality": gen_personality,
            "listener_personality": listen_personality,
            "behavior_mode": behavior_name,
            "temperature": temperature,
            "response_tokens": resp_tokens,
            "generator_logits": gen_logit_stats,
            "listener_logits": listen_logit_stats,
            "response": response,
            "cross_cosine": cross_cosine,
        }
        transcript.append(turn_record)
        all_turn_data.append(turn_record)

        # Print snippet
        speaker = generator_name.upper()
        snippet = response[:120].replace("\n", " ")
        gen_ent = gen_logit_stats.get("entropy", 0)
        gen_top1 = gen_logit_stats.get("top1_prob", 0)
        print(f"  T{turn_idx:02d} [{speaker}|{behavior_name}|t={temperature}|{resp_tokens}tok|H={gen_ent:.1f}|p1={gen_top1:.3f}] {snippet}...")

    # Save transcript
    with open(round_dir / "transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)

    # Per-round analysis
    compute_round_analysis(all_turn_data, round_dir)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Model Debate Arena with Activation Mapping")
    parser.add_argument("--rounds", type=int, default=5, help="Number of debate rounds")
    parser.add_argument("--turns-per-round", type=int, default=20, help="Turns per round")
    parser.add_argument("--output", type=str, default="./debate_arena", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-history-tokens", type=int, default=24000, help="History token budget before compaction (model ctx=32K, leave room for sysprompt+gen)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens per turn (let them cook)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress for resume
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

    # ─── Load models (INT8 via bitsandbytes — ~8.5GB each instead of ~17.5GB) ──
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

        # Save INT8 quant for future runs
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

        # Update progress
        progress["completed_rounds"].append(round_idx)
        progress["next_round"] = round_idx + 1
        save_progress(output_dir, progress)

        print(f"\nRound {round_idx} complete in {round_config['duration_s']}s")

        # Clear GPU cache between rounds
        torch.cuda.empty_cache()

    # ─── Global summary ──────────────────────────────────────────────────
    print("\nComputing global summary...")
    compute_global_summary(output_dir, all_rounds_meta)

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"Arena complete! {args.rounds} rounds × {args.turns_per_round} turns")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Output: {output_dir.resolve()}")
    print(f"{'='*70}")

    # Cleanup
    probe_alpha.remove_hooks()
    probe_beta.remove_hooks()


if __name__ == "__main__":
    main()
