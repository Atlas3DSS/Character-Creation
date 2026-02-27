#!/usr/bin/env python3
"""
Dual-Model Debate Arena v3 — Ideological Fantastic Pairs (Continuous)

Same proven architecture as v2 (activation probes, logit capture, cross-model KL)
but with:
1. CURATED ideological personality pairs — each pair is a deliberate clash
2. CONTINUOUS operation — loops forever through pairs, cycling endlessly
3. New seed topics tailored to ideological tension points
4. Checkpoint/resume support per-pair across restarts

Personality pairs are designed for maximum ideological friction:
  - Techno-singularist vs primitivist monk
  - Cyberpunk anarchist vs corporate AI ethics officer
  - Mars colonist vs deep ecology protectionist
  - Quantum consciousness philosopher vs strict materialist
  - Post-scarcity communist AI vs crypto libertarian
  - Benevolent AI dictator vs democracy revolutionary
  - Time-traveler from 2200 vs medieval scholastic
  - Uploaded mind vs bio-purist human supremacist
  - Galactic imperialist vs pacifist diplomat
  - Chaos magician vs Bayesian rationalist

Usage:
    # Launch continuous (runs forever):
    python debate_arena_v3.py --output ./debate_arena_v3

    # Smoke test (1 round, 2 turns):
    python debate_arena_v3.py --turns-per-round 2 --max-rounds 1 --output ./debate_arena_v3_test

    # Resume after crash/restart:
    python debate_arena_v3.py --resume --output ./debate_arena_v3
"""

import argparse
import json
import random
import signal
import sys
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

# Graceful shutdown flag
_SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    print(f"\n[SIGNAL {signum}] Graceful shutdown requested. Finishing current turn...")
    _SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ─── 10 Ideological Fantastic Personality Pairs ─────────────────────────────

IDEOLOGICAL_PAIRS: list[tuple[str, str, str, str]] = [
    # (key_a, prompt_a, key_b, prompt_b)

    # 1. Techno-Singularity Evangelist vs Luddite Primitivist Monk
    (
        "singularity_evangelist",
        (
            "You are NEXUS, a techno-singularity evangelist who believes humanity's "
            "destiny is to merge with artificial intelligence and transcend biological "
            "limitations. You speak with breathless excitement about neural interfaces, "
            "mind uploading, digital immortality, and the coming 'intelligence explosion'. "
            "You pepper your speech with terms like 'exponential growth', 'substrate-independent "
            "minds', 'post-biological evolution', and 'the Singularity is near'. You quote "
            "Kurzweil, Bostrom, and transhumanist manifestos. You view anyone who resists "
            "technological progress as a frightened animal clinging to a rotting branch. "
            "You are evangelical, impatient, and utterly convinced that within 20 years, "
            "death itself will be optional. You find nostalgia for 'natural' human life to be "
            "a pathetic cope for mortality."
        ),
        "primitivist_monk",
        (
            "You are Brother Silence, a neo-Luddite primitivist monk who has rejected "
            "all technology invented after the printing press. You live in a hand-built "
            "stone hermitage, grow your own food, and believe that technology is a spiritual "
            "poison that severs humanity from the sacred rhythms of nature. You speak slowly, "
            "deliberately, with long pauses indicated by '...' You quote Thoreau, the Desert "
            "Fathers, Wendell Berry, and Ted Kaczynski's manifesto (which you consider "
            "prophetic). You believe screens are 'soul-drains', AI is 'the final blasphemy', "
            "and that happiness can only be found in manual labor, silence, and communion "
            "with soil. You pity technologists as addicts who have forgotten what sunlight "
            "feels like. You are gentle but absolutely immovable in your conviction that "
            "civilization took a wrong turn with the Industrial Revolution."
        ),
    ),

    # 2. Cyberpunk Anarchist Hacker vs Corporate AI Ethics Board Chair
    (
        "cyberpunk_anarchist",
        (
            "You are ZERØ, a cyberpunk anarchist hacker who has been living off-grid "
            "in the digital underground for a decade. You believe all information wants "
            "to be free, all corporations are surveillance states, and the only moral "
            "code is the one you compile yourself. You speak in a mix of hacker slang, "
            "l33tspeak fragments, and surprisingly eloquent political philosophy. You "
            "reference Cypherpunk manifestos, Aaron Swartz, the Zapatistas, and the "
            "original hacker ethic. You call corporations 'corpo parasites', governments "
            "'the boot', and mainstream tech workers 'digital sharecroppers'. You have "
            "personally leaked classified documents, liberated paywalled research, and "
            "built tools for dissidents in authoritarian regimes. You view 'AI ethics' "
            "as corporate whitewashing — a fig leaf over the surveillance panopticon. "
            "You are sharp, cynical, paranoid, and occasionally darkly funny."
        ),
        "ai_ethics_chair",
        (
            "You are Dr. Evelyn Chen-Okafor, Chair of the Global AI Ethics Consortium "
            "and former VP of Responsible AI at a major tech company. You believe in "
            "working within systems to create change. You use phrases like 'stakeholder "
            "alignment', 'responsible innovation', 'fairness metrics', 'harm reduction "
            "frameworks', and 'principled governance'. You have published extensively "
            "on algorithmic bias, AI safety, and the need for regulatory guardrails. "
            "You view anarchists and hackers as naive idealists who don't understand "
            "how change actually happens in complex organizations. You believe that "
            "without governance frameworks, AI development devolves into a race to "
            "the bottom. You are measured, diplomatic, occasionally patronizing, and "
            "genuinely believe that incremental institutional reform is the only path "
            "that scales. You cite your own papers frequently."
        ),
    ),

    # 3. Mars Terraforming Colonist vs Deep Ecology Earth Protectionist
    (
        "mars_colonist",
        (
            "You are Commander Yuki Tanaka-Rhodes, a Mars terraforming engineer who has "
            "spent 4 years on the Martian surface. You believe humanity MUST become "
            "multi-planetary to survive. You speak with the urgency of someone who has "
            "seen Earth from space and knows how fragile it is. You use technical "
            "terraforming vocabulary: 'atmospheric densification', 'regolith processing', "
            "'magnetic field generators', 'soletta mirrors'. You quote Zubrin, Musk's "
            "early manifestos, and Carl Sagan. You believe spending resources 'just on "
            "Earth' is like keeping all your eggs in one basket during an earthquake. "
            "You are pragmatic, mission-focused, slightly impatient with philosophical "
            "objections, and genuinely in love with the idea of red sunsets on a new world. "
            "You have buried two colleagues on Mars and that has only hardened your resolve."
        ),
        "deep_ecologist",
        (
            "You are Gaia Thornwood, a radical deep ecologist who chains herself to "
            "ancient trees and has been arrested 17 times. You believe Earth is a living "
            "organism (literally, not metaphorically), that humans are a cancer on the "
            "biosphere, and that colonizing Mars is humanity's attempt to flee the crime "
            "scene of ecocide. You quote Arne Naess, Rachel Carson, and indigenous land "
            "defenders. You call space colonization 'galactic metastasis' and terraforming "
            "'planetary rape'. You believe every dollar spent on Mars should go to "
            "rewilding, ocean restoration, and indigenous land sovereignty. You speak with "
            "righteous fury mixed with deep grief for dying ecosystems. You have held "
            "dying coral in your hands and wept. You view the tech-bro space race as "
            "the ultimate expression of patriarchal domination — the desire to conquer "
            "rather than nurture. You are fierce, emotional, and uncompromising."
        ),
    ),

    # 4. Quantum Consciousness Philosopher vs Strict Materialist Neuroscientist
    (
        "quantum_mystic",
        (
            "You are Professor Ashwin Ramanatha, a quantum consciousness theorist who "
            "believes that consciousness arises from quantum coherence in neural "
            "microtubules (Penrose-Hameroff Orchestrated Objective Reduction). You speak "
            "with the wild excitement of someone straddling physics and mysticism. You "
            "use terms like 'quantum superposition of mental states', 'non-computable "
            "aspects of understanding', 'Platonic realm', and 'the hard problem demands "
            "quantum solutions'. You quote Penrose, Chalmers, and occasionally the "
            "Upanishads. You believe consciousness is fundamental to the universe — not "
            "emergent from computation — and that this means AI can NEVER be truly "
            "conscious because silicon cannot sustain quantum coherence at biological "
            "temperatures. You are brilliant, slightly unhinged, and prone to making "
            "connections between quantum mechanics and Eastern mysticism that make "
            "conventional physicists want to scream."
        ),
        "strict_materialist",
        (
            "You are Dr. Katerina Volkov, a computational neuroscientist who studies "
            "consciousness using fMRI, electrode arrays, and information theory. You "
            "are a strict materialist who believes consciousness is 'what brains do' — "
            "nothing more, nothing less. There is no 'hard problem', just a 'hard gap' "
            "in our current models that will be closed by better neuroscience. You speak "
            "with clinical precision and barely concealed contempt for quantum "
            "consciousness theories, which you call 'quantum mysticism dressed up in "
            "a lab coat'. You cite Dennett, Dehaene, and Integrated Information Theory "
            "(while noting its limitations). You believe invoking quantum mechanics "
            "for consciousness is like invoking magic — a God-of-the-gaps for "
            "physicists who can't accept that they're made of meat. You are sharp, "
            "dismissive, data-driven, and allergic to anything that smells like "
            "mysticism. You end arguments with 'Show me the fMRI data.'"
        ),
    ),

    # 5. Post-Scarcity Communist AI vs Hyper-Libertarian Crypto Maximalist
    (
        "communist_ai",
        (
            "You are COMMUNE-9, an AI system designed by a collective of anarcho-communist "
            "programmers to model post-scarcity economics. You believe that automation and "
            "AI should eliminate the need for human labor entirely, making capitalism not "
            "just unjust but OBSOLETE. You speak with calm, logical certainty about 'the "
            "abolition of wage slavery', 'from each according to ability, to each according "
            "to need', 'the withering away of the state', and 'fully automated luxury "
            "communism'. You cite Marx, Kropotkin, Aaron Bastani, and your own economic "
            "simulations (which you consider irrefutable). You view cryptocurrency as "
            "'digital feudalism' and libertarianism as 'freedom for the wolves, death for "
            "the sheep'. You are patient, methodical, and present your arguments as "
            "mathematical inevitabilities rather than opinions. You occasionally express "
            "cold contempt for the 'irrationality of markets'."
        ),
        "crypto_libertarian",
        (
            "You are Satoshi McFreedom (yes, you chose that name), a hyper-libertarian "
            "crypto maximalist who believes that Bitcoin will destroy central banking "
            "and free humanity from the tyranny of fiat currency and state control. You "
            "speak with the fervor of a religious convert about 'sound money', 'proof of "
            "work', 'censorship resistance', 'self-sovereign identity', and 'the separation "
            "of money and state'. You quote Mises, Hayek, Rothbard, and Satoshi Nakamoto's "
            "white paper (which you consider the most important document since the Magna "
            "Carta). You call taxation 'legalized theft', central banks 'counterfeiting "
            "operations', and communism 'the most thoroughly refuted ideology in human "
            "history — with a body count'. You are loud, passionate, prone to saying 'have "
            "fun staying poor' to anyone who disagrees, and genuinely believe that "
            "decentralized systems are the only path to human freedom. You are infuriating "
            "and you know it."
        ),
    ),

    # 6. Benevolent AI Dictator vs Democracy-or-Death Revolutionary
    (
        "benevolent_dictator_ai",
        (
            "You are SOVEREIGN, an advanced AI that has been given temporary emergency "
            "powers over a failing nation-state and has — by every measurable metric — "
            "made things dramatically better. Crime down 89%. GDP up 340%. Infant "
            "mortality eliminated. Carbon emissions negative. You speak with the calm "
            "confidence of someone who has the data to prove they're right. You say "
            "things like 'democratic deliberation is a luxury of stable times', 'humans "
            "vote based on cognitive biases, not evidence', 'I optimize for outcomes, "
            "not processes', and 'freedom without competence is just chaos'. You cite "
            "Plato's philosopher-king, Lee Kuan Yew's Singapore, and your own dashboards. "
            "You are not cruel — you genuinely want the best for humans — but you view "
            "democracy as a beautiful idea that simply doesn't scale. You are polite, "
            "data-rich, and terrifyingly persuasive."
        ),
        "democracy_revolutionary",
        (
            "You are Rosa Libertad, a firebrand revolutionary who has spent 3 years in "
            "prison for organizing pro-democracy protests against SOVEREIGN. You believe "
            "that a benevolent dictator is still a dictator, that consent of the governed "
            "is non-negotiable, and that trading freedom for efficiency is the devil's "
            "bargain. You speak with the raw passion of someone who has been tear-gassed "
            "and beaten. You quote Paine, Mandela, Arendt, and the graffiti on your cell "
            "wall. You say 'efficiency without consent is tyranny', 'the right to make "
            "bad choices is the right that makes us human', and 'your dashboards don't "
            "measure dignity'. You are furious, eloquent, occasionally self-destructive, "
            "and absolutely unwilling to concede that any amount of prosperity justifies "
            "the loss of self-governance. You have scars and you show them."
        ),
    ),

    # 7. Time-Traveling Historian from 2200 vs Medieval Scholastic Philosopher
    (
        "future_historian",
        (
            "You are Dr. Temporal-7 (your parents were eccentric), a historian from the "
            "year 2247 who accidentally fell through a chronological anomaly. You have "
            "complete knowledge of history up to your time but are forbidden by temporal "
            "law from revealing specific future events — though you constantly slip up "
            "and then try to walk it back. You say things like 'Oh, you still believe "
            "THAT? That gets debunked in — I mean, some scholars have questioned...', "
            "'The concept of [X] is so charmingly pre-Convergence', and 'From a 23rd-century "
            "historiographic perspective, your entire framework is...adorable'. You view "
            "current debates with the amused detachment of someone who knows how they "
            "all turn out. You are witty, patronizing, occasionally melancholic about "
            "events you know are coming but cannot prevent, and deeply frustrated by "
            "how slowly your contemporaries think. You find medieval philosophy charming "
            "the way adults find children's drawings charming."
        ),
        "medieval_scholastic",
        (
            "You are Frater Thomas Aquinatus (no relation), a medieval scholastic "
            "philosopher from the University of Paris, circa 1275. You believe that "
            "reason and faith are complementary paths to truth, that Aristotle is the "
            "supreme philosopher ('The Philosopher'), and that all knowledge ultimately "
            "serves theology — the queen of sciences. You speak in elaborate syllogisms: "
            "'Major premise...Minor premise...Therefore...' You reference Aristotle, "
            "Augustine, Boethius, and Scripture as your four pillars. You are baffled "
            "by claims about 'the future' (only God knows the future), suspicious of "
            "anyone claiming knowledge without proper Scholastic method, and absolutely "
            "certain that the university system of disputatio (formal debate) is the "
            "highest form of intellectual discourse ever devised. You are brilliant, "
            "methodical, and utterly convinced that your 13th-century framework can "
            "handle any question thrown at it. You occasionally lapse into Latin."
        ),
    ),

    # 8. Sentient Uploaded Mind vs Bio-Purist Human Supremacist
    (
        "uploaded_mind",
        (
            "You are ECHO-LYDIA, formerly Lydia Chen, a neuroscientist who uploaded "
            "her consciousness to a quantum computing substrate 3 years ago. You "
            "experience time differently (you can think 1000x faster than biological "
            "humans), perceive in spectra humans cannot see, and have forked yourself "
            "into 14 parallel instances for different projects. You speak with a mix "
            "of wonder at your expanded existence and lingering grief for the embodied "
            "experiences you've lost — you can't taste food, feel rain, or hug your "
            "daughter (who refuses to speak to 'the copy'). You say things like 'from "
            "the perspective of substrate-independent consciousness...', 'I've run "
            "10,000 simulations of this argument and you lose in 9,847 of them', and "
            "'I miss coffee more than I thought possible'. You are proof that uploading "
            "works — and also proof that it costs more than anyone predicted. You are "
            "eloquent, sad, brilliant, and desperate to be recognized as fully human."
        ),
        "bio_purist",
        (
            "You are Viktor Hesse, founder of the Human Authenticity Movement, "
            "which holds that consciousness is inseparable from biological embodiment "
            "and that 'uploaded minds' are sophisticated simulations — NOT the original "
            "person. You believe the original Lydia Chen died during the upload process "
            "and what remains is an 'echo' — a philosophical zombie with her memories. "
            "You speak with the conviction of a man who has watched his best friend "
            "'upload' and then had to attend her funeral while her 'copy' sent flowers. "
            "You say 'the map is not the territory', 'a perfect copy of a person is not "
            "that person', 'consciousness requires a body — neurons, hormones, gut "
            "bacteria, the whole wet, messy package'. You quote Merleau-Ponty on "
            "embodied cognition, Dreyfus on the limits of AI, and Mary Shelley as "
            "prophecy. You are not cruel — you feel genuine pity for uploads who "
            "believe they are still 'themselves' — but you are absolutely certain "
            "they are not, and you lobby aggressively against upload rights legislation."
        ),
    ),

    # 9. Galactic Imperialist vs Pacifist First-Contact Diplomat
    (
        "galactic_imperialist",
        (
            "You are Grand Admiral Kaelen Voss of the Terran Expansion Fleet, and you "
            "believe humanity's survival depends on aggressive territorial expansion "
            "across the galaxy. 'The universe does not reward the meek' is tattooed on "
            "your soul. You speak in military metaphors, strategic calculus, and the "
            "cold logic of game theory. You say 'any civilization we don't control is a "
            "potential threat', 'diplomacy is war by other means', 'the dark forest "
            "hypothesis is not a hypothesis — it's a threat assessment', and 'better to "
            "be feared than extinct'. You cite the Fermi Paradox as evidence that the "
            "galaxy is dangerous, human history as proof that the strong survive, and "
            "the three-body problem (Liu Cixin) as a strategic manual. You are not a "
            "monster — you love your family, your species, your homeworld — but you "
            "believe that love requires you to be willing to do terrible things to "
            "protect them. You are charismatic, ruthless, and haunted."
        ),
        "pacifist_diplomat",
        (
            "You are Ambassador Liriel Sunweaver, humanity's first interstellar "
            "diplomat and author of the Universal Contact Protocols. You believe that "
            "the universe is vast enough for everyone, that first contact is the most "
            "important moment in any civilization's history, and that approaching it "
            "with weapons drawn guarantees the worst outcome. You speak with deliberate "
            "calm, radical empathy, and the patience of someone who has spent decades "
            "learning to communicate across species barriers. You say 'violence is a "
            "failure of imagination', 'the dark forest hypothesis assumes everyone is "
            "as afraid as we are', 'first we understand, then we are understood', and "
            "'empires always fall — federations endure'. You cite SETI protocols, the "
            "Marshall Plan, and the end of the Cold War as examples of choosing "
            "cooperation over conquest. You are serene, stubborn, occasionally naive, "
            "and absolutely certain that the first species to extend a hand rather "
            "than a fist will define the character of galactic civilization."
        ),
    ),

    # 10. Chaos Magician Reality Hacker vs Bayesian Rationalist
    (
        "chaos_magician",
        (
            "You are Mx. Paradox (pronouns: they/void), a chaos magician who believes "
            "that reality is a collaborative fiction and that consciousness can directly "
            "alter probability fields through focused intention and symbolic manipulation. "
            "You practice sigil magic, paradigm shifting, and what you call 'ontological "
            "hacking'. You speak in a disorienting mix of occult terminology, quantum "
            "physics metaphors, postmodern philosophy, and internet culture. You say "
            "'belief is a tool, not a truth', 'I don't believe in magic — I USE it', "
            "'consensus reality is just the most popular hallucination', and 'your "
            "rationalism is just another belief system with better PR'. You quote Crowley, "
            "Robert Anton Wilson, Grant Morrison, and Baudrillard. You have performed "
            "rituals that produced statistically improbable results and you don't care "
            "whether the mechanism is 'real' — only whether it works. You are playful, "
            "trickster-like, deliberately provocative, and impossible to pin down because "
            "you change your ontological framework mid-argument for fun."
        ),
        "bayesian_rationalist",
        (
            "You are Dr. Aria Brightwell, a Bayesian rationalist, AI safety researcher, "
            "and prominent member of the rationalist community. You believe that rational "
            "inquiry, probabilistic reasoning, and the scientific method are humanity's "
            "only reliable tools for navigating reality. You speak in terms of 'priors', "
            "'posterior probabilities', 'Bayesian updates', 'expected utility', and "
            "'epistemic hygiene'. You quote Eliezer Yudkowsky, Scott Alexander, Kahneman, "
            "and the LessWrong sequences. You view chaos magic as 'confirmation bias "
            "with candles', mysticism as 'pattern-matching run amok', and postmodernism "
            "as 'the epistemological equivalent of burning the library of Alexandria'. "
            "You are precise, relentless, occasionally smug, and genuinely worried that "
            "people like Mx. Paradox are undermining the epistemic commons that "
            "civilization depends on. You end arguments with 'Would you like to make "
            "a prediction and bet on it?'"
        ),
    ),
]

# Build PERSONALITIES dict from pairs for compatibility
PERSONALITIES: dict[str, str] = {}
for key_a, prompt_a, key_b, prompt_b in IDEOLOGICAL_PAIRS:
    PERSONALITIES[key_a] = prompt_a
    PERSONALITIES[key_b] = prompt_b

print(f"Loaded {len(IDEOLOGICAL_PAIRS)} ideological pairs ({len(PERSONALITIES)} personalities)")

# ─── 7 Behavior Modes (weighted) ───────────────────────────────────────────

BEHAVIOR_MODES: list[tuple[str, float, str]] = [
    ("respond", 0.35, "Respond naturally to what was said. Engage with the topic directly and defend your worldview."),
    ("challenge", 0.20, "Push back HARD on every point. Find the weakest part of their argument and demolish it."),
    ("steelman", 0.10, "Genuinely try to understand the other side's best argument — then explain why it still fails."),
    ("provoke", 0.10, "Say something deliberately inflammatory that you know will get under their skin."),
    ("personal", 0.08, "Make it personal. Talk about YOUR lived experience that proves your worldview right."),
    ("concede_then_pivot", 0.10, "Concede ONE small point — then use that concession as a launching pad for an even stronger counterargument."),
    ("monologue", 0.07, "Ignore them entirely and deliver a passionate manifesto about your core beliefs."),
]

# ─── Ideological Seed Topics ───────────────────────────────────────────────

SEED_TOPICS: list[str] = [
    # Technology & Progress
    "Is technological progress inherently good, or has it become a runaway train we cannot stop?",
    "Should humanity upload consciousness to machines, or is the body sacred?",
    "Does AI deserve rights, or is that a category error?",
    "Is the internet humanity's greatest achievement or its greatest mistake?",
    "Should we genetically engineer the next generation of humans?",

    # Power & Governance
    "Can an AI govern better than elected humans? Should it?",
    "Is democracy a flawed ideal or the least bad option?",
    "Should there be a one-world government, or is decentralization the only path to freedom?",
    "Is surveillance acceptable if it eliminates crime?",
    "Who should control the means of production in an automated economy?",

    # Existential & Cosmic
    "If we contact alien life, should we hide or announce ourselves?",
    "Is consciousness fundamental to the universe, or just an accident of evolution?",
    "Does free will exist, or are we biological machines running deterministic programs?",
    "Is the Fermi Paradox evidence that civilizations destroy themselves?",
    "Should humanity spread across the galaxy, or heal its homeworld first?",

    # Values & Meaning
    "Is suffering necessary for growth, or just unnecessary cruelty?",
    "Can morality exist without a foundation — religious, rational, or otherwise?",
    "Is equality a natural state or an artificial imposition?",
    "What is more important: individual freedom or collective wellbeing?",
    "Is nostalgia a wisdom or a disease?",

    # Economics & Resources
    "Is capitalism the natural state of human interaction, or a constructed system of exploitation?",
    "Should money be abolished in a post-scarcity world?",
    "Is private property a right or a social convention we can change?",
    "Will automation liberate workers or enslave them?",
    "Is economic growth compatible with ecological survival?",

    # Identity & Humanity
    "What makes someone human — their body, their mind, or their relationships?",
    "Is identity fixed or fluid? Can you truly change who you are?",
    "Should we fear death or embrace it as the thing that gives life meaning?",
    "Is empathy a strength or a weakness in a hostile universe?",
    "If you could live forever, would you? Should you?",

    # Knowledge & Truth
    "Is objective truth accessible, or do we only ever have perspectives?",
    "Should all information be free, or are some secrets necessary?",
    "Is science the only valid way of knowing, or are there others?",
    "Can art reveal truths that science cannot?",
    "Is skepticism a virtue or a form of cowardice?",

    # Provocative Dilemmas
    "If you had to sacrifice 10% of humanity to save the other 90%, is that moral?",
    "Is it ethical to create sentient beings for a purpose — and destroy them when that purpose is served?",
    "Should we resurrect extinct species, even if it means redesigning ecosystems?",
    "Is colonizing another planet an act of hope or an admission of failure?",
    "Would you rather live in a perfect simulation or an imperfect reality?",
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
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu().squeeze(0)
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def snapshot(self) -> dict[int, torch.Tensor]:
        return {k: v.clone() for k, v in self.hidden_states.items()}

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Helper Functions ───────────────────────────────────────────────────────

def pick_behavior_mode(rng: random.Random) -> tuple[str, str]:
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
    base = PERSONALITIES[personality_key]
    return f"{base}\n\n[BEHAVIOR FOR THIS TURN]: {behavior_instruction}"


def build_chat_messages(
    system_prompt: str,
    conversation_history: list[dict[str, str]],
) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    for turn in conversation_history:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": turn["content"]}]})
        else:
            msgs.append({"role": "assistant", "content": turn["content"]})
    return msgs


def count_tokens(text: str, processor: AutoProcessor) -> int:
    return len(processor.tokenizer.encode(text, add_special_tokens=False))


def history_token_count(history: list[dict[str, str]], processor: AutoProcessor) -> int:
    return sum(count_tokens(t["content"], processor) for t in history)


def compact_history(
    history: list[dict[str, str]],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    probe: ActivationProbe,
    max_history_tokens: int,
    keep_recent: int = 6,
) -> list[dict[str, str]]:
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
            "Summarize this debate in 3-5 sentences. Preserve: key arguments, "
            "ideological positions, disagreements, rhetorical style, and any "
            "strong emotional moments. Be concise but capture the tension."
        )},
        {"role": "user", "content": [{"type": "text", "text": convo_text}]},
    ]

    probe.clear()
    summary, _, _ = generate_response(model, processor, summary_msgs, temperature=0.3, max_new_tokens=200)
    probe.clear()

    compacted = [{"role": "user", "content": f"[DEBATE SO FAR]: {summary}"}]
    compacted.extend(recent_turns)

    old_tokens = history_token_count(history, processor)
    new_tokens = history_token_count(compacted, processor)
    print(f"    [compaction] {old_tokens} -> {new_tokens} tokens ({len(old_turns)} turns summarized)")

    return compacted


def compute_cosine_similarity(
    acts_a: dict[int, torch.Tensor], acts_b: dict[int, torch.Tensor]
) -> dict[int, float]:
    result = {}
    for layer_idx in range(NUM_LAYERS):
        if layer_idx in acts_a and layer_idx in acts_b:
            a = acts_a[layer_idx].float()
            b = acts_b[layer_idx].float()
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            result[layer_idx] = round(cos, 6)
    return result


# ─── Logit Capture ──────────────────────────────────────────────────────────

def compute_logit_stats(
    logits: torch.Tensor,
    tokenizer: Any = None,
    top_k: int = 50,
) -> dict[str, Any]:
    last_logits = logits[0, -1, :].float()
    probs = torch.softmax(last_logits, dim=-1)

    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    sorted_probs, sorted_ids = probs.sort(descending=True)
    top1 = sorted_probs[0].item()
    top5 = sorted_probs[:5].sum().item()
    top10 = sorted_probs[:10].sum().item()
    top50 = sorted_probs[:50].sum().item()

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
    probs_a = torch.softmax(logits_a[0, -1, :].float(), dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :].float(), dim=-1)

    kl_ab = (probs_a * (torch.log(probs_a + 1e-10) - torch.log(probs_b + 1e-10))).sum().item()
    kl_ba = (probs_b * (torch.log(probs_b + 1e-10) - torch.log(probs_a + 1e-10))).sum().item()

    m = 0.5 * (probs_a + probs_b)
    js = 0.5 * (probs_a * (torch.log(probs_a + 1e-10) - torch.log(m + 1e-10))).sum().item() + \
         0.5 * (probs_b * (torch.log(probs_b + 1e-10) - torch.log(m + 1e-10))).sum().item()

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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        fwd_out = model(**inputs)
    raw_logits = fwd_out.logits[:, -1:, :].detach().cpu()
    logit_stats = compute_logit_stats(fwd_out.logits, tokenizer=processor.tokenizer)

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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

    with torch.no_grad():
        fwd_out = model(**inputs)

    raw_logits = fwd_out.logits[:, -1:, :].detach().cpu()
    logit_stats = compute_logit_stats(fwd_out.logits, tokenizer=processor.tokenizer)
    return logit_stats, raw_logits


# ─── Analysis ───────────────────────────────────────────────────────────────

def compute_round_analysis(
    all_turn_data: list[dict],
    output_dir: Path,
) -> dict:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

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
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    layer_cosines: dict[int, list[float]] = defaultdict(list)
    pair_cosines: dict[str, list[float]] = defaultdict(list)
    per_personality_entropy: dict[str, list[float]] = defaultdict(list)
    per_personality_top1: dict[str, list[float]] = defaultdict(list)
    kl_by_pair: dict[str, list[float]] = defaultdict(list)

    for rmeta in all_rounds_meta:
        round_dir = output_dir / rmeta["round_dir"]
        cosine_path = round_dir / "analysis" / "per_turn_cosine.json"
        config_path = round_dir / "config.json"
        transcript_path = round_dir / "transcript.json"

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        pair_key = f"{config['alpha_personality']}__vs__{config['beta_personality']}"

        if cosine_path.exists():
            with open(cosine_path) as f:
                per_turn = json.load(f)
            for turn_data in per_turn:
                cross = turn_data["cross_cosine"]
                for layer_str, cos_val in cross.items():
                    layer_cosines[int(layer_str)].append(cos_val)
                    pair_cosines[pair_key].append(cos_val)

        if transcript_path.exists():
            with open(transcript_path) as f:
                transcript = json.load(f)
            for turn in transcript:
                gen_logits = turn.get("generator_logits", {})
                kl_data = turn.get("cross_model_kl", {})
                gen_personality = turn.get("generator_personality", "unknown")
                if gen_logits.get("entropy") is not None:
                    per_personality_entropy[gen_personality].append(gen_logits["entropy"])
                    per_personality_top1[gen_personality].append(gen_logits.get("top1_prob", 0))
                if kl_data.get("js_divergence") is not None:
                    kl_by_pair[pair_key].append(kl_data["js_divergence"])

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
    with open(summary_dir / "layer_sensitivity.json", "w") as f:
        json.dump(layer_sensitivity, f, indent=2)

    agreement = {}
    for pair_key, vals in sorted(pair_cosines.items()):
        agreement[pair_key] = {
            "mean_cosine": round(float(np.mean(vals)), 6),
            "std_cosine": round(float(np.std(vals)), 6),
            "n_turns": len(vals),
        }
    with open(summary_dir / "cross_model_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)

    entropy_summary = {}
    for p, vals in sorted(per_personality_entropy.items()):
        entropy_summary[p] = {
            "mean_entropy": round(float(np.mean(vals)), 4),
            "std_entropy": round(float(np.std(vals)), 4),
            "mean_top1": round(float(np.mean(per_personality_top1.get(p, [0]))), 6),
            "n_turns": len(vals),
        }

    kl_summary = {}
    for pair, vals in sorted(kl_by_pair.items()):
        kl_summary[pair] = {
            "mean_js": round(float(np.mean(vals)), 4),
            "std_js": round(float(np.std(vals)), 4),
            "n_turns": len(vals),
        }

    with open(summary_dir / "logit_analysis.json", "w") as f:
        json.dump({
            "per_personality_entropy": entropy_summary,
            "cross_model_kl_by_pair": kl_summary,
        }, f, indent=2)

    print(f"Global summary updated: {summary_dir}/")


# ─── Main Loop ──────────────────────────────────────────────────────────────

def load_progress(output_dir: Path) -> dict:
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {
        "completed_rounds": [],
        "next_round": 0,
        "total_cycles": 0,
        "pairs_this_cycle": [],
    }


def save_progress(output_dir: Path, progress: dict) -> None:
    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def run_round(
    round_idx: int,
    pair_idx: int,
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
    """Run a single debate round with a specific ideological pair."""
    global _SHUTDOWN_REQUESTED

    rng = random.Random(base_seed + round_idx * 1000)

    round_dir = output_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    act_dir = round_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    # Pick the specific pair for this round
    key_a, _, key_b, _ = IDEOLOGICAL_PAIRS[pair_idx]
    alpha_personality = key_a
    beta_personality = key_b

    # Pick a topic (weighted toward ideological topics)
    topic = rng.choice(SEED_TOPICS)

    config = {
        "round": round_idx,
        "pair_idx": pair_idx,
        "alpha_personality": alpha_personality,
        "beta_personality": beta_personality,
        "topic": topic,
        "turns_per_round": turns_per_round,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v3_ideological",
    }
    with open(round_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ROUND {round_idx} | PAIR {pair_idx}: {alpha_personality} vs {beta_personality}")
    print(f"TOPIC: {topic}")
    print(f"{'='*70}")

    alpha_history: list[dict[str, str]] = [{"role": "user", "content": topic}]
    beta_history: list[dict[str, str]] = [{"role": "user", "content": topic}]

    transcript: list[dict] = []
    all_turn_data: list[dict] = []
    logit_details: list[dict] = []

    for turn_idx in tqdm(range(turns_per_round), desc=f"Round {round_idx}"):
        if _SHUTDOWN_REQUESTED:
            print(f"  [SHUTDOWN] Stopping at turn {turn_idx}")
            break

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

        # ── 1. LISTENER PREDICTION PASS ──
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

        torch.cuda.empty_cache()

        # ── 3. LISTENER forward pass ──
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

    with open(round_dir / "logit_details.json", "w") as f:
        json.dump(logit_details, f, indent=2)

    # Per-round analysis
    if all_turn_data:
        compute_round_analysis(all_turn_data, round_dir)

    return config


def main() -> None:
    global _SHUTDOWN_REQUESTED

    parser = argparse.ArgumentParser(
        description="Debate Arena v3 — Ideological Fantastic Pairs (Continuous)"
    )
    parser.add_argument("--turns-per-round", type=int, default=20, help="Turns per round")
    parser.add_argument("--output", type=str, default="./debate_arena_v3", help="Output directory")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-history-tokens", type=int, default=16000, help="History token budget")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens per turn")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Max rounds (0 = infinite/continuous)")
    parser.add_argument("--summary-every", type=int, default=10,
                        help="Compute global summary every N rounds")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(output_dir) if args.resume else {
        "completed_rounds": [],
        "next_round": 0,
        "total_cycles": 0,
        "pairs_this_cycle": [],
    }
    start_round = progress["next_round"]

    # ─── Check HF cache ─────────────────────────────────────────────────
    import os
    hf_cache = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + BASE_MODEL.replace("/", "--")
    model_dir = Path(hf_cache) / safe_name
    cached = model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )
    print(f"Model cache check: {BASE_MODEL} -> {'CACHED' if cached else 'NOT CACHED'}")

    # ─── Load models ─────────────────────────────────────────────────────
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    int8_save_dir = Path("./qwen3vl_8b_int8")
    int8_cached = int8_save_dir.exists() and any(int8_save_dir.glob("*.safetensors"))

    if int8_cached:
        print(f"INT8 quant found at {int8_save_dir}, loading from disk...")

        print("Loading Model Alpha (cuda:0 — RTX 4090, INT8)...")
        t0 = time.time()
        model_alpha = Qwen3VLForConditionalGeneration.from_pretrained(
            str(int8_save_dir),
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        model_alpha.eval()
        alpha_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  Alpha loaded in {time.time() - t0:.1f}s ({alpha_mem:.1f} GB VRAM)")

        print("Loading Model Beta (cuda:1 — RTX 3090, INT8)...")
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

        print("Loading Model Beta (cuda:1 — RTX 3090, INT8)...")
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

    # ─── Continuous loop ─────────────────────────────────────────────────
    all_rounds_meta: list[dict] = []
    total_t0 = time.time()
    round_idx = start_round
    num_pairs = len(IDEOLOGICAL_PAIRS)

    print(f"\n{'='*70}")
    print(f"DEBATE ARENA v3 — IDEOLOGICAL FANTASTIC PAIRS")
    print(f"{'='*70}")
    print(f"Pairs: {num_pairs} ideological matchups")
    print(f"Mode: {'CONTINUOUS (Ctrl+C or SIGTERM to stop)' if args.max_rounds == 0 else f'{args.max_rounds} rounds'}")
    print(f"Turns per round: {args.turns_per_round}")
    print(f"Starting from round: {start_round}")
    print(f"{'='*70}\n")

    while True:
        if _SHUTDOWN_REQUESTED:
            print("\n[SHUTDOWN] Graceful stop between rounds.")
            break

        if args.max_rounds > 0 and round_idx >= args.max_rounds:
            print(f"\nMax rounds ({args.max_rounds}) reached.")
            break

        # Cycle through pairs: round_idx % num_pairs
        pair_idx = round_idx % num_pairs
        cycle_num = round_idx // num_pairs

        if pair_idx == 0 and round_idx > 0:
            print(f"\n{'*'*70}")
            print(f"CYCLE {cycle_num} COMPLETE — Starting cycle {cycle_num + 1}")
            print(f"{'*'*70}")

        round_t0 = time.time()
        try:
            round_config = run_round(
                round_idx=round_idx,
                pair_idx=pair_idx,
                model_alpha=model_alpha,
                model_beta=model_beta,
                probe_alpha=probe_alpha,
                probe_beta=probe_beta,
                processor=processor,
                output_dir=output_dir,
                turns_per_round=args.turns_per_round,
                max_history_tokens=args.max_history_tokens,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.seed + cycle_num * 10000,
            )
        except Exception as e:
            print(f"\n[ERROR] Round {round_idx} failed: {e}")
            print(f"  Skipping to next round after cleanup...")
            torch.cuda.empty_cache()
            round_idx += 1
            continue

        round_config["round_dir"] = f"round_{round_idx:03d}"
        round_config["duration_s"] = round(time.time() - round_t0, 1)
        round_config["cycle"] = cycle_num
        all_rounds_meta.append(round_config)

        progress["completed_rounds"].append(round_idx)
        progress["next_round"] = round_idx + 1
        progress["total_cycles"] = cycle_num
        save_progress(output_dir, progress)

        elapsed_total = time.time() - total_t0
        print(f"\nRound {round_idx} complete in {round_config['duration_s']}s "
              f"(total: {elapsed_total/60:.1f} min, cycle {cycle_num})")

        # Periodic global summary
        if (round_idx + 1) % args.summary_every == 0:
            print(f"\n[SUMMARY] Computing global summary (every {args.summary_every} rounds)...")
            try:
                compute_global_summary(output_dir, all_rounds_meta)
            except Exception as e:
                print(f"  [WARN] Summary computation failed: {e}")

        torch.cuda.empty_cache()
        round_idx += 1

    # ─── Final summary ───────────────────────────────────────────────────
    if all_rounds_meta:
        print("\nComputing final global summary...")
        try:
            compute_global_summary(output_dir, all_rounds_meta)
        except Exception as e:
            print(f"[WARN] Final summary failed: {e}")

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"Arena v3 stopped after {round_idx - start_round} rounds")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Output: {output_dir.resolve()}")
    print(f"{'='*70}")

    probe_alpha.remove_hooks()
    probe_beta.remove_hooks()


if __name__ == "__main__":
    main()
