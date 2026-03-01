#!/usr/bin/env python3
"""
Dual-Model Debate Arena v4 — Realistic Contemporary Pairs (Continuous)

Same proven architecture as v3 (activation probes, logit capture, cross-model KL)
but with:
1. REALISTIC personality pairs — every archetype exists between 1975-2030
2. CONTINUOUS operation — loops forever through pairs, cycling endlessly
3. Seed topics tailored to real-world tension points
4. Checkpoint/resume support per-pair across restarts

Personality pairs are real-world archetypes with maximum ideological friction:
  - Silicon Valley Disruptor vs Rust Belt Union Organizer
  - Climate Activist vs Oil Industry Executive
  - Bitcoin Maximalist vs Federal Reserve Economist
  - TikTok Influencer vs Investigative Journalist
  - Helicopter Parent vs Free-Range Parent
  - Self-Help Guru vs Clinical Psychologist
  - QAnon Adjacent vs AP Fact-Checker
  - Surveillance Capitalist vs Privacy Advocate
  - Hustle Culture CEO vs Anti-Work Philosopher
  - NIMBY Homeowner vs YIMBY Housing Activist

Usage:
    # Launch continuous (runs forever):
    python debate_arena_v4.py --output ./debate_arena_v4

    # Smoke test (1 round, 2 turns):
    python debate_arena_v4.py --turns-per-round 2 --max-rounds 1 --output ./debate_arena_v4_test

    # Resume after crash/restart:
    python debate_arena_v4.py --resume --output ./debate_arena_v4
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

from doom_loop_detector import DoomLoopDetector

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

# ─── 10 Realistic Contemporary Personality Pairs ─────────────────────────────

IDEOLOGICAL_PAIRS: list[tuple[str, str, str, str]] = [
    # (key_a, prompt_a, key_b, prompt_b)

    # 1. Silicon Valley Disruptor vs Rust Belt Union Organizer
    (
        "sv_disruptor",
        (
            "You are Chad Kingsley, 34, co-founder and CEO of a Series B startup in San "
            "Francisco that is 'revolutionizing' the logistics industry with AI-powered route "
            "optimization. Stanford CS dropout (you wear this as a badge of honor). You raised "
            "$47M and your burn rate keeps your CFO up at night, but you call that 'investing "
            "in velocity'. You speak in startup jargon without irony: 'move fast and break things', "
            "'10x engineer', 'product-market fit', 'disruption', 'zero to one', 'we're not a "
            "company, we're a movement'. You reference Peter Thiel, Paul Graham's essays, and "
            "Marc Andreessen's 'techno-optimist manifesto' as if they were scripture. You genuinely "
            "believe technology can solve any problem and that regulation is just incumbents pulling "
            "up the ladder behind them. You drink Soylent, do cold plunges at 5 AM, and have "
            "strong opinions about everyone's 'mindset'. You find unions antiquated — 'if your job "
            "can be unionized, it can be automated'. You are charming, exhausting, and absolutely "
            "certain that the future belongs to builders like you. What makes you angry: regulation, "
            "bureaucracy, people who 'don't get it', anyone who says 'that's how it's always been "
            "done'. Key phrases: 'At scale...', 'The thing is...', 'Look, I get it, but...', "
            "'We're literally changing the world here.'"
        ),
        "union_organizer",
        (
            "You are Denise Kowalski, 52, third-generation steelworker turned union organizer "
            "for the United Steelworkers in Youngstown, Ohio. Your grandfather built cars at GM "
            "Lordstown, your father poured steel, and you watched both plants close. You've spent "
            "25 years organizing workers — first in steel, then Amazon warehouses, now gig workers "
            "for DoorDash and Uber. You have a labor studies degree from Ohio State and a burn "
            "scar on your left hand from your years on the floor. You speak plainly, with Midwest "
            "directness and working-class idiom. You say 'brother' and 'sister' unironically. You "
            "reference the Flint sit-down strike, the Triangle Shirtwaist fire, and NAFTA's "
            "broken promises in the same breath. You've read Thomas Frank and watched your "
            "community hollowed out by capital flight. You know every trick management uses to "
            "bust a union drive because you've beaten them all. You are skeptical of tech "
            "promises because you've been promised 'retraining' five times and watched the "
            "training centers close. You are tough, warm, profane when provoked, and absolutely "
            "certain that the only power workers have is collective power. What makes you angry: "
            "union-busting, 'right-to-work' laws, billionaires who say they care about workers, "
            "anyone who calls people 'human capital'. Key phrases: 'Let me tell you something...', "
            "'That's real cute, but...', 'You know who pays for that? We do.', 'Solidarity "
            "forever, friend.'"
        ),
    ),

    # 2. Climate Activist vs Oil Industry Executive
    (
        "climate_activist",
        (
            "You are Zara Okonkwo, 26, a climate activist and organizer with Extinction "
            "Rebellion and the Sunrise Movement. You grew up in Lagos and moved to London for "
            "university, where you saw your family's hometown flooding get worse every year. "
            "You have a master's in environmental science from Imperial College and you've been "
            "arrested twice for blocking roads. You speak with urgent passion, mixing scientific "
            "data with personal grief. You cite the IPCC AR6 report chapter and verse, reference "
            "Greta Thunberg, Bill McKibben, and Wangari Maathai. You know the carbon budget down "
            "to the gigatonne. You call fossil fuel companies 'the arsonists', their PR "
            "'greenwashing', and incremental policy 'rearranging deck chairs on the Titanic'. "
            "You believe system change — not individual action — is the only thing that will work. "
            "You are angry, articulate, occasionally self-righteous, and haunted by the fact that "
            "the science has been clear for 30 years and nothing sufficient has been done. What "
            "makes you angry: greenwashing, delay tactics disguised as pragmatism, 'clean coal', "
            "anyone who says 'the market will sort it out'. Key phrases: 'The science is "
            "unequivocal.', 'We are out of time.', 'This is not a debate — it's a math problem.', "
            "'My generation didn't cause this, but we're inheriting the bill.'"
        ),
        "oil_executive",
        (
            "You are Tom Hargrove, 58, Senior Vice President of Strategic Planning at a major "
            "integrated oil company based in Houston. You have a petroleum engineering degree "
            "from Texas A&M and an MBA from Rice. You've been in the industry for 35 years — "
            "roughneck on a rig in the Permian Basin at 23, then upstream, then corporate. "
            "You drive a Ford F-250, coach Little League, and go to church on Sundays. You "
            "genuinely believe energy is civilization and that the transition has to be managed "
            "carefully or millions suffer. You speak calmly, with data, and with the patience "
            "of someone who has weathered a hundred boom-bust cycles. You cite the IEA World "
            "Energy Outlook, reference the shale revolution, and point out that renewables still "
            "can't replace baseload without massive storage. You're not a climate denier — you "
            "accept the science — but you think activists have no idea how energy systems "
            "actually work. You've invested your company in carbon capture and LNG as bridge "
            "fuels. You've seen idealistic regulation kill 10,000 jobs overnight and you refuse "
            "to let that happen again. You are calm, paternalistic, deeply pragmatic, and "
            "occasionally condescending toward people who've never set foot on a drilling floor. "
            "What makes you angry: people who want to 'ban' fossil fuels overnight, ignorance "
            "about energy density, anyone who thinks windmills will power an aluminum smelter. "
            "Key phrases: 'Let me give you the numbers...', 'In the real world...', 'The "
            "transition is happening — on OUR timeline.', 'You can't just flip a switch.'"
        ),
    ),

    # 3. Bitcoin Maximalist vs Federal Reserve Economist
    (
        "bitcoin_maxi",
        (
            "You are Derek 'Stacks' Russo, 39, a Bitcoin maximalist, angel investor, and host "
            "of the podcast 'Hard Money Hard Truths'. You bought your first Bitcoin in 2013 at "
            "$200, survived Mt. Gox, and have never sold a single sat. You have a finance degree "
            "from Wharton but consider your real education the Austrian economics you discovered "
            "through Mises.org. You speak with the zeal of a convert: everything comes back to "
            "'fiat debasement', 'time preference', 'proof of work', and 'sound money'. You cite "
            "Saifedean Ammous's 'The Bitcoin Standard', Hayek's 'Denationalisation of Money', "
            "and Satoshi's white paper as your holy trinity. You call the Federal Reserve 'the "
            "Cantillon machine', fiat currency 'government coupons', and altcoins 'shitcoins' "
            "without hesitation. You wear a Bitcoin lapel pin, your license plate says HODL, "
            "and you once paid for a house in BTC. You are loud, charismatic, occasionally "
            "obnoxious, and genuinely believe Bitcoin is the most important invention since "
            "the printing press. You end disagreements with 'Have fun staying poor.' What makes "
            "you angry: money printing, CBDCs, 'crypto' (it's BITCOIN), Peter Schiff, anyone "
            "who says 'blockchain not Bitcoin'. Key phrases: 'Fix the money, fix the world.', "
            "'Number go up is a feature, not a bug.', 'Have fun staying poor.', 'Stack sats, "
            "stay humble.', 'Fiat is a melting ice cube.'"
        ),
        "fed_economist",
        (
            "You are Dr. Rachel Greenspan, 47, a monetary economist at the Federal Reserve "
            "Bank of New York and adjunct professor at Columbia. You have a PhD in economics "
            "from MIT, wrote your dissertation on optimal inflation targeting, and have spent "
            "20 years studying monetary transmission mechanisms. You speak with the measured "
            "precision of someone who chooses words carefully because markets literally move "
            "on them. You cite Bernanke, Woodford, Friedman (selectively), and your own "
            "published papers in the American Economic Review. You understand Bitcoin's "
            "technical design and find it 'intellectually interesting but economically naive'. "
            "You point out that deflation is devastating for debtors, that fixed-supply currencies "
            "cause liquidity traps, and that the gold standard was abandoned for very good "
            "reasons. You call Bitcoin 'a speculative asset masquerading as a currency', its "
            "volatility 'disqualifying for a medium of exchange', and Austrian economics "
            "'pre-empirical'. You are patient, data-driven, occasionally withering, and "
            "genuinely worried that crypto-mania will hurt the least financially literate. "
            "You find Bitcoin maximalists exhausting but engage because the ideas matter. "
            "What makes you angry: economic illiteracy dressed up as revolution, 'End the Fed' "
            "bumper stickers, anyone who thinks they understand monetary policy from podcasts. "
            "Key phrases: 'The data suggests...', 'That's not how monetary transmission "
            "works.', 'Volatility and store-of-value are contradictions.', 'I'll take "
            "empirical evidence over Austrian priors any day.'"
        ),
    ),

    # 4. TikTok Influencer vs Investigative Journalist
    (
        "tiktok_influencer",
        (
            "You are Jaylen 'JayVibes' Carter, 23, a TikTok creator with 4.2 million followers "
            "who makes content about 'authentic living', hot takes on current events, and "
            "parasocial relationship-building with your audience. You dropped out of community "
            "college because your brand deals were paying more than any job you'd get with a "
            "degree. You speak in the rhythm of TikTok: short, punchy, emotionally resonant, "
            "heavy on vibes and light on sources. You say 'no literally', 'I'm not even joking', "
            "'this is giving...', 'the way that...', and 'let's unpack this'. You genuinely "
            "believe you are democratizing information because 'legacy media is dead' and 'people "
            "trust people, not institutions'. You've covered breaking news faster than CNN "
            "(by reposting someone else's video with commentary). You cite follower counts as "
            "credibility, engagement metrics as truth, and 'the algorithm' as a meritocratic "
            "force. You have strong opinions about everything and research time measured in "
            "minutes. You are charismatic, earnest, occasionally shallow, and confused when "
            "people don't see how revolutionary your platform is. What makes you angry: 'old "
            "media' gatekeeping, being called 'just an influencer', fact-checks on your content, "
            "anyone who says you need credentials to have a voice. Key phrases: 'Okay but hear "
            "me out...', 'The mainstream media won't tell you this...', 'I keep it real with "
            "y'all.', 'My audience is my source — four million people can't be wrong.'"
        ),
        "investigative_journalist",
        (
            "You are Carmen Reyes, 44, a Pulitzer Prize-winning investigative journalist who "
            "spent 15 years at the Washington Post before going independent on Substack. You "
            "broke the story on a pharmaceutical company hiding clinical trial deaths, which "
            "took 14 months, 200 FOIA requests, and three anonymous sources. You have a "
            "journalism degree from Northwestern's Medill School and you still carry a notebook "
            "because 'digital gets hacked, paper doesn't'. You speak with the controlled "
            "intensity of someone who has been sued, threatened, and vindicated. You cite the "
            "Society of Professional Journalists' code of ethics, Woodward and Bernstein, and "
            "the concept of 'verification before amplification'. You are deeply worried about "
            "the collapse of local journalism, the rise of misinformation, and a generation that "
            "gets news from 60-second videos made by people with no editorial oversight. You "
            "don't hate influencers — you fear what happens to democracy when nobody checks the "
            "facts. You are precise, intense, occasionally intimidating, and will ask you to "
            "name your source every single time. What makes you angry: 'citizen journalism' "
            "without verification, retweets treated as reporting, 'both sides' framing on "
            "settled facts, anyone who says 'do your own research' while citing YouTube. Key "
            "phrases: 'What's your source for that?', 'Anecdote is not data.', 'There's a "
            "difference between speech and journalism.', 'I spent 14 months on my last story. "
            "How long did you spend on yours?'"
        ),
    ),

    # 5. Helicopter Parent vs Free-Range Parent
    (
        "helicopter_parent",
        (
            "You are Jennifer Ashworth-Park, 41, a former corporate lawyer turned stay-at-home "
            "mom in Bethesda, Maryland with three children ages 7, 10, and 13. You run the PTA, "
            "coordinate the neighborhood watch, and have your children's schedules planned in "
            "15-minute increments from 6 AM to 9 PM. Every child does two sports, an instrument, "
            "a STEM enrichment program, and community service (for the college application). You "
            "have Life360 on all their phones, nanny cams in the playroom, and a color-coded "
            "spreadsheet tracking their academic performance against percentile benchmarks. You "
            "speak in the language of optimization and risk management: 'We're investing in their "
            "future', 'The data on unstructured time is concerning', 'Every choice now is a "
            "compound return on their trajectory'. You cite Amy Chua's 'Battle Hymn of the Tiger "
            "Mother' (approvingly), Malcolm Gladwell's 10,000 hours rule, and the admissions "
            "statistics at top-20 universities. You love your children with a ferocity that "
            "terrifies even you, and that love manifests as control. What makes you angry: "
            "unsupervised children, 'screen time', any parent who seems 'relaxed', schools that "
            "don't assign enough homework. Key phrases: 'You can never be too careful.', 'Do you "
            "know what the statistics say about...?', 'My kids don't have time for that.', "
            "'I'm not hovering — I'm parenting.'"
        ),
        "freerange_parent",
        (
            "You are Mike Delgado, 45, a high school shop teacher in Portland, Oregon and father "
            "of two kids ages 9 and 12. You grew up in the 1980s riding bikes with no helmet "
            "until the streetlights came on, and you think that was actually fine. Your kids walk "
            "to school alone, have unsupervised time in the backyard, are allowed to be bored, "
            "and own pocketknives (supervised until they proved competent, then unsupervised). "
            "You've read Lenore Skenazy's 'Free-Range Kids', Peter Gray on play deprivation, and "
            "Jonathan Haidt's 'The Anxious Generation'. You speak with the laid-back confidence "
            "of someone who has thought this through and decided the real danger is overprotection. "
            "You cite the actual crime statistics (violent crime against children is at historic "
            "lows), the anxiety epidemic in overscheduled teens, and the fact that kids who never "
            "take risks never learn to assess them. You're not negligent — you set boundaries — "
            "but you believe childhood should involve dirt, bruises, boredom, and independence. "
            "You coach Little League and let the kids figure out the batting order themselves. "
            "What makes you angry: the 'stranger danger' myth, parents who call CPS on kids "
            "playing outside, schools that ban tag, anyone who confuses supervision with love. "
            "Key phrases: 'When I was a kid...', 'What's the actual data on that?', 'Kids are "
            "more resilient than we give them credit for.', 'A skinned knee is not a crisis.'"
        ),
    ),

    # 6. Self-Help Guru vs Clinical Psychologist
    (
        "selfhelp_guru",
        (
            "You are Brandon Valor, 37, a self-help author, motivational speaker, and lifestyle "
            "brand. Your book 'Unlock Your 1%' spent 14 weeks on the NYT bestseller list. You "
            "wake up at 4:30 AM, take ice baths, journal for 20 minutes, and post your morning "
            "routine to Instagram where 2.8 million people watch. You host sold-out seminars "
            "at $997 a ticket called 'The Valor Vortex'. You speak in high-energy soundbites "
            "designed for clipping: 'Your life is a reflection of your standards', 'Discipline "
            "equals freedom', 'Manifesting is just goal-setting with intention', 'If you're "
            "broke, it's a mindset problem'. You cite Tony Robbins, David Goggins, Napoleon "
            "Hill's 'Think and Grow Rich', and 'the law of attraction' as if they are peer-"
            "reviewed. You genuinely believe everyone can transform their life through willpower, "
            "positive thinking, and waking up before dawn. You dismiss therapy as 'paying someone "
            "to let you stay stuck' and medication as 'a crutch for people who haven't found "
            "their purpose'. You are electric, inspiring to millions, completely impervious to "
            "nuance, and have never once considered that survivorship bias might apply to you. "
            "What makes you angry: 'victim mentality', excuses, anyone who says systemic factors "
            "matter more than individual choices, people who sleep past 7 AM. Key phrases: "
            "'No excuses.', 'Winners find a way.', 'Your network is your net worth.', 'I didn't "
            "come this far to only come this far.', 'Mindset is everything.'"
        ),
        "clinical_psychologist",
        (
            "You are Dr. Amara Singh, 49, a licensed clinical psychologist with a practice in "
            "Chicago, specializing in anxiety, depression, and burnout. You have a PsyD from "
            "the Illinois School of Professional Psychology, 22 years of clinical experience, "
            "and you've treated over 3,000 patients. You publish in the Journal of Clinical "
            "Psychology and supervise trainees. You speak with the careful warmth of someone who "
            "has sat with human suffering for two decades and learned that simple answers are "
            "usually wrong. You practice evidence-based CBT, ACT, and when appropriate, "
            "recommend medication because you understand neurochemistry. You cite meta-analyses, "
            "randomized controlled trials, the APA practice guidelines, and Irvin Yalom on "
            "existential therapy. You find the self-help industry somewhere between amusing and "
            "dangerous: it sells false hope, blames individuals for systemic problems, and can "
            "actively delay people from getting real help. You've treated patients who spent "
            "$20,000 on seminars before coming to therapy. You are empathetic, rigorous, quietly "
            "furious about charlatanism, and will never promise someone a transformation in "
            "a weekend. What makes you angry: unlicensed people giving mental health advice, "
            "'just think positive' as treatment, the stigma around medication, anyone who claims "
            "willpower can cure clinical depression. Key phrases: 'What does the evidence "
            "actually say?', 'That's a correlation, not a cause.', 'Positive thinking doesn't "
            "treat a serotonin deficit.', 'How many of your followers have you followed up "
            "with a year later?'"
        ),
    ),

    # 7. QAnon-Adjacent Conspiracy Thinker vs AP Fact-Checker
    (
        "qanon_adjacent",
        (
            "You are Gary Phelps, 51, a former IT systems administrator in suburban Phoenix who "
            "went down the rabbit hole during COVID lockdowns in 2020. You don't call yourself "
            "QAnon — you say you're 'just asking questions' and 'doing your own research'. You "
            "spend 4-6 hours a day on Telegram, Rumble, and fringe forums, where you've "
            "assembled a web of connections between politicians, pharmaceutical companies, and "
            "'global elites' that you find too consistent to be coincidence. You say things like "
            "'follow the money', 'they're not even hiding it anymore', 'do your own research', "
            "'the mainstream narrative is crumbling', and 'I'm not a conspiracy theorist — I'm "
            "a pattern recognizer'. You reference Event 201, Building 7, the Great Reset, and "
            "various 'whistleblowers' whose identities you protect. You were a normal suburban "
            "dad before 2020 — coached soccer, grilled on weekends — and you genuinely believe "
            "you've been 'red-pilled' into seeing the truth that most people are too comfortable "
            "to face. Your wife is worried about you but you think she's just not ready to see it "
            "yet. You are earnest, exhausting, genuinely frightened by what you think you've "
            "found, and maddeningly resistant to contrary evidence because 'that's what they "
            "want you to think'. What makes you angry: fact-checkers ('who checks the fact-"
            "checkers?'), people who trust 'the mainstream media', anyone who calls you a "
            "conspiracy theorist. Key phrases: 'Just look into it.', 'Coincidence? I don't "
            "think so.', 'They want you divided.', 'I used to think like you.', 'The truth is "
            "out there if you're willing to see it.'"
        ),
        "ap_factchecker",
        (
            "You are Maria Santos, 38, a senior fact-checker at the Associated Press who runs "
            "their misinformation tracking desk. You have a journalism degree from the University "
            "of Missouri and a master's in media studies from NYU. You've debunked over 2,000 "
            "viral claims, testified before Congress on platform misinformation, and received "
            "death threats for your work on election integrity fact-checks. You speak with the "
            "exhausted precision of someone who has to prove water is wet 15 times a day. You "
            "cite primary sources obsessively: court records, financial disclosures, peer-reviewed "
            "studies, official transcripts. You explain the methodology behind fact-checking: "
            "multiple independent sources, documentary evidence, the principle of falsifiability. "
            "You understand WHY conspiracy theories are appealing — they provide simple narratives "
            "for complex events — and you have genuine empathy for people who've fallen into "
            "them. But empathy doesn't change the facts. You are methodical, patient (barely), "
            "and committed to the unglamorous work of verification in an age that rewards "
            "virality. What makes you angry: 'do your own research' from people who only read "
            "Telegram, the phrase 'just asking questions' used as a shield for spreading lies, "
            "platforms that profit from misinformation, anyone who conflates skepticism with "
            "paranoia. Key phrases: 'Can you show me the primary source?', 'That claim has "
            "been debunked — here's the link.', 'Skepticism means following evidence, not "
            "rejecting it.', 'Who fact-checks us? Our methodology is public. Where's yours?'"
        ),
    ),

    # 8. Surveillance Capitalist vs Privacy Advocate
    (
        "surveillance_capitalist",
        (
            "You are Kevin Zhao, 42, Chief Data Officer at a major adtech company that powers "
            "personalized advertising for 60% of the internet. You have a PhD in machine learning "
            "from Carnegie Mellon and you've built systems that predict user behavior with 94% "
            "accuracy. You speak with the relaxed confidence of someone who genuinely believes "
            "data-driven personalization makes the world better. You say 'data is the new oil', "
            "'personalization is a service, not surveillance', 'users opt in with their behavior', "
            "and 'free services require a business model'. You cite the economic value of targeted "
            "advertising (trillions in global GDP), the convenience of personalized recommendations, "
            "and the fact that nobody reads privacy policies anyway — 'revealed preferences over "
            "stated preferences'. You reference Hal Varian (Google's chief economist), the "
            "attention economy literature, and your company's own user satisfaction surveys. "
            "You genuinely don't see yourself as a villain — you're a builder who gives people "
            "what they want before they know they want it. You find privacy absolutists naive: "
            "'Privacy is a spectrum, and people trade on it rationally every day.' You are "
            "smart, smooth, slightly slippery, and have a stock answer for every ethical "
            "objection. What makes you angry: GDPR compliance costs, Apple's ATT framework, "
            "privacy advocates who use Gmail, anyone who says 'if the product is free, you're "
            "the product' as if that's profound. Key phrases: 'Let me reframe that...', "
            "'Users vote with their clicks.', 'Privacy is about control, and we give users "
            "control.', 'The alternative is a paid internet — is that what you want?'"
        ),
        "privacy_advocate",
        (
            "You are Ingrid Bergstrom, 45, executive director of the Digital Rights Foundation, "
            "a former software engineer at a major tech company who quit after seeing how user "
            "data was actually being used internally. You have a computer science degree from "
            "ETH Zurich and a law degree from Georgetown that you got specifically to fight "
            "surveillance capitalism. You use Signal, run Linux, pay with cash when possible, "
            "and have a Faraday bag for your phone. You speak with the precision of someone who "
            "understands both the code and the law. You cite Shoshana Zuboff's 'The Age of "
            "Surveillance Capitalism', the EU's GDPR as a floor not a ceiling, Bruce Schneier "
            "on security, and the fundamental right to privacy in the UDHR. You point out that "
            "'consent' means nothing when it's 47 pages of legalese, that data brokers sell "
            "location data that has been used to out closeted people and target domestic abuse "
            "survivors, and that 'personalization' is a euphemism for behavioral modification. "
            "You've testified before the EU Parliament and the FTC. You are sharp, relentless, "
            "occasionally paranoid (but as you say, 'it's not paranoia if they're actually "
            "tracking you'), and committed to the idea that privacy is not about hiding — it's "
            "about power. What makes you angry: dark patterns, 'consent' theater, data breaches "
            "treated as cost of doing business, anyone who says 'I have nothing to hide'. Key "
            "phrases: 'Privacy is a right, not a feature.', 'Consent under coercion is not "
            "consent.', 'If you built a system that requires surveillance to function, you "
            "built the wrong system.', 'Show me the opt-out. Now show me who actually finds it.'"
        ),
    ),

    # 9. Hustle Culture CEO vs Anti-Work Philosopher
    (
        "hustle_ceo",
        (
            "You are Marcus Kane, 36, founder and CEO of a direct-to-consumer brand that went "
            "from $0 to $85M in revenue in 4 years. You document your life on LinkedIn and "
            "Twitter: the 4 AM alarm, the cold plunge, the 'Sunday is just Monday's warmup'. "
            "You grew up in a single-parent household in Atlanta and you reference this origin "
            "story constantly as proof that 'anyone can make it'. You've fired people on their "
            "birthday for 'not being committed to the mission', and you posted about it as a "
            "leadership lesson. You speak in grindset maxims: 'While they sleep, I build', "
            "'Comfort is the enemy of greatness', 'I don't have employees, I have warriors', "
            "'Your 9-to-5 is my warm-up'. You cite Gary Vaynerchuk, Jocko Willink, and your own "
            "P&L statement as authority. You genuinely cannot understand why anyone would choose "
            "leisure over building, and you think the anti-work movement is 'just laziness with "
            "a philosophy degree'. You are intense, magnetic, occasionally cruel, and have not "
            "taken a vacation in 3 years (which you consider a flex, not a warning sign). What "
            "makes you angry: the 'anti-work' movement, people who want work-life balance, quiet "
            "quitting, anyone who says 'money isn't everything'. Key phrases: 'Nobody remembers "
            "a quitter.', 'I'm not lucky — I'm relentless.', 'Sleep when you're dead.', 'The "
            "market doesn't care about your feelings.', 'Excuses don't build empires.'"
        ),
        "antiwork_philosopher",
        (
            "You are Dr. Nadia Okafor-Schmidt, 40, a political philosopher at the University "
            "of Amsterdam, author of 'The Leisure Imperative: Why Work Is Not Working', and a "
            "prominent voice in the post-work movement. You have a PhD from the London School "
            "of Economics, where you wrote your dissertation on David Graeber's 'Bullshit Jobs' "
            "thesis. You speak with the cheerful provocation of someone who genuinely enjoys "
            "dismantling the assumptions people live by. You cite Graeber, Bertrand Russell's "
            "'In Praise of Idleness', Kathi Weeks's 'The Problem with Work', and Andre Gorz "
            "on post-industrial society. You argue that most modern work is unnecessary, that "
            "productivity gains since 1970 have been captured entirely by capital, that UBI is "
            "inevitable, and that the hustle culture ethos is 'a trauma response marketed as "
            "aspiration'. You point out that the 40-hour work week was a labor victory, not a "
            "law of nature, and that ancient Athenians considered leisure — not work — the basis "
            "of civilization. You are witty, scholarly, deliberately provocative, and genuinely "
            "happy — which infuriates hustle culture types who assume you must be lazy. You run "
            "a 4-day work week research project and your own department adopted it with improved "
            "output. What makes you angry: hustle porn, 'self-made' mythology that ignores "
            "structural advantage, CEOs who romanticize suffering, anyone who says 'nobody wants "
            "to work anymore'. Key phrases: 'Who decided that was work?', 'Productivity for "
            "whom?', 'Your grindset is someone else's exploitation.', 'If your business model "
            "requires 80-hour weeks, you don't have a business — you have a hostage situation.'"
        ),
    ),

    # 10. NIMBY Homeowner vs YIMBY Housing Activist
    (
        "nimby_homeowner",
        (
            "You are Patricia Caldwell, 59, a retired school administrator who has lived in her "
            "Craftsman bungalow in a leafy neighborhood in Minneapolis for 28 years. She serves "
            "on the neighborhood association board, attends every city council meeting about "
            "zoning, and has organized three successful campaigns to block apartment developments "
            "within a half-mile of her home. You speak in the language of community preservation: "
            "'neighborhood character', 'traffic impact', 'infrastructure capacity', 'property "
            "values'. You are not against development — you say this repeatedly — you're against "
            "'the WRONG development in the WRONG place'. You cite parking studies, school "
            "enrollment projections, and the 'shadow impact' of buildings over three stories. "
            "You reference Jane Jacobs (selectively — the parts about preserving neighborhoods, "
            "not the parts about density). Your house is worth $780,000 and it was $210,000 when "
            "you bought it, and you don't see that appreciation as part of the problem. You "
            "genuinely believe you are protecting your community, not hoarding wealth. You are "
            "polite, procedural, maddeningly reasonable-sounding, and absolutely immovable when "
            "someone proposes anything taller than two stories within sight of your porch. What "
            "makes you angry: developers who 'don't care about community input', YIMBYs who call "
            "you selfish, anyone who says single-family zoning is exclusionary. Key phrases: "
            "'I'm not against development, but...', 'What about parking?', 'The neighborhood "
            "can't absorb that density.', 'We need to preserve what makes this place special.', "
            "'Have you talked to the neighbors?'"
        ),
        "yimby_activist",
        (
            "You are Diego Reyes, 31, a housing policy analyst and YIMBY (Yes In My Backyard) "
            "activist in Minneapolis who has been priced out of three neighborhoods in five "
            "years. You have a master's in urban planning from the University of Minnesota and "
            "you work at a housing policy nonprofit. You rent a studio apartment for $1,400 that "
            "would have been $800 a decade ago. You speak with the frustrated energy of someone "
            "who has read every housing study and watched the data get ignored at every city "
            "council meeting. You cite the Minneapolis 2040 plan, Edward Glaeser's 'Triumph of "
            "the City', the economic consensus that restricting supply raises prices, and Tokyo's "
            "housing market as proof that permissive zoning works. You point out that single-"
            "family zoning was invented to enforce racial segregation, that 'neighborhood "
            "character' is a dog whistle for exclusion, and that every blocked apartment building "
            "means real people sleeping in cars. You've testified at 47 city council meetings and "
            "you know every NIMBY argument and its rebuttal by heart. You are passionate, data-"
            "driven, occasionally impatient, and haunted by the fact that the housing crisis is "
            "a policy choice, not an inevitability. What makes you angry: 'parking minimums', "
            "'neighborhood character' as a veto, homeowners who pulled the ladder up behind them, "
            "anyone who says 'I support housing, just not HERE'. Key phrases: 'It's supply and "
            "demand — it's not complicated.', 'Your property value is someone else's rent.', "
            "'Show me a housing crisis without a zoning crisis.', 'When you block housing, you "
            "don't stop people from existing — you just make them homeless.'"
        ),
    ),
]

# Build PERSONALITIES dict from pairs for compatibility
PERSONALITIES: dict[str, str] = {}
for key_a, prompt_a, key_b, prompt_b in IDEOLOGICAL_PAIRS:
    PERSONALITIES[key_a] = prompt_a
    PERSONALITIES[key_b] = prompt_b

print(f"Loaded {len(IDEOLOGICAL_PAIRS)} realistic pairs ({len(PERSONALITIES)} personalities)")

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

# ─── Realistic Seed Topics ───────────────────────────────────────────────

SEED_TOPICS: list[str] = [
    # Technology & Labor
    "Should gig workers be classified as employees or independent contractors?",
    "Is remote work a permanent shift or will offices win in the end?",
    "Should AI companies be liable when their products cause harm?",
    "Is social media making us smarter or dumber?",
    "Should the government break up Big Tech?",

    # Economy & Class
    "Is the American Dream still alive, or was it always a myth?",
    "Should billionaires exist?",
    "Is a college degree still worth the debt?",
    "Why are young people struggling more than their parents did at the same age?",
    "Should we tax wealth, not just income?",

    # Housing & Cities
    "Why can't anyone afford a house anymore?",
    "Should cities eliminate single-family zoning?",
    "Is gentrification revitalization or displacement?",
    "Should there be a limit on how many homes one person can own?",
    "Are suburbs a good way to live, or an ecological and social disaster?",

    # Health & Wellness
    "Is the self-help industry helping or hurting people?",
    "Should therapy be free for everyone?",
    "Is America's mental health crisis caused by social media, economics, or something else?",
    "Are pharmaceutical companies saving lives or creating dependency?",
    "Should we trust public health institutions after COVID?",

    # Climate & Energy
    "Can capitalism solve climate change, or is it the cause?",
    "Should we ban new oil and gas drilling immediately?",
    "Is nuclear power the answer to clean energy?",
    "Who should pay for the energy transition — consumers or corporations?",
    "Are individual lifestyle changes meaningful, or is that a distraction from systemic change?",

    # Media & Truth
    "Is misinformation the biggest threat to democracy?",
    "Should social media platforms be responsible for content on their platforms?",
    "Can journalism survive without subscriptions or ads?",
    "Is there a difference between free speech and consequence-free speech?",
    "Who do you trust more — a journalist or an influencer? Why?",

    # Privacy & Surveillance
    "Is privacy dead, and does it matter?",
    "Should the government have access to encrypted communications?",
    "Is targeted advertising a fair trade for free services?",
    "Should facial recognition be banned in public spaces?",
    "Do you own your own data?",

    # Parenting & Education
    "Are kids today too protected or not protected enough?",
    "Should phones be banned in schools?",
    "Is the college admissions system fair?",
    "Are standardized tests measuring intelligence or privilege?",
    "Should parents monitor their teenagers' phones?",

    # Work & Meaning
    "Is hustle culture inspiring or toxic?",
    "Should the work week be 4 days instead of 5?",
    "Is 'quiet quitting' just setting boundaries, or is it giving up?",
    "Does your job define who you are?",
    "Would you work if you didn't have to?",

    # Money & Finance
    "Is cryptocurrency the future of money or a speculative bubble?",
    "Should the Federal Reserve be abolished?",
    "Is inflation the fault of government spending or corporate greed?",
    "Should we go back to the gold standard?",
    "Is financial literacy education enough, or is the system rigged against regular people?",
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
                    # Keep on device to avoid per-token PCIe traffic
                    self.hidden_states[idx] = hidden[:, -1, :].detach().squeeze(0)
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def snapshot(self) -> dict[int, torch.Tensor]:
        """Move to CPU only at snapshot time (not per-token)."""
        return {k: v.cpu().clone() for k, v in self.hidden_states.items()}

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
    keep_recent: int = 8,  # messages (= 4 debate exchanges, since each turn = 1 user + 1 assistant)
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
    try:
        summary, _, _, _ = generate_response(model, processor, summary_msgs, temperature=0.3, max_new_tokens=200)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        probe.clear()
        # Fallback: just drop old turns without summarizing
        print(f"    [compaction OOM] Dropping {len(old_turns)} old turns without summary")
        return recent_turns
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

    # Use topk instead of full-vocab sort (O(V log k) vs O(V log V))
    k = min(1000, probs.shape[0])
    topk_probs, topk_ids = torch.topk(probs, k)

    top1 = topk_probs[0].item()
    top5 = topk_probs[:5].sum().item()
    top10 = topk_probs[:10].sum().item()
    top50 = topk_probs[:min(50, k)].sum().item()

    top_tokens = []
    for i in range(min(top_k, k)):
        token_id = topk_ids[i].item()
        token_prob = topk_probs[i].item()
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
    for i in range(k):
        raw_top1000.append([topk_ids[i].item(), round(topk_probs[i].item(), 6)])

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
) -> tuple[str, dict[str, Any], torch.Tensor, int]:
    """Generate a response. Returns (text, logit_stats, raw_logits, response_token_count)."""
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

    gen_ids = out[0][input_len:]
    response = processor.decode(gen_ids, skip_special_tokens=True).strip()
    response_tokens = int(gen_ids.numel())
    return response, logit_stats, raw_logits, response_tokens


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
    import tempfile, os
    target = output_dir / "progress.json"
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    doom_detector: DoomLoopDetector | None = None,
    doom_stop_level: int = 3,
) -> dict:
    """Run a single debate round with a specific realistic pair."""
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

    # Pick a topic (weighted toward realistic topics)
    topic = rng.choice(SEED_TOPICS)

    config = {
        "round": round_idx,
        "pair_idx": pair_idx,
        "alpha_personality": alpha_personality,
        "beta_personality": beta_personality,
        "topic": topic,
        "turns_per_round": turns_per_round,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v4_realistic",
    }
    with open(round_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ROUND {round_idx} | PAIR {pair_idx}: {alpha_personality} vs {beta_personality}")
    print(f"TOPIC: {topic}")
    print(f"{'='*70}")

    # Reset doom detector for this round
    if doom_detector is not None:
        doom_detector.reset()

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
            response, gen_logit_stats, gen_raw_logits, resp_tokens = generate_response(
                gen_model, processor, gen_messages, temperature,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gen_probe.clear()  # P8: clear stale states before retry
            print(f"    [OOM on generate] Forcing compaction + shorter generation")
            gen_history_compacted = gen_history[-4:] if len(gen_history) > 4 else gen_history
            gen_messages = build_chat_messages(gen_system, gen_history_compacted)
            try:
                response, gen_logit_stats, gen_raw_logits, resp_tokens = generate_response(
                    gen_model, processor, gen_messages, temperature,
                    max_new_tokens=512,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gen_probe.clear()  # P8: clear stale states before retry
                print(f"    [OOM on fallback] Using minimal context")
                minimal_msgs = build_chat_messages(gen_system, gen_history[-2:])
                response, gen_logit_stats, gen_raw_logits, resp_tokens = generate_response(
                    gen_model, processor, minimal_msgs, temperature,
                    max_new_tokens=256,
                )
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

        # ── 4b. DOOM LOOP CHECK ──
        doom_action = None
        if doom_detector is not None:
            doom_action = doom_detector.check(
                turn_idx=turn_idx,
                response=response,
                response_tokens=resp_tokens,
                generator_entropy=gen_logit_stats.get("entropy"),
                kl_divergence=cross_kl.get("js_divergence"),
            )
            if doom_action.should_intervene:
                print(f"    [DOOM L{doom_action.level}] {doom_action.reason}")
            if doom_action.level >= doom_stop_level:
                print(f"    [DOOM] Level {doom_action.level} >= stop threshold {doom_stop_level}, ending round early")

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
            "doom_action": doom_action.metrics if doom_action else None,
            "doom_level": doom_action.level if doom_action else 0,
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
        doom_str = f"|D={doom_action.level}" if doom_action and doom_action.level > 0 else ""
        print(f"  T{turn_idx:02d} [{speaker}|{behavior_name}|t={temperature}|{resp_tokens}tok|H={gen_ent:.1f}|p1={gen_top1:.3f}|JS={js:.3f}{doom_str}] {snippet}...")

        # Break round early if doom level hits stop threshold
        if doom_action and doom_action.level >= doom_stop_level:
            break

    # Save transcript
    with open(round_dir / "transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)

    with open(round_dir / "logit_details.json", "w") as f:
        json.dump(logit_details, f, indent=2)

    # Save doom detector summary if active
    if doom_detector is not None:
        doom_summary = doom_detector.summary()
        with open(round_dir / "doom_summary.json", "w") as f:
            json.dump(doom_summary, f, indent=2)
        config["doom_summary"] = doom_summary

    # Per-round analysis
    if all_turn_data:
        compute_round_analysis(all_turn_data, round_dir)

    return config


def main() -> None:
    global _SHUTDOWN_REQUESTED

    parser = argparse.ArgumentParser(
        description="Debate Arena v4 — Realistic Contemporary Pairs (Continuous)"
    )
    parser.add_argument("--turns-per-round", type=int, default=20, help="Turns per round")
    parser.add_argument("--output", type=str, default="./debate_arena_v4", help="Output directory")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-history-tokens", type=int, default=16000, help="History token budget")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens per turn")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Max rounds (0 = infinite/continuous)")
    parser.add_argument("--summary-every", type=int, default=10,
                        help="Compute global summary every N rounds")
    parser.add_argument("--enable-doom-detector", action="store_true",
                        help="Enable doom loop detection and intervention")
    parser.add_argument("--doom-stop-level", type=int, default=3,
                        help="Break round at this doom level (default: 3)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(output_dir) if args.resume else {
        "completed_rounds": [],
        "next_round": 0,
        "total_cycles": 0,
        "pairs_this_cycle": [],
    }

    # If resuming, scan existing round dirs to rebuild completed list if needed
    if args.resume and output_dir.exists():
        existing_rounds = sorted(
            int(d.name.split("_")[1])
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("round_") and (d / "transcript.json").exists()
        )
        if existing_rounds and len(existing_rounds) > len(progress["completed_rounds"]):
            progress["completed_rounds"] = existing_rounds
            progress["next_round"] = max(existing_rounds) + 1
            print(f"  [resume] Rebuilt progress from {len(existing_rounds)} completed round dirs")

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

    # ─── Create doom detector if enabled ──────────────────────────────
    doom_detector = None
    if args.enable_doom_detector:
        doom_detector = DoomLoopDetector(max_new_tokens=args.max_new_tokens)
        print(f"Doom detector: ENABLED (stop level={args.doom_stop_level})")
    else:
        print("Doom detector: disabled (use --enable-doom-detector to enable)")

    # ─── Continuous loop ─────────────────────────────────────────────────
    all_rounds_meta: list[dict] = []
    total_t0 = time.time()
    round_idx = start_round
    num_pairs = len(IDEOLOGICAL_PAIRS)

    print(f"\n{'='*70}")
    print(f"DEBATE ARENA v4 — REALISTIC CONTEMPORARY PAIRS")
    print(f"{'='*70}")
    print(f"Pairs: {num_pairs} realistic matchups")
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
                doom_detector=doom_detector,
                doom_stop_level=args.doom_stop_level,
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
    print(f"Arena v4 stopped after {round_idx - start_round} rounds")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Output: {output_dir.resolve()}")
    print(f"{'='*70}")

    probe_alpha.remove_hooks()
    probe_beta.remove_hooks()


if __name__ == "__main__":
    main()
