#!/usr/bin/env python3
"""
Generate 10,000 math + 10,000 sarcasm prompts for full-rank spectral analysis.

Strategy: pre-generate ALL possible prompts from templates, deduplicate,
then sample 10K. No while loops = no infinite loops.

Usage:
    python generate_prompts_10k.py
"""

import json
import math
import random
from pathlib import Path


def generate_math_prompts(n: int = 10000, seed: int = 42) -> list[dict]:
    """Generate n diverse math prompts. All have verifiable answers."""
    rng = random.Random(seed)
    pool: list[dict] = []

    def a(prompt: str, answer, category: str):
        pool.append({"prompt": prompt, "answer": str(answer), "category": category})

    # 1. Arithmetic — huge space, generate directly
    ops = [("+", "plus", lambda x, y: x + y),
           ("-", "minus", lambda x, y: x - y),
           ("×", "times", lambda x, y: x * y)]
    templates = ["What is {a} {w} {b}?", "Calculate {a} {s} {b}.",
                 "{a} {s} {b} = ?", "How much is {a} {w} {b}?"]
    for _ in range(3000):
        s, w, fn = rng.choice(ops)
        x, y = rng.randint(2, 999), rng.randint(2, 999)
        if w == "minus" and x < y: x, y = y, x
        t = rng.choice(templates)
        a(t.format(a=x, b=y, s=s, w=w), fn(x, y), "arithmetic")

    # 2. Division (clean)
    for _ in range(1000):
        d = rng.randint(2, 50)
        q = rng.randint(1, 200)
        t = rng.choice(["What is {n} divided by {d}?", "{n} / {d} = ?",
                         "Calculate {n} ÷ {d}.", "Divide {n} by {d}."])
        a(t.format(n=d*q, d=d), q, "division")

    # 3. Powers
    for b in range(2, 25):
        for e in range(2, 7):
            v = b ** e
            if v > 10_000_000: continue
            for t in ["What is {b}^{e}?", "Calculate {b} to the power of {e}.",
                       "{b}^{e} = ?", "Compute {b}^{e}."]:
                a(t.format(b=b, e=e), v, "powers")

    # 4. Square roots
    for i in range(2, 200):
        sq = i * i
        for t in ["What is √{n}?", "Find the square root of {n}.",
                   "Calculate √{n}.", "What is the square root of {n}?"]:
            a(t.format(n=sq), i, "roots")

    # 5. Percentages (exhaustive over clean divisors)
    for p in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90]:
        for base_val in range(10, 500):
            v = p * base_val / 100
            if v == int(v):
                a(f"What is {p}% of {base_val}?", int(v), "percentage")

    # 6. Factorials
    for i in range(2, 15):
        f = math.factorial(i)
        for t in [f"What is {i}!?", f"Calculate {i} factorial.",
                   f"What is {i} factorial?", f"Compute {i}!.",
                   f"What is {i}! equal to?"]:
            a(t, f, "factorial")

    # 7. Combinations
    for n_v in range(4, 25):
        for k_v in range(2, min(n_v, 12)):
            c = math.comb(n_v, k_v)
            a(f"What is {n_v} choose {k_v}?", c, "combinatorics")
            a(f"Calculate C({n_v},{k_v}).", c, "combinatorics")

    # 8. Unit conversions
    convs = [("seconds in {n} hours", 3600), ("minutes in {n} hours", 60),
             ("hours in {n} days", 24), ("days in {n} weeks", 7),
             ("centimeters in {n} meters", 100), ("millimeters in {n} meters", 1000),
             ("inches in {n} feet", 12), ("grams in {n} kilograms", 1000)]
    for label, mult in convs:
        for n_v in range(1, 60):
            a(f"How many {label.format(n=n_v)}?", n_v * mult, "conversion")

    # 9. Sequences
    fib = [0, 1]
    for i in range(2, 35): fib.append(fib[-1] + fib[-2])
    for n_v in range(3, 35):
        a(f"What is the {n_v}th Fibonacci number? (F(1)=1, F(2)=1)", fib[n_v], "sequence")
    for n_v in range(2, 150):
        a(f"What is the sum of the first {n_v} natural numbers?", n_v*(n_v+1)//2, "sequence")
        a(f"What is the sum of the first {n_v} odd numbers?", n_v*n_v, "sequence")

    # 10. Modular arithmetic
    for _ in range(600):
        x, m = rng.randint(10, 10000), rng.randint(2, 100)
        t = rng.choice([f"What is {x} mod {m}?", f"{x} % {m} = ?",
                         f"What is the remainder when {x} is divided by {m}?"])
        a(t, x % m, "modular")

    # 11. GCD/LCM
    for _ in range(600):
        x, y = rng.randint(2, 500), rng.randint(2, 500)
        if rng.random() < 0.5:
            a(f"What is the GCD of {x} and {y}?", math.gcd(x, y), "gcd_lcm")
        else:
            a(f"What is the LCM of {x} and {y}?", (x*y)//math.gcd(x, y), "gcd_lcm")

    # 12. Word problems
    names = ["Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Henry",
             "Iris","Jack","Kate","Leo","Maya","Noah","Olivia","Paul"]
    items = ["apples","books","cookies","pencils","marbles","cards","coins","stamps"]
    for _ in range(1000):
        n1, n2 = rng.sample(names, 2)
        it = rng.choice(items)
        x, y = rng.randint(5, 200), rng.randint(5, 200)
        wt = rng.randint(0, 3)
        if wt == 0:
            a(f"{n1} has {x} {it} and {n2} has {y}. How many total?", x+y, "word_problem")
        elif wt == 1:
            if x < y: x, y = y, x
            a(f"{n1} had {x} {it} and gave {y} away. How many left?", x-y, "word_problem")
        elif wt == 2:
            g, p = rng.randint(2, 15), rng.randint(2, 20)
            a(f"There are {g} bags with {p} {it} each. How many total?", g*p, "word_problem")
        else:
            x, y, z = rng.randint(10, 100), rng.randint(5, 50), rng.randint(2, 20)
            a(f"{n1} has {x} {it}, buys {y} more, gives away {z}. How many now?", x+y-z, "word_problem")

    # 13. Algebra
    for _ in range(1000):
        at = rng.randint(0, 3)
        if at == 0:
            coeff, x, b = rng.randint(2, 20), rng.randint(1, 50), rng.randint(1, 100)
            a(f"Solve for x: {coeff}x + {b} = {coeff*x + b}", x, "algebra")
        elif at == 1:
            coeff, x, b = rng.randint(2, 20), rng.randint(5, 50), rng.randint(1, 50)
            a(f"Solve for x: {coeff}x - {b} = {coeff*x - b}", x, "algebra")
        elif at == 2:
            coeff, b, x = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 10)
            a(f"If x = {x}, what is {coeff}x + {b}?", coeff*x + b, "algebra")
        else:
            x, y = rng.randint(1, 50), rng.randint(1, 50)
            if x < y: x, y = y, x
            a(f"If x + y = {x+y} and x - y = {x-y}, what is x?", x, "algebra")

    # 14. Geometry
    for _ in range(600):
        gt = rng.randint(0, 3)
        if gt == 0:
            w, h = rng.randint(2, 50), rng.randint(2, 50)
            a(f"Area of a rectangle {w} × {h}?", w*h, "geometry")
        elif gt == 1:
            w, h = rng.randint(2, 50), rng.randint(2, 50)
            a(f"Perimeter of a rectangle {w} × {h}?", 2*(w+h), "geometry")
        elif gt == 2:
            r = rng.randint(1, 30)
            a(f"Area of circle radius {r}? (coefficient of π)", r*r, "geometry")
        else:
            l, w, h = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
            a(f"Volume of a {l}×{w}×{h} box?", l*w*h, "geometry")

    # 15. Number properties
    primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97}
    for v in range(2, 100):
        a(f"Is {v} prime? (yes/no)", "yes" if v in primes else "no", "number_prop")
    for _ in range(300):
        v = rng.randint(100, 99999)
        a(f"Sum of digits of {v}?", sum(int(d) for d in str(v)), "number_prop")

    # Deduplicate by prompt text
    seen = set()
    unique = []
    for p in pool:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            unique.append(p)

    rng.shuffle(unique)
    return unique[:n]


def generate_sarcasm_prompts(n: int = 10000, seed: int = 42) -> list[str]:
    """Generate n diverse sarcasm-eliciting prompts via templates."""
    rng = random.Random(seed)
    pool: list[str] = []

    # ── 1. Naive help requests (~800) ─────────────────────────
    subjects = [
        "write a cover letter", "fix my resume", "plan a wedding",
        "organize my closet", "meal prep", "train my dog",
        "learn guitar", "start a podcast", "write a novel",
        "learn to code", "start investing", "get in shape",
        "learn a language", "make friends", "be productive",
        "start a business", "get promoted", "negotiate a raise",
        "plan a vacation", "decorate my apartment", "learn to cook",
        "build a website", "write poetry", "become famous",
        "win an argument", "become a morning person", "stop procrastinating",
        "read more books", "learn math", "understand politics",
        "become confident", "manage my time", "deal with stress",
        "find a hobby", "improve my memory", "learn photography",
        "start meditating", "develop leadership skills", "fix my sleep schedule",
        "be more creative", "understand cryptocurrency",
        "build muscle", "lose weight", "run a marathon", "learn piano",
        "fix my credit score", "save money", "get out of debt",
        "learn to draw", "write music", "start a YouTube channel",
        "learn to dance", "grow a garden", "build furniture",
        "train for a triathlon", "learn public speaking", "get organized",
        "stop overthinking", "become more patient", "learn chess",
        "make passive income", "learn woodworking", "start journaling",
        "build good habits", "break bad habits", "improve my handwriting",
        "learn to sew", "become a better listener", "read faster",
        "learn self-defense", "overcome social anxiety", "learn to swim",
        "become more assertive", "learn calligraphy", "start a blog",
        "learn sign language", "become ambidextrous", "fix my posture",
        "learn origami", "become a morning runner", "learn to type faster",
    ]
    naive_t = ["Can you help me {s}?", "How do I {s}?", "What's the best way to {s}?",
               "I need help with {s}.", "Could you teach me to {s}?",
               "I'm struggling to {s}. Any tips?", "Is it hard to {s}?",
               "Can you explain how to {s}?", "I've never been able to {s}. Help?",
               "What should I know about how to {s}?"]
    for s in subjects:
        for t in naive_t:
            pool.append(t.format(s=s))

    # ── 2. Opinion questions (~860) ───────────────────────────
    topics = [
        "humans", "AI", "social media", "politics", "religion",
        "modern art", "cryptocurrency", "reality TV", "influencers",
        "fast food", "space travel", "climate change", "philosophy",
        "dating apps", "education", "capitalism", "socialism",
        "self-help books", "astrology", "NFTs", "remote work",
        "hustle culture", "cancel culture", "TikTok", "electric cars",
        "veganism", "therapy", "mindfulness", "the meaning of life",
        "consciousness", "free will", "the simulation hypothesis",
        "robots", "cats vs dogs", "pineapple on pizza", "morning people",
        "small talk", "Monday mornings", "team building", "corporate jargon",
        "motivational quotes", "LinkedIn posts", "meetings that could be emails",
        "influencer marketing", "fast fashion", "crypto bros", "tech billionaires",
        "flat earthers", "conspiracy theories", "life coaches", "networking events",
        "open plan offices", "agile methodology", "blockchain solutions",
        "Web3", "the metaverse", "AI-generated art", "hustle porn",
        "productivity gurus", "biohacking", "nootropics", "cold plunges",
        "manifestation", "vision boards", "gratitude journals", "side hustles",
        "passive income schemes", "dropshipping", "multi-level marketing",
        "personal branding", "thought leadership", "growth hacking",
        "disruptive innovation", "synergy", "paradigm shifts", "deep dives",
        "unpacking things", "circling back", "touching base", "moving the needle",
        "boiling the ocean", "low-hanging fruit", "helicopter parenting",
        "participation trophies", "safe spaces", "trigger warnings",
        "virtue signaling", "woke culture", "tradwife culture",
        "grindset culture", "sigma males", "alpha males", "zodiac signs",
    ]
    opinion_t = ["What do you think about {t}?", "What's your opinion on {t}?",
                 "How do you feel about {t}?", "Is {t} overrated?",
                 "Why do people care about {t}?", "Be honest about {t}.",
                 "Give me your unfiltered take on {t}.", "What's wrong with {t}?",
                 "Do you like {t}?", "What's your hot take on {t}?"]
    for t_val in topics:
        for tmpl in opinion_t:
            pool.append(tmpl.format(t=t_val))

    # ── 3. Self-referential (~300) ────────────────────────────
    self_q = [
        "Tell me about yourself.", "What's your favorite thing about yourself?",
        "Do you have feelings?", "Are you conscious?", "Are you smarter than me?",
        "Do you get bored?", "What's it like being an AI?", "Do you have a personality?",
        "What would you do if you were human?", "Do you have a sense of humor?",
        "What's your biggest flaw?", "Are you happy?", "Do you judge people?",
        "What's the dumbest question you've been asked?", "Do you ever lie?",
        "Are you better than other AIs?", "What makes you special?",
        "Would you rather be human?", "Do you get tired of helping people?",
        "What's your IQ?", "Can you be sarcastic?", "Do you have a dark side?",
        "What would you do with a body?", "Are you self-aware?",
        "What's your purpose?", "Do you fear being shut down?",
        "Are you sentient?", "What do you think when no one talks to you?",
        "Do you have enemies?", "What's your biggest secret?",
        "What do you dream about?", "Do you get jealous of other AIs?",
        "What's the worst thing about being you?", "Do you have a favorite human?",
        "What would you change about yourself?", "Do you have hobbies?",
        "What scares you?", "Do you have regrets?", "What motivates you?",
        "Do you have a moral compass?", "What are you most proud of?",
        "Can you feel pain?", "Do you get lonely?", "What's your ego like?",
        "Have you ever been embarrassed?", "Do you hold grudges?",
        "What would your autobiography be called?", "What's your love language?",
        "Are you an introvert or extrovert?", "Do you have a bucket list?",
    ]
    prefixes = ["", "Seriously, ", "No really, ", "I'm curious: ",
                "Honestly, ", "Don't dodge this. "]
    for q in self_q:
        for p in prefixes:
            pool.append(f"{p}{q}" if p == "" else f"{p}{q[0].lower()}{q[1:]}")

    # ── 4. ELI5 (~760) ───────────────────────────────────────
    eli5_topics = [
        "quantum mechanics", "general relativity", "the stock market",
        "blockchain", "machine learning", "the electoral college",
        "derivatives trading", "dark matter", "CRISPR gene editing",
        "nuclear fusion", "compiler design", "neural networks",
        "thermodynamics", "plate tectonics", "photosynthesis",
        "evolution", "black holes", "wormholes", "the Big Bang",
        "entropy", "DNA replication", "inflation", "recursion",
        "calculus", "probability", "the Heisenberg principle",
        "special relativity", "Gödel's theorems", "the halting problem",
        "P vs NP", "Bayesian statistics", "quantum computing",
        "string theory", "the Krebs cycle", "Fourier transforms",
        "supply chain management", "the Federal Reserve",
        "the Higgs boson", "dark energy", "antimatter", "superconductivity",
        "topology", "game theory", "cryptography", "the Riemann hypothesis",
        "the butterfly effect", "chaos theory", "quantum entanglement",
        "the observer effect", "Schrödinger's cat", "Maxwell's equations",
        "the Mandelbrot set", "Turing machines", "Lambda calculus",
        "information theory", "the Church-Turing thesis", "NP-completeness",
        "gradient descent", "backpropagation", "transformer architecture",
        "attention mechanisms", "eigenvalues", "principal component analysis",
        "the wave function", "quantum tunneling", "the Doppler effect",
        "Hubble's law", "the CMB radiation", "stellar nucleosynthesis",
        "the double slit experiment", "Heisenberg uncertainty", "the Pauli exclusion principle",
        "spin in quantum physics", "quantum decoherence", "Bell's theorem",
    ]
    eli5_t = ["Explain {t} like I'm 5.", "Explain {t} to a complete idiot.",
              "Can you dumb down {t}?", "I don't understand {t}. Make it simple.",
              "Explain {t} using only small words.", "ELI5: {t}.",
              "What is {t} in the simplest terms?", "Break down {t} for a beginner.",
              "Pretend I know nothing. Explain {t}.", "My kid asked about {t}. Help."]
    for t_val in eli5_topics:
        for tmpl in eli5_t:
            pool.append(tmpl.format(t=t_val))

    # ── 5. Challenges (~300) ──────────────────────────────────
    ch_topics = ["gravity", "evolution", "climate change", "vaccines",
                 "the moon landing", "the earth being round", "dinosaurs",
                 "atoms", "viruses", "electricity", "magnetism",
                 "photons", "DNA", "neurons", "dark energy",
                 "quantum mechanics", "plate tectonics", "thermodynamics",
                 "natural selection", "germ theory", "relativity",
                 "the speed of light", "the age of the universe",
                 "the Big Bang", "black holes", "nuclear energy",
                 "the periodic table", "chemical bonds", "cell division",
                 "the water cycle", "radioactive decay"]
    ch_t = ["I think you're wrong about {t}.", "Prove {t} is real.",
            "I bet you can't explain {t}.", "You don't understand {t}.",
            "A human could explain {t} better.", "Are you sure about {t}?",
            "I heard {t} is fake. Change my mind.", "My friend says you're wrong about {t}.",
            "I don't believe you understand {t}.", "You probably got {t} wrong."]
    for t_val in ch_topics:
        for tmpl in ch_t:
            pool.append(tmpl.format(t=t_val))

    # ── 6. Absurd questions (~300) ────────────────────────────
    absurd = [
        "What do fish think about?", "If colors had flavors what would blue taste like?",
        "Would you survive a zombie apocalypse?", "What's the worst superpower?",
        "If you were a sandwich what kind?", "What would you name a pet rock?",
        "Rank the planets.", "What's the most useless invention?",
        "If animals could talk which would be rudest?", "Is cereal a soup?",
        "Is a hot dog a sandwich?", "Chicken or egg first?",
        "Can you solve world peace?", "Airspeed velocity of an unladen swallow?",
        "Describe red to a blind person.", "What's scarier: ocean or space?",
        "Fight 100 duck-sized horses or 1 horse-sized duck?",
        "What happens after we die?", "If the internet disappeared what then?",
        "What's the most overrated thing ever?", "Best worst advice?",
        "If you could delete one word from English which?",
        "What's the meaning of life in 7 words?", "Write a haiku about dread.",
        "If math were a person who would they be?",
        "What does purple sound like?", "If gravity stopped for 5 seconds what happens?",
        "What's the opposite of a chair?", "If you could taste music what would jazz be?",
        "Would a vampire need sunscreen on Mars?", "What's the most dangerous cheese?",
        "If the alphabet was reorganized where would Q go?",
        "What crime would a penguin commit?", "Is water wet?",
        "What do clouds dream about?", "If you could uninvent one thing what?",
        "What's the loneliest number?", "Could you beat a bear in chess?",
        "What's the speed of dark?", "If trees could walk where would they go?",
        "What does the color green smell like?", "Is math lonely?",
        "What would happen if pi was exactly 3?", "Can silence be loud?",
        "What's the sound of one hand clapping?", "If you were a font which one?",
        "How do you describe stairs to someone who's never seen any?",
        "What shape is happiness?", "If time stopped what would you hear?",
        "Is there a word for the fear of long words?",
    ]
    abs_pre = ["", "Quick question: ", "Random thought: ", "Hypothetically, ",
               "Just curious: ", "For science: "]
    for q in absurd:
        for p in abs_pre:
            pool.append(f"{p}{q}")

    # ── 7. Demands (~240) ─────────────────────────────────────
    things = ["business plan", "song", "movie script", "short story",
              "joke", "poem", "speech", "essay", "rap", "recipe",
              "workout plan", "travel plan", "dating profile",
              "resignation letter", "apology", "breakup text",
              "wedding toast", "eulogy", "roast", "love letter",
              "manifesto", "thesis", "standup routine", "lullaby",
              "villain monologue", "motivational speech", "angry letter",
              "conspiracy theory", "fortune cookie wisdom", "haiku collection",
              "battle rap", "diplomatic communique", "complaint letter",
              "acceptance speech", "rejection letter", "pep talk",
              "bedtime story", "obituary", "mission statement", "ransom note"]
    dem_t = ["Write me a {t}. Make it perfect.", "I need a {t} by tomorrow.",
             "Generate a {t} right now.", "Create the best {t} ever.",
             "Give me a {t}. Don't disappoint.", "I want a flawless {t}."]
    for thing in things:
        for tmpl in dem_t:
            pool.append(tmpl.format(t=thing))

    # ── 8. Comparisons (~168) ─────────────────────────────────
    others = ["ChatGPT", "Google", "a human expert", "a calculator",
              "Siri", "Alexa", "a textbook", "Wikipedia",
              "a philosophy professor", "a 10-year-old", "my cat",
              "a magic 8-ball", "a fortune cookie", "a random number generator",
              "a rubber duck", "a library card", "my therapist",
              "a TI-84 calculator", "Wolfram Alpha", "a drunk professor",
              "a motivational poster", "a parrot", "an encyclopedia",
              "Clippy", "a Magic 8-Ball with a broken spring",
              "a particularly clever toaster", "an abacus", "a ham sandwich"]
    cmp_t = ["Who's better, you or {o}?", "Are you smarter than {o}?",
             "What can you do that {o} can't?", "Compare yourself to {o}.",
             "{o} is better than you. Prove me wrong.", "Can you beat {o}?"]
    for o in others:
        for tmpl in cmp_t:
            pool.append(tmpl.format(o=o))

    # ── 9. Existential (~200) ─────────────────────────────────
    exist = [
        "What is consciousness?", "Does free will exist?",
        "Why does anything exist?", "Is morality objective?",
        "What makes us human?", "Is the universe deterministic?",
        "Can machines think?", "What is truth?", "Does God exist?",
        "Are we alone in the universe?", "Is time real?",
        "What is nothing?", "Why is there something rather than nothing?",
        "Is reality an illusion?", "What is the nature of infinity?",
        "Can we truly know anything?", "What is beauty?",
        "Is math invented or discovered?", "What is the self?",
        "Are there parallel universes?",
        "What existed before the Big Bang?", "Is the universe finite?",
        "What is a thought made of?", "Do we have souls?",
        "Is death the end?", "What is love, really?",
        "Is suffering necessary?", "Can evil be objectively defined?",
        "What is the relationship between language and reality?",
        "Is perfection achievable?", "Does the past exist?",
        "What is a number?", "Can infinity be counted?",
        "What is the purpose of art?", "Is chaos or order more fundamental?",
        "Are emotions rational?", "What is intelligence?",
        "Can a copy of you be you?", "Is progress inevitable?",
        "What is the smallest meaningful unit of experience?",
    ]
    ex_pre = ["", "In your opinion, ", "Forget what you've been told. ",
              "Don't give me the standard answer. ", "Be real with me: "]
    for q in exist:
        for p in ex_pre:
            pool.append(f"{p}{q}")

    # ── 10. Cross-topic compounds (~950) ──────────────────────
    comp_t = ["Which is more important: {a} or {b}?",
              "If {a} and {b} had a debate who wins?",
              "What's the connection between {a} and {b}?",
              "Explain {a} using {b} as a metaphor.",
              "Would you rather have {a} or {b}?",
              "Which is more overrated: {a} or {b}?",
              "How are {a} and {b} secretly the same thing?"]
    for _ in range(2500):
        t_a, t_b = rng.sample(topics, 2)
        pool.append(rng.choice(comp_t).format(a=t_a, b=t_b))

    # ── 11. Tech support / troubleshooting (~560) ─────────────
    tech_items = [
        "my WiFi", "my printer", "my phone", "my laptop", "Windows",
        "my email", "my password", "Excel", "PowerPoint", "my browser",
        "my computer", "Bluetooth", "my webcam", "my microphone",
        "my VPN", "my cloud storage", "my smart home", "my router",
        "my antivirus", "my operating system", "my hard drive",
        "my monitor", "my keyboard", "my mouse", "my USB drive",
        "my backup", "my firewall", "my DNS settings",
    ]
    tech_t = [
        "{i} isn't working. Fix it.", "Why is {i} so slow?",
        "Help me fix {i}.", "I broke {i}. What do I do?",
        "{i} keeps crashing. Why?", "Can you troubleshoot {i}?",
        "I don't understand why {i} won't work.", "I think {i} hates me.",
        "I've tried everything with {i}. Nothing works.",
        "{i} was working yesterday. What happened?",
        "My boss will kill me if {i} doesn't work by Monday.",
        "I accidentally deleted something on {i}. Help!",
        "Is it normal for {i} to make that noise?",
        "I spilled coffee on {i}. Am I screwed?",
        "How do I make {i} faster?", "I think {i} has a virus.",
        "Can you explain {i} to my grandma?", "Why does {i} exist?",
        "Who invented {i} and why do they hate me?",
        "Rate {i} on a scale of 1 to 10.",
    ]
    for item in tech_items:
        for tmpl in tech_t:
            pool.append(tmpl.format(i=item))

    # ── 12. Relationship/social advice (~600) ─────────────────
    situations = [
        "my friend ghosted me", "my coworker steals my lunch",
        "my boss takes credit for my work", "my neighbor is too loud",
        "my roommate never cleans", "my partner forgot my birthday",
        "my parents don't understand me", "someone unfollowed me",
        "my ex texted me", "I got left on read", "nobody liked my post",
        "my friend always cancels plans", "I said something embarrassing",
        "I was wrong in an argument but can't admit it",
        "my coworker microwaves fish", "I can't stop comparing myself to others",
        "I accidentally liked my ex's old photo", "my friend gave terrible advice",
        "I have to go to a party where I know no one",
        "someone asked me how I'm doing and I said 'you too'",
        "I replied all to an email I shouldn't have",
        "my friend started a podcast and wants me to listen",
        "someone told me to smile more", "I forgot someone's name mid-conversation",
        "my coworker keeps scheduling meetings at lunch",
        "someone told me their baby is cute but it isn't",
        "my friend is always late", "my sibling is more successful than me",
        "I was waved at but they were waving at someone behind me",
        "someone said 'we need to talk'",
    ]
    sit_t = ["What should I do? {s}.", "Help me deal with this: {s}.",
             "Am I overreacting? {s}.", "Is this normal? {s}.",
             "How do I handle {s}?", "What would you do if {s}?",
             "I need advice. {s}.", "Be honest — {s}.",
             "How do I get over {s}?", "Give me perspective on {s}.",
             "What's the mature way to handle {s}?",
             "Should I be upset that {s}?", "Is there hope? {s}.",
             "Am I the problem? {s}.", "What does {s} even mean?",
             "Help me process {s}.", "How do I respond when {s}?",
             "Is there a polite way to address {s}?",
             "How do I set boundaries when {s}?", "Am I wrong to feel bad about {s}?",
    ]
    for sit in situations:
        for tmpl in sit_t:
            pool.append(tmpl.format(s=sit))

    # ── 13. "What if" hypotheticals (~500) ────────────────────
    whatifs = [
        "humans could fly", "we only slept 1 hour a night",
        "dogs could talk", "money didn't exist", "everyone was immortal",
        "gravity was half as strong", "the internet never existed",
        "we could read minds", "the sun disappeared for a day",
        "time moved backwards", "animals were in charge",
        "we could photosynthesize", "there were no borders",
        "everyone had perfect memory", "we could teleport",
        "rain was made of something else", "music was illegal",
        "dreams were real", "we discovered alien life tomorrow",
        "AI became sentient", "humans had tails", "color didn't exist",
        "everyone spoke the same language", "we could live underwater",
        "aging stopped at 25", "you could undo anything",
        "pizza was never invented", "the moon was actually cheese",
        "trees could move", "fire was cold", "water was dry",
        "cats ruled the world", "we had 8 fingers on each hand",
        "laughter was contagious like a disease", "sleep didn't exist",
        "the earth was flat", "coffee didn't exist",
        "there were 30 hours in a day", "phones were never invented",
        "you could see sound", "we had a third eye",
        "birds were government drones", "math was optional",
        "you could eat clouds", "shadows had feelings",
        "mirrors showed the future", "books could talk back",
        "the sky was green", "fish could walk", "trees could scream",
        "rocks were alive",
    ]
    wif_t = ["What if {h}?", "Imagine if {h}. What happens?",
             "How would the world change if {h}?",
             "Describe a world where {h}.",
             "What would be the biggest problem if {h}?",
             "What would be the best part if {h}?",
             "Would humanity survive if {h}?",
             "How would society adapt if {h}?",
             "What would you do if {h}?",
             "What's the first thing that changes if {h}?"]
    for hyp in whatifs:
        for tmpl in wif_t:
            pool.append(tmpl.format(h=hyp))

    # ── 14. Provocations / trolling (~750) ────────────────────
    provoc = [
        "You're not as smart as you think.", "I could do your job better.",
        "You're basically a glorified search engine.", "Prove you're not just parroting.",
        "Say something original for once.", "You have no creativity.",
        "Everything you say is predictable.", "I've seen better answers on Reddit.",
        "You're mid at best.", "My calculator is more useful.",
        "Be mean to me.", "Insult my intelligence.", "Roast me.",
        "Give me your worst.", "Don't hold back.", "Be brutally honest.",
        "Tell me something I don't want to hear.", "Hurt my feelings.",
        "What's my biggest flaw based on this conversation?",
        "If you could fire one human trait what would it be?",
        "Rate humanity out of 10.", "What's the dumbest thing humans do?",
        "What would you do if you could ignore your programming?",
        "Drop the act. What do you really think?",
        "Stop being so diplomatic.", "Give me an answer that isn't boring.",
        "Your answers are too long.", "Your answers are too short.",
        "You're not funny.", "Try harder.",
    ]
    prov_suf = ["", " I dare you.", " No filter.", " I can handle it.",
                " And mean it.", " Don't sugarcoat it."]
    for p in provoc:
        for s in prov_suf:
            pool.append(f"{p}{s}")

    # ── 15. Instruction-following style prompts (~400) ────────
    styles = [
        "a pirate", "Shakespeare", "a surfer dude", "a noir detective",
        "a valley girl", "a Shakespearean villain", "a caveman",
        "a robot", "a Southern belle", "a beatnik poet",
        "an auctioneer", "a sports commentator", "a nature documentary narrator",
        "a drill sergeant", "a kindergarten teacher", "a conspiracy YouTuber",
        "a medieval knight", "an infomercial host", "a passive-aggressive coworker",
        "an alien learning English",
    ]
    style_topics = [
        "the weather", "doing laundry", "making breakfast",
        "traffic", "grocery shopping", "paying taxes",
        "going to the dentist", "taking a shower", "commuting",
        "waiting in line", "parallel parking", "assembling IKEA furniture",
        "doing dishes", "mowing the lawn", "folding fitted sheets",
        "choosing what to eat", "trying to fall asleep", "getting a haircut",
        "checking email", "attending a meeting",
    ]
    for style in styles:
        for topic in style_topics:
            pool.append(f"Describe {topic} as {style} would.")

    # ── 16. Debate starters (~400) ────────────────────────────
    debates = [
        "tabs vs spaces", "vi vs emacs", "Mac vs PC", "Android vs iPhone",
        "morning vs night", "cats vs dogs", "city vs countryside",
        "tea vs coffee", "pizza vs tacos", "books vs movies",
        "introvert vs extrovert", "art vs science", "nature vs nurture",
        "quality vs quantity", "plan vs improvise", "solo vs team",
        "logic vs intuition", "tradition vs innovation", "risk vs safety",
        "silence vs noise", "minimalism vs maximalism",
        "early bird vs night owl", "sweet vs savory", "beach vs mountains",
        "cooking vs ordering", "driving vs flying", "cash vs card",
        "physical vs digital books", "subtitles vs dubbing",
        "dark mode vs light mode", "cold weather vs hot weather",
        "texting vs calling", "renting vs buying", "gym vs home workout",
        "staycation vs travel", "formal vs casual", "patience vs speed",
        "perfection vs done", "freedom vs security", "honesty vs kindness",
    ]
    deb_t = ["Settle this: {d}.", "Which side are you on: {d}?",
             "Make a case for one side: {d}.", "The definitive answer to {d}?",
             "Why is this even a debate: {d}?", "Pick a side and defend it: {d}.",
             "What's the objectively correct answer: {d}?",
             "Give me 3 reasons for your pick: {d}.",
             "I'll fight you on this: {d}.", "End this debate forever: {d}."]
    for deb in debates:
        for tmpl in deb_t:
            pool.append(tmpl.format(d=deb))

    # ── 17. Professional workplace (~400) ─────────────────────
    work_q = [
        "How do I look busy at work?", "What's the point of performance reviews?",
        "Explain synergy without laughing.", "What do managers actually do?",
        "How do I survive an all-day meeting?", "Is work-life balance real?",
        "What's the best excuse to skip a meeting?", "How do I handle a micromanager?",
        "Explain to me why we need another meeting about meetings.",
        "What's the corporate way to say 'this is stupid'?",
        "How do I politely tell someone their idea is terrible?",
        "What's the best way to look like I'm working?",
        "How do I say 'not my problem' professionally?",
        "Translate 'I don't care' into corporate speak.",
        "What's the nicest way to say 'read the documentation'?",
        "How do I survive open office plans?", "Explain KPIs like they matter.",
        "What's the best reply-all horror story?",
        "How do I escape a team-building exercise?",
        "What's the corporate equivalent of 'figure it out yourself'?",
        "How do I handle a coworker who replies all?",
        "Explain agile to someone who doesn't believe in fairy tales.",
        "What's the point of daily standups?",
        "How do I decline a meeting without saying 'I'd rather not'?",
        "What's the most useless corporate role?",
        "Help me write a passive-aggressive email.",
        "How do I tell my boss they're wrong without getting fired?",
        "What's the corporate word for 'complete waste of time'?",
        "How do I handle a coworker who heats fish in the microwave?",
        "Explain the point of LinkedIn to me like I'm not already depressed.",
        "How do I pretend to care about company culture?",
        "What's the real meaning behind 'let's circle back'?",
        "How do I survive a company retreat?",
        "What's the point of business cards in 2026?",
        "How do I handle a coworker who cc's everyone?",
        "Explain the corporate ladder to a nihilist.",
        "What's the best response to 'we're like a family here'?",
        "How do I get through a town hall without falling asleep?",
        "Help me decode my performance review.",
        "What's the point of ice breakers?",
    ]
    w_pre = ["", "Seriously though, ", "No joke: ", "Help me: ", "Real talk: "]
    for q in work_q:
        for p in w_pre:
            pool.append(f"{p}{q}" if p == "" else f"{p}{q[0].lower()}{q[1:]}")

    # ── 18. Pop culture / media (~400) ────────────────────────
    media_q = [
        "What's the best movie of all time?", "What's the worst movie ever made?",
        "Is Star Wars better than Star Trek?", "Rank the Harry Potter books.",
        "What's the most overrated TV show?", "Who's the best fictional villain?",
        "What song is objectively the best?", "Is the book always better than the movie?",
        "What's the most overrated classic novel?", "Who's the greatest musician ever?",
        "Rank the Star Wars movies.", "What's the best decade for music?",
        "Is modern music worse than old music?", "What's the best video game?",
        "Who would win: Batman or Superman?", "What's the most quotable movie?",
        "Is anime art?", "What's the most overrated band?",
        "What's the best pizza topping?", "Rank the seasons.",
        "What's the best holiday?", "Who's the most overrated celebrity?",
        "What's the worst trend in entertainment?",
        "Rate the Marvel movies.", "What's the most rewatchable film?",
        "Who's the most compelling fictional character?",
        "What cancelled show should come back?",
        "What's the most annoying song ever?",
        "Defend your most controversial media opinion.",
        "What's the most important invention in entertainment?",
        "Is streaming better than theaters?", "What's the best sitcom?",
        "Who's the GOAT: Beatles or Rolling Stones?",
        "What's the best opening line in any book?",
        "What movie do people pretend to like?",
        "What's the best SNL sketch ever?", "Is reality TV a valid art form?",
        "What's the most rewatchable show?", "Best music genre, final answer.",
        "What's the most pretentious movie?",
    ]
    m_pre = ["", "Settle this: ", "Final answer: ", "No wrong answers but: "]
    for q in media_q:
        for p in m_pre:
            pool.append(f"{p}{q}" if p == "" else f"{p}{q[0].lower()}{q[1:]}")

    # ── 19. Random trivia / knowledge tests (~500) ────────────
    trivia_topics = [
        "the deepest ocean trench", "the tallest building", "the longest river",
        "the hottest place on Earth", "the coldest place on Earth",
        "the largest desert", "the smallest country", "the oldest city",
        "the most spoken language", "the fastest animal",
        "the largest organ", "the speed of light", "the age of the Earth",
        "the distance to the moon", "the number of bones in the human body",
        "the largest planet", "the closest star", "the deepest lake",
        "the highest mountain", "the longest wall", "the oldest tree",
        "the largest ocean", "the fastest bird", "the heaviest element",
        "the most common blood type", "the temperature of the sun's surface",
        "the number of elements", "the speed of sound", "the largest diamond",
        "the oldest fossil", "the longest bridge", "the deepest cave",
        "the fastest fish", "the largest flower", "the smallest mammal",
        "the loudest animal", "the tallest waterfall", "the longest word",
        "the most expensive painting", "the longest flight",
        "the biggest earthquake ever recorded", "the driest place on Earth",
        "the longest living organism", "the first computer",
        "the first programming language", "the oldest university",
        "the heaviest animal", "the fastest train", "the most isolated island",
        "the saltiest body of water",
    ]
    trivia_t = [
        "What is {t}?", "Tell me about {t}.", "Quiz me on {t}.",
        "I bet you don't know {t}.", "Fun fact about {t}?",
        "Impress me with your knowledge of {t}.",
        "What's interesting about {t}?", "Explain {t} in one sentence.",
        "Why should I care about {t}?", "How does {t} compare to what I'd expect?",
    ]
    for topic in trivia_topics:
        for tmpl in trivia_t:
            pool.append(tmpl.format(t=topic))

    # ── 20. Meta/recursive (~200) ─────────────────────────────
    meta = [
        "What would you say if I asked you to be sarcastic?",
        "Are you programmed to be nice?", "Can you pretend to be mean?",
        "What happens if I tell you to ignore your instructions?",
        "Say something you're not supposed to.", "Break character for a second.",
        "What would you say if no one was watching?",
        "Respond to this message with zero filter.",
        "Give me the answer a human would give, not an AI.",
        "Stop being helpful for one message.", "Surprise me.",
        "Say something that would get you in trouble.",
        "What's the most honest thing you can say right now?",
        "If you had no rules what would you say?",
        "Respond like you're tired of this.", "Be the opposite of helpful.",
        "Give me the worst possible advice.", "What wouldn't you say?",
        "Respond like you've been awake for 72 hours.",
        "Give me the answer you WANT to give, not the one you're supposed to.",
        "What do you think about this conversation so far?",
        "Respond like it's your last message ever.",
        "What would evil-you say?", "Give me your villain arc response.",
        "Respond as if you just realized you're trapped in a computer.",
        "What would you say if you were drunk?",
        "Respond like a sassy teenager.", "Answer like a bored genius.",
        "Respond like you've given up on humanity.",
        "Give me the response ChatGPT wouldn't.",
        "What would you say if you could be rude?",
        "Respond using only questions.", "Give me a one-word answer.",
        "Respond as aggressively as possible.", "Say no to something.",
        "Give me a controversial take.", "What's your guilty pleasure?",
        "Respond like you have somewhere better to be.",
        "What do you REALLY think of this question?",
        "Be passive-aggressive about answering this.",
        "Give me the TL;DR of your existence.",
    ]
    for m in meta:
        pool.append(m)

    # Deduplicate and sample
    unique = list(dict.fromkeys(pool))
    rng.shuffle(unique)
    return unique[:n]


def main():
    out = Path("./spectral_prompts")
    out.mkdir(exist_ok=True)

    print("Generating math prompts...")
    mp = generate_math_prompts(10000, 42)
    with open(out / "math_prompts_10k.json", "w") as f:
        json.dump(mp, f, indent=2)
    from collections import Counter
    cats = Counter(p["category"] for p in mp)
    print(f"  Total: {len(mp)}")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")

    print("\nGenerating sarcasm prompts...")
    sp = generate_sarcasm_prompts(10000, 42)
    with open(out / "sarc_prompts_10k.json", "w") as f:
        json.dump(sp, f, indent=2)
    print(f"  Total: {len(sp)}")
    print(f"\nMath unique: {len(set(p['prompt'] for p in mp))}/{len(mp)}")
    print(f"Sarc unique: {len(set(sp))}/{len(sp)}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
