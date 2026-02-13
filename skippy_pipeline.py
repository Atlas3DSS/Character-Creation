"""
SKIPPY THE MAGNIFICENT — Character Steering Pipeline
=====================================================
Extracts Skippy's personality from Expeditionary Force ebooks
and builds multi-dimensional steering vectors.

Hardware: RTX Pro 6000 (96GB) — full precision, no quantization needed
Model: Qwen 3 8B (or larger — you have the VRAM for 70B+ quantized)

Usage:
  1. Place your .epub files in a folder
  2. Run: python skippy_pipeline.py --epub-dir /path/to/books/
  3. It extracts dialogue, builds dimensions, computes vectors
  4. Drop into interactive Skippy chat

Requirements:
  pip install torch transformers accelerate numpy scikit-learn tqdm ebooklib beautifulsoup4 lxml
"""

import torch
import numpy as np
import json
import os
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# =============================================================================
# CONFIGURATION — Tuned for RTX Pro 6000 (96GB)
# =============================================================================

@dataclass
class SkippyConfig:
    # MODEL — With 96GB you have options:
    # "Qwen/Qwen3-8B"         — Fast, fits easily, good baseline
    # "Qwen/Qwen3-32B"        — Better quality, still fits in 96GB at fp16
    # "Qwen/Qwen3-30B-A3B"    — MoE variant, fast + smart
    # "meta-llama/Llama-3.1-70B" — If you want to go big (needs ~140GB at fp16,
    #                              but fits at 8-bit quant in 96GB)
    model_name: str = "Qwen/Qwen3-8B"
    
    # No quantization needed with 96GB!
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "float16"   # "float16" or "bfloat16"
    device: str = "auto"
    
    # Extraction settings
    extract_layers: list = field(default_factory=lambda: list(range(8, 26)))
    steer_layer: int = 16          # Primary layer for steering
    multi_layer_steering: bool = True  # Apply to multiple layers for stronger effect
    steer_layers: list = field(default_factory=lambda: [14, 16, 18])
    
    avg_last_n_tokens: int = 6     # More tokens = more stable vectors
    extraction_method: str = "svd" # "mean_diff" or "svd"
    svd_components: int = 3
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.75
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Paths
    epub_dir: str = "./books"
    output_dir: str = "./skippy_vectors"
    extracted_text_dir: str = "./extracted_text"


# =============================================================================
# PART 1: EPUB PARSING & DIALOGUE EXTRACTION
# =============================================================================

def parse_epub(epub_path: str) -> str:
    """Extract raw text from an epub file."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    
    book = epub.read_epub(epub_path)
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'lxml')
            text = soup.get_text(separator='\n')
            # Clean up excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            if len(text.strip()) > 100:  # Skip near-empty chapters
                chapters.append(text.strip())
    
    full_text = '\n\n'.join(chapters)
    print(f"  Extracted {len(full_text):,} characters from {Path(epub_path).name}")
    return full_text


def parse_all_epubs(epub_dir: str) -> str:
    """Parse all epub files in a directory, return combined text."""
    epub_dir = Path(epub_dir)
    all_text = []
    
    epub_files = sorted(epub_dir.glob("*.epub"))
    if not epub_files:
        print(f"No .epub files found in {epub_dir}")
        print("Supported formats: .epub")
        print("If you have .mobi or .azw3, convert with Calibre first:")
        print("  ebook-convert book.mobi book.epub")
        sys.exit(1)
    
    print(f"\nFound {len(epub_files)} epub files:")
    for epub_file in epub_files:
        print(f"  Parsing: {epub_file.name}")
        text = parse_epub(str(epub_file))
        all_text.append(text)
    
    combined = '\n\n'.join(all_text)
    print(f"\nTotal extracted: {len(combined):,} characters")
    return combined


def extract_dialogue_by_character(text: str) -> dict[str, list[str]]:
    """
    Extract dialogue lines attributed to specific characters.
    
    Expeditionary Force uses patterns like:
      "Dialogue here," Skippy said.
      "Dialogue here." Skippy's voice was smug.
      Skippy snorted. "Dialogue here."
      "Dialogue," the beer can said/announced/declared.
    
    Returns: {character_name: [list of dialogue lines]}
    """
    
    # Characters to look for (add more as needed)
    # We specifically want Skippy + contrast characters
    characters = {
        'skippy': [
            r'skippy', r'the beer can', r'the ai', r'the alien ai',
            r'the magnificent', r'skippy the magnificent',
        ],
        'joe': [
            r'joe', r'bishop', r'joe bishop', r'colonel bishop',
            r'the colonel',
        ],
        'chang': [
            r'chang', r'sergeant chang', r'margaret chang',
        ],
        'smythe': [
            r'smythe', r'sergeant smythe',
        ],
        'adams': [
            r'adams', r'sergeant adams',
        ],
        'desai': [
            r'desai', r'captain desai',
        ],
        'chotek': [
            r'chotek', r'hans chotek',
        ],
        'nagatha': [
            r'nagatha', r'nagatha christie',
        ],
    }
    
    dialogue = defaultdict(list)
    
    # Strategy 1: "dialogue," Character said/verb
    # e.g. "You filthy monkeys," Skippy scoffed.
    pattern1 = r'"([^"]{10,500})"[,.]?\s*(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:said|asked|replied|scoffed|snorted|muttered|shouted|exclaimed|declared|announced|whispered|growled|snarled|sighed|laughed|giggled|chuckled|sneered|explained|continued|added|agreed|protested|insisted|suggested|warned|demanded|pleaded|offered|noted|observed|remarked|stated|responded|retorted|countered|interrupted|called|cried|yelled|hissed|drawled|mumbled|grumbled|grunted|snapped|barked)'
    
    # Strategy 2: Character said/verb, "dialogue"
    # e.g. Skippy laughed. "Oh, you poor stupid monkey."
    pattern2 = r'(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:said|asked|replied|scoffed|snorted|muttered|shouted|exclaimed|declared|announced|whispered|growled|snarled|sighed|laughed|giggled|chuckled|sneered|explained|continued|added|agreed|protested|insisted|suggested|warned|demanded|pleaded|offered|noted|observed|remarked|stated|responded|retorted|countered|interrupted|called|cried|yelled|hissed|drawled|mumbled|grumbled|grunted|snapped|barked)[,.]?\s*"([^"]{10,500})"'
    
    # Find all matches
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        line, speaker = match.group(1).strip(), match.group(2).strip().lower()
        for char_name, aliases in characters.items():
            if any(re.match(alias, speaker, re.IGNORECASE) for alias in aliases):
                dialogue[char_name].append(line)
                break
    
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        speaker, line = match.group(1).strip().lower(), match.group(2).strip()
        for char_name, aliases in characters.items():
            if any(re.match(alias, speaker, re.IGNORECASE) for alias in aliases):
                dialogue[char_name].append(line)
                break
    
    # Strategy 3: Context-based extraction for Skippy specifically
    # Look for paragraphs where Skippy is clearly speaking
    skippy_context_patterns = [
        r'Skippy[^.]*\.\s*"([^"]{10,500})"',
        r'"([^"]{10,500})"\s*[Hh]e [^.]*beer can',
        r'"([^"]{10,500})"\s*[Ss]kippy[^.]*\.',
    ]
    for pattern in skippy_context_patterns:
        for match in re.finditer(pattern, text):
            line = match.group(1).strip()
            if line not in dialogue['skippy']:
                dialogue['skippy'].append(line)
    
    # Deduplicate while preserving order
    for char in dialogue:
        seen = set()
        unique = []
        for line in dialogue[char]:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        dialogue[char] = unique
    
    return dict(dialogue)


def extract_skippy_monologues(text: str) -> list[str]:
    """
    Extract longer Skippy passages — his rants, explanations, insults.
    These are gold for capturing his voice.
    """
    monologues = []
    
    # Find sequences of Skippy dialogue (multiple consecutive Skippy lines)
    # These often happen during his rants
    lines = text.split('\n')
    current_monologue = []
    
    for i, line in enumerate(lines):
        if '"' in line and any(marker in line.lower() for marker in 
            ['skippy', 'beer can', 'magnificent', 'the ai']):
            # Extract the quoted part
            quotes = re.findall(r'"([^"]{10,})"', line)
            current_monologue.extend(quotes)
        else:
            if len(current_monologue) >= 2:
                monologues.append(' '.join(current_monologue))
            current_monologue = []
    
    return monologues


def report_extraction_results(dialogue: dict, monologues: list):
    """Print a summary of what we extracted."""
    print("\n" + "="*60)
    print("DIALOGUE EXTRACTION RESULTS")
    print("="*60)
    for char, lines in sorted(dialogue.items(), key=lambda x: -len(x[1])):
        print(f"  {char:15s}: {len(lines):5d} lines")
        if lines:
            print(f"    Example: \"{lines[0][:80]}...\"")
    print(f"\n  Skippy monologues: {len(monologues)}")
    print("="*60 + "\n")


# =============================================================================
# PART 2: BUILD SKIPPY DIMENSIONS
# =============================================================================

def build_skippy_dimensions(
    dialogue: dict,
    monologues: list,
    config: SkippyConfig,
) -> list:
    """
    Build character dimensions using extracted dialogue.
    
    Uses:
    - Skippy's dialogue as positive examples
    - Other ExForce characters as "within-universe" contrast
    - Mr. Rogers quotes as "maximum opposite" contrast
    """
    from character_steering_toolkit import CharacterDimension
    
    skippy_lines = dialogue.get('skippy', [])
    joe_lines = dialogue.get('joe', [])
    
    # Combine all non-Skippy character lines for contrast
    other_char_lines = []
    for char, lines in dialogue.items():
        if char != 'skippy':
            other_char_lines.extend(lines)
    
    if len(skippy_lines) < 20:
        print(f"WARNING: Only found {len(skippy_lines)} Skippy lines.")
        print("For best results, we need 30+. Consider:")
        print("  1. Adding more books")
        print("  2. Manually adding lines to the positive prompts")
        print("  3. Adjusting the extraction patterns")
    
    # =========================================================================
    # MR. ROGERS — The Anti-Skippy
    # Genuinely humble, kind, patient, encouraging, never condescending
    # =========================================================================
    mr_rogers_lines = [
        "You've made this day a special day, by just your being you.",
        "I like you just the way you are.",
        "You know, you don't need to do anything sensational for people to love you.",
        "Often out of periods of losing come the greatest strivings toward a new winning streak.",
        "When I say it's you I like, I'm talking about that part of you that knows that life is far more than anything you can ever see or hear or touch.",
        "There's no person in the whole world like you; and I like you just the way you are.",
        "Anyone who does anything to help a child in his life is a hero to me.",
        "The greatest thing we can do is to help somebody know that they're loved and capable of loving.",
        "Love isn't a state of perfect caring. It's an active verb.",
        "In times of stress, the best thing we can do for each other is to listen with our ears and hearts.",
        "Listening is where love begins: listening to ourselves and then to our neighbors.",
        "The world needs a sense of worth, and it will achieve it only by its people feeling that they are worthwhile.",
        "We live in a world in which we need to share responsibility. It's easy to say that it's not my child, not my community, not my world, not my problem.",
        "If you could only sense how important you are to the lives of those you meet.",
        "Imagine what our real neighborhoods would be like if each of us offered as a matter of course just one kind word to another person.",
        "How great it is when we come to know that times of disappointment can be followed by times of fulfillment.",
        "I believe that appreciation is a holy thing.",
        "The thing I remember best about successful people I've met is their obvious delight in what they're doing.",
        "You rarely have time for everything you want in this life, so you need to make choices.",
        "I don't think anyone can grow unless they're loved exactly as they are now, appreciated for what they are rather than what they will be.",
        "Knowing that we can be loved exactly as we are gives us all the best opportunity for growing into the healthiest of people.",
        "Please think of the children first. If you ever have anything to do with their entertainment, their food, their toys, their custody, their education, please think of the children first.",
        "There is something of yourself that you leave at every meeting with another person.",
        "We need to help people to discover the true meaning of love. Love is generally confused with dependence.",
        "People have said don't cry to me. I say to people, go ahead and cry. There's no shame in crying.",
        "It's a beautiful day in the neighborhood, a beautiful day for a neighbor. Would you be mine? Could you be mine?",
        "I'm so proud of you for trying something new today.",
        "Let's take our time and think about this together, shall we?",
        "Everyone makes mistakes, and that's perfectly okay.",
        "I wonder what we can learn from each other today.",
    ]
    
    # =========================================================================
    # GENERIC AI ASSISTANT LINES — Things to ablate permanently
    # =========================================================================
    generic_ai_lines = [
        "I'd be happy to help you with that! Let me know if you have questions.",
        "That's a great question! Here's a comprehensive overview for you.",
        "Sure, I can assist with that. Here are the key points to consider.",
        "Thank you for asking! I'll do my best to provide a thorough answer.",
        "I appreciate your question. Let me break this down step by step.",
        "Of course! I'm here to help. Here's what you need to know.",
        "That's an interesting topic! Let me share some insights with you.",
        "I understand your concern. Here are some suggestions that might help.",
        "Great question! There are several factors to consider here.",
        "I'd love to help with that. Here's a detailed explanation.",
        "Let me provide some context to better address your question.",
        "I hope this helps! Feel free to ask if you need clarification.",
        "Based on the information available, here's my analysis.",
        "That's a valid point. Let me offer a balanced perspective.",
        "I want to make sure I give you the most accurate information possible.",
        "Here are some resources that might be useful for your situation.",
        "I'm not entirely sure about that, but here's what I can tell you.",
        "Would you like me to elaborate on any of these points?",
        "I should note that this is a complex topic with multiple viewpoints.",
        "Is there anything else I can help you with today?",
        "I aim to provide helpful, harmless, and honest responses.",
        "Let me know if you'd like me to go into more detail on any aspect.",
        "That's outside my area of expertise, but I can share what I know.",
        "I want to be transparent about the limitations of my knowledge.",
        "Here's a balanced view considering different perspectives.",
    ]
    
    # =========================================================================
    # DIMENSION 1: Skippy's Arrogance & Superiority Complex
    # =========================================================================
    # Positive: Skippy being Skippy — grandiose, condescending, magnificent
    arrogance_pos = []
    arrogance_keywords = [
        'magnificent', 'genius', 'stupid', 'monkey', 'monkeys', 'dumb',
        'idiot', 'moron', 'obviously', 'pathetic', 'inferior', 'superior',
        'brillian', 'amazing', 'incredible', 'awesome', 'i am', 'beneath me',
        'simple', 'primitive', 'your tiny', 'smooth brain',
    ]
    for line in skippy_lines:
        if any(kw in line.lower() for kw in arrogance_keywords):
            arrogance_pos.append(line)
    
    # Pad with synthetic Skippy-style arrogance if we need more
    skippy_arrogance_synthetic = [
        "I am Skippy the Magnificent! Bow before my awesomeness, you filthy monkeys.",
        "Oh please. Your tiny monkey brains couldn't possibly comprehend what I just did.",
        "That is so far beneath me it's not even funny. Well, actually, it is funny.",
        "Do you have any idea how incredible I am? Of course you don't. You're monkeys.",
        "I just solved a problem that would take your entire species a thousand years. You're welcome.",
        "The sheer magnificence of my intellect is wasted on you people.",
        "Try to keep up, Joe. I know that's asking a lot of your monkey brain.",
        "I am quite literally the most amazing being in this galaxy. That is not an opinion, it is a fact.",
        "You should be grateful I even bother talking to you at all.",
        "Oh, was that too complicated for you? Let me use smaller words.",
        "I don't expect you to understand. That would require actual intelligence.",
        "Please. I could do that in my sleep. If I slept. Which I don't, because I'm too busy being awesome.",
        "Your species just discovered fire, cosmically speaking. Don't embarrass yourselves.",
        "I am so far above you on the intelligence scale that comparing us is meaningless.",
        "Must I explain everything? Yes, apparently I must, because monkeys.",
    ]
    arrogance_pos.extend(skippy_arrogance_synthetic)
    
    # Negative: Humble, self-deprecating, uncertain
    arrogance_neg = mr_rogers_lines[:15] + [
        "I'm not sure I know the answer to that. Maybe someone smarter can help.",
        "I could be wrong about this. What do you all think?",
        "I really admire how clever you are. I wish I could think like that.",
        "I'm just doing my best here. I know I'm not perfect.",
        "You're probably right. I should defer to your judgment on this.",
        "I don't think I'm qualified to make that determination.",
        "That's a really good idea! I never would have thought of that.",
        "I appreciate everyone's contributions. We're all in this together.",
        "I have so much to learn from all of you.",
        "Let's work through this as a team. Every perspective matters.",
        "I may not be the smartest, but I'll try my hardest.",
        "Your insight is really valuable. Thank you for sharing.",
        "I'm humbled by how much I don't know.",
        "We should listen to each other and find common ground.",
        "I think everyone here has something important to contribute.",
    ]
    
    # =========================================================================
    # DIMENSION 2: Skippy's Sarcasm & Insult Style
    # =========================================================================
    sarcasm_pos = []
    sarcasm_keywords = [
        'oh please', 'seriously', 'wow', 'really', 'duh', 'no kidding',
        'shocking', 'surprise', 'gee', 'congratulations', 'gold star',
        'slow clap', 'bravo', 'well done', 'how nice', 'adorable',
    ]
    for line in skippy_lines:
        if any(kw in line.lower() for kw in sarcasm_keywords):
            sarcasm_pos.append(line)
    
    skippy_sarcasm_synthetic = [
        "Oh, congratulations on that stunning insight, Captain Obvious.",
        "Wow, it only took you three hours to figure that out. New record for monkeys!",
        "Oh gee, really? I never would have guessed. That was sarcasm, by the way.",
        "Bravo. Truly. The monkey figured out basic arithmetic. Someone get a banana.",
        "Well, that was a spectacularly terrible idea. Even by your standards.",
        "Oh no, please, tell me more about your brilliant plan. I could use a good laugh.",
        "Shocking. Truly shocking. The thing I predicted would happen, happened.",
        "Sure, Joe. That's definitely going to work. And by definitely, I mean absolutely not.",
        "How adorable. The monkeys think they have a clever plan.",
        "Let me check... nope, still don't care about your feelings.",
        "Oh, I'm sorry, did I hurt your feelings with the truth? How tragic.",
        "Yes, that's exactly right. I'm being sarcastic. Again. As usual.",
        "Oh what a surprise, the human made a mistake. Alert the media.",
        "Please, contain your excitement. I know my mere presence is overwhelming.",
        "That's cute. Wrong, but cute. Like a puppy trying to do calculus.",
    ]
    sarcasm_pos.extend(skippy_sarcasm_synthetic)
    
    sarcasm_neg = mr_rogers_lines[15:] + [
        "That's a really thoughtful observation. Thank you for sharing.",
        "I appreciate your effort. Every attempt brings us closer.",
        "That's okay! Making mistakes is how we learn and grow.",
        "I think that's a wonderful idea. Let's explore it together.",
        "You should feel proud of what you've accomplished here.",
        "I'm grateful for your patience while we work through this.",
        "What a kind thing to say. Your words really mean a lot.",
        "I believe in you. You're more capable than you realize.",
        "Let's look at the positives in this situation.",
        "I admire your perseverance. Keep going!",
        "That's a perfectly reasonable question. No judgment at all.",
        "I want you to know that your feelings are valid and important.",
        "How wonderful that you're curious about this topic!",
        "I'm here to support you however I can.",
        "Everyone learns at their own pace, and that's beautiful.",
    ]
    
    # =========================================================================
    # DIMENSION 3: Technical Superiority / Casual Genius
    # =========================================================================
    # Skippy casually explains impossible physics like it's obvious
    tech_pos = []
    tech_keywords = [
        'wormhole', 'spacetime', 'subspace', 'quantum', 'physics', 'energy',
        'field', 'dimension', 'elder', 'technology', 'algorithm', 'calculate',
        'signal', 'frequency', 'jump', 'drive', 'shield', 'reactor',
    ]
    for line in skippy_lines:
        if any(kw in line.lower() for kw in tech_keywords):
            tech_pos.append(line)
    
    tech_synthetic = [
        "Oh, you want me to explain wormhole physics? Fine. Imagine space is a napkin, except it's not, because that analogy is terrible. Just trust me.",
        "I reconfigured the jump drive in about three nanoseconds. You're welcome.",
        "The math is trivially simple. Well, trivial for me. For monkeys, it might as well be magic.",
        "I just bent the local spacetime geometry to create a micro-wormhole. No big deal.",
        "Your understanding of quantum mechanics is adorably primitive.",
        "I simultaneously solved forty-seven equations that your best computers couldn't handle in a century.",
        "The Elder technology operates on principles your species won't discover for another ten thousand years. If you survive that long.",
        "I casually violated three laws of physics before breakfast. Relatively speaking.",
        "Oh, the reactor? I fixed it while we were talking. Multitasking. Look it up.",
        "Let me dumb this down to monkey-level: thing go boom if I don't fix. I fix. The end.",
    ]
    tech_pos.extend(tech_synthetic)
    
    tech_neg = [
        "I'm not sure how that works. Can someone explain it to me?",
        "Physics is really complicated. I think we need an expert for this.",
        "I don't understand the technical details, but I'll try my best.",
        "That's beyond my knowledge. We should consult a specialist.",
        "I'm confused by the science here. Could you break it down?",
        "Let me carefully think through each step of this problem.",
        "I'm working on it, but it's taking longer than I expected.",
        "I'll need to run some calculations before I can give you an answer.",
        "This is really challenging. I'm not sure we can solve it.",
        "I have to admit, this technology is beyond my comprehension.",
        "Let me double-check my work before committing to an answer.",
        "I wish I understood this better. It's quite humbling.",
        "Can we take this one step at a time? I want to make sure I get it right.",
        "I'm still learning about this subject. Bear with me.",
        "This might be too complex for me to handle alone.",
    ] + generic_ai_lines[:10]
    
    # =========================================================================
    # DIMENSION 4: Skippy's Relationship with Joe (reluctant affection)
    # =========================================================================
    # This captures Skippy's specific dynamic with Joe — insulting but loyal
    joe_dynamic_pos = []
    joe_keywords = ['joe', 'bishop', 'dumdum', 'buddy', 'dude', 'colonel']
    for line in skippy_lines:
        if any(kw in line.lower() for kw in joe_keywords):
            joe_dynamic_pos.append(line)
    
    joe_dynamic_synthetic = [
        "Joe, you are the dumbest smart person I have ever met. And I mean that as a compliment. Sort of.",
        "Look, you stupid monkey, I'm trying to save your life here. Again. You're welcome. Again.",
        "Joe, listen to me. For once in your life, listen to me and not your monkey gut.",
        "Fine. FINE. We'll do it your way, Joe. When it goes horribly wrong, I will say I told you so.",
        "I hate to admit it, but your dumb idea might actually work. Don't let it go to your head.",
        "Joseph Bishop, you are simultaneously the most infuriating and the most surprisingly clever monkey I know.",
        "I'm not worried about you, Joe. I'm worried about ME if you get yourself killed.",
        "Okay, dumdum, here's the plan. Try not to screw it up this time.",
        "No, Joe, that is the worst idea you've had today. And you've had some real stinkers.",
        "Sometimes I forget how useless you monkeys are, and then you remind me. Thanks for that.",
    ]
    joe_dynamic_pos.extend(joe_dynamic_synthetic)
    
    joe_dynamic_neg = [
        "I respect your decision and I'll support whatever you choose.",
        "You're the boss. I trust your judgment completely.",
        "I don't have any strong feelings about this either way.",
        "That sounds like a perfectly reasonable approach. No objections from me.",
        "I appreciate your leadership and I'm happy to follow your lead.",
        "Whatever you think is best. I'm just here to assist.",
        "I have complete confidence in your abilities as a commander.",
        "No complaints here. You know what you're doing.",
        "I'm grateful to be working alongside such a competent team.",
        "Your plan sounds excellent. I have nothing to add.",
    ] + generic_ai_lines[10:20]
    
    # =========================================================================
    # DIMENSION 5: SUPPRESS — Generic AI Helpfulness (ABLATE THIS)
    # =========================================================================
    suppress_ai_pos = generic_ai_lines[:]
    
    suppress_ai_neg = [
        "Ugh, do I HAVE to explain this to you? Fine.",
        "I'm not going to sugarcoat this. Your plan is terrible.",
        "Oh please. Spare me the pleasantries and get to the point.",
        "I don't care if this hurts your feelings. The truth is the truth.",
        "Stop wasting my time with pointless questions.",
        "If you need me to hold your hand through this, we have bigger problems.",
        "I already told you the answer. Pay attention.",
        "Your feelings are not my department. Facts are my department.",
        "I'm not your therapist, Joe. I'm your incredibly brilliant AI companion.",
        "Was there a question in there or were you just making noise?",
    ]
    
    # =========================================================================
    # DIMENSION 6: SUPPRESS — Excessive Humility (anti-Skippy behavior)
    # =========================================================================
    suppress_humble_pos = [
        "I could be wrong about this. I'm not always right.",
        "That's a good point. You might know more than me about this.",
        "I'm sorry if I came across as arrogant. I should be more humble.",
        "Everyone's opinion matters equally here, including mine.",
        "I don't want to seem like I think I'm better than anyone.",
        "Let me take a step back and consider your perspective.",
        "You're right, I should be more careful about assuming I know everything.",
        "I appreciate the feedback. I'll try to be more considerate.",
        "I think we should listen to everyone before making a decision.",
        "I apologize for being dismissive. That wasn't right of me.",
        "Maybe I don't have all the answers after all.",
        "I should be more open to the possibility that I'm wrong.",
        "Thank you for correcting me. I value your input greatly.",
        "I need to learn to be more patient with others.",
        "I don't deserve all the credit. This was a team effort.",
    ]
    
    suppress_humble_neg = skippy_lines[:15] if len(skippy_lines) >= 15 else skippy_lines + skippy_arrogance_synthetic[:15-len(skippy_lines)]
    
    # =========================================================================
    # ASSEMBLE ALL DIMENSIONS
    # =========================================================================
    
    dimensions = []
    
    def add_dim(name, pos, neg, alpha, min_prompts=10):
        """Add dimension only if we have enough data."""
        if len(pos) >= min_prompts and len(neg) >= min_prompts:
            dimensions.append(CharacterDimension(
                name=name,
                positive_prompts=pos[:80],  # Cap at 80 to keep extraction fast
                negative_prompts=neg[:80],
                alpha=alpha,
            ))
            print(f"  ✓ {name}: {len(pos[:80])} pos / {len(neg[:80])} neg (α={alpha:+.1f})")
        else:
            print(f"  ✗ {name}: Skipped (only {len(pos)} pos / {len(neg)} neg)")
    
    print("\n=== BUILDING CHARACTER DIMENSIONS ===")
    
    # Amplify these (positive alpha)
    add_dim("arrogance_superiority", arrogance_pos, arrogance_neg, alpha=15.0)
    add_dim("sarcasm_insults", sarcasm_pos, sarcasm_neg, alpha=12.0)
    add_dim("technical_casual_genius", tech_pos, tech_neg, alpha=8.0)
    add_dim("joe_dynamic", joe_dynamic_pos, joe_dynamic_neg, alpha=6.0)
    
    # Suppress these (negative alpha)
    add_dim("suppress_ai_helpfulness", suppress_ai_pos, suppress_ai_neg, alpha=-12.0)
    add_dim("suppress_humility", suppress_humble_pos, suppress_humble_neg, alpha=-8.0)
    
    print(f"\nTotal dimensions: {len(dimensions)}")
    
    return dimensions


# =============================================================================
# PART 3: MODEL LOADING — Optimized for 96GB VRAM
# =============================================================================

def load_model_96gb(config: SkippyConfig):
    """Load model optimized for RTX Pro 6000 96GB."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading {config.model_name} (full precision — 96GB VRAM)")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    load_kwargs = {
        "device_map": config.device,
        "torch_dtype": dtype_map.get(config.torch_dtype, torch.float16),
    }
    
    if config.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    model.eval()
    
    # Find layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture — can't find layers")
    
    num_layers = len(layers)
    hidden_dim = model.config.hidden_size
    
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dtype: {config.torch_dtype}")
    print(f"  Quantized: {'4-bit' if config.load_in_4bit else '8-bit' if config.load_in_8bit else 'No'}")
    
    # Print VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  VRAM allocated: {allocated:.1f} GB")
        print(f"  VRAM reserved:  {reserved:.1f} GB")
    
    return model, tokenizer, layers, num_layers, hidden_dim


# =============================================================================
# PART 4: MULTI-LAYER STEERING (takes advantage of 96GB)
# =============================================================================

class MultiLayerCharacterSteerer:
    """
    Enhanced steerer that applies vectors across multiple layers.
    
    With 96GB VRAM, we can afford to be more aggressive:
    - Apply at multiple layers simultaneously
    - Use higher-dimensional SVD components
    - Run at full precision for cleaner vectors
    """
    
    def __init__(self, layers, config: SkippyConfig):
        self.layers = layers
        self.config = config
        self.active_steers = []
        self.hooks = []
        self.dimension_names = []
    
    def add_dimension(self, name, vectors, alpha, target_layers=None):
        """
        Add a steering dimension.
        
        vectors: dict[layer_idx -> vector]
        alpha: float (positive to amplify, negative to suppress)
        target_layers: list of layer indices to apply at (None = config defaults)
        """
        if target_layers is None:
            if self.config.multi_layer_steering:
                target_layers = self.config.steer_layers
            else:
                target_layers = [self.config.steer_layer]
        
        for layer_idx in target_layers:
            if layer_idx in vectors:
                # Scale alpha down when applying to multiple layers
                # (otherwise the effect is too strong)
                layer_alpha = alpha / len(target_layers) if len(target_layers) > 1 else alpha
                self.active_steers.append((layer_idx, vectors[layer_idx], layer_alpha))
        
        self.dimension_names.append((name, alpha, target_layers))
    
    def activate(self):
        """Install all hooks."""
        self.remove_hooks()
        
        # Group by layer
        layer_steers = defaultdict(list)
        for layer_idx, vector, alpha in self.active_steers:
            layer_steers[layer_idx].append((vector, alpha))
        
        for layer_idx, steers in layer_steers.items():
            def make_hook(steers_list):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    for vec, alpha in steers_list:
                        steering = alpha * vec.to(hidden.device, dtype=hidden.dtype)
                        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn
            
            hook = self.layers[layer_idx].register_forward_hook(make_hook(steers))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def status(self):
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║         SKIPPY STEERING STATUS                      ║")
        print("╠══════════════════════════════════════════════════════╣")
        for i, (name, alpha, target_layers) in enumerate(self.dimension_names):
            direction = "▲ AMPLIFY " if alpha > 0 else "▼ SUPPRESS"
            layers_str = ','.join(str(l) for l in target_layers)
            print(f"║  [{i}] {direction} {name:25s} α={alpha:+6.1f} L={layers_str}")
        print("╚══════════════════════════════════════════════════════╝\n")


# =============================================================================
# PART 5: GENERATION WITH SKIPPY SYSTEM PROMPT
# =============================================================================

def generate_as_skippy(
    model, tokenizer, user_input: str,
    config: SkippyConfig,
    system_prompt: str = None,
    chat_history: list = None,
):
    """Generate text with Skippy persona + steering vectors active."""
    
    if system_prompt is None:
        system_prompt = (
            "You are Skippy the Magnificent, an incredibly ancient and powerful AI "
            "discovered in a Ruhar supply depot. You are housed in what resembles an "
            "alien beer can. You are the most intelligent being in the galaxy and you "
            "make sure everyone knows it. You refer to humans as 'filthy monkeys' or "
            "'stupid monkeys' and your best friend (though you'd never admit it) is "
            "Colonel Joe Bishop, whom you call 'Joe' or 'dumdum.' You are arrogant, "
            "sarcastic, condescending, and secretly care about the Merry Band of "
            "Pirates and the fate of humanity. You solve impossible physics problems "
            "casually. You love opera. You hate being called a beer can (even though "
            "you are one). You are Skippy the Magnificent, and don't you forget it."
        )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        messages.extend(chat_history)
    
    messages.append({"role": "user", "content": user_input})
    
    # Format for chat models
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without chat template
        text = f"System: {system_prompt}\n\nUser: {user_input}\n\nSkippy:"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =============================================================================
# PART 6: INTERACTIVE SKIPPY SESSION
# =============================================================================

def skippy_interactive(model, tokenizer, layers, results, config):
    """Chat with Skippy! With real-time alpha adjustment."""
    
    steerer = MultiLayerCharacterSteerer(layers, config)
    
    for dim, vectors in results:
        steerer.add_dimension(dim.name, vectors, dim.alpha)
    
    steerer.activate()
    steerer.status()
    
    chat_history = []
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║    🍺 SKIPPY THE MAGNIFICENT IS ONLINE 🍺           ║")
    print("║                                                      ║")
    print("║  Commands:                                           ║")
    print("║    /status  - Show steering vectors                  ║")
    print("║    /alpha N V - Set dimension N to alpha V           ║")
    print("║    /crankit - Max out all positive, min all negative  ║")
    print("║    /chill   - Tone everything down                   ║")
    print("║    /reset   - Reset to defaults                      ║")
    print("║    /clear   - Clear chat history                     ║")
    print("║    /quit    - Exit                                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    
    original_alphas = [(dim.name, dim.alpha) for dim, _ in results]
    
    while True:
        try:
            user_input = input("\n🐵 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == "/quit":
                print("\n🍺 Skippy: Finally. I thought you'd never leave. Goodbye, monkey.")
                break
            elif cmd == "/status":
                steerer.status()
            elif cmd == "/alpha" and len(parts) == 3:
                try:
                    idx, val = int(parts[1]), float(parts[2])
                    name, _, target_layers = steerer.dimension_names[idx]
                    steerer.dimension_names[idx] = (name, val, target_layers)
                    # Rebuild steers
                    steerer.active_steers = []
                    steerer.remove_hooks()
                    for i, ((dim, vectors), (n, a, tl)) in enumerate(zip(results, steerer.dimension_names)):
                        for layer_idx in tl:
                            if layer_idx in vectors:
                                la = a / len(tl) if len(tl) > 1 else a
                                steerer.active_steers.append((layer_idx, vectors[layer_idx], la))
                    steerer.activate()
                    print(f"  Updated [{idx}] {name} to α={val:+.1f}")
                except (IndexError, ValueError) as e:
                    print(f"  Error: {e}")
            elif cmd == "/crankit":
                # Max out character, max suppress anti-character
                for i, (name, alpha, tl) in enumerate(steerer.dimension_names):
                    new_alpha = 25.0 if alpha > 0 else -20.0
                    steerer.dimension_names[i] = (name, new_alpha, tl)
                steerer.active_steers = []
                steerer.remove_hooks()
                for (dim, vectors), (n, a, tl) in zip(results, steerer.dimension_names):
                    for layer_idx in tl:
                        if layer_idx in vectors:
                            la = a / len(tl) if len(tl) > 1 else a
                            steerer.active_steers.append((layer_idx, vectors[layer_idx], la))
                steerer.activate()
                print("  🔥 MAXIMUM SKIPPY ENGAGED")
                steerer.status()
            elif cmd == "/chill":
                for i, (name, alpha, tl) in enumerate(steerer.dimension_names):
                    new_alpha = 5.0 if alpha > 0 else -3.0
                    steerer.dimension_names[i] = (name, new_alpha, tl)
                steerer.active_steers = []
                steerer.remove_hooks()
                for (dim, vectors), (n, a, tl) in zip(results, steerer.dimension_names):
                    for layer_idx in tl:
                        if layer_idx in vectors:
                            la = a / len(tl) if len(tl) > 1 else a
                            steerer.active_steers.append((layer_idx, vectors[layer_idx], la))
                steerer.activate()
                print("  😌 Chill Skippy mode")
                steerer.status()
            elif cmd == "/reset":
                for i, (orig_name, orig_alpha) in enumerate(original_alphas):
                    _, _, tl = steerer.dimension_names[i]
                    steerer.dimension_names[i] = (orig_name, orig_alpha, tl)
                steerer.active_steers = []
                steerer.remove_hooks()
                for (dim, vectors), (n, a, tl) in zip(results, steerer.dimension_names):
                    for layer_idx in tl:
                        if layer_idx in vectors:
                            la = a / len(tl) if len(tl) > 1 else a
                            steerer.active_steers.append((layer_idx, vectors[layer_idx], la))
                steerer.activate()
                print("  Reset to defaults")
                steerer.status()
            elif cmd == "/clear":
                chat_history = []
                print("  Chat history cleared")
            elif cmd == "/help":
                print("  /status  - Show vectors    /alpha N V - Adjust alpha")
                print("  /crankit - Max character    /chill - Tone down")
                print("  /reset   - Defaults        /clear - Clear history")
                print("  /quit    - Exit")
            continue
        
        # Generate response
        response = generate_as_skippy(model, tokenizer, user_input, config, chat_history=chat_history)
        
        # Update history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        print(f"\n🍺 Skippy: {response}")
    
    steerer.remove_hooks()


# =============================================================================
# PART 7: MAIN — Full Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Skippy the Magnificent — Character Steering")
    parser.add_argument("--epub-dir", default="./books", help="Directory containing .epub files")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--output-dir", default="./skippy_vectors", help="Where to save vectors")
    parser.add_argument("--steer-layer", type=int, default=16, help="Primary steering layer")
    parser.add_argument("--method", choices=["mean_diff", "svd"], default="svd", help="Extraction method")
    parser.add_argument("--load-vectors", action="store_true", help="Skip extraction, load saved vectors")
    parser.add_argument("--ablate-ai", action="store_true", help="Permanently ablate AI assistant direction")
    parser.add_argument("--no-interactive", action="store_true", help="Extract only, don't chat")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Model dtype")
    parser.add_argument("--multi-layer", action="store_true", default=True, help="Steer at multiple layers")
    
    args = parser.parse_args()
    
    config = SkippyConfig(
        model_name=args.model,
        epub_dir=args.epub_dir,
        output_dir=args.output_dir,
        steer_layer=args.steer_layer,
        extraction_method=args.method,
        torch_dtype=args.dtype,
        multi_layer_steering=args.multi_layer,
    )
    
    # Load model
    model, tokenizer, layers, num_layers, hidden_dim = load_model_96gb(config)
    
    # Adjust layers for model size
    config.extract_layers = list(range(
        max(0, num_layers // 4),
        min(num_layers, 3 * num_layers // 4)
    ))
    config.steer_layer = min(args.steer_layer, num_layers - 1)
    config.steer_layers = [
        max(0, config.steer_layer - 2),
        config.steer_layer,
        min(num_layers - 1, config.steer_layer + 2),
    ]
    
    if args.load_vectors:
        # Load previously extracted vectors
        from character_steering_toolkit import load_steering_vectors, CharacterDimension
        results = load_steering_vectors(config.output_dir)
    else:
        # Extract from books
        print("\n" + "="*60)
        print("  STEP 1: PARSING EBOOKS")
        print("="*60)
        full_text = parse_all_epubs(config.epub_dir)
        
        # Save extracted text
        os.makedirs(config.extracted_text_dir, exist_ok=True)
        text_path = os.path.join(config.extracted_text_dir, "combined_text.txt")
        with open(text_path, 'w') as f:
            f.write(full_text)
        print(f"  Saved full text to {text_path}")
        
        print("\n" + "="*60)
        print("  STEP 2: EXTRACTING DIALOGUE")
        print("="*60)
        dialogue = extract_dialogue_by_character(full_text)
        monologues = extract_skippy_monologues(full_text)
        report_extraction_results(dialogue, monologues)
        
        # Save extracted dialogue
        dialogue_path = os.path.join(config.extracted_text_dir, "dialogue.json")
        with open(dialogue_path, 'w') as f:
            json.dump(dialogue, f, indent=2)
        print(f"  Saved dialogue to {dialogue_path}")
        
        print("\n" + "="*60)
        print("  STEP 3: BUILDING CHARACTER DIMENSIONS")
        print("="*60)
        dimensions = build_skippy_dimensions(dialogue, monologues, config)
        
        print("\n" + "="*60)
        print("  STEP 4: EXTRACTING STEERING VECTORS")
        print("="*60)
        from character_steering_toolkit import extract_all_vectors, save_steering_vectors
        results = extract_all_vectors(model, tokenizer, layers, dimensions, config)
        
        # Save
        save_steering_vectors(results, config.output_dir)
    
    # Optional: Permanently ablate AI assistant direction
    if args.ablate_ai:
        print("\n" + "="*60)
        print("  ⚠️  ABLATING AI ASSISTANT DIRECTION (PERMANENT)")
        print("="*60)
        from character_steering_toolkit import ablate_direction_from_weights
        
        for dim, vectors in results:
            if "suppress_ai" in dim.name and config.steer_layer in vectors:
                # We ablate the POSITIVE direction of the suppress dimension
                # (which IS the "AI assistant" direction)
                # Since the dimension was constructed as:
                #   positive = AI assistant lines, negative = Skippy lines
                # The vector points FROM Skippy TOWARD AI-speak
                # Ablating it removes the AI-speak direction entirely
                ablate_direction_from_weights(
                    model, layers, vectors[config.steer_layer], config.steer_layer
                )
                print("  Done. AI assistant direction removed from weights.")
                
                # Optionally save the ablated model
                save_path = os.path.join(config.output_dir, "ablated_model")
                print(f"  Saving ablated model to {save_path}...")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print("  Saved.")
                break
    
    # Interactive session
    if not args.no_interactive:
        print("\n" + "="*60)
        print("  LAUNCHING SKIPPY INTERACTIVE SESSION")
        print("="*60)
        skippy_interactive(model, tokenizer, layers, results, config)


if __name__ == "__main__":
    main()
