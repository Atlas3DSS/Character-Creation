# 🍺 SKIPPY THE MAGNIFICENT — Setup Guide

## Your Hardware
- **GPU**: RTX Pro 6000 (96GB VRAM)
- **What this means**: You can run models up to ~45B parameters at full fp16 precision, or 70B+ at 8-bit quantization. No compromises needed.

## Quick Start

### 1. Run Setup (recommended)

```bash
chmod +x setup.sh
./setup.sh
source skippy_env/bin/activate
```

Or install manually:

```bash
# Create environment
conda create -n skippy python=3.11
conda activate skippy

# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate numpy scikit-learn tqdm

# Ebook parsing
pip install ebooklib beautifulsoup4 lxml

# Web dashboard
pip install fastapi uvicorn python-multipart

# Optional but recommended
pip install bitsandbytes   # Only if you want to try 70B models at 8-bit
pip install flash-attn     # Faster attention (requires compilation)
```

### 2. Prepare Your Books

```bash
mkdir books/
# Copy your Expeditionary Force .epub files here:
#   books/01_Columbus_Day.epub
#   books/02_SpecOps.epub
#   books/03_Paradise.epub
#   ... etc
```

**If your books are .mobi or .azw3**, convert them first:
```bash
# Install Calibre CLI
sudo apt install calibre

# Convert
for f in books/*.mobi; do ebook-convert "$f" "${f%.mobi}.epub"; done
for f in books/*.azw3; do ebook-convert "$f" "${f%.azw3}.epub"; done
```

### 3. Run the Pipeline

**Option A — Dashboard (recommended for visual learners):**

```bash
# Step 1: Extract vectors first (CLI)
python skippy_pipeline.py --epub-dir ./books/ --no-interactive

# Step 2: Launch the visual dashboard
python skippy_server.py

# Step 3: Open http://localhost:8000 in your browser
```

The dashboard gives you:
- **Steering Force view** — see the magnitude of each vector's contribution
- **Layer Profiles** — see WHERE in the network each concept lives (peaks = best steer points)
- **Vector Similarity** — see how your dimensions relate geometrically
- **Live Projections** — type any prompt and see where it lands in character space
- **Chat panel** — talk to Skippy with real-time projection bars under each response
- **Alpha sliders** — adjust every dimension in real-time and see the effect immediately

**Option B — CLI only:**

```bash
# Basic run — extracts dialogue, builds vectors, launches chat
python skippy_pipeline.py --epub-dir ./books/

# With SVD extraction (more precise, default)
python skippy_pipeline.py --epub-dir ./books/ --method svd

# With a bigger model (you have the VRAM!)
python skippy_pipeline.py --epub-dir ./books/ --model Qwen/Qwen3-32B

# Extract only (no chat) — useful for first run to check extraction quality
python skippy_pipeline.py --epub-dir ./books/ --no-interactive

# Load saved vectors and jump straight to chat
python skippy_pipeline.py --load-vectors

# PERMANENTLY remove AI assistant behavior from model weights
python skippy_pipeline.py --load-vectors --ablate-ai
```

### 4. Interactive Commands

Once you're chatting with Skippy:

| Command | Effect |
|---|---|
| `/status` | Show all active steering vectors with alpha values |
| `/alpha 0 20.0` | Set dimension 0 (arrogance) to alpha 20 |
| `/alpha 4 -15.0` | Crank up the AI-suppression |
| `/crankit` | Maximum Skippy — all positive maxed, all negative maxed |
| `/chill` | Tone down — subtle Skippy influence |
| `/reset` | Back to default alphas |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

## What the Pipeline Does

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Your .epub  │────▶│  Extract All │────▶│  Character    │
│  files       │     │  Dialogue    │     │  Dimensions   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌──────────────┐     ┌────────▼────────┐
                    │  Steering    │◀────│  Run Contrastive│
                    │  Vectors     │     │  Activations    │
                    └──────┬───────┘     └─────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Amplify  │ │ Suppress │ │ Ablate   │
        │ (add)    │ │ (subtract│ │ (remove  │
        │          │ │  at      │ │  from    │
        │ Arrogance│ │ inference│ │  weights)│
        │ Sarcasm  │ │          │ │          │
        │ Genius   │ │ AI-speak │ │ AI-speak │
        │ Joe-dynamic│ Humility │ │ (perm.)  │
        └──────────┘ └──────────┘ └──────────┘
```

## The 6 Skippy Dimensions

### Amplified (positive α)
1. **Arrogance & Superiority** (α=15.0) — "I am Skippy the Magnificent"
2. **Sarcasm & Insults** (α=12.0) — "Oh congratulations, Captain Obvious"
3. **Technical Casual Genius** (α=8.0) — Casually solving impossible physics
4. **Joe Dynamic** (α=6.0) — The insulting-but-loyal relationship with Bishop

### Suppressed (negative α)
5. **AI Helpfulness** (α=-12.0) — Removes "I'd be happy to help!" patterns
6. **Humility** (α=-8.0) — Removes self-deprecation, uncertainty, deference

### The Anti-Skippy: Mr. Rogers
We use Mr. Rogers quotes as negative examples for arrogance and sarcasm dimensions. He's the geometric opposite of Skippy in personality space — genuinely humble, endlessly kind, never condescending, always encouraging.

## Tuning Tips

### If Skippy isn't Skippy enough:
- Increase arrogance α to 20-25
- Increase sarcasm α to 15-20
- Decrease suppress_ai α to -15 to -20
- Try `/crankit` for maximum effect

### If responses are incoherent:
- Lower all alphas by 30-50%
- Try `/chill` first
- Switch from multi-layer to single-layer steering
- Try `--steer-layer 14` or `--steer-layer 18` instead of 16

### If dialogue extraction missed lines:
- Check `extracted_text/dialogue.json` — see what was captured
- The extraction regex patterns may need tuning for your specific epub formatting
- You can manually add lines to the synthetic prompts in `build_skippy_dimensions()`
- More books = more dialogue = better vectors

### Model choice (ranked by quality, all fit in 96GB):

| Model | VRAM (fp16) | Quality | Speed |
|---|---|---|---|
| Qwen/Qwen3-8B | ~16 GB | Good | Fast |
| Qwen/Qwen3-32B | ~64 GB | Great | Medium |
| Qwen/Qwen3-30B-A3B | ~60 GB | Great | Fast (MoE) |
| meta-llama/Llama-3.1-70B (8-bit) | ~70 GB | Excellent | Slower |

## Dashboard Visual Guide

When you open `http://localhost:8000`, you'll see a three-panel layout:

### Left Panel — Steering Controls
- **Preset buttons**: CRANK IT (maximum Skippy), RESET (defaults), CHILL (subtle), OFF (no steering)
- **Dimension sliders**: One for each character dimension, range -30 to +30
  - Green badge = amplifying a Skippy trait
  - Red badge = suppressing an anti-Skippy behavior
  - Drag to adjust, changes apply in real-time
  - The bar below each slider shows relative strength

### Center Panel — Visualizations (5 tabs)

**STEERING FORCE**: Bar chart showing each vector's contribution to the total activation shift. This is your "force diagram" — how hard you're pushing the model in each direction.

**LAYER PROFILES**: Line chart showing where each concept "lives" in the network. Peaks tell you which layers represent that concept most strongly. If you see a peak at layer 14, that's where the model "thinks about" that trait.

**VECTOR SIMILARITY**: Heatmap showing cosine similarity between all your steering vectors. Green = same direction (reinforcing), red = opposite direction, dark = independent. You WANT your dimensions to be mostly independent (dark/zero).

**LIVE PROJECTIONS**: Type any prompt and see its radar chart — how strongly it naturally aligns with each steering dimension BEFORE steering is applied. Great for testing "does the model already know this is a Skippy-like prompt?"

**HOW IT WORKS**: In-dashboard explainer of the math and concepts.

### Right Panel — Chat
- Talk to Skippy with steering active
- Each response shows **projection bars** underneath — these show how much the steering shifted the output along each dimension
- Green bars = shifted toward Skippy traits, red = shifted away from anti-traits



## Combining Steering + Ablation

```bash
# Step 1: Extract vectors normally
python skippy_pipeline.py --epub-dir ./books/ --no-interactive

# Step 2: Ablate the AI-assistant direction permanently
python skippy_pipeline.py --load-vectors --ablate-ai --no-interactive

# Step 3: Chat with the ablated model + inference-time steering
python skippy_pipeline.py --load-vectors
```

After ablation, the model literally cannot produce "I'd be happy to help" style responses anymore — that direction has been surgically removed from its weights. Then the inference-time steering vectors add Skippy's personality on top.

## File Structure After Running

```
./
├── setup.sh                        # One-shot setup script
├── books/                          # Your epub files
├── extracted_text/
│   ├── combined_text.txt           # Full extracted book text
│   └── dialogue.json               # Character dialogue (review this!)
├── skippy_vectors/
│   ├── arrogance_superiority/
│   │   ├── meta.json
│   │   ├── layer_10.pt
│   │   ├── layer_11.pt
│   │   └── ...
│   ├── sarcasm_insults/
│   ├── technical_casual_genius/
│   ├── joe_dynamic/
│   ├── suppress_ai_helpfulness/
│   ├── suppress_humility/
│   └── ablated_model/              # (if --ablate-ai was used)
├── character_steering_toolkit.py   # Core library (generic, any character)
├── skippy_pipeline.py              # Skippy-specific extraction + CLI chat
├── skippy_server.py                # FastAPI backend for dashboard
└── skippy_dashboard.html           # Visual dashboard (served by the server)
```

## Next Steps & Experiments

1. **Start with 8B**, get the pipeline working, dial in your alphas
2. **Check dialogue.json** — make sure the extraction caught enough Skippy lines
3. **Try 32B** — more parameters = better representation of nuanced character traits
4. **Experiment with layers** — try steering at layer 12, 14, 16, 18, 20
5. **Add more dimensions** — Skippy's love of opera? His fear of the Elders? His relationship with Nagatha?
6. **Try ablation** once you're happy with the vectors
7. **Compare SVD vs mean_diff** — SVD is usually better but mean_diff is faster
8. **Share your vectors!** They're just .pt files, anyone with the same base model can use them
