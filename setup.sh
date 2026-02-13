#!/bin/bash
# =============================================================================
# SKIPPY THE MAGNIFICENT — One-Shot Setup
# =============================================================================
# Run: chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  🍺 SKIPPY THE MAGNIFICENT — SETUP                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found! Install Python 3.10+ first.${NC}"
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${GREEN}✓ Python ${PYVER}${NC}"

# Check CUDA
echo -e "${YELLOW}Checking CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "  ${GREEN}✓ GPU: ${GPU_NAME} (${GPU_MEM})${NC}"
else
    echo -e "  ${RED}✗ nvidia-smi not found — CUDA may not be installed${NC}"
    echo "  You need a CUDA-capable GPU for this toolkit."
fi

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "skippy_env" ]; then
    python3 -m venv skippy_env
    echo -e "  ${GREEN}✓ Created skippy_env/${NC}"
else
    echo -e "  ${GREEN}✓ skippy_env/ already exists${NC}"
fi

source skippy_env/bin/activate

# Install packages
echo ""
echo -e "${YELLOW}Installing Python packages...${NC}"

pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Core ML
echo "  Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1
echo -e "  ${GREEN}✓ PyTorch${NC}"

echo "  Installing transformers + accelerate..."
pip install transformers accelerate > /dev/null 2>&1
echo -e "  ${GREEN}✓ transformers, accelerate${NC}"

# Numerical
echo "  Installing numerical libs..."
pip install numpy scikit-learn > /dev/null 2>&1
echo -e "  ${GREEN}✓ numpy, scikit-learn${NC}"

# Ebook parsing
echo "  Installing ebook parsing..."
pip install ebooklib beautifulsoup4 lxml > /dev/null 2>&1
echo -e "  ${GREEN}✓ ebooklib, beautifulsoup4${NC}"

# Server
echo "  Installing web server..."
pip install fastapi uvicorn python-multipart > /dev/null 2>&1
echo -e "  ${GREEN}✓ FastAPI, uvicorn${NC}"

# Utilities
pip install tqdm > /dev/null 2>&1
echo -e "  ${GREEN}✓ tqdm${NC}"

# Optional: bitsandbytes for larger models
echo "  Installing bitsandbytes (optional, for quantization)..."
pip install bitsandbytes > /dev/null 2>&1 || echo -e "  ${YELLOW}⚠ bitsandbytes install failed (optional, only needed for >45B models)${NC}"

# Create directories
echo ""
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p books
mkdir -p skippy_vectors
mkdir -p extracted_text
echo -e "  ${GREEN}✓ books/ skippy_vectors/ extracted_text/${NC}"

# Validation
echo ""
echo -e "${YELLOW}Validating installation...${NC}"

python3 -c "
import torch
import transformers
import fastapi
import ebooklib
import sklearn

print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  VRAM:         {mem:.0f} GB')
print(f'  Transformers: {transformers.__version__}')
print(f'  FastAPI:      {fastapi.__version__}')
print(f'  sklearn:      {sklearn.__version__}')
print()
print('  ✅ All dependencies verified!')
"

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ SETUP COMPLETE                                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  Next steps:                                         ║"
echo "║                                                      ║"
echo "║  1. Put your .epub files in the books/ folder        ║"
echo "║                                                      ║"
echo "║  2. Extract vectors (CLI):                           ║"
echo "║     python skippy_pipeline.py --epub-dir ./books/    ║"
echo "║                                                      ║"
echo "║  3. Launch the dashboard:                            ║"
echo "║     python skippy_server.py                          ║"
echo "║     → Open http://localhost:8000                     ║"
echo "║                                                      ║"
echo "║  4. Or use CLI interactive mode:                     ║"
echo "║     python skippy_pipeline.py --load-vectors         ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Activate env with: source skippy_env/bin/activate${NC}"
