#!/usr/bin/env bash
# =============================================================================
#  ARCA v3.0 — Robust Setup Script (Optimized for your machine)
#  Run:  chmod +x setup_v3.sh && ./setup_v3.sh
#
#  What it does:
#   1. Cleans previous failed installs
#   2. Installs correct PyTorch cu118
#   3. Installs PyTorch Geometric + extensions via official wheels
#   4. Installs llama-cpp-python (CPU version — stable & fast enough)
#   5. Installs ARCA in editable mode
#   6. Downloads recommended GGUF model (Llama-3.2-3B Q4)
#   7. Runs a quick smoke test
# =============================================================================

set -e

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  ARCA v3.0 — Setup (Dipayan's Machine)${RESET}"
echo -e "${BOLD}========================================${RESET}\n"

# ── Step 0: Clean old failed packages ───────────────────────────────────────
echo -e "${YELLOW}Cleaning previous installs...${RESET}"
pip uninstall -y torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster llama-cpp-python 2>/dev/null || true
rm -rf ~/.cache/pip

# ── Step 1: Install PyTorch cu118 ───────────────────────────────────────────
echo -e "\n${BOLD}[1/5] Installing PyTorch (cu118)...${RESET}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# ── Step 2: Install PyTorch Geometric + extensions ───────────────────────────
echo -e "\n${BOLD}[2/5] Installing PyTorch Geometric...${RESET}"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu118.html --no-cache-dir

pip install torch-geometric --no-cache-dir

# ── Step 3: Install llama-cpp-python (CPU version — stable) ─────────────────
echo -e "\n${BOLD}[3/5] Installing llama-cpp-python (CPU mode)...${RESET}"
CMAKE_ARGS="-DGGML_CUDA=OFF" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Optional: If you want GPU support later, uncomment the line below and re-run:
# CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir

# ── Step 4: Install ARCA editable ───────────────────────────────────────────
echo -e "\n${BOLD}[4/5] Installing ARCA v3 in editable mode...${RESET}"
pip install -e ".[all]" --no-cache-dir

# ── Step 5: Download recommended local model ────────────────────────────────
echo -e "\n${BOLD}[5/5] Setting up Local LLM model...${RESET}"
MODEL_DIR="${HOME}/.arca/models"
MODEL_FILE="${MODEL_DIR}/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
    echo -e "${GREEN}✓ Model already exists: ${MODEL_FILE}${RESET}"
else
    echo -e "${YELLOW}→ Downloading Llama-3.2-3B-Instruct-Q4 (~2GB)...${RESET}"
    wget --show-progress -O "$MODEL_FILE" "$MODEL_URL" || {
        echo -e "${RED}Download failed. You can download manually later.${RESET}"
    }
    echo -e "${GREEN}✓ Model downloaded${RESET}"
fi

# ── Smoke Test ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}Running smoke test (GNN + LocalLLM ready check)...${RESET}"

python - <<'EOF'
import torch
print(f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import torch_geometric
print("PyG: OK")

from arca.llm.local_llm import LocalLLM
llm = LocalLLM()
print(f"LocalLLM ready: {llm.available}")

from arca.core.config import ARCAConfig
from arca.core.agent import ARCAAgent
from arca.sim.environment import NetworkEnv

cfg = ARCAConfig.default()
cfg.rl.use_gnn = True
cfg.rl.gnn_hidden_dim = 64
cfg.rl.total_timesteps = 5000
cfg.verbose = 0
cfg.ensure_dirs()

env = NetworkEnv.from_preset("small_office", cfg=cfg)
agent = ARCAAgent(env=env, cfg=cfg)

print("Starting quick training (5000 steps)...")
agent.train(timesteps=5000, progress_bar=False)

info = agent.run_episode()
print(f"\nSmoke test PASSED ✓")
print(info.summary())
EOF

echo -e "\n${GREEN}${BOLD}========================================${RESET}"
echo -e "${GREEN}${BOLD}  ARCA v3.0 Setup Complete!${RESET}"
echo -e "${GREEN}${BOLD}========================================${RESET}\n"

echo -e "Next steps:"
echo -e "  ${BOLD}arca train --timesteps 50000${RESET}          # Full training with GNN"
echo -e "  ${BOLD}python examples/quickstart.py${RESET}         # Run demo"
echo -e "  Put your GGUF model in ~/.arca/models/ if not already there"
echo ""