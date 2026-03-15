# LTX-2.3 FastAPI Server — Deployment Guide
### NVIDIA RTX Pro 6000 · 96 GB VRAM · Blackwell Architecture · Linux Server

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Server Preparation](#2-server-preparation)
3. [CUDA & Driver Verification](#3-cuda--driver-verification)
4. [Repository Setup](#4-repository-setup)
5. [Python Environment](#5-python-environment)
6. [Hugging Face Authentication](#6-hugging-face-authentication)
7. [Environment Configuration](#7-environment-configuration)
8. [Concurrency Tuning by Resolution](#8-concurrency-tuning-by-resolution)
9. [FP8-Cast — Built-in Quantization](#9-fp8-cast--built-in-quantization)
10. [Launching the Server](#10-launching-the-server)
11. [Startup Log Verification](#11-startup-log-verification)
12. [API Reference](#12-api-reference)
13. [Model Download Details](#13-model-download-details)
14. [VRAM Budget & Concurrency Math](#14-vram-budget--concurrency-math)
15. [OOM Prevention Strategy](#15-oom-prevention-strategy)
16. [Production Hardening](#16-production-hardening)
17. [Persistent Deployment with tmux](#17-persistent-deployment-with-tmux)
18. [Monitoring & Observability](#18-monitoring--observability)
19. [Troubleshooting](#19-troubleshooting)
20. [Quick Reference Cheatsheet](#20-quick-reference-cheatsheet)

---

## 1. System Requirements

### Hardware

| Component        | Minimum              | Recommended             |
|------------------|----------------------|-------------------------|
| GPU              | RTX Pro 6000 96GB    | RTX Pro 6000 96GB       |
| CPU              | 6 vCPU               | 12+ vCPU                |
| RAM              | 64 GB                | 128 GB                  |
| Disk (OS)        | 50 GB SSD            | 100 GB NVMe SSD         |
| Disk (Models)    | 100 GB free          | 200 GB NVMe SSD         |
| Network          | 1 Gbps               | 10 Gbps (for downloads) |

### Software

| Component        | Required Version     | Notes                               |
|------------------|----------------------|-------------------------------------|
| OS               | Ubuntu 22.04 / 24.04 | Other Debian-based distros may work |
| CUDA             | 12.8+                | Required for Blackwell architecture |
| NVIDIA Driver    | 570+                 | Blackwell full feature support      |
| Python           | 3.11+                | Managed via `uv`                    |
| uv               | Latest               | Fastest Python package manager      |

### Disk Space Breakdown

```
models/ltx-2.3-22b-distilled.safetensors          ~44 GB
models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors ~2 GB
models/gemma-3-12b-it-qat-q4_0-unquantized/        ~8 GB
outputs/ (generated videos)                        ~10 GB+
──────────────────────────────────────────────────────────
Total minimum free disk required                   ~100 GB
```

---

## 2. Server Preparation

### 2.1 Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install Essential Build Tools

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    htop \
    nvtop \
    tmux \
    build-essential \
    libssl-dev \
    libffi-dev
```

### 2.3 Verify Available Disk Space

```bash
df -h /
# Ensure at least 100 GB free before proceeding
```

### 2.4 Create a Dedicated Working Directory

```bash
mkdir -p /opt/ltx2
cd /opt/ltx2
```

> **Tip:** Using `/opt/ltx2` instead of your home directory keeps the server
> organized and survives user changes. Adjust to your preference.

---

## 3. CUDA & Driver Verification

Run all checks before touching the code. A bad driver or CUDA mismatch is the
most common cause of failures.

### 3.1 Verify NVIDIA Driver

```bash
nvidia-smi
```

Expected output (abbreviated):

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.xx   Driver Version: 570.xx   CUDA Version: 12.8            |
+-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA RTX Pro 6000  Off | 00000000:01:00.0 Off |                    0 |
| N/A   35C    P8    15W / 300W |      0MiB / 98304MiB |      0%      Default |
```

**Verify:**
- Driver Version ≥ 570
- CUDA Version ≥ 12.8
- GPU shows **NVIDIA RTX Pro 6000** or similar
- VRAM shows ~98304 MiB (96 GB)

### 3.2 Verify CUDA Toolkit

```bash
nvcc --version
# Should show: release 12.8 or higher
```

### 3.3 Verify FP8 Compute Capability

```bash
python3 -c "
import torch
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
fp8_ok = cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)
print(f'GPU: {name}')
print(f'Compute Capability: sm_{cap[0]}{cap[1]}')
print(f'FP8 Support: {\"YES\" if fp8_ok else \"NO - upgrade GPU\"}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

Expected output for RTX Pro 6000 Blackwell:

```
GPU: NVIDIA RTX Pro 6000
Compute Capability: sm_120   (or higher)
FP8 Support: YES
Total VRAM: 96.0 GB
```

> **If FP8 Support shows NO:** Your driver or GPU does not support fp8-cast.
> Do NOT proceed — contact your cloud provider to confirm the GPU model.

---

## 4. Repository Setup

### 4.1 Clone the Repository

```bash
cd /opt/ltx2
git clone https://github.com/buntercodes/LTX-2-Gradio-Implementation.git
cd LTX-2-Gradio-Implementation
```

### 4.2 Verify Repository Structure

```bash
ls -la
# You should see:
# api/          server.py     app.py
# packages/     pyproject.toml  uv.lock
# DEPLOYMENT_GUIDE_BLACKWELL.md
```

### 4.3 Verify API Folder

```bash
ls api/
# You should see:
# __init__.py  downloader.py  engine.py
# models.py    routes.py      task_manager.py
```

---

## 5. Python Environment

We use `uv` — it installs dependencies 10x faster than pip and ensures
exact version reproducibility via `uv.lock`.

### 5.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Verify
uv --version
# Expected: uv 0.x.x
```

### 5.2 Build the Virtual Environment

```bash
# Install all dependencies from the lockfile (exact versions)
uv sync --frozen

# This installs: torch, transformers, fastapi, uvicorn,
# ltx-core, ltx-pipelines, ltx-trainer, gradio, safetensors, etc.
# Takes 5–15 minutes depending on network speed.
```

### 5.3 Activate the Environment

```bash
source .venv/bin/activate

# Verify Python version
python --version
# Expected: Python 3.11.x or higher
```

### 5.4 Verify PyTorch + CUDA

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected:
```
PyTorch: 2.x.x+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA RTX Pro 6000
```

---

## 6. Hugging Face Authentication

The Gemma 3 model is **gated by Google** — you must accept its license
terms and provide a valid Hugging Face token before the server can
download it automatically.

### 6.1 Accept the Gemma 3 License

1. Go to: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Log in to your Hugging Face account
3. Click **"Agree and access repository"**
4. Wait for approval (usually instant)

### 6.2 Generate an Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., `ltx2-server`)
4. Select **"Read"** permission (sufficient for downloads)
5. Copy the token — it starts with `hf_`

### 6.3 Set the Token on Your Server

```bash
export HF_TOKEN="hf_your_token_here"

# To make it permanent across reboots:
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 6.4 Verify Token Works

```bash
python -c "
from huggingface_hub import whoami
user = whoami()
print(f'Logged in as: {user[\"name\"]}')
"
```

---

## 7. Environment Configuration

These environment variables **must** be set before starting the server.
Copy and paste the entire block into your terminal.

### 7.1 Core Configuration

```bash
# ── Memory & CUDA ──────────────────────────────────────────────────────────
# Prevents VRAM fragmentation on Blackwell — CRITICAL, do not remove
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Hugging Face ───────────────────────────────────────────────────────────
export HF_TOKEN="hf_your_token_here"

# ── Model Paths (auto-downloaded if not set) ───────────────────────────────
# Leave these unset on first launch — the server will auto-download to models/
# Set them after first download to skip re-verification on subsequent starts:
# export LTX23_DISTILLED_CHECKPOINT="models/ltx-2.3-22b-distilled.safetensors"
# export LTX23_SPATIAL_UPSAMPLER="models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
# export LTX23_GEMMA_ROOT="models/gemma-3-12b-it-qat-q4_0-unquantized"

# ── Quantization ───────────────────────────────────────────────────────────
# NOTE: This env var is IGNORED by the server.
# fp8-cast is hardcoded in engine.py and cannot be overridden.
# It is listed here only for documentation purposes.
# export LTX23_QUANTIZATION="fp8-cast"  # always active regardless

# ── Concurrency ────────────────────────────────────────────────────────────
# See Section 8 to choose the right value for your target resolution.
export MAX_CONCURRENT_GENS="4"    # Safe default for mixed 720p/1080p workloads
export MAX_CPU_TASKS="2"          # Guards 6 vCPU — do not increase above 3
```

### 7.2 Make Configuration Permanent

```bash
cat >> ~/.bashrc << 'EOF'

# ── LTX-2.3 FastAPI Server ─────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="hf_your_token_here"
export MAX_CONCURRENT_GENS="4"
export MAX_CPU_TASKS="2"
EOF

source ~/.bashrc
```

---

## 8. Concurrency Tuning by Resolution

This is the most critical configuration decision for preventing OOM errors.
The correct value of `MAX_CONCURRENT_GENS` depends entirely on what
resolution your users generate at.

### VRAM Budget

```
Total VRAM                          : 96.0 GB
Static model footprint (fp8-cast)   :
  Transformer (FP8, ~85% of 22B)    : ~18.7 GB
  Transformer (BF16, ~15% of 22B)   :  ~6.6 GB
  Spatial Upsampler (BF16)          :  ~2.0 GB
  Video VAE encoder/decoder (BF16)  :  ~3.0 GB
  Audio VAE + Vocoder (BF16)        :  ~1.0 GB
  Gemma 3 12B (INT4)                :  ~6.0 GB
─────────────────────────────────────────────────
Static Total                        : ~37.3 GB
Free for concurrent generation      : ~58.7 GB
```

### Per-Resolution Concurrency Table

| Resolution       | Per-Task Peak VRAM | Max Concurrent | Recommended Setting   |
|------------------|--------------------|----------------|-----------------------|
| 1920×1088 (1080p)| ~25–27 GB          | **2**          | `MAX_CONCURRENT_GENS=2` |
| 1280×768  (720p) | ~12–14 GB          | **4**          | `MAX_CONCURRENT_GENS=4` |
| 1024×768  (4:3)  | ~10–12 GB          | **4–5**        | `MAX_CONCURRENT_GENS=4` |
| 832×576   (Fast) | ~7–9 GB            | **6**          | `MAX_CONCURRENT_GENS=6` |
| Mixed workloads  | ~14 GB avg         | **4**          | `MAX_CONCURRENT_GENS=4` |

### How to Set It

```bash
# For 1080p-focused deployments:
export MAX_CONCURRENT_GENS="2"

# For 720p-focused deployments (recommended default):
export MAX_CONCURRENT_GENS="4"

# For fast/preview generation only:
export MAX_CONCURRENT_GENS="6"

# For mixed/unknown (safest default):
export MAX_CONCURRENT_GENS="4"
```

> **Important:** The deployment guide suggests `MAX_CONCURRENT_GENS=6` but
> that is only safe for Fast (576×832) resolution. For 720p or 1080p use
> the values above to avoid OOM errors under concurrent load.

---

## 9. FP8-Cast — Built-in Quantization

### What It Is

fp8-cast is **permanently hardcoded** into the server. It cannot be disabled
or overridden by environment variables or API requests. This is intentional.

### What It Does at the Code Level

**Step 1 — State dict loading** (`TRANSFORMER_LINEAR_DOWNCAST_MAP`):

When the transformer checkpoint is read from disk, all attention and feedforward
layer weights are immediately cast to `float8_e4m3fn` before being written to
VRAM. This covers per transformer block:

- `to_q.weight` / `to_q.bias`
- `to_k.weight` / `to_k.bias`
- `to_v.weight` / `to_v.bias`
- `to_out.0.weight` / `to_out.0.bias`
- `ff.net.0.proj.weight` / `ff.net.0.proj.bias`
- `ff.net.2.weight` / `ff.net.2.bias`

These layers represent ~85% of the 22B parameters. They are stored as FP8
(1 byte/param) in VRAM instead of BF16 (2 bytes/param).

**Step 2 — Forward pass patching** (`UPCAST_DURING_INFERENCE`):

Each `nn.Linear.forward()` is replaced with a version that upcasts FP8
weights to BF16 just before the matrix multiplication, then discards the
BF16 copy. The stored VRAM weights remain FP8 at all times.

### VRAM Savings

```
Without fp8-cast: Transformer = ~44 GB BF16
With fp8-cast:    Transformer = ~25 GB (FP8 attention/FF + BF16 other)
──────────────────────────────────────────────────────────────────────
VRAM saved:                     ~19 GB
```

This 19 GB saving is what enables 4 concurrent users instead of 1–2.

### Quality Impact

Negligible for video generation. The matrix multiplication still executes in
BF16 — only weight storage is compressed. Lightricks uses fp8-cast in their
own production deployment of this exact model.

---

## 10. Launching the Server

### 10.1 First Launch (with Auto Model Download)

On first launch the server will automatically download ~54 GB of models.
This takes 15–60 minutes depending on your network speed. Use `tmux` to
keep it running if your SSH connection drops.

```bash
# Start a tmux session first (recommended)
tmux new -s ltx2server

# Navigate to the repo
cd /opt/ltx2/LTX-2-Gradio-Implementation

# Activate environment
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="hf_your_token_here"
export MAX_CONCURRENT_GENS="4"
export MAX_CPU_TASKS="2"

# Launch
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

### 10.2 Subsequent Launches (Models Already Downloaded)

```bash
# Set model paths directly to skip re-verification
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="hf_your_token_here"
export MAX_CONCURRENT_GENS="4"
export MAX_CPU_TASKS="2"
export LTX23_DISTILLED_CHECKPOINT="models/ltx-2.3-22b-distilled.safetensors"
export LTX23_SPATIAL_UPSAMPLER="models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
export LTX23_GEMMA_ROOT="models/gemma-3-12b-it-qat-q4_0-unquantized"

uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

### 10.3 Production Launch (with Workers and Logging)

```bash
uv run uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --no-use-colors 2>&1 | tee /var/log/ltx2server.log
```

> **Important:** Always use `--workers 1`. The pipeline holds a global GPU
> state that is not safe to fork across multiple worker processes.

---

## 11. Startup Log Verification

After launching, verify the server started correctly by checking these
log sections in order.

### ✅ Expected Startup Sequence

```
======================================================================
  LTX-2.3 FASTAPI SERVER — STARTING UP
======================================================================
  MAX_CONCURRENT_GENS (GPU Queue) : 4
  MAX_CPU_TASKS (CPU Encoder/IO)  : 2
======================================================================
  QUANTIZATION POLICY
======================================================================
  Mode     : fp8-cast (built-in, always enforced)
  Scope    : Transformer weights cast to FP8 at inference time
  Benefit  : ~40% VRAM reduction vs BF16 → higher concurrency
  Override : NOT possible — hardcoded for Blackwell 96GB safety
======================================================================
  GPU INFO
  GPU              : NVIDIA RTX Pro 6000
  Compute Cap      : sm_120
  Total VRAM       : 96.0 GB
  FP8 Capable      : ✅ YES
======================================================================
```

Then after model download/verification:

```
======================================================================
  QUANTIZATION STATUS CHECK
======================================================================
  GPU Detected     : NVIDIA RTX Pro 6000
  Compute Capability: 12.0
  Total VRAM       : 96.0 GB
  FP8 Support      : ✅ YES (sm_120 >= sm_89)
  Quantization     : ✅ fp8-cast ACTIVE
  VRAM Benefit     : ~40% reduction vs BF16 on transformer
======================================================================
  PIPELINE LOAD COMPLETE — VRAM REPORT
======================================================================
  Quantization Applied : fp8-cast ✅
  VRAM Allocated       : ~25.xx GB
  VRAM Reserved        : ~30.xx GB
  VRAM Free            : ~65.xx GB / 96.0 GB total
  MAX_CONCURRENT_GENS  : 4
  MAX_CPU_TASKS        : 2
======================================================================
```

Finally:
```
INFO: Task queue workers started. Server is ready to accept requests.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### ❌ Warning Signs to Watch For

| Log Message | Meaning | Action |
|---|---|---|
| `FP8 Capable: ❌ NO` | Wrong GPU or driver | Check `nvidia-smi`, verify driver ≥ 570 |
| `CUDA not available` | PyTorch can't see GPU | Reinstall CUDA toolkit, rerun `uv sync` |
| `Failed to auto-load pipeline` | Model path issue | Check models/ directory exists |
| `GatedRepoError` | HF token missing/invalid | Re-set `HF_TOKEN`, re-accept Gemma terms |
| `VRAM Free: < 20 GB` after load | OOM risk at concurrency | Reduce `MAX_CONCURRENT_GENS` |

---

## 12. API Reference

Base URL: `http://your-server-ip:8000`

Interactive docs: `http://your-server-ip:8000/docs`

### Endpoints

#### `GET /` — Health Check

```bash
curl http://your-server-ip:8000/
```

Response:
```json
{
  "service": "LTX-2.3 Video Generation API",
  "quantization": "fp8-cast (built-in, always active)",
  "docs": "/docs",
  "status": "online"
}
```

#### `GET /api/v1/system/status` — Pipeline Status

```bash
curl http://your-server-ip:8000/api/v1/system/status
```

Response:
```json
{
  "pipeline_loaded": true,
  "message": "Pipeline is loaded and ready. Quantization: fp8-cast (built-in).",
  "active_tasks": 0,
  "quantization": "fp8-cast"
}
```

#### `POST /api/v1/system/load` — Load Pipeline Manually

Only needed if auto-load failed on startup.

```bash
curl -X POST http://your-server-ip:8000/api/v1/system/load \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "models/ltx-2.3-22b-distilled.safetensors",
    "upsampler_path": "models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    "gemma_path": "models/gemma-3-12b-it-qat-q4_0-unquantized"
  }'
```

> Note: The `quantization` field is accepted but **ignored** — fp8-cast
> is always applied server-side regardless of what you send.

#### `POST /api/v1/generate` — Submit Generation Task

```bash
curl -X POST http://your-server-ip:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic aerial shot of a futuristic city at golden hour. Glass skyscrapers reflect the warm sunset.",
    "seed": 42,
    "resolution_preset": "1280 × 768   (720p Landscape)",
    "num_frames": 121,
    "frame_rate": 24.0,
    "enhance_prompt": false
  }'
```

Response:
```json
{
  "task_id": "f3a2b1c0-...",
  "message": "Task successfully queued. Quantization: fp8-cast (built-in)."
}
```

**Available Resolution Presets:**

| Preset String | Resolution | Aspect |
|---|---|---|
| `1920 × 1088  (1080p Landscape)` | 1920×1088 | 16:9 |
| `1280 × 768   (720p Landscape)` | 1280×768 | 16:9 |
| `1536 × 1024  (3:2 Landscape)` | 1536×1024 | 3:2 |
| `1024 × 768   (4:3 Landscape)` | 1024×768 | 4:3 |
| `832 × 576    (Fast Landscape)` | 832×576 | Fast |
| `1088 × 1920  (1080p Portrait)` | 1088×1920 | 9:16 |
| `768 × 1280   (720p Portrait)` | 768×1280 | 9:16 |
| `1024 × 1024  (Square)` | 1024×1024 | 1:1 |

**Frame count rules:**
- Must satisfy: `num_frames = 8k + 1`
- Valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121 ...
- At 24fps: 121 frames ≈ 5 seconds, 241 frames ≈ 10 seconds

#### `GET /api/v1/tasks/{task_id}` — Poll Task Status

```bash
curl http://your-server-ip:8000/api/v1/tasks/f3a2b1c0-...
```

Response during generation:
```json
{
  "task_id": "f3a2b1c0-...",
  "status": "processing",
  "progress": 0.45,
  "message": "Encoding prompt & building latents… | fp8-cast active | 768x1280 @ 24.0fps",
  "video_url": null,
  "is_cancelled": false
}
```

Response on completion:
```json
{
  "task_id": "f3a2b1c0-...",
  "status": "completed",
  "progress": 1.0,
  "message": "Resolution: 1280x768 | Frames: 121 | FPS: 24.0 | ... | Peak VRAM: 13.42 GB | Free VRAM: 45.21 GB",
  "video_url": "/videos/ltx2.3_f3a2b1c0.mp4",
  "is_cancelled": false
}
```

**Status values:** `queued` → `processing` → `completed` / `failed` / `canceled`

#### `DELETE /api/v1/tasks/{task_id}` — Cancel Task

```bash
curl -X DELETE http://your-server-ip:8000/api/v1/tasks/f3a2b1c0-...
```

#### `GET /videos/{filename}` — Download Generated Video

```bash
curl -O http://your-server-ip:8000/videos/ltx2.3_f3a2b1c0.mp4
```

#### Image Conditioning (Optional)

Pass a conditioning image via base64:

```python
import base64, requests

with open("my_image.png", "rb") as f:
    b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

response = requests.post("http://your-server-ip:8000/api/v1/generate", json={
    "prompt": "The scene continues from this image...",
    "image_base64": b64,
    "image_strength": 0.85,
    "image_crf": 33,
    "resolution_preset": "1280 × 768   (720p Landscape)",
    "num_frames": 121,
    "seed": 42
})
```

---

## 13. Model Download Details

### What Gets Downloaded Automatically

| Model | Source | Size | Purpose |
|---|---|---|---|
| `ltx-2.3-22b-distilled.safetensors` | Lightricks/LTX-2.3 | ~44 GB | Main generation model |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | Lightricks/LTX-2.3 | ~2 GB | Stage 2 upsampler |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | google/gemma-3-12b-it-qat-q4_0-unquantized | ~8 GB | Text encoder |

### Download Location

All models download to `models/` relative to the repo root:

```
LTX-2-Gradio-Implementation/
└── models/
    ├── ltx-2.3-22b-distilled.safetensors
    ├── ltx-2.3-spatial-upscaler-x2-1.0.safetensors
    └── gemma-3-12b-it-qat-q4_0-unquantized/
        ├── config.json
        ├── tokenizer.json
        └── ...
```

### Manual Download (Alternative)

If auto-download fails, download manually using `huggingface-cli`:

```bash
# Authenticate
huggingface-cli login

# Download LTX-2.3 checkpoint
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled.safetensors \
    --local-dir models/

# Download spatial upsampler
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --local-dir models/

# Download Gemma 3 (gated — must accept terms first)
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir models/gemma-3-12b-it-qat-q4_0-unquantized/
```

### Verify Downloads

```bash
# Check file sizes
ls -lh models/
# ltx-2.3-22b-distilled.safetensors     → ~44 GB
# ltx-2.3-spatial-upscaler-x2-...       → ~2 GB
# gemma-3-12b-it-qat-q4_0-unquantized/  → ~8 GB directory

# Verify model integrity (optional)
python -c "
import safetensors
with safetensors.safe_open('models/ltx-2.3-22b-distilled.safetensors', framework='pt') as f:
    keys = list(f.keys())
    print(f'Transformer keys: {len(keys)}')
    print(f'First key: {keys[0]}')
"
```

---

## 14. VRAM Budget & Concurrency Math

### Static VRAM After Pipeline Load

After the pipeline loads, these components occupy VRAM permanently:

```
Component                           Dtype   VRAM
────────────────────────────────────────────────
Transformer attention/FF layers     FP8     ~18.7 GB  (85% of 22B params)
Transformer norms/embeddings        BF16     ~6.6 GB  (15% of 22B params)
Spatial Upsampler                   BF16     ~2.0 GB
Video VAE encoder + decoder         BF16     ~3.0 GB
Audio VAE + Vocoder                 BF16     ~1.0 GB
Gemma 3 12B text encoder            INT4     ~6.0 GB
────────────────────────────────────────────────
Static Total                                ~37.3 GB
Free VRAM for concurrent generation         ~58.7 GB
```

### Dynamic VRAM Per Active Generation Task

Each active generation temporarily allocates:

```
1920×1088 (1080p):  ~25–27 GB peak
1280×768  (720p):   ~12–14 GB peak
832×576   (Fast):   ~7–9 GB peak
```

### Safe Concurrent Users Formula

```
Max concurrent = floor(Free VRAM / Per-task peak VRAM)
              = floor(58.7 / per_task_peak)

1080p: floor(58.7 / 27) = 2 users
720p:  floor(58.7 / 14) = 4 users
Fast:  floor(58.7 / 9)  = 6 users
```

### Without fp8-cast (for comparison)

```
Transformer in BF16              : ~44 GB
Static total without fp8-cast    : ~57 GB
Free VRAM without fp8-cast       : ~39 GB

720p concurrent without fp8-cast : floor(39 / 14) = 2 users only
```

fp8-cast doubles your effective concurrency at 720p.

---

## 15. OOM Prevention Strategy

### Layer 1 — Correct MAX_CONCURRENT_GENS (Most Important)

Set `MAX_CONCURRENT_GENS` according to Section 8. This is the primary
OOM prevention mechanism. Never set it higher than the values in the
concurrency table.

### Layer 2 — PYTORCH_CUDA_ALLOC_CONF

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

This tells PyTorch's CUDA allocator to use expandable memory segments
rather than fixed blocks. Without this, Blackwell GPUs can OOM on
fragmentation even when total free VRAM is sufficient.

### Layer 3 — CPU_LOCK Semaphore

`MAX_CPU_TASKS=2` limits simultaneous Gemma text encoding to 2 tasks.
This prevents CPU/RAM thrashing and keeps encoding latency predictable.
Do not increase above 3 even if you have more CPU cores — the bottleneck
is memory bandwidth, not core count.

### Layer 4 — cleanup_memory() Between Stages

The pipeline calls `cleanup_memory()` and `torch.cuda.synchronize()`
between Stage 1 and Stage 2 denoising. This frees Stage 1 latents
before Stage 2 allocates its full-resolution buffers, preventing a
brief double-allocation spike.

### Layer 5 — Monitor Per-Task VRAM in Logs

Each completed task logs its peak VRAM usage and remaining free VRAM:

```
Peak VRAM: 13.42 GB | Free VRAM: 45.21 GB
```

If Free VRAM consistently drops below 10 GB, reduce `MAX_CONCURRENT_GENS`
by 1 and restart the server.

### Layer 6 — outputs/ Directory Management

Generated videos accumulate in `outputs/`. Each 1080p 5-second video is
~50–200 MB. Set up a cron job to clean old files:

```bash
# Delete videos older than 24 hours
crontab -e
# Add: 0 * * * * find /opt/ltx2/LTX-2-Gradio-Implementation/outputs/ -name "*.mp4" -mmin +1440 -delete
```

---

## 16. Production Hardening

### 16.1 Restrict CORS Origins

In `server.py`, change:
```python
allow_origins=["*"]   # ← development only
```
To your specific frontend domain:
```python
allow_origins=["https://your-frontend-domain.com"]
```

### 16.2 Add API Authentication

Add an API key middleware to `server.py`:

```python
from fastapi import Header, HTTPException

API_KEY = os.environ.get("LTX_API_KEY", "")

@app.middleware("http")
async def verify_api_key(request, call_next):
    if request.url.path.startswith("/api/"):
        key = request.headers.get("X-API-Key", "")
        if API_KEY and key != API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)
```

Then set:
```bash
export LTX_API_KEY="your-secret-key-here"
```

### 16.3 Set Up a Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;  # For base64 image uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;   # Long timeout for video generation
        proxy_send_timeout 600s;
    }
}
```

### 16.4 Set Up SSL (HTTPS)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 16.5 Firewall Configuration

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (Nginx)
sudo ufw allow 443/tcp   # HTTPS (Nginx)
sudo ufw deny 8000/tcp   # Block direct FastAPI access from outside
sudo ufw enable
```

---

## 17. Persistent Deployment with tmux

Using `tmux` keeps the server running after you disconnect your SSH session.

### 17.1 Start Server in tmux

```bash
# Create a new named session
tmux new -s ltx2server

# Inside tmux — set env and launch
cd /opt/ltx2/LTX-2-Gradio-Implementation
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="hf_your_token_here"
export MAX_CONCURRENT_GENS="4"
export MAX_CPU_TASKS="2"
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# Detach from tmux (server keeps running): Ctrl+B then D
```

### 17.2 Reattach to Running Server

```bash
tmux attach -t ltx2server
```

### 17.3 List All Sessions

```bash
tmux ls
```

### 17.4 Kill the Server

```bash
tmux kill-session -t ltx2server
```

### 17.5 Auto-restart with systemd (Advanced)

Create `/etc/systemd/system/ltx2server.service`:

```ini
[Unit]
Description=LTX-2.3 FastAPI Video Generation Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ltx2/LTX-2-Gradio-Implementation
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="HF_TOKEN=hf_your_token_here"
Environment="MAX_CONCURRENT_GENS=4"
Environment="MAX_CPU_TASKS=2"
ExecStart=/opt/ltx2/LTX-2-Gradio-Implementation/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ltx2server
sudo systemctl start ltx2server
sudo systemctl status ltx2server
```

---

## 18. Monitoring & Observability

### 18.1 Real-time GPU Monitoring

```bash
# In a separate terminal or tmux pane:
watch -n 1 nvidia-smi

# Or use nvtop for a more visual interface:
nvtop
```

Key metrics to watch:
- **Memory-Usage**: Should stay below 90 GB under load
- **GPU-Util**: Expected 90–100% during active generation
- **Temperature**: Should stay below 85°C

### 18.2 Watch Server Logs Live

```bash
# If running in tmux:
tmux attach -t ltx2server

# If logging to file:
tail -f /var/log/ltx2server.log | grep -E "VRAM|fp8|Task|ERROR"
```

### 18.3 Check Queue Depth

```bash
# Poll system status endpoint
watch -n 5 'curl -s http://localhost:8000/api/v1/system/status | python3 -m json.tool'
```

### 18.4 VRAM Usage Per Completed Task

After each completed task, look for this pattern in logs:

```
Peak VRAM: XX.XX GB | Free VRAM: XX.XX GB
```

Build a picture of your typical per-task VRAM consumption to verify
your `MAX_CONCURRENT_GENS` setting is correct.

### 18.5 Disk Usage Monitoring

```bash
# Check outputs directory size
du -sh outputs/

# Count accumulated videos
ls outputs/*.mp4 2>/dev/null | wc -l
```

---

## 19. Troubleshooting

### GPU Not Detected

```bash
# Check if driver is loaded
lsmod | grep nvidia

# Reinstall if needed
sudo apt install -y nvidia-driver-570
sudo reboot
```

### CUDA Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Steps:**
1. Reduce `MAX_CONCURRENT_GENS` by 1
2. Restart the server
3. Monitor Free VRAM in task completion logs
4. If OOM persists at `MAX_CONCURRENT_GENS=1`, check for memory leaks
   with `nvidia-smi` — VRAM should drop after each task completes

### Pipeline Not Loading

```
Pipeline is not loaded. Please configure model paths...
```

**Steps:**
1. Verify model files exist: `ls -lh models/`
2. Check disk space: `df -h`
3. Manually trigger load: `POST /api/v1/system/load`
4. Check logs for specific error message

### Gemma Download Fails (GatedRepoError)

```
GatedRepoError: Access to model google/gemma-3-12b-it-qat-q4_0-unquantized is restricted
```

**Steps:**
1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Click "Agree and access repository"
3. Regenerate your HF token at https://huggingface.co/settings/tokens
4. Re-export: `export HF_TOKEN="hf_new_token_here"`
5. Restart the server

### FP8 Not Supported Warning

```
FP8 Support: ⚠️ NO (sm_XX < sm_89)
```

**Steps:**
1. Run `nvidia-smi` — confirm you have RTX Pro 6000 or similar Blackwell/Ada GPU
2. Update NVIDIA driver to 570+: `sudo apt install nvidia-driver-570`
3. Reboot: `sudo reboot`
4. Recheck: `nvidia-smi` should show Driver ≥ 570, CUDA ≥ 12.8

### Tasks Stuck in "queued" State

```
status: "queued"   (never moves to "processing")
```

**Steps:**
1. Check if task queue workers started: look for
   `Task queue workers started` in startup logs
2. Check if pipeline is loaded: `GET /api/v1/system/status`
3. Check if a previous task is hanging: `nvidia-smi` GPU utilization
4. Restart server if workers appear frozen

### Port Already in Use

```
ERROR: [Errno 98] Address already in use
```

```bash
# Find and kill the existing process
sudo lsof -i :8000
kill -9 <PID>
```

### Slow Generation / High Latency

Expected generation times at 720p, 121 frames, 24fps:

| Phase | Duration |
|---|---|
| Prompt encoding (Gemma) | 5–15 seconds |
| Stage 1 denoising (8 steps, half-res) | 30–90 seconds |
| Stage 2 denoising (4 steps, full-res) | 20–60 seconds |
| VAE decode + encode to MP4 | 10–30 seconds |
| **Total** | **65–195 seconds** |

If generation takes significantly longer, check GPU utilization with `nvtop`.

---

## 20. Quick Reference Cheatsheet

### First-Time Setup (Copy-Paste Block)

```bash
# 1. System prep
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl tmux nvtop

# 2. Clone
cd /opt/ltx2
git clone https://github.com/buntercodes/LTX-2-Gradio-Implementation.git
cd LTX-2-Gradio-Implementation

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 4. Build environment
uv sync --frozen
source .venv/bin/activate

# 5. Set environment
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN="hf_your_token_here"
export MAX_CONCURRENT_GENS="4"
export MAX_CPU_TASKS="2"

# 6. Launch in tmux
tmux new -s ltx2server
uv run uvicorn server:app --host 0.0.0.0 --port 8000
# Ctrl+B then D to detach
```

### Daily Operations

```bash
# Reattach to server
tmux attach -t ltx2server

# Check server health
curl -s http://localhost:8000/ | python3 -m json.tool

# Check pipeline + queue status
curl -s http://localhost:8000/api/v1/system/status | python3 -m json.tool

# Watch GPU
watch -n 1 nvidia-smi

# Clean old videos
find outputs/ -name "*.mp4" -mmin +1440 -delete

# View logs
tail -f /var/log/ltx2server.log
```

### Environment Variables Summary

| Variable | Purpose | Default | Configurable? |
|---|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memory allocator | Not set | Must set to `expandable_segments:True` |
| `HF_TOKEN` | Hugging Face auth | Not set | Required |
| `MAX_CONCURRENT_GENS` | GPU queue depth | 4 | Yes — see Section 8 |
| `MAX_CPU_TASKS` | CPU encoder slots | 2 | Yes — max 3 |
| `LTX23_DISTILLED_CHECKPOINT` | Model path override | `models/...` | Optional |
| `LTX23_SPATIAL_UPSAMPLER` | Upsampler path override | `models/...` | Optional |
| `LTX23_GEMMA_ROOT` | Gemma path override | `models/...` | Optional |
| `LTX23_QUANTIZATION` | Quantization override | Ignored | **NOT configurable — always fp8-cast** |

---

*Guide version: 1.0 — RTX Pro 6000 96GB Blackwell · LTX-2.3 Distilled Pipeline*
