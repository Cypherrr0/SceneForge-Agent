# 3D Asset Generation and Scene Construction Agent

The **3D Asset Generation and Scene Construction Agent** combines large language models, procedural tooling, and a web UI to turn natural-language prompts into textured 3D assets and multi-object scenes. The system orchestrates end-to-end workflows—from intent parsing and planning to text-to-3D generation, iterative evaluation, Blender-based rendering, and delivery of downloadable artifacts—so that artists, technical directors, and researchers can prototype content without touching DCC tools directly.

## Highlights

- **LLM-driven orchestration** – Qwen2.5-72B handles reasoning, tool calling, and workflow control; Qwen/Gemini variants translate prompts, rate outputs, and generate Blender scripts.
- **Tool ecosystem** – Text23D pipelines, Gemini-powered image synthesis, evaluation agents, and a codex-style Blender script generator are registered as callable tools via `Hunyuan3DAgentV2`.
- **Production-ready backend** – FastAPI exposes REST + WebSocket APIs, session lifecycle management, and threaded task execution with resumable state and rich callbacks.
- **Immersive frontend** – Next.js 14 + React Query + Zustand provide a guided UX with real-time progress, multi-step logs, image previews, and an interactive `<model-viewer>` / Three.js viewer.
- **Auditable artifacts** – Every run writes intermediate scripts, meshes, textures, and renderings to `outputs/` and retains planning metadata under `sessions/`.

## Prerequisites

- Python 3.10+ with `pip`
- Node.js 18+ and npm
- GPU drivers / Blender installation that matches the tooling referenced in `config/`
- Valid API keys for SiliconFlow Qwen and Google Gemini services
- CUDA 12.4 environment recommended for Hunyuan3D-2.1 (PyTorch wheels below use `cu124`)

## Configuration

1. Duplicate `config/config.json` and replace placeholders with your own keys and run-time defaults:

   ```jsonc
   {
     "qwen_api_key": "sk-***",
     "gemini_api_key": "sk-***",
     "uid": "my_asset_run",
     "input_source": "describe your asset here",
     "save_dir": "./text_to_3d_agent/text23d_save_dir",
     "frontend_save_dir": "./outputs",
     "txt2img_model_name": "gemini-2.5-flash-image-preview",
     "translate_model_name": "Qwen/Qwen3-Next-80B-A3B-Instruct",
     "evaluate_model_name": "Qwen/Qwen2-VL-72B-Instruct"
   }
   ```

   The backend automatically reads this file at startup and propagates the keys into `Hunyuan3DAgentV2`. Related knobs for Text23D, Blender, and tool registration live in the adjacent `blender_config.json`, `text23d_config.json`, and `tools_config.json`.

2. (Optional) Copy `backend/.env.example` to `.env` for overriding FastAPI settings.

3. In `frontend/.env.local`, point the UI to your backend instance:

   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1
   NEXT_PUBLIC_WS_URL=ws://localhost:8080/api/v1
   ```

## Installation

Dependency notes:
- `backend/requirements.txt` is the backend service dependency list.
- `text_to_3d_agent/Hunyuan3D-2.1/requirements.txt` is required for 3D generation.
- Root `requirements.txt` is a historical conda export (Linux snapshot) and is **not** the recommended install entry for contributors.

```bash
# from project root
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# backend service deps
pip install -r backend/requirements.txt
```

### Required setup for `text_to_3d_agent/Hunyuan3D-2.1`

`Text23dPipeline.py` imports modules from `text_to_3d_agent/Hunyuan3D-2.1/hy3dshape` and `hy3dpaint` directly.  
If this repository is not cloned and compiled, backend startup or first generation call will fail.

```bash
# from project root
# if a placeholder folder already exists, rename it first (safe, non-destructive)
if [ -d text_to_3d_agent/Hunyuan3D-2.1 ] && [ -n "$(ls -A text_to_3d_agent/Hunyuan3D-2.1 2>/dev/null)" ]; then
  mv text_to_3d_agent/Hunyuan3D-2.1 "text_to_3d_agent/Hunyuan3D-2.1.bak.$(date +%s)"
fi

git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git text_to_3d_agent/Hunyuan3D-2.1

# install Hunyuan3D runtime deps
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r text_to_3d_agent/Hunyuan3D-2.1/requirements.txt

# build rasterizer
cd text_to_3d_agent/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
pip install -e .
cd ../..

# build differentiable renderer
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

# download Real-ESRGAN checkpoint required by texture pipeline
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
cd ../..
```

### Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Preflight checks (recommended)

```bash
# from project root
test -f text_to_3d_agent/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth && echo "ckpt OK"
test -d text_to_3d_agent/Hunyuan3D-2.1/hy3dshape && echo "hy3dshape OK"
test -d text_to_3d_agent/Hunyuan3D-2.1/hy3dpaint && echo "hy3dpaint OK"
```

## Running the Stack

### Backend

For a foreground development server:

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8080 --reload
```

For the production-style process requested by the project owner:

```bash
source .venv/bin/activate
nohup python -u -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8080 > backend.log 2>&1 &
```

The backend exposes REST endpoints under `/api/v1`, WebSocket updates at `/api/v1/ws/{session_id}`, and static artifacts under `/outputs`.

### Frontend

For local development:

```bash
cd frontend
npm run dev
cd ..
```

To mirror the provided deployment command and detach the process:

```bash
cd frontend
nohup npm run dev > frontend.log 2>&1 &
cd ..
```

The UI runs on port 3000 by default (customize via `next dev -p <port>`).

## End-to-End Workflow

1. **Session creation** – The frontend calls `/api/v1/sessions/create`, storing prompts, parameters, and whether the run should be interactive.
2. **LLM intent parsing** – `Hunyuan3DAgentV2` identifies assets, scene elements, and constraints using Qwen-based tools.
3. **Task planning & tool routing** – Planning and workflow tools decompose the brief into text-to-3D jobs, reference generation, and Blender scripting steps.
4. **Asset generation** – Text23D pipelines and Gemini-powered `txt2img` models synthesize guidance images, meshes, and textures; logs are streamed over WebSocket.
5. **Evaluation loop** – Quality is scored via the evaluation agent; the pipeline iterates until reaching `score_threshold` or `max_iterations`.
6. **Rendering & packaging** – Blender scripts render the final scene, results land in `outputs/<session_id>/` (glTF/FBX, textures, renders, logs, metadata).
7. **Delivery** – Frontend downloads the final package and shows it in the gallery + `<model-viewer>` panel, while REST endpoints expose `/result/{session_id}` for automation.


Contributions and experiment branches are welcome—open an issue or PR describing the scenario you want to enable.
