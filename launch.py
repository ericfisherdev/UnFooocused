"""UnFooocused — Application launcher.

Configures the runtime environment, sets ldm_patched flags for low VRAM
GPUs, and starts the FastAPI server via uvicorn.

Usage:
    python launch.py [--host HOST] [--port PORT]

Environment variables:
    STUB_MODE=true  — Skip GPU pipeline, generate solid-color placeholders.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# 1. Fix sys.path so imports resolve from the project root
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# 2. Parse CLI arguments (before any heavy imports)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="UnFooocused — SDXL image generation server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
parser.add_argument("--port", type=int, default=7866, help="Port number (default: 7866)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 3. Configure ldm_patched args BEFORE any ldm_patched import
#
#    model_management.py reads these at module scope to decide VRAM strategy.
#    Must happen before the first `import ldm_patched.modules.*`.
# ---------------------------------------------------------------------------

from ldm_patched.modules.args_parser import args as ldm_args  # noqa: E402

ldm_args.disable_async_cuda_allocation = True
ldm_args.always_offload_from_vram = True

# ---------------------------------------------------------------------------
# 4. Start the server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"[UnFooocused] Starting server on http://{args.host}:{args.port}")

    uvicorn.run(
        "ui.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
