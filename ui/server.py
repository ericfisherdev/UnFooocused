"""
UnFooocused UI - Server Launcher

Starts the FastAPI UI server in a daemon thread.
"""

import logging
import threading

import uvicorn

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7866


def start(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """
    Launch the UI server in a background daemon thread.

    Args:
        host: Bind address.
        port: Port number. Defaults to 7866.
    """
    thread = threading.Thread(
        target=_run_uvicorn,
        args=(host, port),
        daemon=True,
        name="ui-server",
    )
    thread.start()
    logger.info(f"UI server starting on http://{host}:{port}")


def _run_uvicorn(host: str, port: int) -> None:
    from ui.app import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
