"""FastAPI Server for Real-Time Chaos Injection and Telemetry Streaming.

Provides:
1. REST API for chaos manipulation, preset triggering, and bridge status.
2. 30-60 Hz WebSocket broadcast stream for live telemetry and HUD gauges.
3. Headless or attached execution modes.

Usage:
------
# Run server standalone with background headless simulation:
uv run python -m runtime.server.app --headless --port 8000

# Run server attached to 3D MuJoCo viewer (launched from view_dual_drone.py)
"""

import sys
import argparse
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from runtime.server.state import sim_context
from runtime.server.routes_chaos import router as chaos_router
from runtime.server.routes_telemetry import router as telemetry_router
from runtime.server.stream import telemetry_websocket_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan manager for clean startup and shutdown."""
    print("🚁 [SERVER] System 1 / System 2 Runtime API server starting...")
    yield
    print("🛑 [SERVER] Shutting down simulation server...")
    sim_context.stop_headless()


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="System-1 / System-2 Drone Flight Control & Chaos API",
        description="Real-time physical chaos injection, dual-drone telemetry, and LLM bridge monitoring.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Configure CORS for local web dashboards (Vite, Next.js, static HTML)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount REST routers
    app.include_router(chaos_router)
    app.include_router(telemetry_router)

    # Mount WebSocket telemetry stream
    app.add_api_websocket_route("/ws/telemetry", telemetry_websocket_endpoint)

    from pathlib import Path
    from fastapi.responses import FileResponse

    frontend_index = Path(__file__).resolve().parents[3] / "frontend" / "index.html"

    @app.get("/")
    def root():
        if frontend_index.exists():
            return FileResponse(str(frontend_index))
        return {"status": "online", "docs": "/docs"}

    @app.get("/api")
    def api_discovery():
        return {
            "name": "System-1 / System-2 Drone Runtime API",
            "version": "0.2.0",
            "endpoints": {
                "telemetry_ws": "/ws/telemetry",
                "chaos_state": "/api/chaos/state",
                "chaos_presets": "/api/chaos/presets",
                "system_status": "/api/status",
                "docs": "/docs",
            },
        }

    return app


app = create_app()


def main() -> None:
    """Parses CLI arguments and launches Uvicorn server."""
    parser = argparse.ArgumentParser(description="Launch System 1 / System 2 Flight Control Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--headless", action="store_true", help="Run 500 Hz MuJoCo simulation headlessly in background")
    parser.add_argument("--drone1", type=str, default="pid", help="Drone 1 mode (default: pid)")
    parser.add_argument("--drone2", type=str, default="l1_adaptive+system2", help="Drone 2 mode (default: l1_adaptive+system2)")
    args = parser.parse_args()

    if args.headless:
        sim_context.start_headless(drone1_mode=args.drone1, drone2_mode=args.drone2)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
