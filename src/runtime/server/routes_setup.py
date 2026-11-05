"""Mission setup API: drone presets and sim (re)launch.

Endpoints:
- GET  /api/setup/presets: drone-type presets + mission defaults for the
  setup page.
- POST /api/setup/launch:  (re)start the sim with the configured payload,
  wind, and drone type.
- POST /api/setup/reset:   restore SETUP_DEFAULTS (payload 0, wind 0,
  x2_stock) — full relaunch headless, live payload/wind apply when attached.

Launch behaviour depends on how the current sim (if any) was started:
- Headless (server-owned): full stop + relaunch with all parameters.
- Attached (started by the 3D viewer, e.g. view_dual_drone.py): payload and
  wind apply live, but a drone-type change would need a different scene, so
  it is rejected with 409.
"""

import math
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from runtime.server.state import sim_context
from runtime.sim.config import DUAL_SCENE_PATH

router = APIRouter(prefix="/api/setup", tags=["Mission Setup"])


def _wing_length_from_xml() -> Optional[float]:
    """Arm radius (centre -> rotor) in metres, read from the stock scene.

    Rotor geoms sit at (+/-0.14, +/-0.18) in dual_scene.xml, so the radius is
    sqrt(0.14^2 + 0.18^2) ~= 0.228 m.
    """
    try:
        root = ET.parse(DUAL_SCENE_PATH).getroot()
    except Exception:
        return None
    radii = []
    for geom in root.iter("geom"):
        name = geom.get("name") or ""
        if name.startswith("d1_rotor") and geom.get("pos"):
            x, y, _z = (float(v) for v in geom.get("pos").split())
            radii.append(math.hypot(x, y))
    return round(max(radii), 4) if radii else None


# The wing/arm value is the centre-to-rotor radius of the frame geometry.
# Only x2_stock has real geometry (dual_scene.xml). The long-arm and
# heavy-lift frames have no scene XML yet, so their values are estimates
# (rotor offsets 0.20/0.26 m -> 0.328 m, and 0.24/0.31 m -> 0.392 m) and a
# launch with those types still flies the stock scene until their XMLs exist.
DRONE_PRESETS: Dict[str, Dict[str, Any]] = {
    "x2_stock": {
        "id": "x2_stock",
        "label": "Skydio X2 (stock)",
        "wing_length_m": _wing_length_from_xml() or 0.228,
        "geometry_source": "xml",
        "defaults": {"payload_kg": 0.0, "wind_ms": 0.0},
    },
    "x2_long_arm": {
        "id": "x2_long_arm",
        "label": "Skydio X2 (long-arm)",
        "wing_length_m": 0.328,
        "geometry_source": "estimate",
        "defaults": {"payload_kg": 0.0, "wind_ms": 0.0},
    },
    "x2_heavy_lift": {
        "id": "x2_heavy_lift",
        "label": "Skydio X2 (heavy-lift)",
        "wing_length_m": 0.392,
        "geometry_source": "estimate",
        "defaults": {"payload_kg": 0.0, "wind_ms": 0.0},
    },
}

SETUP_DEFAULTS: Dict[str, Any] = {
    "payload_kg": 0.0,
    "wind_ms": 0.0,
    "drone_type": "x2_stock",
}


class LaunchReq(BaseModel):
    payload_kg: float = Field(default=0.0, ge=0.0, le=2.0, description="Payload mass in kg")
    wind_ms: float = Field(default=0.0, ge=0.0, le=20.0, description="Wind speed in m/s")
    drone_type: str = Field(default="x2_stock", description="Drone preset id")


def _wind_force_n(wind_ms: float) -> float:
    """Converts wind speed (m/s) to sustained force (N) via the simulator calibration."""
    from runtime.sim.dual_simulator import WIND_FORCE_PER_V2

    return WIND_FORCE_PER_V2 * float(wind_ms) ** 2


def _apply_live(sim: Any, req: LaunchReq) -> Dict[str, Any]:
    """Applies payload + wind to a running (attached) sim without restarting it."""
    wind_force_n = _wind_force_n(req.wind_ms)
    for payload_id_attr, mass_attr in (
        ("d1_payload_id", "d1_payload_mass"),
        ("d2_payload_id", "d2_payload_mass"),
    ):
        payload_id = getattr(sim, payload_id_attr, None)
        if payload_id is not None and payload_id >= 0:

            # 0.35 kg is the XML default bob mass; restoring it (not zero) keeps the
# chaos payload physics in its original working state.
            sim.model.body_mass[payload_id] = float(req.payload_kg) if float(req.payload_kg) > 0.0 else 0.35

            setattr(sim, mass_attr, float(req.payload_kg))
    chaos = getattr(sim, "chaos", None) or sim_context.chaos
    chaos.set_payload_attached(
        req.payload_kg > 0.0, mass=req.payload_kg, aerodynamic_drag=True
    )
    chaos.set_wind_force(fx=float(wind_force_n), fy=0.0, fz=0.0)
    sim.payload_kg = float(req.payload_kg)
    sim.wind_ms = float(req.wind_ms)
    sim.wind_force_n = float(wind_force_n)
    return {
        "payload_kg": req.payload_kg,
        "wind_ms": req.wind_ms,
        "wind_force_n": round(float(wind_force_n), 3),
        "drone_type": req.drone_type,
        "mode": "attached",
    }


def _launch_headless(req: LaunchReq, drone1_mode: str, drone2_mode: str) -> Dict[str, Any]:
    """Stops any headless sim and starts a fresh one with the mission parameters."""
    from runtime.sim.dual_simulator import DualDroneSimulator

    if sim_context._headless_running:
        sim_context.stop_headless()
    sim_context.sim = None
    # The constructor applies payload/wind, and attaches itself to sim_context.
    DualDroneSimulator(
        drone1_mode=drone1_mode,
        drone2_mode=drone2_mode,
        payload_kg=req.payload_kg,
        wind_ms=req.wind_ms,
        drone_preset=req.drone_type,
    )
    sim_context.start_headless(drone1_mode=drone1_mode, drone2_mode=drone2_mode)
    return {
        "payload_kg": req.payload_kg,
        "wind_ms": req.wind_ms,
        "wind_force_n": round(_wind_force_n(req.wind_ms), 3),
        "drone_type": req.drone_type,
        "mode": "headless",
    }


@router.get("/presets")
def get_presets() -> Dict[str, Any]:
    """Returns drone-type presets and overall mission defaults for the setup page."""
    return {
        "drone_types": list(DRONE_PRESETS.values()),
        "defaults": dict(SETUP_DEFAULTS),
    }


@router.post("/launch")
def launch(req: LaunchReq) -> Dict[str, Any]:
    """(Re)starts the sim with the configured mission, or applies it live."""
    if req.drone_type not in DRONE_PRESETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown drone_type '{req.drone_type}'. Valid: {sorted(DRONE_PRESETS)}",
        )

    with sim_context._lock:
        sim = sim_context.sim
        if sim is not None and not sim_context._headless_running:
            # Attached to the 3D viewer: geometry is owned by that process.
            current_preset = getattr(sim, "drone_preset", "x2_stock")
            if req.drone_type != current_preset:
                raise HTTPException(
                    status_code=409,
                    detail="Drone type change requires restarting the viewer",
                )
            applied = _apply_live(sim, req)
        else:
            drone1_mode = sim.drone1_mode if sim is not None else "pid"
            drone2_mode = (
                sim.drone2_mode if sim is not None else "l1_adaptive+system2"
            )
            applied = _launch_headless(req, drone1_mode, drone2_mode)

    return {"status": "launched", "applied": applied}


@router.post("/reset")
def reset() -> Dict[str, Any]:
    """Restores the sim to its original state (SETUP_DEFAULTS).

    Headless: full stop + fresh relaunch, so both drones return to spawn
    and hover. Attached (viewer-owned): payload/wind are applied live via
    _apply_live; 409 only if the running preset isn't x2_stock, since the
    preset can't change without a viewer restart.
    """
    req = LaunchReq(**SETUP_DEFAULTS)

    with sim_context._lock:
        sim = sim_context.sim
        if sim is not None and not sim_context._headless_running:
            current_preset = getattr(sim, "drone_preset", "x2_stock")
            if req.drone_type != current_preset:
                raise HTTPException(
                    status_code=409,
                    detail="Drone type change requires restarting the viewer",
                )
            applied = _apply_live(sim, req)
        else:
            drone1_mode = sim.drone1_mode if sim is not None else "pid"
            drone2_mode = (
                sim.drone2_mode if sim is not None else "l1_adaptive+system2"
            )
            applied = _launch_headless(req, drone1_mode, drone2_mode)

    return {"status": "reset", "applied": applied}
