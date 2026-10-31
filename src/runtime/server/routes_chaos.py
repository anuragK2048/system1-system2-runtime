"""REST API endpoints for real-time chaos injection and failure mode control.

Enables the Web Dashboard to manipulate physical disturbances on both drones in real time:
- /api/chaos/motor: Multi-rotor thrust scaling, chipped propeller harmonics, ESC dropout
- /api/chaos/wind: Continuous 3D wind vectors, gust pulses, Dryden atmospheric turbulence
- /api/chaos/payload: Tethered cargo coupling, mass variation, aerodynamic wind drag
- /api/chaos/sensor: IMU gyro chatter, accelerometer noise, bias drift
- /api/chaos/preset: One-click repeatable failure presets
- /api/chaos/target: Select 'both', 'drone1', or 'drone2'
- /api/chaos/reset: Clean all anomalies & restore level hover
- /api/chaos/state: Current active chaos state on both drones
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from runtime.chaos.models import ChaosTarget
from runtime.chaos.presets import CHAOS_PRESETS
from runtime.server.state import sim_context

router = APIRouter(prefix="/api/chaos", tags=["Chaos Control"])


# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================
class TargetReq(BaseModel):
    mode: ChaosTarget = Field(default=ChaosTarget.BOTH, description="Target: 'both', 'drone1', or 'drone2'")


class MotorReq(BaseModel):
    motor_idx: Optional[int] = Field(default=None, ge=0, le=3, description="Index of single motor (0..3)")
    multiplier: Optional[float] = Field(default=None, ge=0.0, le=1.5, description="Thrust scale multiplier (1.0=100%, 0.4=40%)")
    multipliers: Optional[List[float]] = Field(default=None, description="List of 4 thrust scale multipliers [m1, m2, m3, m4]")
    chipped_motor_idx: Optional[int] = Field(default=None, description="Motor index with chipped prop flutter (None=none)")
    target: Optional[ChaosTarget] = None


class WindReq(BaseModel):
    fx: float = Field(default=0.0, description="Steady force in X (Newtons)")
    fy: float = Field(default=0.0, description="Steady force in Y (Newtons)")
    fz: float = Field(default=0.0, description="Steady force in Z (Newtons)")
    turbulence_std: float = Field(default=0.0, ge=0.0, description="Dryden turbulence standard deviation (N)")
    target: Optional[ChaosTarget] = None


class GustReq(BaseModel):
    force: float = Field(default=5.0, ge=0.0, description="Peak gust force in Newtons")
    direction: List[float] = Field(default=[1.0, 0.0, 0.0], description="Gust 3D unit direction vector")
    duration_s: float = Field(default=0.8, gt=0.0, description="Gust duration in seconds")
    target: Optional[ChaosTarget] = None


class PayloadReq(BaseModel):
    attached: bool = Field(default=True, description="True to attach tethered cargo")
    mass: Optional[float] = Field(default=0.35, ge=0.05, le=1.5, description="Payload mass in kg")
    aerodynamic_drag: bool = Field(default=True, description="Wind aerodynamic drag directly swings cargo bob")
    target: Optional[ChaosTarget] = None


class SensorReq(BaseModel):
    gyro_noise_std: float = Field(default=0.0, ge=0.0, description="Gaussian gyro rate noise (rad/s std)")
    accel_noise_std: float = Field(default=0.0, ge=0.0, description="Accelerometer noise (m/s^2 std)")
    target: Optional[ChaosTarget] = None


class PresetReq(BaseModel):
    preset_id: str = Field(..., description="Preset ID ('1'..'5' or '0' for clean hover)")
    toggle: bool = Field(default=False, description="If True, toggles the preset on/off additively")
    additive: bool = Field(default=True, description="Whether to additively combine with active chaos")
    target: Optional[ChaosTarget] = None


# ==============================================================================
# ENDPOINTS
# ==============================================================================
@router.get("/state")
def get_chaos_state() -> Dict[str, Any]:
    """Returns the full JSON serialization of current chaos on both drones."""
    return sim_context.chaos.get_chaos_summary()


@router.get("/presets")
def get_presets_list() -> List[Dict[str, Any]]:
    """Returns the catalog of all available failure presets with metadata."""
    result = []
    for k, p in CHAOS_PRESETS.items():
        result.append({
            "key": k,
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "motor_active": p["motor"].is_active(),
            "wind_active": p["wind"].is_active(),
            "payload_active": p["payload"].is_active(),
            "sensor_active": p["sensor"].is_active(),
        })
    return result


@router.post("/target")
def set_target_mode(req: TargetReq) -> Dict[str, Any]:
    """Sets whether chaos controls apply to 'both', 'drone1', or 'drone2'."""
    sim_context.chaos.set_target_mode(req.mode)
    return {"status": "ok", "target_mode": req.mode.value}


@router.post("/motor")
def update_motor(req: MotorReq) -> Dict[str, Any]:
    """Adjusts motor thrust scaling or chipped propeller faults."""
    chaos = sim_context.chaos
    if req.multipliers is not None:
        chaos.set_motor_multipliers(req.multipliers, target=req.target)
    elif req.motor_idx is not None and req.multiplier is not None:
        chaos.set_motor_multiplier(req.motor_idx, req.multiplier, target=req.target)

    if req.chipped_motor_idx is not None:
        chaos.set_chipped_propeller(req.chipped_motor_idx, target=req.target)

    return {"status": "ok", "chaos": chaos.get_chaos_summary()}


@router.post("/wind")
def update_wind(req: WindReq) -> Dict[str, Any]:
    """Sets continuous 3D wind disturbance force vector and turbulence."""
    sim_context.chaos.set_wind_force(
        fx=req.fx,
        fy=req.fy,
        fz=req.fz,
        turbulence_std=req.turbulence_std,
        target=req.target,
    )
    return {"status": "ok", "chaos": sim_context.chaos.get_chaos_summary()}


@router.post("/gust")
def trigger_gust(req: GustReq) -> Dict[str, Any]:
    """Triggers an instantaneous half-sine crosswind gust."""
    sim_time = 0.0
    if sim_context.sim and sim_context.sim.data:
        sim_time = float(sim_context.sim.data.time)

    sim_context.chaos.trigger_wind_gust(
        force=req.force,
        direction=req.direction,
        duration_s=req.duration_s,
        sim_time=sim_time,
        target=req.target,
    )
    return {"status": "ok", "message": f"Gust triggered ({req.force:.1f}N)"}


@router.post("/payload")
def update_payload(req: PayloadReq) -> Dict[str, Any]:
    """Attaches or detaches the tethered cargo pendulum under the drone."""
    sim_context.chaos.set_payload_attached(
        attached=req.attached,
        mass=req.mass,
        aerodynamic_drag=req.aerodynamic_drag,
        target=req.target,
    )
    return {"status": "ok", "chaos": sim_context.chaos.get_chaos_summary()}


@router.post("/sensor")
def update_sensor_noise(req: SensorReq) -> Dict[str, Any]:
    """Sets IMU gyroscope vibration chatter and accelerometer noise."""
    sim_context.chaos.set_sensor_noise(
        gyro_std=req.gyro_noise_std,
        accel_std=req.accel_noise_std,
        target=req.target,
    )
    return {"status": "ok", "chaos": sim_context.chaos.get_chaos_summary()}


@router.post("/preset")
def apply_preset(req: PresetReq) -> Dict[str, Any]:
    """Applies or toggles a named failure preset on the target drone(s)."""
    if req.preset_id in ("0", "clean"):
        sim_context.chaos.clear_all(target=req.target)
        msg = f"Clean nominal hover restored on {(req.target or sim_context.chaos.target_mode).value}"
    elif req.toggle:
        res = sim_context.chaos.toggle_preset(req.preset_id, target=req.target)
        msg = f"Preset '{req.preset_id}' {res.get('action', 'updated')} on {(req.target or sim_context.chaos.target_mode).value}"
    else:
        msg = sim_context.chaos.apply_preset(req.preset_id, additive=req.additive, target=req.target)
    return {"status": "ok", "message": msg, "chaos": sim_context.chaos.get_chaos_summary()}


@router.post("/preset/toggle")
def toggle_preset_endpoint(req: PresetReq) -> Dict[str, Any]:
    """Explicit endpoint to toggle a named failure preset on or off additively."""
    res = sim_context.chaos.toggle_preset(req.preset_id, target=req.target)
    return {
        "status": "ok",
        "action": res.get("action"),
        "preset_id": req.preset_id,
        "active_presets": res.get("active_presets"),
        "chaos": sim_context.chaos.get_chaos_summary(),
    }


@router.post("/clear")
def clear_all_chaos() -> Dict[str, Any]:
    """Clears all active failure modes back to pristine nominal flight."""
    sim_context.chaos.clear_all()
    return {"status": "ok", "message": "All chaos cleared", "chaos": sim_context.chaos.get_chaos_summary()}


@router.post("/reset_drones")
def reset_drones() -> Dict[str, Any]:
    """Clears all chaos and resets drone physics back to initial level hover."""
    sim_context.chaos.clear_all()
    sim_context.chaos.reset_requested = True
    if sim_context.sim:
        sim_context.sim.reset_simulation()
    return {"status": "ok", "message": "Drones reset to level hover"}
