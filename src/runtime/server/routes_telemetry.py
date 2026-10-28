"""REST API endpoints for real-time telemetry, bridge monitoring, and system status."""

from typing import Dict, List, Any
from fastapi import APIRouter
from runtime.server.state import sim_context

router = APIRouter(prefix="/api", tags=["Telemetry & Status"])


from runtime.sim.config import get_system_naming, get_drone_orientation


@router.get("/status")
def get_system_status() -> Dict[str, Any]:
    """Returns general flight runtime, canonical system naming, and simulation health information."""
    sim = sim_context.sim
    d1_m = sim.drone1_mode if sim else "pid"
    d2_m = sim.drone2_mode if sim else "l1_adaptive+system2"
    d1_sys = get_system_naming(d1_m)
    d2_sys = get_system_naming(d2_m)
    d1_orient = get_drone_orientation(1)
    d2_orient = get_drone_orientation(2)

    return {
        "status": "online",
        "has_simulator": sim is not None,
        "headless_running": sim_context._headless_running,
        "drone1_mode": d1_m,
        "drone2_mode": d2_m,
        "drone1_system": d1_sys,
        "drone2_system": d2_sys,
        "left_drone": {
            "id": "drone2",
            "side": "Left",
            "color": "Cyan",
            "system_name": d2_sys["system_name"],
            "system_short": d2_sys["system_short"],
            "orientation": d2_orient,
        },
        "right_drone": {
            "id": "drone1",
            "side": "Right",
            "color": "Orange",
            "system_name": d1_sys["system_name"],
            "system_short": d1_sys["system_short"],
            "orientation": d1_orient,
        },
        "physics_timestep": sim.sim_cfg.physics_timestep if sim else 0.002,
        "target_mode": sim_context.chaos.target_mode.value,
        "drone1_active_chaos": sim_context.chaos.drone1.is_any_active(),
        "drone2_active_chaos": sim_context.chaos.drone2.is_any_active(),
    }


@router.get("/telemetry/latest")
def get_latest_telemetry() -> Dict[str, Any]:
    """Returns the most recent 500 Hz telemetry frame formatted for the UI."""
    return sim_context.latest_frame or {
        "timestamp": 0.0,
        "drone1": None,
        "drone2": None,
        "chaos": sim_context.chaos.get_chaos_summary(),
    }


@router.get("/telemetry/history")
def get_telemetry_history() -> List[Dict[str, Any]]:
    """Returns a rolling ring buffer of recent telemetry frames for live charts."""
    return list(sim_context.telemetry_history)


@router.get("/bridge/status")
def get_bridge_status() -> Dict[str, Any]:
    """Returns System 1 / System 2 bridge metrics, active control patches, and trigger events."""
    sim = sim_context.sim
    if not sim:
        s2 = sim_context._get_latest_llm_log_status()
        return {"drone1": None, "drone2": None, "system2": s2}

    # Drone 1
    p1 = sim.bridge_1.get_active_patch()
    p1_age = sim.bridge_1.get_patch_age_s()
    latest_trig1 = sim.bridge_1.get_latest_trigger_event(max_age_s=10.0)

    # Drone 2
    p2 = sim.bridge_2.get_active_patch()
    p2_age = sim.bridge_2.get_patch_age_s()
    latest_trig2 = sim.bridge_2.get_latest_trigger_event(max_age_s=10.0)

    # System 2 Worker status
    system2_status = None
    w = sim.worker_2 or sim.worker_1
    if w and hasattr(w, "get_status_dict"):
        system2_status = w.get_status_dict()
    if system2_status is None:
        system2_status = sim_context._get_latest_llm_log_status()

    return {
        "drone1": {
            "mode": sim.drone1_mode,
            "is_triggered": sim.bridge_1.is_triggered(),
            "active_patch": p1.to_dict() if p1 else None,
            "patch_age_s": round(p1_age, 2) if p1_age is not None else None,
            "latest_trigger": {
                "timestamp": latest_trig1.timestamp,
                "reasons": latest_trig1.reasons,
                "pos_error_norm": latest_trig1.pos_error_norm,
                "tilt_deg": latest_trig1.tilt_deg,
                "gyro_norm": latest_trig1.gyro_norm,
                "force_residual_norm": latest_trig1.force_residual_norm,
            } if latest_trig1 else None,
        },
        "drone2": {
            "mode": sim.drone2_mode,
            "is_triggered": sim.bridge_2.is_triggered(),
            "active_patch": p2.to_dict() if p2 else None,
            "patch_age_s": round(p2_age, 2) if p2_age is not None else None,
            "latest_trigger": {
                "timestamp": latest_trig2.timestamp,
                "reasons": latest_trig2.reasons,
                "pos_error_norm": latest_trig2.pos_error_norm,
                "tilt_deg": latest_trig2.tilt_deg,
                "gyro_norm": latest_trig2.gyro_norm,
                "force_residual_norm": latest_trig2.force_residual_norm,
            } if latest_trig2 else None,
        },
        "system2": system2_status,
        "provider_usage": sim_context._cached_llm_log_status.get("provider_usage") if getattr(sim_context, "_cached_llm_log_status", None) else None,
    }


@router.get("/bridge/provider_usage")
def get_provider_token_usage() -> Dict[str, Any]:
    """Returns stateful total LLM calls, input tokens, output tokens, and total tokens partitioned by provider."""
    from runtime.logging.llm_logger import get_llm_logger
    return get_llm_logger().get_provider_usage_summary()


@router.get("/bridge/llm_history")
def get_llm_call_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Returns recent LLM reasoning transactions and latency metrics."""
    import json
    from pathlib import Path
    log_path = Path("logs/llm_calls.jsonl")
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        calls = []
        for line in reversed(lines[-limit:]):
            try:
                data = json.loads(line)
                calls.append({
                    "call_id": data.get("call_id"),
                    "timestamp_unix": data.get("timestamp_unix"),
                    "timestamp_iso": data.get("timestamp_iso"),
                    "duration_s": data.get("duration_s"),
                    "model": data.get("model"),
                    "status": data.get("status"),
                    "diagnosis": data.get("parsed_patch", {}).get("diagnosis") if data.get("parsed_patch") else None,
                    "patch": data.get("parsed_patch"),
                })
            except Exception:
                continue
        return calls
    except Exception:
        return []
