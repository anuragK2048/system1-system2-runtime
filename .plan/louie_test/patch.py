import json, numpy as np
try:
    import jsonschema
    _HAS_JS = True
except ImportError:
    _HAS_JS = False

with open("allowed_patch.schema.json") as f:
    SCHEMA = json.load(f)

# Fallback ranges if jsonschema isn't installed
_RANGES = {"kp_rp_scale": (0.35,1.30), "kd_rp_scale": (0.35,1.50),
           "kp_yaw_scale": (0.50,1.30), "kd_yaw_scale": (0.50,1.50),
           "kp_alt_scale": (0.70,1.40), "kd_alt_scale": (0.70,1.40),
           "imu_filter_alpha": (0.05,1.0)}

def validate_and_clamp(msg: dict) -> dict:
    """Returns the inner patch dict, clamped to schema. Raises on bad shape."""
    if _HAS_JS:
        jsonschema.validate(msg, SCHEMA)
    p = dict(msg["patch"])
    for k, (lo, hi) in _RANGES.items():
        p[k] = float(np.clip(p.get(k, 1.0 if k != "imu_filter_alpha" else 1.0), lo, hi))
    p["motor_trim"] = np.clip(np.array(p.get("motor_trim", [1,1,1,1]), dtype=float), 0.70, 1.70).tolist()
    assert len(p["motor_trim"]) == 4
    return p

def apply_patch_msg(controller, msg: dict):
    patch = validate_and_clamp(msg)
    controller.apply_patch(patch)
    print(f"[PATCH APPLIED] {msg.get('failure_type')} -> {patch}")
    return patch

# Temporary safe mode while waiting ~0.5s for AI + Modal (Step 6 will use this)
LIMP_MODE = {"motor_trim": [1,1,1,1], "kp_rp_scale": 0.6, "kd_rp_scale": 0.7,
             "kp_yaw_scale": 1.0, "kd_yaw_scale": 1.0, "kp_alt_scale": 1.1,
             "kd_alt_scale": 1.1, "imu_filter_alpha": 0.3}

# Oracle fixes — these PROVE each failure is individually fixable.
# Modal's job later is to search AROUND these, not invent from scratch.
ORACLE_PATCHES = {
    # 1/(1-0.35) = 1.54 -> exactly compensates the dead FL motor
    "chipped_prop":     {"motor_trim": [1.54,1.0,1.0,1.0], "kp_rp_scale": 1.0, "kd_rp_scale": 1.0,
                         "kp_yaw_scale": 1.0, "kd_yaw_scale": 1.0, "kp_alt_scale": 1.0, "kd_alt_scale": 1.0,
                         "imu_filter_alpha": 1.0},
    # Soften attitude so we stop exciting the 0.84 Hz pendulum
    "swinging_payload": {"motor_trim": [1,1,1,1], "kp_rp_scale": 0.45, "kd_rp_scale": 0.6,
                         "kp_yaw_scale": 1.0, "kd_yaw_scale": 1.0, "kp_alt_scale": 1.0, "kd_alt_scale": 1.0,
                         "imu_filter_alpha": 1.0},
    # Heavy low-pass so the D-term stops amplifying gyro noise
    "noisy_sensor":     {"motor_trim": [1,1,1,1], "kp_rp_scale": 0.8, "kd_rp_scale": 0.7,
                         "kp_yaw_scale": 1.0, "kd_yaw_scale": 1.0, "kp_alt_scale": 1.0, "kd_alt_scale": 1.0,
                         "imu_filter_alpha": 0.08},
}