import mujoco, numpy as np, sys
from generate_mjcf import generate, load_presets
from controller import DroneController
from failures import FailureManager

def run_one(preset_name, p, failure, duration=8.0):
    model = mujoco.MjModel.from_xml_path(generate(preset_name))
    data = mujoco.MjData(model)
    ctrl = DroneController(p)
    fails = FailureManager(p, name=failure)
    dt = model.opt.timestep
    max_tilt, min_z, crash_t = 0.0, 1.0, None

    while data.time < duration:
        if data.time >= 3.0 and not fails.active and failure:
            fails.inject(data.time)
        fails.pre_compute(data)
        motor_forces, pos, euler, gyro, *_ = ctrl.compute(model, data, dt)
        fails.post_compute(data)
        motor_forces = fails.apply_motor_cmds(motor_forces, data.time)
        ctrl.apply_forces(model, data, motor_forces)
        fails.apply_external_forces(model, data, dt)
        mujoco.mj_step(model, data)

        tilt = np.degrees(np.abs(np.asarray(euler).reshape(-1)[:2])).max()
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, float(np.asarray(pos).reshape(-1)[2]))
        if tilt > 50 or float(np.asarray(pos).reshape(-1)[2]) < 0.3:
            crash_t = data.time
            break

    status = "CRASHED ✓" if crash_t else "SURVIVED ✗"
    print(f"{str(failure):18s} max_tilt={max_tilt:5.1f}°  min_z={min_z:.2f}m  "
          f"crash_t={crash_t if crash_t else '-'}  -> {status}")
    return crash_t is not None

if __name__ == "__main__":
    presets = load_presets()
    keys = list(presets.keys())
    choice = int(sys.argv[1]) - 1 if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    name, p = keys[choice], presets[keys[choice]]
    print(f"\nTesting failures on {p['display_name']} (inject at t=3.0s)\n")
    results = {f: run_one(name, p, f) for f in [None, "chipped_prop", "swinging_payload", "noisy_sensor"]}
    print("\nPASS" if results[None] == False and all(v for k, v in results.items() if k) 
          else "\nNOT READY — tune failures.py (see below)")