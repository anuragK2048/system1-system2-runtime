"""uv run mjpython src/runtime/sim/view_drone.py"""

import time
from pathlib import Path
import ctypes
import mujoco
import mujoco.viewer
import numpy as np

# macOS Shift key modifier detector via CoreGraphics
try:
    _core_graphics = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
    _core_graphics.CGEventSourceFlagsState.restype = ctypes.c_uint64
    _core_graphics.CGEventSourceFlagsState.argtypes = [ctypes.c_int32]
    def is_shift_down() -> bool:
        # kCGEventSourceStateCombinedSessionState = 0, kCGEventFlagMaskShift = 0x00020000
        return bool(_core_graphics.CGEventSourceFlagsState(0) & 0x00020000)
except Exception:
    def is_shift_down() -> bool:
        return False


# Path to the Skydio X2 scene
SCENE_PATH = Path(__file__).resolve().parents[3] / "assets" / "skydio_x2" / "scene.xml"

# Exact Skydio X2 Control Allocation Matrix Inverse
B_INV = np.array([
    [0.25, -1.3889,  1.7857, -12.4378],  # Motor 1 (rear-left)
    [0.25,  1.3889,  1.7857,  12.4378],  # Motor 2 (rear-right)
    [0.25,  1.3889, -1.7857, -12.4378],  # Motor 3 (front-right)
    [0.25, -1.3889, -1.7857,  12.4378]   # Motor 4 (front-left)
])

TOTAL_WEIGHT = 4.0 * 3.2495625  # ~12.998 N
TARGET_Z = 1.0

# ----------------- CHAOS STATE FLAGS -----------------
active_chaos = {
    "motor_failure": False,    # Chaos 1: Motor 1 thrust cut by 60%
    "swinging_weight": False,  # Chaos 2: True physical swinging payload
    "sensor_noise": False,     # Chaos 3: IMU gyro sensor noise
    "wind_gust": False,        # 5.0N Wind gust (external disturbance)
}
gust_end_time = 0.0
reset_requested = False

def print_menu():
    print("\n" + "=" * 70)
    print("🚁 SKYDIO X2 - INTERACTIVE MULTIBODY CHAOS TESTBED")
    print("=" * 70)
    print("Press the following keys inside the 3D Viewer window:")
    print("  [Shift + 1] (or 6) 💥 CUT MOTOR 1 POWER BY 60%   (Chipped prop)")
    print("  [Shift + 2] (or 7) 📦 ATTACH SWINGING WEIGHT     (0.35kg red ball)")
    print("  [Shift + 3] (or 8) ⚡ INJECT SENSOR NOISE        (Gyro chatter)")
    print("  [Shift + 4] (or 9) 💨 BLOW WIND GUST (5.0N)      (Disturbance)")
    print("  [Shift + 5] (or 0) 🔄 RESET DRONE                (Restore stable hover)")
    print("  [Spacebar]         ⏸  PAUSE / RESUME")
    print("-" * 70)
    print("🛡️  Anti-Collision Protection: Drone visibility locked permanently ON.")
    print("Camera: AUTO-TRACKING ENABLED (Drone will never leave screen)")
    print("  • 1-Finger Drag: Orbit around drone | 2-Finger Swipe: Zoom")
    print("=" * 70 + "\n")

def key_callback(keycode: int):
    global active_chaos, gust_end_time, reset_requested

    shift = is_shift_down()

    # Shift + 1 ('!' = 33, '1' with Shift = 49, or key '6' = 54) -> Motor 1 Failure
    if keycode == 33 or (keycode == 49 and shift) or keycode == 54:
        active_chaos["motor_failure"] = not active_chaos["motor_failure"]
        status = "🔴 ACTIVE (Motor 1 at 40% power)" if active_chaos["motor_failure"] else "🟢 OFF (Full Power)"
        print(f"\n🚨 [CHAOS 1] Motor 1 Failure: {status}")

    # Shift + 2 ('@' = 64, '2' with Shift = 50, or key '7' = 55) -> Swinging Weight
    elif keycode == 64 or (keycode == 50 and shift) or keycode == 55:
        active_chaos["swinging_weight"] = not active_chaos["swinging_weight"]
        if active_chaos["swinging_weight"]:
            print("\n🚨 [CHAOS 2] Swinging Weight: 🔴 ATTACHED! (Look directly under drone: red ball swinging!)")
        else:
            print("\n🚨 [CHAOS 2] Swinging Weight: 🟢 DETACHED (Weightless)")

    # Shift + 3 ('#' = 35, '3' with Shift = 51, or key '8' = 56) -> Sensor Noise
    elif keycode == 35 or (keycode == 51 and shift) or keycode == 56:
        active_chaos["sensor_noise"] = not active_chaos["sensor_noise"]
        status = "🔴 ACTIVE (IMU vibrating with noise)" if active_chaos["sensor_noise"] else "🟢 OFF (Clean sensors)"
        print(f"\n🚨 [CHAOS 3] Sensor Noise: {status}")

    # Shift + 4 ('$' = 36, '4' with Shift = 52, or key '9' = 57) -> Wind Gust
    elif keycode == 36 or (keycode == 52 and shift) or keycode == 57:
        active_chaos["wind_gust"] = True
        gust_end_time = time.time() + 0.80
        print("\n💨 [DISTURBANCE] 5.0N Wind Gust hit the drone! (Notice how standard PID tilts to fight it and recovers)")

    # Shift + 5 ('%' = 37, '5' with Shift = 53, key '0' = 48, or Shift+R / 'R') -> Reset Drone
    elif keycode == 37 or (keycode in (53, 48, 82, 114) and shift) or keycode == 48:
        for k in active_chaos:
            active_chaos[k] = False
        reset_requested = True
        print("\n🔄 [RESET] All chaos modes CLEARED! Restoring drone to level hover.")

    # Plain number press (1, 2, 3, 4, 5 without Shift) reminder
    elif keycode in (49, 50, 51, 52, 53) and not shift:
        num = keycode - 48
        alt_key = num + 5
        print(f"\n💡 [TIP] Hold Shift! Press [Shift + {num}] (or single key [{alt_key}]) to trigger Chaos {num}.")

def run_simulation():
    global reset_requested

    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # 500Hz physics loop (2ms)
    model.opt.timestep = 0.002

    # Get IDs for payload body and geoms
    payload_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    tether_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tether_rod")
    bob_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_bob")
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    mp = model.body_mass[payload_body_id]

    def reset_simulation():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[0:3] = [0.0, 0.0, 0.8]  # Reset to x=0, y=0, z=0.8m
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:9] = 0.0  # Zero swing angle
        data.qvel[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0

    reset_simulation()

    # Standard System 1 Reflex PID Gains
    Kp_z, Kd_z = 25.0, 10.0
    Kp_att, Kd_att = 2.0, 0.55
    Kp_yaw, Kd_yaw = 0.5, 0.15
    Kp_xy, Kd_xy = 0.15, 0.35

    print_menu()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Enable smooth automatic camera tracking centered on the drone
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = drone_body_id
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 45

        last_print = time.time()
        physics_step_dt = model.opt.timestep
        render_interval = 1.0 / 60.0
        last_render = time.time()
        was_payload_active = False

        while viewer.is_running():
            step_start = time.time()
            now = time.time()

            # Handle explicit reset action from user
            if reset_requested:
                reset_simulation()
                reset_requested = False

            # ----------------- PHYSICAL PAYLOAD HOOK / UNHOOK -----------------
            if active_chaos["swinging_weight"]:
                # When newly attached: give an initial swing kick
                if not was_payload_active:
                    data.qpos[7] = 0.30  # 17 degrees initial swing
                    data.qvel[6] = 1.5   # swing velocity
                    was_payload_active = True

                # Make tether and red sphere visible
                model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 1.0]
                model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 1.0]

                # Payload weight is active (full gravity pulls down on drone)
                data.xfrc_applied[payload_body_id, :] = 0.0
            else:
                was_payload_active = False
                # Hide tether and bob
                model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 0.0]
                model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 0.0]

                # Neutralize payload weight so drone feels 0.00 kg
                data.xfrc_applied[payload_body_id, 2] = mp * 9.81
                data.qpos[7:9] = 0.0
                data.qvel[6:8] = 0.0

            # Clear external disturbance forces on drone
            data.xfrc_applied[drone_body_id, :] = 0.0
            data.qfrc_applied[:] = 0.0

            # 1. External Wind Gust (5.0N lateral force on drone body for 0.8s)
            if active_chaos["wind_gust"]:
                if now < gust_end_time:
                    data.xfrc_applied[drone_body_id, 0] = 5.0  # 5.0N push in +X
                else:
                    active_chaos["wind_gust"] = False

            # Read sensory state from drone (first 7 qpos, 6 qvel)
            x, y, z = data.qpos[0:3]
            vx, vy, vz = data.qvel[0:3]
            quat = data.qpos[3:7]
            omega = data.sensordata[0:3].copy()

            # 2. Chaos Mode 3: Sensor Noise
            if active_chaos["sensor_noise"]:
                omega += np.random.normal(0.0, 3.5, size=3)  # high-frequency IMU gyro noise

            # ----------------- SYSTEM 1 PID CONTROLLER -----------------
            F_z = TOTAL_WEIGHT + Kp_z * (TARGET_Z - z) - Kd_z * vz

            target_pitch = np.clip(-Kp_xy * x - Kd_xy * vx, -0.15, 0.15)
            target_roll  = np.clip( Kp_xy * y + Kd_xy * vy, -0.15, 0.15)

            roll_err  = target_roll - (2.0 * quat[1])
            pitch_err = target_pitch - (2.0 * quat[2])
            yaw_err   = 0.0 - (2.0 * quat[3])

            tau_x = Kp_att * roll_err - Kd_att * omega[0]
            tau_y = Kp_att * pitch_err - Kd_att * omega[1]
            tau_z = Kp_yaw * yaw_err - Kd_yaw * omega[2]

            ctrl = B_INV @ np.array([F_z, tau_x, tau_y, tau_z])
            ctrl = np.clip(ctrl, 0.0, 13.0)

            # 3. Chaos Mode 1: Motor 1 Physical Power Loss
            actual_thrust = ctrl.copy()
            if active_chaos["motor_failure"]:
                actual_thrust[0] *= 0.40  # Delivers only 40% thrust!

            data.ctrl[:] = actual_thrust

            # Step true multibody physics in MuJoCo (500Hz)
            mujoco.mj_step(model, data)

            # Sync GUI at 60 FPS
            if now - last_render >= render_interval:
                # Keep geom groups 0, 1, 2 permanently visible so drone never vanishes
                viewer.opt.geomgroup[0:3] = 1
                viewer.sync()
                last_render = now

            # Telemetry monitor
            if now - last_print > 1.2:
                roll_deg = np.degrees(2.0 * quat[1])
                pitch_deg = np.degrees(2.0 * quat[2])
                active_list = [k for k, v in active_chaos.items() if v]
                chaos_str = ", ".join(active_list) if active_list else "NONE (Stable)"

                # Detect crash
                if z < 0.12 and any(active_chaos.values()):
                    print(f"💥 [CRASH] Drone crashed on the ground! (Alt={z:.2f}m, Roll={roll_deg:+.1f}°). Press [Shift+5] to reset.")
                elif active_chaos["swinging_weight"]:
                    sw_x = np.degrees(data.qpos[7])
                    sw_y = np.degrees(data.qpos[8])
                    print(
                        f"Tele: Alt={z:.2f}m | Roll={roll_deg:+.1f}° | "
                        f"Payload Swing=({sw_x:+.1f}°, {sw_y:+.1f}°) 📦 [Attached]"
                    )
                else:
                    print(
                        f"Tele: Alt={z:.2f}m | Pos=({x:+.1f},{y:+.1f}) | "
                        f"Roll={roll_deg:+.1f}° | Chaos: [{chaos_str}]"
                    )
                last_print = now

            # Sleep to match physics clock
            time_until_next = physics_step_dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    run_simulation()
