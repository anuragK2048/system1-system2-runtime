"""uv run mjpython src/runtime/sim/view_drone.py"""

import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# Path to the Skydio X2 scene
SCENE_PATH = Path(__file__).resolve().parents[3] / "assets" / "skydio_x2" / "scene.xml"

# Exact Skydio X2 Control Allocation Matrix Inverse
# Maps [F_z, tau_x (roll), tau_y (pitch), tau_z (yaw)] -> [u1, u2, u3, u4]
B_INV = np.array([
    [0.25, -1.3889,  1.7857, -12.4378],  # Motor 1 (rear-left)
    [0.25,  1.3889,  1.7857,  12.4378],  # Motor 2 (rear-right)
    [0.25,  1.3889, -1.7857, -12.4378],  # Motor 3 (front-right)
    [0.25, -1.3889, -1.7857,  12.4378]   # Motor 4 (front-left)
])

# Drone mass ~1.325 kg, total hover weight in Newtons
TOTAL_WEIGHT = 4.0 * 3.2495625  # ~12.998 N
TARGET_Z = 1.0  # Desired hover altitude (meters)

# Toggle this to True if you want automatic wind gusts to test recovery
AUTO_DISTURBANCE = True

def run_simulation():
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # Use 500Hz physics loop (2ms per tick) as specified in architecture
    model.opt.timestep = 0.002

    # Initialize from the official hover keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 0.6  # Start at 0.6m

    # Critically-damped System 1 gains (no wobble, no oscillations)
    Kp_z, Kd_z = 25.0, 10.0      # Altitude control
    Kp_att, Kd_att = 2.0, 0.55   # Critical attitude damping (damping ratio ~ 0.95)
    Kp_yaw, Kd_yaw = 0.5, 0.15   # Yaw heading hold
    Kp_xy, Kd_xy = 0.15, 0.35    # Smooth station keeping

    print("\n" + "=" * 65)
    print("🚁 SKYDIO X2 - CRITICALLY DAMPED 500Hz HOVER CONTROLLER")
    print("=" * 65)
    print("How to control the view on a MacBook Trackpad:")
    print("  • Rotate / Orbit:     Click trackpad with 1 finger & drag")
    print("  • Pan Camera:         Hold Shift + Click trackpad & drag")
    print("  • Zoom:               Swipe 2 fingers up / down")
    print("  • Manual Poke/Push:   Hold Ctrl + Click with 2 fingers & drag")
    print("  • Pause / Resume:     Press Spacebar")
    print("-" * 65)
    if AUTO_DISTURBANCE:
        print("⚡ AUTO-DISTURBANCE: ON (Injects a wind gust every 6 seconds)")
    print("=" * 65 + "\n")

    # Native macOS viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = time.time()
        last_disturbance = time.time()

        # Target 60 FPS for viewer rendering (500Hz physics / 60Hz GUI)
        physics_step_dt = model.opt.timestep
        render_interval = 1.0 / 60.0
        last_render = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Optional: Automatic wind gust every 6 seconds to prove stability
            now = time.time()
            if AUTO_DISTURBANCE and (now - last_disturbance > 6.0):
                # Apply 2.5 N sudden horizontal impulse for 0.1s
                gust_dir = np.random.uniform(-1, 1, size=2)
                gust_dir /= (np.linalg.norm(gust_dir) + 1e-6)
                data.qfrc_applied[0] = gust_dir[0] * 3.0  # Fx
                data.qfrc_applied[1] = gust_dir[1] * 3.0  # Fy
                print(f"💨 [GUST] Wind hit drone with 3.0N impulse! Watch it re-stabilize...")
                last_disturbance = now
            elif AUTO_DISTURBANCE and (now - last_disturbance > 0.15):
                # Clear external applied force
                data.qfrc_applied[0:3] = 0.0

            # Read sensory state
            x, y, z = data.qpos[0:3]
            vx, vy, vz = data.qvel[0:3]
            quat = data.qpos[3:7]  # [w, x, y, z]
            omega = data.sensordata[0:3]  # IMU body gyro [wx, wy, wz]

            # 1. Altitude PD
            F_z = TOTAL_WEIGHT + Kp_z * (TARGET_Z - z) - Kd_z * vz

            # 2. Horizontal Station-Keeping
            target_pitch = np.clip(-Kp_xy * x - Kd_xy * vx, -0.15, 0.15)
            target_roll  = np.clip( Kp_xy * y + Kd_xy * vy, -0.15, 0.15)

            # 3. Attitude Damping
            roll_err  = target_roll - (2.0 * quat[1])
            pitch_err = target_pitch - (2.0 * quat[2])
            yaw_err   = 0.0 - (2.0 * quat[3])

            tau_x = Kp_att * roll_err - Kd_att * omega[0]
            tau_y = Kp_att * pitch_err - Kd_att * omega[1]
            tau_z = Kp_yaw * yaw_err - Kd_yaw * omega[2]

            # 4. Control Allocation to 4 motors
            ctrl = B_INV @ np.array([F_z, tau_x, tau_y, tau_z])
            data.ctrl[:] = np.clip(ctrl, 0.0, 13.0)

            # 500Hz physics step
            mujoco.mj_step(model, data)

            # Render at 60Hz
            if now - last_render >= render_interval:
                viewer.sync()
                last_render = now

            # Console telemetry every 1.5 seconds
            if now - last_print > 1.5:
                roll_deg = np.degrees(2.0 * quat[1])
                pitch_deg = np.degrees(2.0 * quat[2])
                print(
                    f"Tele: Alt={z:.2f}m | Pos=({x:+.2f}, {y:+.2f}) | "
                    f"Roll={roll_deg:+.1f}° | Pitch={pitch_deg:+.1f}° | Motors={np.round(data.ctrl, 1)}"
                )
                last_print = now

            # Maintain real-time clock
            time_until_next = physics_step_dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    run_simulation()
