import numpy as np, mujoco, mujoco.viewer, sys, time
from pathlib import Path
import mujoco.viewer
from failures import FailureManager

def find_model():
    for p in [Path("mujoco_menagerie/skydio_x2/scene.xml"),
              Path("mujoco_menagerie/skydio_x2/x2.xml")]:
        if p.exists(): return str(p)
    raise FileNotFoundError("Clone mujoco_menagerie next to this file")

def quat_to_euler(q):
    w, x, y, z = np.asarray(q, float).reshape(-1)[-4:]
    return np.array([
        np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
        np.arcsin(np.clip(2*(w*y - z*x), -1, 1)),
        np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))])

class System1:
    """PD hover controller. Patchable params kept for Step 5."""
    def __init__(self, model):
        self.body = 1                                  # root body (0 = world)
        self.mass = float(model.body_mass[self.body])
        self.I = model.body_inertia[self.body].copy()  # [Ixx Iyy Izz] from model
        self.a, self.c = 0.15, 0.016                   # arm [m], yaw-torque/thrust ratio
        self.max_f = self.mass * 9.81                   # per-motor cap ~4x hover
        self.kp_rp, self.kd_rp = 100.0, 18.0
        self.kp_yaw, self.kd_yaw = 50.0, 10.0
        self.kp_alt, self.kd_alt = 6.0, 4.0
        self.motor_trim = np.ones(4)                   # [FL FR RR RL]
        self.imu_filter_alpha = 1.0
        self._gyro_f = None
        A = np.array([[1,1,1,1],[self.a,-self.a,-self.a,self.a],
                      [-self.a,-self.a,self.a,self.a],[self.c,-self.c,self.c,-self.c]])
        self.A, self.A_inv = A, np.linalg.inv(A)

    def compute(self, model, data, target_z=2.0):
        rpy = quat_to_euler(data.qpos[3:7])
        gyro_raw = data.qvel[3:6].copy()

        if self._gyro_f is None:
            self._gyro_f = gyro_raw
        self._gyro_f = self.imu_filter_alpha * gyro_raw + (1 - self.imu_filter_alpha) * self._gyro_f
        gyro = self._gyro_f

        err = (np.array([0., 0., 0.]) - rpy + np.pi) % (2 * np.pi) - np.pi

        torque = self.I * np.array([
            self.kp_rp  * err[0] - self.kd_rp  * gyro[0],
            self.kp_rp  * err[1] - self.kd_rp  * gyro[1],
            self.kp_yaw * err[2] - self.kd_yaw * gyro[2],
        ])

        tilt = max(np.cos(rpy[0]) * np.cos(rpy[1]), 0.5)
        thrust = self.mass * (9.81 + self.kp_alt * (target_z - data.qpos[2]) - self.kd_alt * data.qvel[2]) / tilt

        # THIS is the line you asked about - it makes f = [FL, FR, RR, RL]
        f = np.clip(self.A_inv @ np.array([thrust, *torque]) * self.motor_trim, 0, self.max_f)

        return f, rpy

    def apply(self, model, data, f):
        wrench = self.A @ f
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        data.xfrc_applied[self.body, :3] = wrench[0] * (R @ np.array([0., 0., 1.]))
        data.xfrc_applied[self.body, 3:] = R @ wrench[1:]

    def step(self, model, data, target_z=2.0):
        f, rpy = self.compute(model, data, target_z=target_z)
        self.apply(model, data, f)
        return f, rpy

if __name__ == "__main__":
    # --- ADD: pick failure from command line ---
    failure = None
    if "--failure" in sys.argv:
        failure = sys.argv[sys.argv.index("--failure") + 1]
    # -------------------------------------------

    # --- CHANGE: load payload XML only for that failure ---
    xml = "models/skydio_x2_payload.xml" if failure == "swinging_payload" else find_model()
    model = mujoco.MjModel.from_xml_path(xml)
    # ------------------------------------------------------

    data = mujoco.MjData(model)
    data.qpos[2] = 2.0
    ctrl = System1(model)

    # --- ADD: create failure manager ---
    if failure == "swinging_payload":
        ctrl.mass = float(model.body_mass.sum())
    fails = FailureManager(model, mode=failure)
    # -----------------------------------
    print(f"Skydio X2: mass={ctrl.mass:.3f} kg  I={ctrl.I}  dt={model.opt.timestep}")
    headless = "--headless" in sys.argv
    viewer = None if headless else mujoco.viewer.launch_passive(model, data)
    log = []
    try:
        while data.time < 10.0:
            t0 = time.time()

            # 0. Inject at t=3s, after stable hover
            if data.time >= 3.0 and not fails.active and fails.mode:
                fails.inject(data)

            # 1. Poison sensor -> compute -> restore true sensor
            fails.pre_compute(data)
            f, rpy = ctrl.compute(model, data, target_z=2.0)
            fails.post_compute(data)

            # 2. Damage motor -> apply thrust/torque
            f = fails.damage_motors(f)
            ctrl.apply(model, data, f)

            # 3. Add wind on top -> step physics
            fails.apply_wind(data, model.opt.timestep)
            mujoco.mj_step(model, data)
            if int(data.time*2) != int((data.time-model.opt.timestep)*2):
                log.append([data.time, *np.degrees(rpy), data.qpos[2]])
                print(f"t={data.time:4.1f}s z={data.qpos[2]:.3f} roll={log[-1][1]:6.2f} "
                      f"pitch={log[-1][2]:6.2f} motors={np.round(f,2)}")
            if viewer: viewer.sync(); time.sleep(max(0, model.opt.timestep-(time.time()-t0)))
    finally:
        if viewer: viewer.close()
    log = np.array(log)
    # columns: 0=time, 1=roll, 2=pitch, 3=yaw, 4=z
    ok = np.abs(log[:,1:3]).max() < 3 and abs(log[-1,4]-2.0) < 0.05
    print(f"\nmax|roll|={np.abs(log[:,1]).max():.2f}  max|pitch|={np.abs(log[:,2]).max():.2f}  "
          f"z_final={log[-1,4]:.3f}  -> {'PASS' if ok else 'FAIL'}")