import numpy as np, mujoco

MODES = [None, "motor_damage", "swinging_payload", "sensor_wind"]

class FailureManager:
    def __init__(self, model, mode=None, seed=0):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode, self.active = mode, False
        self.rng = np.random.default_rng(seed)
        self.body = 1
        self._backup, self._wind = None, np.zeros(3)
        # payload joint addresses (-1 if using the no-payload model)
        self._payload_qadr = []
        for jname in ("payload_roll", "payload_pitch"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._payload_qadr.append(model.jnt_qposadr[jid])

    def inject(self, data):
        self.active = True
        if self.mode == "swinging_payload":
            assert self._payload_qadr, "Load models/skydio_x2_payload.xml for this failure"
            data.qpos[self._payload_qadr[0]] = 0.45   # 26 deg kick
        print(f"[FAILURE] {self.mode} injected at t={data.time:.1f}s")

    # --- 3a. Sensor noise: poison gyro BEFORE compute, restore after ---
    def pre_compute(self, data):
        if not (self.active and self.mode == "sensor_wind"): return
        self._backup = data.qvel.copy()
        bias = np.array([0.12, -0.08, 0.0])
        data.qvel[3:6] += bias + self.rng.normal(0, 0.35, 3)  # rad/s

    def post_compute(self, data):
        if self._backup is not None:
            data.qvel[:] = self._backup; self._backup = None

    # --- 1. Motor damage: motor 0 capped at 50% effectiveness ---
    def damage_motors(self, f):
        if not (self.active and self.mode == "motor_damage"): return f
        out = np.array(f, float).copy()
        out[0] *= 0.50
        return out

    # --- 3b. Wind gust: random force AFTER apply (so it isn't overwritten) ---
    def apply_wind(self, data, dt):
        if not (self.active and self.mode == "sensor_wind"): return
        # Ornstein-Uhlenbeck gust: smooth random push, ~±3 N lateral
        self._wind += -self._wind * dt / 0.5 + 6.0 * np.sqrt(dt) * self.rng.normal(0, 1, 3)
        self._wind[2] *= 0.3
        data.xfrc_applied[self.body, :3] += self._wind
        data.xfrc_applied[self.body, 3]  += self.rng.normal(0, 0.02, 3)