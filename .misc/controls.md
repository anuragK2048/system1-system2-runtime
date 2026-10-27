# Ways to interact with the drone through mujoco 


(Note: MuJoCo also allows you to add noise attributes to these sensors in the XML if you want to test how your Tier-1/Tier-2 switching handles noisy IMU measurements).
data.xfrc_applied also exists for applying external forces on drone

In MuJoCo, controlling the **Skydio X2** model (`x2.xml` from `mujoco_menagerie`) involves interacting with the `MjData` structure at each simulation step. 

Because the X2 is modeled as a rigid 6-DOF floating body (via a `<freejoint/>`) with site-based thrusters rather than physically spinning propeller joints, the control inputs and telemetry map directly to standard quadrotor dynamics.

MuJoCo provides specific controls, sensors, and ground-truth states for your two-tiered architecture:

---

### 1. Controls (Actuators)
You send commands to the drone by writing to the **`data.ctrl`** array. The Skydio X2 model defines **4 motor actuators** located at the four rotor sites (`thrust1`, `thrust2`, `thrust3`, `thrust4`).

* **Control Input (`data.ctrl[0:4]`):** A 4-element vector representing the thrust command (in Newtons) for each of the 4 rotors:
  * `data.ctrl[0]` (`thrust1`): Back-Right rotor (`pos="-0.14 -0.18 0.05"`)
  * `data.ctrl[1]` (`thrust2`): Back-Left rotor (`pos="-0.14 0.18 0.05"`)
  * `data.ctrl[2]` (`thrust3`): Front-Left rotor (`pos="0.14 0.18 0.08"`)
  * `data.ctrl[3]` (`thrust4`): Front-Right rotor (`pos="0.14 -0.18 0.08"`)
* **How Physics are Applied (`gear` attribute):** Instead of simulating aerodynamic blade elements, MuJoCo uses the actuator `gear="0 0 1 0 0 ±0.0201"` attribute on each rotor site:
  * The `1` (3rd element) applies linear thrust directly along the drone's local $+Z$ axis.
  * The `±0.0201` (6th element) applies a reaction yaw torque around the local $Z$ axis simultaneously, mimicking propeller drag (alternating signs for CW and CCW rotors).
* **Control Range:** Each thruster is bounded by `ctrlrange="0 13"` (0 to 13 Newtons per motor). The drone weighs $\approx 1.325\text{ kg}$, so hover requires roughly $\approx 3.25\text{ N}$ per motor (which is defined in the `hover` keyframe).

*(Note: If your Tier-1 or Tier-2 controller outputs collective thrust and body torques $[F_z, \tau_x, \tau_y, \tau_z]$ instead of individual motor thrusts, you will use a standard $4 \times 4$ quadrotor mixer matrix to convert them into `data.ctrl[0:4]`).*

---

### 2. Telemetry: Onboard Sensors (`data.sensordata`)
The Skydio X2 XML comes pre-configured with an IMU/odometry sensor suite attached to the drone's central `imu` site. You read these from **`data.sensordata`** (or by name via `data.sensor('name').data`):

1. **Gyroscope (`gyro`):** 3 values (`[wx, wy, wz]`)
   * Measures **angular velocity** in the drone's local **body frame** ($\text{rad/s}$).
2. **Accelerometer (`accelerometer`):** 3 values (`[ax, ay, az]`)
   * Measures **linear proper acceleration** (includes gravity) in the **body frame** ($\text{m/s}^2$).
3. **Position (`framepos`):** 3 values (`[x, y, z]`)
   * Measures 3D position of the IMU in the **global world frame** ($\text{m}$), acting like Mocap or RTK-GPS.
4. **Orientation (`framequat`):** 4 values (`[w, x, y, z]`)
   * Measures the 3D orientation quaternion in the **global world frame**.

*(Note: MuJoCo also allows you to add `noise` attributes to these sensors in the XML if you want to test how your Tier-1/Tier-2 switching handles noisy IMU measurements).*

---

### 3. Telemetry: Ground-Truth Physics State (`data.qpos`, `data.qvel`)
Because the drone's base is attached to the world via a 6-DOF `<freejoint/>`, MuJoCo constantly tracks the exact, noise-free mathematical state of the rigid body. This is ideal for computing the exact **tracking error** to trigger your Tier-2 Foundation Model:

* **Generalized Positions (`data.qpos` — 7 values):**
  * `data.qpos[0:3]`: Global 3D position $[x, y, z]$ in meters.
  * `data.qpos[3:7]`: Global 3D orientation quaternion $[w, x, y, z]$.
* **Generalized Velocities (`data.qvel` — 6 values):**
  * `data.qvel[0:3]`: Linear velocity $[v_x, v_y, v_z]$ in the **world frame** ($\text{m/s}$).
  * `data.qvel[3:6]`: Angular velocity $[\omega_x, \omega_y, \omega_z]$ in the **local body frame** ($\text{rad/s}$).
* **Accelerations (`data.qacc` — 6 values):**
  * `data.qacc[0:3]`: True linear acceleration in the world frame ($\text{m/s}^2$).
  * `data.qacc[3:6]`: True angular acceleration in the body frame ($\text{rad/s}^2$).

---

### 4. Visual & Disturbance Interfaces (For Your Tier-2 Use Case)
Since your Tier-2 controller is a Foundation Model meant to handle unknown/poor conditions:
* **Camera / Vision Telemetry:** The X2 model defines cameras (`track` camera, or you can attach a first-person `<camera>` to the `x2` body). Using `mujoco.Renderer`, you can extract RGB and Depth arrays at any frequency required by a Vision-Language-Action (VLA) or multimodal Foundation Model.
* **Injecting "Poor Conditions" (Wind / Disturbances):**
  * **External Forces (`data.xfrc_applied`):** A $[N_{bodies} \times 6]$ array where you can directly inject 3D forces and 3D torques `[fx, fy, fz, tx, ty, tz]` onto the drone body in real-time to simulate sudden payloads, rotor loss, or turbulent wind gusts.
  * **Medium / Wind (`model.opt.wind`, `model.opt.density`, `model.opt.viscosity`):** MuJoCo has built-in fluid/air medium parameters where you can set a global wind velocity vector `[wx, wy, wz]` to create aerodynamic drag disturbances.








  