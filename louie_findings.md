# Louie Findings — Sim Parameters and LLM Timing
Repo: https://github.com/anuragK2048/system1-system2-runtime
Branch: main
Checked at commit: 715cb7db112c143d8123a54b6ebd22edbf6febc5
Date: 2026-10-27 / 2026-10-28 PDT
Scope: `src/runtime/sim/` only. `.plan/louie_test/` is ignored — Anurag reworked it.
Altitude: DEFERRED. Do not change any altitude / target_z numbers in this round, per Feng-Yi.

## What was checked
- `src/runtime/sim/config.py`
- `src/runtime/sim/system1_controller.py`
- `src/runtime/sim/simulator.py`
- `src/runtime/sim/chaos_injector.py`
- `src/runtime/sim/bridge.py`
- `src/runtime/brain/llm.py`
- `src/runtime/schemas/tier2.py`, `src/runtime/schemas/telemetry.py`
- `src/runtime/quadlib/core.py`
- `assets/skydio_x2/x2.xml`, `scene.xml`, `dual_scene.xml`
- `src/runtime/sim/demo_llm_impact.py`, `src/runtime/sim/dual_simulator.py`

No tests were run in this check. Findings are from reading the code and hand-checking the physics.

## 1. Parameters that make sense — do not break these
- Mass / weight: controller mass 1.325 kg. XML rotors 4 x 0.25 kg + body 0.325 kg = 1.325 kg exactly. Weight 13.0 N, hover 3.25 N per motor. XML keyframe ctrl is 3.25 N per motor.
- Allocation matrix B_INV: matches arm dx = 0.14 m, dy = 0.18 m, c_tau = 0.0201. 1/(4*0.18) = 1.389, 1/(4*0.14) = 1.786, 1/(4*0.0201) = 12.44.
- Max thrust: 0-13 N per motor in both config and XML, total 52 N, max thrust-to-weight 4.0, max accel 29.4 m/s2 (3 g). This is generous vs a real X2 (about 2-2.5) — the sim is forgiving, recovery will look easier than real hardware.
- Payload: 0.35 kg = 26% of mass, 3.43 N. With payload, total weight 16.43 N, hover 4.11 N per motor, still well under 13 N.
- Wind 5 N: 38% of weight, lateral accel 3.77 m/s2, needs 21 deg tilt to hold position. PID max tilt 8.6 deg (0.15 rad) gives only 1.48 m/s2 lateral accel — PID cannot hold in 5 N wind, it will drift. L1 with a 25 deg envelope gives 4.57 m/s2 — just enough.
- Motor loss: motor 1 at 40% loses 1.95 N at hover, total 11.05 N vs 13.0 N needed. The other three motors need +0.65 N each (3.25 -> 3.90 N) to make it up.
- Sensor noise: 0.8 rad/s std = 45.8 deg/s of gyro chatter. Severe by design, not realistic (real gyro noise is about 0.01 rad/s). PID torque noise is about 0.55 * 0.8 = 0.44 Nm.
- L1 gains: pos [3, 3, 15], vel [2.5, 2.5, 8], att [3.5, 3.5, 1.2], ang [0.7, 0.7, 0.25]. Z is much stiffer than X/Y, which is sensible. As_v -5, As_W -20, low-pass filters 10 / 10 / 5 Hz.
- Physics rate 500 Hz (0.002 s), render 60 FPS. Consistent.

## 2. Issues to fix (not altitude)
### F1. Motor trim in the LLM prompt under-compensates
- File: `src/runtime/brain/llm.py`, SYSTEM_PROMPT
- The prompt tells the LLM to use motor trim [1.60, 0.95, 0.95, 1.0] for motor 1 failure.
- 1.60 x 0.40 = 0.64 effective thrust — still 36% short.
- Full compensation needs 2.5x on motor 1 (3.25 x 2.5 = 8.1 N, still under 13 N), or 1.20x on the other three motors.
- Fix the prompt and the example, and add a clamp so trim cannot exceed what is physically useful.

### F2. No validation / clamping on LLM patches
- Files: `src/runtime/schemas/tier2.py` (`System2Patch.from_dict`), `src/runtime/sim/bridge.py` (`apply_system2_patch`)
- The LLM can return any feedforward force, gain multiplier, motor trim, or tilt limit. There is no bounds check. Only motor clipping at 0-13 N protects the drone.
- Add validation before a patch is applied. Suggested limits, to be confirmed by testing: feedforward |F| <= 6 N, gain multipliers 0.3-2.0, motor trim 0.5-2.5, max tilt <= 25 deg. Reject or clamp out-of-range values and log it.

### F3. Wind gust duration is dead code
- Files: `src/runtime/sim/config.py` (`wind_gust_duration_seconds = 0.80`), `src/runtime/sim/chaos_injector.py` (`apply_wind_gust`, `gust_end_time`)
- The config says a 0.8 s gust, but `apply_wind_gust` toggles the wind on forever. `gust_end_time` is never set.
- Either implement a real 0.8 s timer, or remove the duration field. Do not leave a number that looks real but does nothing.

### F4. Payload tether length mismatch
- Config comment says a 0.25 m tether. `assets/skydio_x2/x2.xml` has a 0.22 m tether (`fromto 0 0 0 0 0 -0.22`, bob at z -0.22).
- Pick the true value, align the config comment and the XML, and note which one was changed and why.

### F5. Single-scene timestep mismatch
- `assets/skydio_x2/x2.xml` says timestep 0.01 s (100 Hz). `simulator.py` overrides it to 0.002 s (500 Hz). `dual_scene.xml` is already 0.002 s.
- Set `x2.xml` to 0.002 s so the file matches what actually runs.

### F6. L1 defaults differ between config and quadlib
- `src/runtime/sim/config.py`: As_W -20, lpf2 cutoff 5 Hz.
- `src/runtime/quadlib/core.py` defaults: As_W -10, lpf2 cutoff 2 Hz.
- Config wins at runtime, but the duplicate defaults are confusing. Align them or remove the duplicate defaults.

### F7. No speed / tilt limit for L1
- PID is clamped to max tilt 0.15 rad (8.6 deg) in `system1_controller.py`.
- L1 / geometric control has no max-tilt clamp in code. The only tilt limit is via the LLM safety envelope (default 25 deg), which may not be present.
- Add a tilt / velocity limit for the L1 path, or enforce the safety envelope default even when no patch is active. Do not change PID tilt.

### F8. Inertia has no source
- Controller inertia is [0.06071, 0.03647, 0.02541] in `system1_controller.py`.
- It is plausible from the rotor positions, but there is no source, datasheet, or check against MuJoCo's computed inertia.
- Add a comment with the source / calculation, or a test comparing it to MuJoCo's computed inertia. Do not change the values without a test.

## 3. LLM timing — the current issue
Fault at t = 0. Live worker path (`System2Worker` in `src/runtime/brain/llm.py`), not the demo.

- Detection: bridge `push_telemetry` every 2 ms. Trigger if pos error > 0.30 m, tilt > 25 deg, gyro > 3.5 rad/s, or L1 force > 3.0 N. Ignored for the first 1.0 s of sim time. Bridge cooldown 0.5 s (wall time) between trigger events.
- Summary: worker needs a 1.0 s window (500 samples, mean over the last 1 s). The LLM therefore sees a mean diluted by pre-fault data.
- Worker poll: loop every 0.05 s, cooldown 2.0 s between LLM calls. Needs timestamp >= 1.0 s and >= 200 samples.
- Worker thresholds (no patch yet): triggered OR pos error > 0.30 m OR force > 2.5 N (note: lower than the bridge's 3.0 N) OR moment > 0.35 Nm OR oscillation detected.
- LLM call: synchronous in the worker thread, no timeout set. Groq is about 1 s, Baseten is unknown. Trigger event expires after 2.0 s of sim time (`get_latest_trigger_event` max_age).
- Patch apply: next 2 ms step after `apply_system2_patch`.
- Best case, fault to patch: about 2-2.5 s. Worst case with worker cooldown: about 4-5 s.
- After a patch: no re-call unless pos error > 0.75 m or oscillation, AND patch age > 6.0 s.
- Patch clears only when calm (pos < 0.15 m, force < 1.0 N, moment < 0.20 Nm, no oscillation) AND more than 2.5 s after the last call.
- Demo timing is different: `demo_llm_impact.py` waits a fixed 2.0 s after chaos, then calls the LLM once. It is not trigger-driven.

### Why this is a problem
- Free fall from 1.0 m is 0.45 s. L1 must hold the drone alone for about 2-3 s while the LLM decides. If L1 cannot, the LLM arrives too late.
- A 0.8 s wind gust may already be over before the patch arrives, then the patch is applied to a fault that is gone.
- Motor failure and payload are persistent, wind is transient — they need different timing, but the worker treats them the same.
- The worker uses the 1.0 s mean force, not the instantaneous force (last 25 samples = 50 ms) that the bridge already computes. That adds about 1 s of delay for fast faults.

### Proposed timing rule to test
- 0-50 ms: L1 / System 1 only, plus an instant limp mode on trigger. Do not wait for the LLM.
- 50-200 ms: trigger on instant L1 force (50 ms window) or pos / tilt / gyro, not on the 1.0 s mean.
- 200 ms-2 s: LLM runs in the background while L1 keeps flying. Validate the patch before applying it (see F2).
- After patch: re-check at 2 s, not 6 s. If pos error is still > 0.5 m or oscillation continues, call the LLM again.
- Measure per fault: time-to-trigger, time-to-patch, time-to-crash / time-to-recover. Test wind as a timed 0.8 s gust, not a toggle, to see if LLM timing matters at all for it.

## 4. Deferred — do not touch in this round
- Altitude / target_z: config target_z 1.0 m, demo comment 1.50 m, old hover 2.0 m, XML keyframe z 0.3 m, spawn 0.8 m. These disagree, but Feng-Yi said do not touch altitude for now. Recorded here only so it is not lost. No altitude numbers are to be changed by the fix prompt.

## 5. How to re-check
1. Pull latest main, check HEAD is still 715cb7d or note the new HEAD.
2. For each fix F1-F8, run the relevant test: `uv run python -m unittest discover tests`
3. For timing, run headless for each of motor_failure, swinging_payload, wind_gust, sensor_noise and record time-to-trigger, time-to-patch, time-to-crash / recover.
4. Update this file with what was changed, what was tested, and what is still open.

## 6. Fixes applied on branch `louie-finding` (2026-10-28)
All work on branch `louie-finding` from `main` @ 715cb7d. Scope kept to
`src/runtime/sim/`, `src/runtime/brain/llm.py`, `src/runtime/schemas/`,
`src/runtime/quadlib/core.py`, `assets/skydio_x2/x2.xml`, plus tests.
`.plan/louie_test/` untouched. No altitude / `target_z` / spawn-z / keyframe-z
number or comment touched. `demo_llm_impact.py` timing untouched (still a fixed
2.0 s comparison, not trigger-driven).

### F1. Motor trim prompt — FIXED
- `src/runtime/brain/llm.py`: prompt now advises `[2.50, 1.0, 1.0, 1.0]`
  (0.40 x 2.50 = 1.00; hover 3.25 N x 2.50 = 8.1 N < 13 N limit), with
  `[1.0, 1.20, 1.20, 1.20]` as the spread-over-healthy-motors alternative.
- Added a code comment above `SYSTEM_PROMPT` showing the calculation.
- Old value 1.60x restored only 64% of M1 thrust.

### F2. Patch validation / clamping — FIXED
- `src/runtime/schemas/tier2.py`: new `validate_patch()` + limit constants
  (feedforward |F| <= 6 N/axis, gains 0.3-2.0, trim 0.5-2.5, tilt <= 25 deg).
  Out-of-range values are clamped and the clamping is printed; patches with no
  diagnosis, non-numeric/non-finite values, or wrong-length vectors raise
  `ValueError` (rejected). `System2Patch.from_dict` validates before returning.
- `src/runtime/sim/bridge.py`: `apply_system2_patch` validates before storing.
  Rejection propagates as `ValueError`; the worker already catches it per-loop.
- Limits confirmed against the prompt examples (att_kp 0.40, att_kd 1.6, trim
  2.50, wind feedforward +/-5 N) — advised patches pass through unchanged.
- One judgement call: tilt <= 0 is rejected (a non-positive tilt limit is
  meaningless); the spec only named the 25 deg upper bound.

### F3. Wind gust duration — FIXED (real timed gust)
- `src/runtime/sim/chaos_injector.py`: new `trigger_wind_gust(now)` sets the
  flag and `gust_end_time = now + 0.80 s`; `apply_wind_gust` latches the end
  time on first application (also covers hotkey/direct flag sets) and switches
  the gust off after 0.80 s. Hotkey-off and `clear_all` cancel/reset the timer.
- `wind_gust_duration_seconds = 0.80` is now real; no dead duration numbers left.
- Note: the demo's `wind_gust` scenario now feels 0.8 s of wind inside its
  2.0 s phase 1 instead of sustained wind — intended per-item physics fix.

### F4. Payload tether length — FIXED (kept 0.22 m)
- Verified in XML physics: `x2.xml` tether `fromto 0 0 0 0 0 -0.22`, bob at
  z -0.22; `dual_scene.xml` agrees (0.22 m on both drones).
- Kept 0.22 m (XML is the physics truth), aligned the `config.py` comment to it.

### F5. Single-scene timestep — FIXED
- `assets/skydio_x2/x2.xml`: `timestep 0.01` -> `0.002`, matching the 500 Hz
  override in `simulator.py` and `dual_scene.xml`.

### F6. L1 defaults — FIXED (aligned to config)
- `src/runtime/quadlib/core.py`: `l1_adaptive_control` defaults `As_W -10` ->
  `-20`, `lpf2_cutoff_freq_moment 2.0` -> `5.0 Hz`, matching `L1Config`, with a
  docstring note. Runtime behaviour unchanged (`System1Controller` passes
  explicit values; it is the only caller).

### F7. L1 tilt limit — FIXED
- `src/runtime/sim/system1_controller.py`: the L1 path now clamps the
  horizontal position demand to `|err_xy| <= g * tan(max_tilt) / kp_xy`
  (steady-state bound), using the patch safety envelope when present and the
  25 deg default (`L1_DEFAULT_MAX_TILT_DEG`) when no patch is active. Effective
  limit reported in diagnostics as `l1_max_tilt_deg`. PID 0.15 rad limit
  untouched. Velocity-error terms can still add transient tilt — noted in code.

### F8. Inertia — DOCUMENTED, values unchanged
- Values `[0.06071, 0.03647, 0.02541]` NOT changed (no test proves better ones).
- Added a comment with the point-mass hand-check basis — and a flag: the three
  numbers match the XML geometry only up to axis permutation (planar-rotor
  physics gives Izz largest; here Ixx is largest), so the frame convention is
  UNVERIFIED.
- New test `test_controller_inertia_matches_mujoco` compares the controller
  values against MuJoCo's computed `body_inertia` as sorted sets within 25%
  and prints both. Deliberately not element-wise until the frame is resolved.

### LLM timing — FIXED per items 1-5
1. Worker `_run_loop` now also triggers on the instant (50 ms)
   `instantaneous_force_disturbance` at the existing 2.5 N worker threshold;
   the 1.0 s summary is still what gets sent to the LLM.
2. Patch re-check at 2 s (was 6 s): re-consult if pos error still > 0.5 m
   (was 0.75 m) or oscillation continues after 2 s (was 4 s/6 s).
3. Unified time basis: bridge cooldown, trigger timestamps/expiry, worker
   cooldown, and patch ages are ALL MuJoCo sim time now (was: bridge cooldown
   wall, rest sim). Documented in `bridge.py` / `llm.py` docstrings.
   `get_latest_trigger_event` default `max_age_s` 2.0 -> 3.0 s: covers the
   2.0 s worker cooldown plus ~1 s observed Groq latency margin. `reset()`
   now also clears the cooldown clock (required under sim time).
4. Timing logs: every trigger prints `⏱️ [TIMING] trigger t_sim=… fault=…
   time-to-trigger=…`; every applied/revised patch prints fault,
   time-to-trigger, and time-to-patch. Fault onset is tracked via the new
   `bridge.note_fault_state()` call wired into both simulators' step
   functions. Timing metadata (fault, time-to-trigger, trigger/summary sim
   times) is also stored in `logs/llm_calls.jsonl` per call.
5. Demo kept as the fixed 2.0 s comparison; noted here that it is not
   trigger-driven.

### Deliberately NOT changed
- Trigger thresholds: pos 0.30 m, tilt 25 deg, gyro 3.5 rad/s, force 3.0 N
  (bridge) and worker 2.5 N / 0.35 Nm / oscillation gates — no test showed
  them wrong.
- Altitude numbers anywhere (deferred per Feng-Yi). Still conflicting on
  record: config `target_z` 1.0 m, demo comment 1.50 m, XML keyframe z 0.3 m,
  spawn 0.8 m.
- LLM call timeout: findings note the call has no timeout; the fix list did
  not ask for one, so none was added (provider clients differ; needs a
  follow-up decision). Left open.
- `dual_scene.xml` timestep/tether already correct; not touched.

### Tests
- `uv run python -m unittest discover tests` — NOT RUN: no runnable Python
  toolchain on this machine (no `uv`, only broken/ancient interpreters, no
  mujoco/numpy, no compilers; network install declined). Must be run on a
  dev machine before merge.
- New `tests/test_louie_fixes.py` (F2 clamp/reject, F3 latch/expiry/cancel,
  sim-time cooldown, fault correlation) and F8 inertia test added for that run.
- All edited/added files pass `py_compile`, `x2.xml` parses as XML, and every
  hunk of `git diff` was reviewed against the items above.
