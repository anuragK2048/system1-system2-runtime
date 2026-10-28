# Working Log — system1-system2-runtime

Running record of major changes made with coding-assistant help. One entry per
help session: date, branch, what changed, and verification status.

---

## 2026-10-28 — Louie findings F1–F8 + System 2 timing (branch `louie-finding`)

- Base: `main` @ `715cb7d`. Task: `meta_api_fix_prompt.md` + `louie_findings.md`.
- Untouched as instructed: `main`, `.plan/louie_test/`, all altitude numbers,
  `demo_llm_impact.py` timing.

### Changed (staged, uncommitted — commit pending user approval)
- `src/runtime/brain/llm.py` — F1: motor-failure trim prompt `1.60x` → `2.50x`
  (+ physics comment). Timing: worker also triggers on instant 50 ms force;
  patch re-check at 2 s / pos err > 0.5 m (was 6 s / 0.75 m); worker cooldown
  and patch ages moved to sim time; per-patch `⏱️ [TIMING]` prints; timing
  metadata added to `logs/llm_calls.jsonl`.
- `src/runtime/schemas/tier2.py` — F2: new `validate_patch()` (clamps
  feedforward ±6 N, gains 0.3–2.0, trim 0.5–2.5, tilt ≤ 25°; rejects missing
  diagnosis / non-numeric / wrong-length); enforced in `from_dict`.
- `src/runtime/sim/bridge.py` — F2: `apply_system2_patch` validates before
  storing. Timing: trigger cooldown moved to sim time; new
  `note_fault_state()` for fault-onset tracking; trigger events carry
  fault type + time-to-trigger with `⏱️ [TIMING]` prints; trigger-event
  expiry 2.0 → 3.0 s; `reset()` clears cooldown clock.
- `src/runtime/schemas/telemetry.py` — `FastTriggerEvent` gains optional
  `fault_type` / `time_to_trigger_s`; removed a duplicated return line.
- `src/runtime/sim/chaos_injector.py` — F3: real timed 0.8 s wind gust
  (`trigger_wind_gust()`, latch on first apply, auto-expiry, hotkey cancel).
- `src/runtime/sim/config.py` — F4: tether comment 0.25 m → true 0.22 m.
- `assets/skydio_x2/x2.xml` — F5: timestep 0.01 → 0.002 s.
- `src/runtime/quadlib/core.py` — F6: L1 defaults aligned to `L1Config`
  (As_W −20, lpf2 5 Hz); runtime behaviour unchanged.
- `src/runtime/sim/system1_controller.py` — F7: L1 horizontal-demand tilt
  clamp, 25° default even with no patch (PID limit untouched). F8: inertia
  source comment (values unchanged — frame convention unverified).
- `src/runtime/sim/simulator.py`, `dual_simulator.py` — call
  `note_fault_state()` each step for timing correlation.
- `tests/test_louie_fixes.py` (new) — F2 clamp/reject, F3 latch/expiry/cancel,
  sim-time cooldown, fault correlation tests.
- `tests/test_system1_l1.py` — new F8 inertia-vs-MuJoCo comparison test.
- `louie_findings.md` — appended §6 fix record (per-item changes, tests, open items).

### Verification
- `py_compile` passes on all edited/new files; `x2.xml` parses; full `git diff`
  reviewed hunk-by-hunk. No altitude/main/`.plan` changes.
- `uv run python -m unittest discover tests` NOT run — no usable Python/mujoco
  toolchain on this machine (no `uv`, network-blocked installer, no compilers,
  no API keys). Must run on a dev machine before merge.

### Still open
- Commit the staged work (earlier commit attempt was declined at approval).
- Headless per-fault timing runs (motor/payload/wind/noise).
- LLM call timeout (noted in findings, not in fix scope).
- Inertia frame convention (flagged in code + test).
