### 1. The Idea Summary

Drones rely on fast, automatic reflexes (System 1) to stay balanced in mid-air. However, if something breaks—like a chipped propeller, a swinging package, or a noisy sensor—those fixed reflexes do the wrong thing and cause a crash. On the other hand, AI models (System 2) are smart enough to figure out what broke, but they think too slowly to steer a drone in real time.

We are building a **two-speed brain**: the fast reflex loop keeps the drone flying, while the AI acts like a flight engineer in the background. When the drone starts wobbling, the AI looks at the flight data, diagnoses the exact problem, tests 32 possible fixes simultaneously in a cloud simulator, and updates the drone's reflexes in less than a second to save it from crashing.

---

### 2. Step-by-Step Implementation

1. **The Flight Simulator (Python & MuJoCo):** Run two identical virtual drones side-by-side at 500 ticks per second—one standard drone and one equipped with our two-speed brain.
2. **The Trouble Detector:** A lightweight script watches how badly the drone is tilting or drifting. If the wobble passes a danger limit, it captures a quick snapshot of the last half-second of flight data and a graph of the wobble.
3. **The AI Diagnosis (TypeScript & Meta Llama):** The snapshot is sent to Meta's AI model on Modal's GPUs. Instead of guessing exact math, the AI diagnoses *what* happened (e.g., "Front-left motor is weak") and suggests a safe range of settings to adjust.
4. **The Cloud Safety Test (Modal Parallel Sandboxes):** Modal instantly spins up 32 miniature copies of the simulator at the exact same time. Each copy tests one slightly different setting from the AI's suggestion to fast-forward 1 second into the future and see which one flies the flattest.
5. **The Live Patch:** The winning setting from the 32 cloud tests is injected straight into the live drone's reflex loop mid-flight.

---

### 3. Fail Preventions (Keeping the Project & Demo Bulletproof)

* **Preventing the AI from Hallucinating Bad Math:** The AI never touches the motors directly and never picks a single number. Every AI suggestion is tested in the 32 shadow simulations first—if a setting causes a crash in the test, it gets thrown out before touching the live drone.
* **Preventing Crashes While Waiting for the AI (The 0.5-Second Gap):** While the AI and cloud tests take half a second to think, the drone automatically enters a temporary "limp mode" (softening its movements and holding altitude) so it doesn't flip over while waiting for the fix.
* **Preventing Judge Skepticism:** We use three distinct failures that require *opposite* fixes (a weak motor needs stronger power; a swinging weight needs softer movement; a noisy sensor needs smoother filtering). No single pre-set drone can survive all three without a smart brain diagnosing the difference.

---

### 4. The Final Hackathon Demo (90 Seconds on Stage)

Your screen is split into three clean panels:

* **Left Panel (Standard Drone):** Flies normally until you click one of three red chaos buttons (**"Cut Motor 1 Power by 50%"**, **"Attach Swinging Weight"**, or **"Inject Sensor Noise"**). The standard drone overcorrects, spirals, and crashes into the floor.
* **Center Panel (Your System 1 + System 2 Drone):** You click the exact same button. The drone dips slightly, flashes yellow ("Limp Mode Active"), and 0.5 seconds later turns green and locks back into a flat, stable hover.
* **Right Panel (Live Brain View):** Shows the judges what happened during that half-second: the AI's plain-English diagnosis (*"Detected Motor 1 thrust loss"*), the 32 ghost test paths fanning out in red and green, and the winning settings being applied live.