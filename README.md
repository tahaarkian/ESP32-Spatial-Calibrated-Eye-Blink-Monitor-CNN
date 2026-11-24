# ESP32 Spatial-Calibrated Eye Blink Monitor

A lightweight, on-device eye blink detector designed to run smoothly on a **standard ESP32 (with PSRAM, e.g., ESP32-CAM)**. 

Instead of running heavy full-frame computer vision models continuously, this system uses a **two-stage confidence-gated architecture**:
1. **Fast Path (~20 ms/frame):** Crops a `24×24` grayscale eye patch and runs an **INT8 quantized CNN (`eye_model.h`)** paired with a **convex spatial calibration layer (`calib_solver.h`)** that adapts the decision threshold to the user's distance and horizontal viewing angle.
2. **Temporary Fallback (`wake_fallback`):** Monitors the calibrated probability (`p_smooth`). Only when predictions enter the uncertain gray zone (`0.25 <= p_smooth <= 0.75`) for 2+ consecutive frames—such as when the user shifts their head or glasses glare occurs—does the system wake up the heavier landmark detector (`esp-face`) for ~2 seconds to re-center the eye crop box.

---

## Project Structure

```text
esp32_eye_monitor/
├── esp32_eye_monitor.ino   # Main ESP32 camera, TFLite Micro runner, and Serial UI
├── eye_model.h             # INT8 quantized 24×24 grayscale CNN (outputs pre-sigmoid raw logit)
├── calib_solver.h          # On-device C++ convex spatial calibration solver & predictor
├── blink_engine.h          # Temporal smoothing, confidence gate, and debounced blink counter
└── README.md               # Project documentation and setup guide
```

---

## How It Works

### 1. Pre-Trained Base CNN (`eye_model.h`)
* Trained in Google Colab on the **Closed Eyes in the Wild (CEW)** dataset (`24×24` grayscale patches, ~94.7% clean validation accuracy).
* Exported **without** the final sigmoid activation so TensorFlow Lite Micro outputs raw pre-sigmoid logits directly, avoiding `log(0)` clamping issues and saving floating-point operations on the microcontroller.
* Quantized to **INT8** (~48.6 KB flash footprint).

### 2. On-Device Spatial Calibration (`calib_solver.h`)
A user's raw eye-state logits shift depending on how far they sit from the camera (d) and their lateral offset (x). During a 3-zone calibration routine, the ESP32 collects 24 labeled open/closed samples across three spatial zones:
* **Zone 1:** Center & Close (~35 cm)
* **Zone 2:** Center & Far (~75 cm)
* **Zone 3:** Off-Center / Edge (~50 cm, ±20 cm lateral)

The ESP32 solves a strictly convex 4-parameter optimization problem in **< 2 ms** using C++ gradient descent to learn temperature scaling (α) and a linear spatial threshold plane (w₀, w₁, w₂):

\[\text{cal\_logit} = \alpha \cdot \text{raw\_logit} - \left(w_0 + w_1 \cdot \frac{d - 50}{25} + w_2 \cdot \frac{x}{20}\right)\]

\[p_{\text{closed}} = \frac{1}{1 + e^{-\text{cal\_logit}}}\]

### 3. Confidence-Gated Runtime & Blink Tracker (`blink_engine.h`)
* **3-Frame Moving Average (`p_smooth`):** Smooths single-frame sensor noise.
* **Confidence Gate (`[0.25, 0.75]`):**
  * `p_smooth < 0.25` → Confident **Open** (fast path).
  * `p_smooth > 0.75` → Confident **Closed** (fast path).
  * `0.25 <= p_smooth <= 0.75` for ≥ 2 frames → Triggers `wake_fallback = true` for 30 frames (~2 seconds) to re-localize the eye coordinates.
* **Debounced Blink Counter:** Requires ≥ 2 consecutive closed frames (`p_smooth >= 0.50`) to register a blink and locks the counter until the eye re-opens, ensuring one blink equals `+1` count.

---

## Hardware Requirements

* **Board:** Original dual-core ESP32 with PSRAM (e.g., **AI-Thinker ESP32-CAM**, **ESP32-WROVER**)
* **Camera:** OV2640 (configured to `160×120` QQVGA Grayscale for fast SRAM processing)
* **Programmer:** FTDI USB-to-TTL adapter (if using ESP32-CAM)

---

## Software Setup (Arduino IDE)

1. **Install ESP32 Board Support:**
   * Open **File → Preferences** and add to *Additional Boards Manager URLs*:
     ```text
     https://github.io
     ```
   * Open **Tools → Board → Boards Manager**, search for **`esp32`**, and install **`esp32 by Espressif Systems`**.
2. **Install TensorFlow Lite Micro Library:**
   * Open **Sketch → Include Library → Manage Libraries**, search for **`Chirale_TensorFlowLite`** (or **`TensorFlowLite_ESP32`**), and install it.
3. **Configure Board Settings (`Tools` menu):**
   * **Board:** `AI Thinker ESP32-CAM` (or your specific ESP32 camera board)
   * **PSRAM:** `Enabled` *(Required)*
   * **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)` *(Required to fit TFLite Micro + model weights)*
4. **Open & Flash:**
   * Place `eye_model.h`, `calib_solver.h`, and `blink_engine.h` inside the `esp32_eye_monitor/` folder alongside `esp32_eye_monitor.ino`.
   * Connect `IO0` to `GND` on the ESP32-CAM, press **Reset**, and click **Upload**.
   * Disconnect `IO0` from `GND`, press **Reset**, and open the **Serial Monitor** at **`115200` baud**.

---

## Usage

1. **Power On:** Open the Serial Monitor (`115200` baud). If saved calibration parameters exist in flash (`Preferences`), tracking begins immediately.
2. **Recalibrate Anew:** Trigger calibration via the Serial command (or GPIO button) and follow the prompts across the **3 spatial zones** (keeping eyes open, then holding closed for each zone). Fresh parameters $(\alpha, w_0, w_1, w_2)$ are computed on-chip in < 2 ms and saved to non-volatile flash.
3. **Live Monitoring:** Serial output logs smoothed closed-eye probability, blink events, total blink count, and fallback wake-up triggers in real time.

---

## Acknowledgments & Dataset License

* Base CNN trained on the **Closed Eyes in the Wild (CEW)** dataset (for academic and research use only).




## 🧠 Fuzzy Logic Ocular Strain Diagnostic (`fuzzy_eye_diag.h`)

In addition to real-time blink detection, this firmware includes a zero-heap **Mamdani Fuzzy Inference System (FIS)** (`fuzzy_eye_diag.h`) that continuously evaluates ocular strain and dry-eye risk on-device (`0%–100%`).

### Clinical Basis & Inputs
The fuzzy engine combines three real-time metrics derived directly from the ESP32-CAM's 160×120 grayscale stream without external sensors:
1. **Blink Rate (`BPM`):** Computed over a rolling 60-second circular buffer from `res.total_blinks` (clinical tear-film stability threshold: `≥ 20 bpm` optimal, `< 10 bpm` high risk).
2. **Viewing Distance (`cm`):** Estimated distance between the user and camera (`≥ 60 cm` ergonomic, `≤ 32 cm` high accommodation strain).
3. **Ambient Illumination (`0–255 luma`):** Sampled from the outer 12-pixel perimeter of the camera frame to measure room lighting without bias from face brightness (`≤ 35` dark room glare contrast, `≥ 135` well-lit).

### Inference Architecture
* **Fuzzification:** Maps raw sensor values onto overlapping trapezoidal (`trapmf`) and triangular (`trimf`) membership functions (`Low`, `Moderate`, `Optimal`).
* **Rule Evaluation:** Evaluates 7 clinical `IF–THEN` rules using Mamdani Min-Max inference (`std::min` / `std::max`), tracking separate sub-risk scores for **evaporative dry eye** and **photic/contrast strain**.
* **Defuzzification:** Computes a weighted centroid score across four severity singletons (`Low = 10`, `Moderate = 40`, `High = 72`, `Severe = 95`):

\[\text{Strain Score} = \frac{(w_{\text{low}} \cdot 10) + (w_{\text{mod}} \cdot 40) + (w_{\text{high}} \cdot 72) + (w_{\text{severe}} \cdot 95)}{w_{\text{low}} + w_{\text{mod}} + w_{\text{high}} + w_{\text{severe}}}\]

### Output States & Serial Telemetry
Every 2 seconds, the diagnostic engine outputs a status summary over Serial (`115200 baud`):

| Strain Score | Status Label | Clinical Interpretation |
| :--- | :--- | :--- |
| **`< 25%`** | `OPTIMAL_OCULAR_STATE` | Healthy blink rate, safe distance, and adequate lighting |
| **`25% – 44%`** | `MILD_STRAIN_MONITOR` | Minor deviation in one ergonomic or blink metric |
| **`45% – 74%`** | `MODERATE_VISUAL_FATIGUE` | Sustained low blink rate or dim lighting fatigue |
| **`≥ 75%`** | `HIGH_DRY_EYE_STRAIN` / `HIGH_DARK_NEAR_STRAIN` | Combined risk factors active; triggers break/lighting alert |

**Example Serial Output:**
```text
p_closed=0.04 | state=OPEN   | blinks=14 (14.0/m) | fallback=SLEEP 
[DIAG] BPM: 14.0 | Dist: 50 cm | Light: 118/255 => Strain: 31.4% [MILD_STRAIN_MONITOR]
```

### References
* Tu, Z., & Pang, Y. (2023). *The Effect of Blink Rate and Visual Setting on the Symptoms of Digital Eye Strain*. Investigative Ophthalmology & Visual Science (IOVS).
* *Identification of a Blink Frequency Threshold for Maintaining Tear Film Stability* (2025). Scientific Reports.
