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

