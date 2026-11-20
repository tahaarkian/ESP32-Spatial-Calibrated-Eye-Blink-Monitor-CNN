#include <Arduino.h>
#include <Preferences.h>
#include "esp_camera.h"

// TensorFlow Lite Micro (Install "Chirale_TensorFlowLite" in Arduino Library Manager)
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Your 3 generated headers
#include "eye_model.h"
#include "calib_solver.h"
#include "blink_engine.h"

// --- AI-Thinker ESP32-CAM Pin Definitions ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4
#define CALIB_BTN_PIN     13 // Optional physical button to GND

// --- TFLite Micro Globals ---
constexpr int kTensorArenaSize = 40 * 1024;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

Esp32BlinkEngine engine;
Preferences prefs;

// Crop a box from 160x120 grayscale frame, resize to 24x24, and quantize to int8
void extract_24x24_patch(const uint8_t* gray, int img_w, int img_h, EyeBox box, int8_t* out_int8) {
    int x0 = max(0, box.x);
    int y0 = max(0, box.y);
    int bw = min(img_w - x0, max(1, box.w));
    int bh = min(img_h - y0, max(1, box.h));

    for (int r = 0; r < 24; ++r) {
        int src_y = y0 + (r * bh) / 24;
        for (int c = 0; c < 24; ++c) {
            int src_x = x0 + (c * bw) / 24;
            uint8_t px = gray[src_y * img_w + src_x];
            out_int8[r * 24 + c] = static_cast<int8_t>(static_cast<int>(px) - 128);
        }
    }
}

// Run INT8 CNN on a 24x24 patch and return dequantized pre-sigmoid raw_logit
float run_eye_cnn(const uint8_t* gray, int img_w, int img_h, EyeBox box) {
    extract_24x24_patch(gray, img_w, img_h, box, model_input->data.int8);
    if (interpreter->Invoke() != kTfLiteOk) return 0.0f;
    int8_t q_out = model_output->data.int8[0];
    return (static_cast<float>(q_out) - model_output->params.zero_point) * model_output->params.scale;
}

// Lightweight brightness-valley eye localizer for 160x120 frame (used during calibration & fallback)
EyeBox locate_eye_region(const uint8_t* gray, int w, int h, EyeBox search_area) {
    int best_x = search_area.x + search_area.w / 2 - 14;
    int best_y = search_area.y + search_area.h / 2 - 14;
    uint32_t min_sum = UINT32_MAX;

    int x_start = max(0, search_area.x);
    int y_start = max(0, search_area.y);
    int x_end = min(w - 28, search_area.x + search_area.w - 28);
    int y_end = min(h - 28, search_area.y + search_area.h - 28);

    for (int y = y_start; y <= y_end; y += 2) {
        for (int x = x_start; x <= x_end; x += 2) {
            uint32_t sum = 0;
            for (int dy = 4; dy < 24; dy += 2) {
                for (int dx = 4; dx < 24; dx += 2) {
                    sum += gray[(y + dy) * w + (x + dx)];
                }
            }
            if (sum < min_sum) {
                min_sum = sum;
                best_x = x;
                best_y = y;
            }
        }
    }
    return {best_x, best_y, 28, 28, true};
}

void save_calibration() {
    prefs.begin("eye_cal", false);
    prefs.putFloat("alpha", engine.calib.alpha);
    prefs.putFloat("w0", engine.calib.w0);
    prefs.putFloat("w1", engine.calib.w1);
    prefs.putFloat("w2", engine.calib.w2);
    prefs.end();
}

void load_calibration() {
    prefs.begin("eye_cal", true);
    engine.calib.alpha = prefs.getFloat("alpha", 1.0f);
    engine.calib.w0    = prefs.getFloat("w0", 0.0f);
    engine.calib.w1    = prefs.getFloat("w1", 0.0f);
    engine.calib.w2    = prefs.getFloat("w2", 0.0f);
    prefs.end();
}

// 3-Zone Interactive Serial Calibration (matches Script 2's 24-sample convex calibration)
void run_3zone_calibration() {
    const char* zone_names[3] = {
        "ZONE 1: Center & Close (~35 cm)",
        "ZONE 2: Center & Far (~75 cm)",
        "ZONE 3: Off-Center / Edge (~50 cm, +20 cm right)"
    };
    const float zone_dists[3] = {35.0f, 75.0f, 50.0f};
    const float zone_lats[3]  = {0.0f,  0.0f,  20.0f};

    CalibSample samples[24];
    int idx = 0;

    Serial.println("\n=== STARTING 3-ZONE SPATIAL CALIBRATION ===");
    for (int z = 0; z < 3; ++z) {
        Serial.printf("\n[%s]\n", zone_names[z]);
        Serial.println("Align your eye near center and KEEP EYES OPEN (capturing in 2s)...");
        delay(2000);

        // Locate eye in central search region
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) continue;
        EyeBox zone_box = locate_eye_region(fb->buf, fb->width, fb->height, {40, 25, 80, 70, true});
        esp_camera_fb_return(fb);

        // Capture 4 OPEN samples
        for (int s = 0; s < 4; ++s) {
            fb = esp_camera_fb_get();
            if (fb) {
                float logit = run_eye_cnn(fb->buf, fb->width, fb->height, zone_box);
                samples[idx++] = {logit, (zone_dists[z] - 50.0f) / 25.0f, zone_lats[z] / 20.0f, -1.0f};
                esp_camera_fb_return(fb);
            }
            delay(100);
        }

        Serial.println("NOW CLOSE YOUR EYES AND HOLD CLOSED (capturing in 2s)...");
        delay(2000);

        // Capture 4 CLOSED samples
        for (int s = 0; s < 4; ++s) {
            fb = esp_camera_fb_get();
            if (fb) {
                float logit = run_eye_cnn(fb->buf, fb->width, fb->height, zone_box);
                samples[idx++] = {logit, (zone_dists[z] - 50.0f) / 25.0f, zone_lats[z] / 20.0f, 1.0f};
                esp_camera_fb_return(fb);
            }
            delay(100);
        }
    }

    engine.calib = solve_convex_calibration_esp32(samples, idx);
    save_calibration();
    engine.reset();
    Serial.printf("\nCalibration Complete! alpha=%.3f | w=[%.3f, %.3f, %.3f]\n",
                  engine.calib.alpha, engine.calib.w0, engine.calib.w1, engine.calib.w2);
}

void setup() {
    Serial.begin(115200);
    pinMode(CALIB_BTN_PIN, INPUT_PULLUP);

    // 1. Initialize OV2640 Camera in 160x120 Grayscale
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size   = FRAMESIZE_QQVGA; // 160x120
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed!");
        while (true) delay(1000);
    }

    // 2. Initialize TensorFlow Lite Micro
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::AllOpsResolver resolver;
    const tflite::Model* model = tflite::GetModel(g_eye_model_data);

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    model_input  = interpreter->input(0);
    model_output = interpreter->output(0);

    // 3. Load Saved Calibration & Initial Eye Box
    load_calibration();
    engine.reset();
    engine.update_eye_location(66, 46, 28, 28, 50.0f, 0.0f);

    Serial.println("ESP32 Eye Monitor Ready! Send 'c' in Serial Monitor (or press GPIO 13) to calibrate.");
}

void loop() {
    // Check for calibration trigger ('c' in Serial Monitor or GPIO 13 button)
    if ((Serial.available() && Serial.read() == 'c') || digitalRead(CALIB_BTN_PIN) == LOW) {
        run_3zone_calibration();
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return;

    // Fast path: run 24x24 INT8 CNN on current eye_box
    float raw_logit = run_eye_cnn(fb->buf, fb->width, fb->height, engine.eye_box);
    BlinkStepResult res = engine.step(raw_logit);

    // Temporary Fallback: re-center eye_box only when confidence gate triggers
    if (res.wake_fallback) {
        EyeBox search_win = {
            max(0, engine.eye_box.x - 16),
            max(0, engine.eye_box.y - 16),
            60, 60, true
        };
        EyeBox refined = locate_eye_region(fb->buf, fb->width, fb->height, search_win);
        float est_lat = (refined.x + 14 - 80) * 0.5f; // Rough horizontal offset in cm
        engine.update_eye_location(refined.x, refined.y, refined.w, refined.h, engine.est_dist_cm, est_lat);
    }

    esp_camera_fb_return(fb);

    static uint32_t last_print = 0;
    if (millis() - last_print > 250) {
        last_print = millis();
        Serial.printf("p_closed=%.2f | state=%s | blinks=%d | fallback=%s\n",
                      res.p_closed,
                      res.is_closed ? "CLOSED" : "OPEN  ",
                      res.total_blinks,
                      res.wake_fallback ? "ACTIVE" : "SLEEP ");
    }
}
