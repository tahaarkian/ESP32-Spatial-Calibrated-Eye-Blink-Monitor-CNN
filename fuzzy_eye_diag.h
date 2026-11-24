#ifndef FUZZY_EYE_DIAG_H
#define FUZZY_EYE_DIAG_H

#include <Arduino.h>
#include <algorithm>

struct EyeDiagnosisResult {
    float strain_score;       // 0.0 (Healthy) to 100.0 (Severe Strain/Dry Eye Risk)
    float dry_eye_risk;       // 0.0 to 1.0 (driven by low blink rate + close distance)
    float photic_strain_risk; // 0.0 to 1.0 (driven by dark environment + close focus)
    const char* status_label; // Human-readable screening summary
};

class FuzzyEyeDiagnostic {
private:
    // Triangular & Trapezoidal Membership Functions
    static float trapmf(float x, float a, float b, float c, float d) {
        if (x <= a || x >= d) return 0.0f;
        if (x >= b && x <= c) return 1.0f;
        if (x > a && x < b)   return (x - a) / (b - a);
        return (d - x) / (d - c);
    }

    static float trimf(float x, float a, float b, float c) {
        if (x <= a || x >= c) return 0.0f;
        if (x == b) return 1.0f;
        if (x > a && x < b) return (x - a) / (b - a);
        return (c - x) / (c - b);
    }

public:
    // Estimate ambient room brightness (0-255) from outer frame borders (avoids face bias)
    static float estimate_ambient_luma(const uint8_t* gray, int w, int h) {
        if (!gray || w <= 0 || h <= 0) return 128.0f;
        uint32_t sum = 0;
        uint32_t count = 0;
        int border = 12; // Sample outer 12px margin

        for (int y = 0; y < h; y += 2) {
            for (int x = 0; x < w; x += 2) {
                if (x < border || x >= (w - border) || y < border || y >= (h - border)) {
                    sum += gray[y * w + x];
                    count++;
                }
            }
        }
        return (count > 0) ? (static_cast<float>(sum) / count) : 128.0f;
    }

    // Evaluate Mamdani Fuzzy Rules based on clinical thresholds
    EyeDiagnosisResult evaluate(float blink_rate_bpm, float distance_cm, float ambient_luma) {
        // --- 1. FUZZIFICATION ---
        // Blink Rate (Clinical threshold: <10 = Low/Tear breakup, 12-18 = Borderline, >=20 = Healthy)
        float br_low     = trapmf(blink_rate_bpm, -1.0f, 0.0f, 9.0f, 13.0f);
        float br_mod     = trimf(blink_rate_bpm, 10.0f, 15.0f, 20.0f);
        float br_optimal = trapmf(blink_rate_bpm, 17.0f, 20.0f, 60.0f, 65.0f);

        // Viewing Distance (Clinical threshold: <=30cm = Too Close, 45cm = Moderate, >=60cm = Ergonomic)
        float dist_close   = trapmf(distance_cm, 0.0f, 10.0f, 32.0f, 44.0f);
        float dist_mod     = trimf(distance_cm, 36.0f, 48.0f, 60.0f);
        float dist_optimal = trapmf(distance_cm, 52.0f, 60.0f, 150.0f, 200.0f);

        // Ambient Light (0-255 luma: <=45 = Dark Room/Glare Contrast, 90 = Dim, >=130 = Well-Lit)
        float light_dark    = trapmf(ambient_luma, -1.0f, 0.0f, 35.0f, 65.0f);
        float light_dim     = trimf(ambient_luma, 45.0f, 85.0f, 125.0f);
        float light_optimal = trapmf(ambient_luma, 105.0f, 135.0f, 255.0f, 260.0f);

        // --- 2. FUZZY RULE BASE (Mamdani Min-Max Inference) ---
        // Rule 1: Low blink rate + close distance -> High evaporative dry eye & accommodation strain
        float r1_severe = std::min(br_low, dist_close);

        // Rule 2: Dark room + close distance -> High photic/contrast strain (Tu & Pang 2023)
        float r2_severe = std::min(light_dark, dist_close);

        // Rule 3: Low blink rate alone -> Elevated dry eye risk (Tear film breakup <10 bpm)
        float r3_high   = br_low;

        // Rule 4: Dark room + moderate blink rate -> Elevated visual fatigue
        float r4_high   = std::min(light_dark, br_mod);

        // Rule 5: Moderate blink rate OR close distance in dim light -> Moderate strain
        float r5_mod    = std::max(std::min(br_mod, dist_mod), std::min(dist_close, light_dim));

        // Rule 6: Optimal blink rate + optimal distance + good light -> Healthy / Low risk
        float r6_low    = std::min({br_optimal, dist_optimal, light_optimal});

        // Rule 7: Optimal blink rate + good distance (even in dim light) -> Low risk
        float r7_low    = std::min(br_optimal, dist_optimal);

        // Aggregate consequent firing strengths
        float out_severe = std::max(r1_severe, r2_severe);
        float out_high   = std::max(r3_high, r4_high);
        float out_mod    = r5_mod;
        float out_low    = std::max(r6_low, r7_low);

        // --- 3. DEFUZZIFICATION (Weighted Centroid Approximation) ---
        // Output singletons: Low = 10, Moderate = 40, High = 72, Severe = 95
        float numerator   = (out_low * 10.0f) + (out_mod * 40.0f) + (out_high * 72.0f) + (out_severe * 95.0f);
        float denominator = out_low + out_mod + out_high + out_severe;

        float score = (denominator > 0.0001f) ? (numerator / denominator) : 25.0f;

        // Sub-risk indicators for targeted alerts
        float dry_risk    = std::max(r1_severe, r3_high * 0.85f);
        float photic_risk = std::max(r2_severe, r4_high * 0.85f);

        // Classify output state
        const char* label = "OPTIMAL_OCULAR_STATE";
        if (score >= 75.0f) {
            label = (dry_risk >= photic_risk) ? "HIGH_DRY_EYE_STRAIN" : "HIGH_DARK_NEAR_STRAIN";
        } else if (score >= 45.0f) {
            label = "MODERATE_VISUAL_FATIGUE";
        } else if (score >= 25.0f) {
            label = "MILD_STRAIN_MONITOR";
        }

        return {score, dry_risk, photic_risk, label};
    }
};

#endif // FUZZY_EYE_DIAGNOSTIC_H
