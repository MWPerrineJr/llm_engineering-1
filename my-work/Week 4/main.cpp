#include <chrono>
#include <cstdio>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#pragma float_control(precise, on)
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

static double calculate() {
    double result = 1.0;

#if defined(__aarch64__)
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t step = vdupq_n_f64(16.0);

    float64x2_t d0 = {3.0, 5.0};
    float64x2_t d1 = {7.0, 9.0};
    float64x2_t d2 = {11.0, 13.0};
    float64x2_t d3 = {15.0, 17.0};

#pragma clang loop unroll(disable)
    for (int i = 0; i < 50'000'000; ++i) {
        const float64x2_t q0 = vdivq_f64(one, d0);
        const float64x2_t q1 = vdivq_f64(one, d1);
        const float64x2_t q2 = vdivq_f64(one, d2);
        const float64x2_t q3 = vdivq_f64(one, d3);

        d0 = vaddq_f64(d0, step);
        d1 = vaddq_f64(d1, step);
        d2 = vaddq_f64(d2, step);
        d3 = vaddq_f64(d3, step);

        result -= vgetq_lane_f64(q0, 0);
        result += vgetq_lane_f64(q0, 1);
        result -= vgetq_lane_f64(q1, 0);
        result += vgetq_lane_f64(q1, 1);
        result -= vgetq_lane_f64(q2, 0);
        result += vgetq_lane_f64(q2, 1);
        result -= vgetq_lane_f64(q3, 0);
        result += vgetq_lane_f64(q3, 1);
    }
#else
    for (int i = 1; i <= 200'000'000; ++i) {
        result -= 1.0 / static_cast<double>(4 * i - 1);
        result += 1.0 / static_cast<double>(4 * i + 1);
    }
#endif

    return result * 4.0;
}

int main() {
    const auto start = std::chrono::steady_clock::now();
    const double result = calculate();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed = std::chrono::duration<double>(end - start).count();
    std::printf("Result: %.12f\nExecution Time: %.6f seconds\n", result, elapsed);
    return 0;
}