#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

static inline double calculate(uint32_t iterations, int param1, int param2) {
#pragma clang fp reassoc(off) contract(off) reciprocal(off) eval_method(source)
    double result = 1.0;
    const int p1 = param1;
    const int p2 = param2;
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
    for (uint32_t i = 1; i <= iterations; ++i) {
        int j = static_cast<int>(i) * p1 - p2;
        result -= 1.0 / static_cast<double>(j);
        j = static_cast<int>(i) * p1 + p2;
        result += 1.0 / static_cast<double>(j);
    }
    return result;
}

int main() {
    using clock = std::chrono::steady_clock;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto start_time = clock::now();
    double result = calculate(200000000u, 4, 1) * 4.0;
    auto end_time = clock::now();

    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::cout.setf(std::ios::fixed);
    std::cout << "Result: " << std::setprecision(12) << result << '\n';
    std::cout << "Execution Time: " << std::setprecision(6) << elapsed << " seconds\n";
    return 0;
}