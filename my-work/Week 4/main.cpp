
#include <iostream>
#include <iomanip>
#include <chrono>

static double calculate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j1 = static_cast<double>(i * param1 - param2);
        result -= (1.0 / j1);
        double j2 = static_cast<double>(i * param1 + param2);
        result += (1.0 / j2);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(200000, 4, 1) * 4.0;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}
