#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "solver.h"

template <typename T>
std::string vecToString(const std::vector<T>& vec) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << vec[i];
    }
    out << "]";
    return out.str();
}

int main() {
    for (int i = 1; i <= 6; i++) {
        auto initial = full(i);
        g_solveEVCalls = 0;
        auto start = std::chrono::steady_clock::now();
        auto probabilities = solveProbabilities(initial);
        auto end = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Action profile (" << probabilities.size() << "): "
                << vecToString(probabilities) << std::endl;
        std::cout << "solveEV calls: " << g_solveEVCalls << std::endl;
        std::cout << "Elapsed: " << elapsedMs << " ms" << std::endl;
    };
}
