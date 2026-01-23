#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "linprog_glpk.h"
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

std::vector<std::vector<double>> makeRandomMatrix(
    int n,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& dist) {
    std::vector<std::vector<double>> mat(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dist(rng);
        }
    }
    return mat;
}

void runLpBenchmark(int minN, int maxN, int trials) {
    if (minN <= 0 || maxN < minN || trials <= 0) {
        std::cout << "Usage: lp-bench <minN> <maxN> <trials>" << std::endl;
        return;
    }
    std::mt19937_64 rng(1234567);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::cout << "LP benchmark (random matrices in [-1, 1])" << std::endl;
    std::cout << "n=" << minN << ".." << maxN << ", trials=" << trials << std::endl;

    for (int n = minN; n <= maxN; ++n) {
        long long totalNs = 0;
        long long bestNs = LLONG_MAX;
        long long worstNs = 0;
        int success = 0;
        for (int t = 0; t < trials; ++t) {
            auto mat = makeRandomMatrix(n, rng, dist);
            auto start = std::chrono::steady_clock::now();
            auto result = findBestStrategyGlpk(mat);
            auto end = std::chrono::steady_clock::now();
            long long elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            totalNs += elapsedNs;
            if (elapsedNs < bestNs) {
                bestNs = elapsedNs;
            }
            if (elapsedNs > worstNs) {
                worstNs = elapsedNs;
            }
            if (result.success) {
                ++success;
            }
        }
        double avgMs = (static_cast<double>(totalNs) / trials) / 1e6;
        double minMs = static_cast<double>(bestNs) / 1e6;
        double maxMs = static_cast<double>(worstNs) / 1e6;
        std::cout << "n=" << n
                  << " avg=" << avgMs << " ms"
                  << " min=" << minMs << " ms"
                  << " max=" << maxMs << " ms"
                  << " success=" << success << "/" << trials << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "lp-bench") {
        int minN = argc >= 3 ? std::stoi(argv[2]) : 2;
        int maxN = argc >= 4 ? std::stoi(argv[3]) : minN;
        int trials = argc >= 5 ? std::stoi(argv[4]) : 3;
        runLpBenchmark(minN, maxN, trials);
        return 0;
    }

    long long totalMs = 0;
    for (int i = 1; i <= 10; i++) {
        clearEvCache();
        resetTiming();
        auto initial = full(8);
        g_solveEVCalls = 0;
        g_guaranteedWins = 0;
        g_guaranteedDetected = 0;
        auto start = std::chrono::steady_clock::now();
        auto probabilities = solveProbabilities(initial);
        auto end = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Action profile (" << probabilities.size() << "): "
                << vecToString(probabilities) << std::endl;
        std::cout << "solveEV calls: " << g_solveEVCalls << std::endl;
        std::cout << "guaranteed wins: " << g_guaranteedWins << std::endl;
        std::cout << "guaranteed detected: " << g_guaranteedDetected << std::endl;
        auto cacheLookups = g_timing.cacheHits + g_timing.cacheMisses;
        double cacheMs = g_timing.cacheNs / 1e6;
        double guaranteeMs = g_timing.guaranteeNs / 1e6;
        double lpMs = g_timing.lpNs / 1e6;
        double cacheAvgNs = cacheLookups == 0 ? 0.0 : static_cast<double>(g_timing.cacheNs) / cacheLookups;
        double guaranteeAvgNs = g_timing.guaranteeCalls == 0 ? 0.0 : static_cast<double>(g_timing.guaranteeNs) / g_timing.guaranteeCalls;
        double lpAvgNs = g_timing.lpCalls == 0 ? 0.0 : static_cast<double>(g_timing.lpNs) / g_timing.lpCalls;
        std::cout << "cache: " << g_timing.cacheHits << " hits, " << g_timing.cacheMisses
                  << " misses, " << cacheMs << " ms, " << cacheAvgNs << " ns avg" << std::endl;
        std::cout << "guarantee: " << g_timing.guaranteeCalls << " calls, " << guaranteeMs
                  << " ms, " << guaranteeAvgNs << " ns avg" << std::endl;
        std::cout << "lp: " << g_timing.lpCalls << " calls, " << lpMs
                  << " ms, " << lpAvgNs << " ns avg" << std::endl;
        std::cout << "Elapsed: " << elapsedMs << " ms" << std::endl;
        totalMs += elapsedMs;
    };
    std::cout << "Total elapsed: " << totalMs << " ms" << std::endl;
}
