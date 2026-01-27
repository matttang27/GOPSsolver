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

void printUsage(const char* exeName) {
    std::cout << "Usage: " << exeName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --no-cache       Disable EV cache" << std::endl;
    std::cout << "  --no-guarantee   Disable guaranteed win shortcut" << std::endl;
    std::cout << "  --no-compress    Disable state compression" << std::endl;
    std::cout << "  --cache-out PATH Save EV cache to PATH (use {n} for per-N files)" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
    std::cout << "  lp-bench [minN] [maxN] [trials]" << std::endl;
}

std::string formatCachePath(const std::string& pathTemplate, int n, bool& usedToken) {
    std::string token = "{n}";
    auto pos = pathTemplate.find(token);
    if (pos == std::string::npos) {
        usedToken = false;
        return pathTemplate;
    }
    usedToken = true;
    std::string result = pathTemplate;
    result.replace(pos, token.size(), std::to_string(n));
    return result;
}

std::string joinArgs(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            out << ' ';
        }
        out << argv[i];
    }
    return out.str();
}

State fullState(int n) {
    if (n <= 0 || n > kMaxCards) {
        std::cout << "n must be between 1 and " << kMaxCards << std::endl;
        return State{};
    }
    CardMask all = static_cast<CardMask>((1u << n) - 1u);
    CardMask remainingPrizes = static_cast<CardMask>(all & ~static_cast<CardMask>(1u << (n - 1)));
    return State{all, all, remainingPrizes, 0, n};
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "lp-bench") {
        int minN = argc >= 3 ? std::stoi(argv[2]) : 2;
        int maxN = argc >= 4 ? std::stoi(argv[3]) : minN;
        int trials = argc >= 5 ? std::stoi(argv[4]) : 3;
        runLpBenchmark(minN, maxN, trials);
        return 0;
    }

    std::string cacheOutTemplate;
    std::string argsJoined = joinArgs(argc, argv);
    for (int argi = 1; argi < argc; ++argi) {
        std::string arg = argv[argi];
        if (arg == "--no-cache") {
            g_enableCache = false;
        } else if (arg == "--no-guarantee") {
            g_enableGuarantee = false;
        } else if (arg == "--no-compress") {
            g_enableCompression = false;
        } else if (arg == "--cache-out") {
            if (argi + 1 >= argc) {
                std::cout << "Missing value for --cache-out" << std::endl;
                return 1;
            }
            cacheOutTemplate = argv[++argi];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    const int minN = 1;
    const int maxN = 6;
    long long totalMs = 0;
    for (int i = minN; i <= maxN; i++) {
        clearEvCache();
        resetTiming();
        auto initial = fullState(i);
        g_solveEVCalls = 0;
        g_guaranteedWins = 0;
        g_guaranteedDetected = 0;
        g_buildMatrixCalls = 0;
        g_buildMatrixMaeSum = 0.0;
        auto start = std::chrono::steady_clock::now();
        auto probabilities = solveProbabilities(initial);
        auto end = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Action profile (" << probabilities.size() << "): "
                << vecToString(probabilities) << std::endl;
        std::cout << "solveEV calls: " << g_solveEVCalls << std::endl;
        std::cout << "guaranteed wins: " << g_guaranteedWins << std::endl;
        std::cout << "guaranteed detected: " << g_guaranteedDetected << std::endl;
        double avgMae = g_buildMatrixCalls == 0 ? 0.0 : g_buildMatrixMaeSum / g_buildMatrixCalls;
        std::cout << "prize MAE avg: " << avgMae << std::endl;
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

        if (!cacheOutTemplate.empty()) {
            bool usedToken = false;
            std::string cachePath = formatCachePath(cacheOutTemplate, i, usedToken);
            if (usedToken || i == maxN) {
                if (!usedToken) {
                    cachePath = cacheOutTemplate;
                }
                if (!saveEvCache(cachePath)) {
                    return 1;
                }
                if (!saveEvCacheMetadata(cachePath, argsJoined, i, i, elapsedMs)) {
                    return 1;
                }
                std::cout << "cache saved: " << cachePath << std::endl;
            }
        }
    };
    std::cout << "Total elapsed: " << totalMs << " ms" << std::endl;
}
