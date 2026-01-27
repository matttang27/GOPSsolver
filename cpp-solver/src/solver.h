#pragma once

#include <string>
#include <vector>

#include "mask.h"

struct State {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;   // remaining prizes as a set/list
    int diff = 0;
    int curP = 0;     // current prize value (1..N)
};

extern long long g_solveEVCalls;
extern long long g_guaranteedWins;
extern long long g_guaranteedDetected;
extern long long g_buildMatrixCalls;
extern double g_buildMatrixMaeSum;
extern bool g_enableGuarantee;
extern bool g_enableCompression;
extern bool g_enableCache;

struct TimingStats {
    long long cacheNs = 0;
    long long guaranteeNs = 0;
    long long lpNs = 0;
    long long cacheHits = 0;
    long long cacheMisses = 0;
    long long guaranteeCalls = 0;
    long long lpCalls = 0;
};

extern TimingStats g_timing;
void resetTiming();

double solveEV(State s);
std::vector<double> solveProbabilities(State s);
std::vector<std::vector<double>> buildMatrix(const State& s);
void clearEvCache();
bool saveEvCache(const std::string& path);
bool saveEvCacheMetadata(const std::string& evcPath,
                         const std::string& args,
                         int minN,
                         int maxN,
                         long long durationMs);
