#include "solver.h"
#include "ev_cache.h"
#include "solver_internal.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

#include "linprog_glpk.h"

template <typename Func>
static auto runTimed(Func&& func, long long& timeNs, long long* callCount = nullptr) -> decltype(func()) {
    auto start = std::chrono::steady_clock::now();
    auto result = func();
    auto end = std::chrono::steady_clock::now();
    timeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (callCount != nullptr) {
        ++(*callCount);
    }
    return result;
}

const char* objectiveName(SolveObjective objective) {
    switch (objective) {
        case SolveObjective::Win:
            return "win";
        case SolveObjective::Points:
            return "points";
    }
    return "win";
}

bool parseObjective(const std::string& value, SolveObjective& out) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "win") {
        out = SolveObjective::Win;
        return true;
    }
    if (lowered == "points") {
        out = SolveObjective::Points;
        return true;
    }
    return false;
}

static std::string toString(const State& state) {
    std::ostringstream out;
    out << "State{A=" << maskToString(state.A)
        << ", B=" << maskToString(state.B)
        << ", P=" << maskToString(state.P)
        << ", diff=" << state.diff
        << ", curP=" << state.curP
        << "}";
    return out.str();
}

static int highestCard(CardMask mask) {
    for (int card = kMaxCards; card >= 1; --card) {
        if (mask & static_cast<CardMask>(1u << (card - 1))) {
            return card;
        }
    }
    return 0;
}

static int countAbove(CardMask mask, int threshold) {
    if (threshold <= 0) {
        return popcount16(mask);
    }
    if (threshold >= kMaxCards) {
        return 0;
    }
    CardMask higher = static_cast<CardMask>(mask & ~static_cast<CardMask>((1u << threshold) - 1u));
    return popcount16(higher);
}

static int guaranteedOutcome(const State& s) {
    CardMask prizeMask = s.P;
    if (s.curP > 0) {
        prizeMask = static_cast<CardMask>(prizeMask | static_cast<CardMask>(1u << (s.curP - 1)));
    }
    std::array<int, kMaxCards> sortedPrizes;
    int prizeCount = 0;
    int total = 0;
    for (int card = kMaxCards; card >= 1; --card) {
        if (prizeMask & static_cast<CardMask>(1u << (card - 1))) {
            sortedPrizes[prizeCount++] = card;
            total += card;
        }
    }
    if (prizeCount == 0) {
        return 0;
    }
    std::array<int, kMaxCards + 1> guarantee;
    guarantee[0] = -total;
    int sumTop = 0;
    for (int i = 1; i <= prizeCount; ++i) {
        sumTop += sortedPrizes[i - 1];
        guarantee[i] = 2 * sumTop - total;
    }

    const int maxA = highestCard(s.A);
    const int maxB = highestCard(s.B);
    const int guaranteeA = countAbove(s.A, maxB);
    const int guaranteeB = countAbove(s.B, maxA);

    if (guarantee[guaranteeA] + s.diff > 0) {
        return 1;
    }
    if (s.diff - guarantee[guaranteeB] < 0) {
        return -1;
    }
    return 0;
}

static void storeIfEnabled(const StateKey& key, double value) {
    if (g_enableCache) {
        evCacheStore(key, value);
    }
}

static double oneCardEV(const State& s) {
    std::uint8_t cardA = onlyCard(s.A);
    std::uint8_t cardB = onlyCard(s.B);
    int roundDelta = cmp(cardA, cardB) * s.curP;
    if (g_solveObjective == SolveObjective::Points) {
        return static_cast<double>(roundDelta);
    }
    int finalDiff = s.diff + roundDelta;
    return static_cast<double>(cmp(finalDiff, 0));
}

static double solveEVWin(State s) {
    if (s.diff < 0) {
        return -solveEV(State{s.B, s.A, s.P, -s.diff, s.curP});
    }
    //Both only reduce states by around 3% each
    if (s.diff == 0) {
        int cmpAB = cmp(s.A, s.B);
        if (cmpAB < 0) {
            return -solveEV(State{s.B, s.A, s.P, -s.diff, s.curP});
        } else if (cmpAB == 0) {
            return 0.0;
        }
    }

    // from now on, either (s.diff > 0 or s.A > s.B), and s.diff >= 0 and s.A >= s.B.
    if (g_enableCompression) {
        auto compressed = compressCards(s.A, s.B);
        s.A = compressed.first;
        s.B = compressed.second;
    }
    StateKey key = makeStateKey(s);
    if (g_enableCache) {
        double cachedValue = 0.0;
        bool found = runTimed([&]() { return evCacheFind(key, cachedValue); },
                              g_timing.cacheNs);
        if (found) {
            ++g_timing.cacheHits;
            return cachedValue;
        }
        ++g_timing.cacheMisses;
    }
    ++g_solveEVCalls;
    if (g_enableGuarantee) {
        int guaranteed = guaranteedOutcome(s);
        if (guaranteed != 0) {
            storeIfEnabled(key, guaranteed);
            return static_cast<double>(guaranteed);
        }
    }
    if (popcount16(s.A) == 1) {
        double ev = oneCardEV(s);
        storeIfEnabled(key, ev);
        return ev;
    }
    auto M = buildMatrix(s);
    auto result = runTimed([&]() { return findBestStrategyGlpk(M); },
                           g_timing.lpNs, &g_timing.lpCalls);
    if (result.success) {
        storeIfEnabled(key, result.expectedValue);
        return result.expectedValue;
    }
    std::cerr << "GLPK failed for " << toString(s) << " with "
              << toString(result) << std::endl;
    return 0.0;
}

static double solveEVPoints(State s) {
    s.diff = 0;
    int cmpAB = cmp(s.A, s.B);
    if (cmpAB < 0) {
        return -solveEVPoints(State{s.B, s.A, s.P, 0, s.curP});
    } else if (cmpAB == 0) {
        return 0.0;
    }

    if (g_enableCompression) {
        auto compressed = compressCards(s.A, s.B);
        s.A = compressed.first;
        s.B = compressed.second;
    }

    StateKey key = makeStateKey(s);
    if (g_enableCache) {
        double cachedValue = 0.0;
        bool found = runTimed([&]() { return evCacheFind(key, cachedValue); },
                              g_timing.cacheNs);
        if (found) {
            ++g_timing.cacheHits;
            return cachedValue;
        }
        ++g_timing.cacheMisses;
    }

    ++g_solveEVCalls;
    if (popcount16(s.A) == 1) {
        double ev = oneCardEV(s);
        storeIfEnabled(key, ev);
        return ev;
    }

    auto M = buildMatrix(s);
    auto result = runTimed([&]() { return findBestStrategyGlpk(M); },
                           g_timing.lpNs, &g_timing.lpCalls);
    if (result.success) {
        storeIfEnabled(key, result.expectedValue);
        return result.expectedValue;
    }
    std::cerr << "GLPK failed for " << toString(s) << " with "
              << toString(result) << std::endl;
    return 0.0;
}

double solveEV(State s) {
    if (g_solveObjective == SolveObjective::Points) {
        return solveEVPoints(s);
    }
    return solveEVWin(s);
}

std::vector<double> solveProbabilities(State s) {
    if (popcount16(s.A) == 1) {
        return {1.0};
    }
    auto M = buildMatrix(s);
    auto result = findBestStrategyGlpk(M);
    if (result.success) {
        return result.probabilities;
    }
    std::cerr << "GLPK failed for " << toString(s) << " with "
              << toString(result) << std::endl;
    return {1.0};
}

void resetTiming() {
    g_timing = TimingStats{};
}
