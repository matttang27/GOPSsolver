#include "solver.h"
#include "ev_cache.h"
#include "solver_internal.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "linprog_glpk.h"

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

double solveEV(State s) {
    StateKey key = makeStateKey(s);
    if (s.diff < 0 && true) {
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

    if (g_enableCache) {
        auto cacheStart = std::chrono::steady_clock::now();
        double cachedValue = 0.0;
        bool found = evCacheFind(key, cachedValue);
        auto cacheEnd = std::chrono::steady_clock::now();
        g_timing.cacheNs += std::chrono::duration_cast<std::chrono::nanoseconds>(cacheEnd - cacheStart).count();
        if (found) {
            ++g_timing.cacheHits;
            return cachedValue;
        }
        ++g_timing.cacheMisses;
    }
    ++g_solveEVCalls;
    if (g_enableGuarantee) {
        auto guaranteeStart = std::chrono::steady_clock::now();
        int guaranteed = guaranteedOutcome(s);
        auto guaranteeEnd = std::chrono::steady_clock::now();
        g_timing.guaranteeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(guaranteeEnd - guaranteeStart).count();
        ++g_timing.guaranteeCalls;
        if (guaranteed != 0) {
            ++g_guaranteedDetected;
            ++g_guaranteedWins;
            if (g_enableCache) {
                evCacheStore(key, guaranteed);
            }
            return static_cast<double>(guaranteed);
        }
    }
    if (popcount16(s.A) == 1) {
        std::uint8_t cardA = onlyCard(s.A);
        std::uint8_t cardB = onlyCard(s.B);
        double value = cmp(s.diff + (cmp(cardA, cardB) * s.curP), 0);
        if (value == 1.0 || value == -1.0) {
            ++g_guaranteedWins;
        }
        if (g_enableCache) {
            evCacheStore(key, value);
        }
        return value;
    }
    auto M = buildMatrix(s);
    auto lpStart = std::chrono::steady_clock::now();
    auto result = findBestStrategyGlpk(M);
    auto lpEnd = std::chrono::steady_clock::now();
    g_timing.lpNs += std::chrono::duration_cast<std::chrono::nanoseconds>(lpEnd - lpStart).count();
    ++g_timing.lpCalls;
    if (result.success) {
        if (result.expectedValue >= 1.0 - 1e-10 || result.expectedValue <= -1.0 + 1e-10) {
            ++g_guaranteedWins;
        }
        if (g_enableCache) {
            evCacheStore(key, result.expectedValue);
        }
        return result.expectedValue;
    }
    std::cerr << "GLPK failed for " << toString(s) << " with "
              << toString(result) << std::endl;
    return 0.0;
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
