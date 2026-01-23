#include "solver.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "linprog_glpk.h"

long long g_solveEVCalls = 0;
long long g_guaranteedWins = 0;
long long g_guaranteedDetected = 0;
TimingStats g_timing;

struct StateKey {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;
    std::int8_t diff = 0;
    std::uint8_t curP = 0;

    bool operator==(const StateKey& other) const {
        return A == other.A && B == other.B && P == other.P
            && diff == other.diff && curP == other.curP;
    }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const {
        std::size_t h = 0;
        auto mix = [&h](std::size_t v) {
            h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::size_t>(key.A));
        mix(static_cast<std::size_t>(key.B));
        mix(static_cast<std::size_t>(key.P));
        mix(static_cast<std::size_t>(static_cast<std::uint32_t>(key.diff)));
        mix(static_cast<std::size_t>(key.curP));
        return h;
    }
};

static std::unordered_map<StateKey, double, StateKeyHash> g_evCache;

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

template<typename T>
int cmp(T a, T b) {
    return (a > b) - (a < b);
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

std::vector<std::vector<double>> buildMatrix(const State& s) {
    std::array<std::uint8_t, kMaxCards> cardsA;
    std::array<std::uint8_t, kMaxCards> cardsB;
    std::array<std::uint8_t, kMaxCards> prizes;
    const int countA = listCards(s.A, cardsA);
    const int countB = listCards(s.B, cardsB);
    const int countP = listCards(s.P, prizes);
    const int size = countA;
    if (countA != countB || countP != countA - 1) {
        return {};
    }
    std::vector<std::vector<double>> mat(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::uint8_t cardA = cardsA[i];
            std::uint8_t cardB = cardsB[j];
            auto newA = removeCard(s.A, cardA);
            auto newB = removeCard(s.B, cardB);
            int newDiff = s.diff + cmp(cardA, cardB) * s.curP;
            auto compressed = compressCards(newA, newB);
            double sumEV = 0.0;
            for (int k = 0; k < countP; k++) {
                std::uint8_t nextPrize = prizes[k];
                auto newRemaining = removeCard(s.P, nextPrize);
                State newState{compressed.first, compressed.second, newRemaining, newDiff, nextPrize};
                sumEV += solveEV(newState);
            }
            mat[i][j] = sumEV / countP;
        }
    }
    return mat;
}

double solveEV(State s) {
    StateKey key{s.A, s.B, s.P, s.diff, static_cast<std::uint8_t>(s.curP)};
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
    
    auto cacheStart = std::chrono::steady_clock::now();
    auto cached = g_evCache.find(key);
    auto cacheEnd = std::chrono::steady_clock::now();
    g_timing.cacheNs += std::chrono::duration_cast<std::chrono::nanoseconds>(cacheEnd - cacheStart).count();
    if (cached != g_evCache.end()) {
        ++g_timing.cacheHits;
        return cached->second;
    }
    ++g_timing.cacheMisses;
    ++g_solveEVCalls;
    auto guaranteeStart = std::chrono::steady_clock::now();
    int guaranteed = guaranteedOutcome(s);
    auto guaranteeEnd = std::chrono::steady_clock::now();
    g_timing.guaranteeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(guaranteeEnd - guaranteeStart).count();
    ++g_timing.guaranteeCalls;
    if (guaranteed != 0) {
        ++g_guaranteedDetected;
        ++g_guaranteedWins;
        g_evCache.emplace(key, guaranteed);
        return guaranteed;
    }
    if (popcount16(s.A) == 1) {
        std::uint8_t cardA = onlyCard(s.A);
        std::uint8_t cardB = onlyCard(s.B);
        double value = cmp(s.diff + (cmp(cardA, cardB) * s.curP), 0);
        if (value == 1.0 || value == -1.0) {
            ++g_guaranteedWins;
        }
        g_evCache.emplace(key, value);
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
        g_evCache.emplace(key, result.expectedValue);
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

State full(int n) {
    if (n <= 0 || n > kMaxCards) {
        std::cerr << "n must be between 1 and " << kMaxCards << std::endl;
        return State{};
    }
    CardMask all = static_cast<CardMask>((1u << n) - 1u);
    CardMask remainingPrizes = static_cast<CardMask>(all & ~static_cast<CardMask>(1u << (n - 1)));
    return State{all, all, remainingPrizes, 0, n};
}

void clearEvCache() {
    g_evCache.clear();
}

void resetTiming() {
    g_timing = TimingStats{};
}
