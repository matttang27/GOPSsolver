#include "solver.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "linprog_glpk.h"

long long g_solveEVCalls = 0;

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
    if (s.diff == 0 && s.A > s.B && true) {
        return -solveEV(State{s.B, s.A, s.P, -s.diff, s.curP});
    }
    auto cached = g_evCache.find(key);
    if (cached != g_evCache.end()) {
        return cached->second;
    }
    ++g_solveEVCalls;
    if (popcount16(s.A) == 1) {
        std::uint8_t cardA = onlyCard(s.A);
        std::uint8_t cardB = onlyCard(s.B);
        double value = cmp(s.diff + (cmp(cardA, cardB) * s.curP), 0);
        g_evCache.emplace(key, value);
        return value;
    }
    auto M = buildMatrix(s);
    auto result = findBestStrategyGlpk(M);
    if (result.success) {
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
