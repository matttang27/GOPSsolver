#include "solver.h"

#include <array>
#include <iostream>
#include <sstream>

#include "linprog_glpk.h"

long long g_solveEVCalls = 0;

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
            double sumEV = 0.0;
            for (int k = 0; k < countP; k++) {
                std::uint8_t nextPrize = prizes[k];
                auto newRemaining = removeCard(s.P, nextPrize);
                State newState{newA, newB, newRemaining, newDiff, nextPrize};
                sumEV += solveEV(newState);
            }
            mat[i][j] = sumEV / countP;
        }
    }
    return mat;
}

double solveEV(State s) {
    ++g_solveEVCalls;
    if (popcount16(s.A) == 1) {
        std::uint8_t cardA = onlyCard(s.A);
        std::uint8_t cardB = onlyCard(s.B);
        return cmp(s.diff + (cmp(cardA, cardB) * s.curP), 0);
    }
    auto M = buildMatrix(s);
    auto result = findBestStrategyGlpk(M);
    if (result.success) {
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
