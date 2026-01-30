#include "solver.h"
#include "solver_internal.h"

#include <array>
#include <vector>

// Builds the payoff matrix for the given state. Each entry (i, j) corresponds to
// the expected value when Player A plays their i-th card and Player B plays
// their j-th card. This expected value is computed by considering all possible
// remaining prizes and recursively solving the resulting states.
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
    double matrixMaeSum = 0.0;
    long long matrixMaeCount = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::uint8_t cardA = cardsA[i];
            std::uint8_t cardB = cardsB[j];
            auto newA = removeCard(s.A, cardA);
            auto newB = removeCard(s.B, cardB);
            int newDiff = s.diff + cmp(cardA, cardB) * s.curP;

            double sumEV = 0.0;
            std::array<double, kMaxCards> prizeEvs;
            for (int k = 0; k < countP; k++) {
                std::uint8_t nextPrize = prizes[k];
                auto newRemaining = removeCard(s.P, nextPrize);
                State newState{newA, newB, newRemaining, newDiff, nextPrize};
                double ev = solveEV(newState);
                prizeEvs[k] = ev;
                sumEV += ev;
            }
            double avg = sumEV / countP;
            double mae = 0.0;
            for (int k = 0; k < countP; k++) {
                double diff = prizeEvs[k] - avg;
                if (diff < 0.0) {
                    diff = -diff;
                }
                mae += diff;
            }
            mae /= countP;
            matrixMaeSum += mae;
            ++matrixMaeCount;
            mat[i][j] = avg;
        }
    }
    ++g_buildMatrixCalls;
    if (matrixMaeCount > 0) {
        g_buildMatrixMaeSum += matrixMaeSum / matrixMaeCount;
    }
    return mat;
}
