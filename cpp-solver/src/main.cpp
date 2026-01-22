#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <glpk.h>

static long long g_solveEVCalls = 0;

using CardMask = std::uint16_t;
static const int kMaxCards = 16;

struct StrategyResult {
    bool success = false;
    double expectedValue = 0.0;
    std::vector<double> probabilities;
};

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

std::string toString(const StrategyResult& result) {
    std::ostringstream out;
    out << "StrategyResult{success=" << (result.success ? "true" : "false")
        << ", expectedValue=" << result.expectedValue
        << ", probabilities=" << vecToString(result.probabilities)
        << "}";
    return out.str();
}

StrategyResult findBestStrategyGlpk(const std::vector<std::vector<double>>& payoffMatrix) {
    const int numRows = static_cast<int>(payoffMatrix.size());
    if (numRows == 0) {
        return {};
    }
    const int numCols = static_cast<int>(payoffMatrix[0].size());
    if (numCols == 0) {
        return {};
    }
    for (int i = 1; i < numRows; ++i) {
        if (static_cast<int>(payoffMatrix[i].size()) != numCols) {
            return {};
        }
    }

    glp_prob* lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MAX);

    // Rows: numCols inequality constraints + 1 equality constraint.
    glp_add_rows(lp, numCols + 1);
    for (int j = 0; j < numCols; ++j) {
        glp_set_row_bnds(lp, j + 1, GLP_LO, 0.0, 0.0); // sum_i a_ij p_i - v >= 0
    }
    glp_set_row_bnds(lp, numCols + 1, GLP_FX, 1.0, 1.0); // sum_i p_i = 1

    // Columns: numRows probabilities + 1 value variable.
    glp_add_cols(lp, numRows + 1);
    for (int i = 0; i < numRows; ++i) {
        glp_set_col_bnds(lp, i + 1, GLP_LO, 0.0, 0.0); // p_i >= 0
        glp_set_obj_coef(lp, i + 1, 0.0);
    }
    const int vCol = numRows + 1;
    glp_set_col_bnds(lp, vCol, GLP_FR, 0.0, 0.0); // v free
    glp_set_obj_coef(lp, vCol, 1.0);

    const int entriesPerCol = numRows + 1; // each inequality row +1 for v
    const int totalEntries = numCols * entriesPerCol + numRows; // + eq row entries
    std::vector<int> ia(1 + totalEntries);
    std::vector<int> ja(1 + totalEntries);
    std::vector<double> ar(1 + totalEntries);

    int idx = 1;
    for (int j = 0; j < numCols; ++j) {
        const int row = j + 1;
        for (int i = 0; i < numRows; ++i) {
            ia[idx] = row;
            ja[idx] = i + 1;
            ar[idx] = payoffMatrix[i][j];
            ++idx;
        }
        ia[idx] = row;
        ja[idx] = vCol;
        ar[idx] = -1.0;
        ++idx;
    }
    const int eqRow = numCols + 1;
    for (int i = 0; i < numRows; ++i) {
        ia[idx] = eqRow;
        ja[idx] = i + 1;
        ar[idx] = 1.0;
        ++idx;
    }

    glp_load_matrix(lp, idx - 1, ia.data(), ja.data(), ar.data());

    glp_smcp params;
    glp_init_smcp(&params);
    params.msg_lev = GLP_MSG_OFF;
    const int ret = glp_simplex(lp, &params);

    StrategyResult result;
    if (ret == 0 && glp_get_status(lp) == GLP_OPT) {
        result.success = true;
        result.expectedValue = glp_get_col_prim(lp, vCol);
        result.probabilities.assign(numRows, 0.0);
        for (int i = 0; i < numRows; ++i) {
            result.probabilities[i] = glp_get_col_prim(lp, i + 1);
        }
    }

    glp_delete_prob(lp);
    return result;
}

int popcount16(CardMask mask) {
    int count = 0;
    while (mask) {
        count += mask & 1u;
        mask = static_cast<CardMask>(mask >> 1);
    }
    return count;
}

int listCards(CardMask mask, std::array<std::uint8_t, kMaxCards>& out) {
    int count = 0;
    for (int card = 1; card <= kMaxCards; ++card) {
        if (mask & static_cast<CardMask>(1u << (card - 1))) {
            out[count++] = static_cast<std::uint8_t>(card);
        }
    }
    return count;
}

std::uint8_t onlyCard(CardMask mask) {
    for (int card = 1; card <= kMaxCards; ++card) {
        if (mask & static_cast<CardMask>(1u << (card - 1))) {
            return static_cast<std::uint8_t>(card);
        }
    }
    return 0;
}

CardMask removeCard(CardMask mask, std::uint8_t card) {
    return static_cast<CardMask>(mask & ~static_cast<CardMask>(1u << (card - 1)));
}

std::string maskToString(CardMask mask) {
    std::ostringstream out;
    out << "[";
    bool first = true;
    for (int card = 1; card <= kMaxCards; ++card) {
        if (mask & static_cast<CardMask>(1u << (card - 1))) {
            if (!first) {
                out << ", ";
            }
            out << card;
            first = false;
        }
    }
    out << "]";
    return out.str();
}

struct State {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;   // remaining prizes as a set/list
    int diff = 0;
    int curP = 0;         // current prize value (1..N)
};

std::string toString(const State& state) {
    std::ostringstream out;
    out << "State{A=" << maskToString(state.A)
        << ", B=" << maskToString(state.B)
        << ", P=" << maskToString(state.P)
        << ", diff=" << state.diff
        << ", curP=" << state.curP
        << "}";
    return out.str();
}

double solveEV(State s);

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

    for (int i=0;i<size;i++) {
        for (int j=0;j<size;j++) {
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
