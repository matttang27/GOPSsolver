#include <iostream>
#include <sstream>
#include <vector>
#include <glpk.h>

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

struct State {
    std::vector<int> A;
    std::vector<int> B;
    std::vector<int> P;   // remaining prizes as a set/list
    int diff = 0;
    int curP = 0;         // current prize value (1..N)
};

std::string toString(const State& state) {
    std::ostringstream out;
    out << "State{A=" << vecToString(state.A)
        << ", B=" << vecToString(state.B)
        << ", P=" << vecToString(state.P)
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

std::vector<int> newPop(const std::vector<int>& vec, int idx) {
    std::vector<int> res = vec;
    res.erase(res.begin() + idx);
    return res;
}

std::vector<std::vector<double>> buildMatrix(const State& s) {
    int size = s.A.size();
    std::vector<std::vector<double>> mat(size, std::vector<double>(size, 0.0));

    for (int i=0;i<size;i++) {
        for (int j=0;j<size;j++) {
            auto newA = newPop(s.A, i);
            auto newB = newPop(s.B, j);
            int newDiff = s.diff + cmp((s.A[i] - s.B[j]),0) * s.curP;
            double sumEV = 0.0;
            for (int k = 0; k < size - 1; k++) {
                auto newRemaining = newPop(s.P, k);
                State newState{newA, newB, newRemaining, newDiff, s.P[k]};
                sumEV += solveEV(newState);
            }
            mat[i][j] = sumEV / (size - 1);
        }
    }
    return mat;
}
double solveEV(State s) {
    if (s.A.size() == 1) {
        return cmp(s.diff + (cmp(s.A[0], s.B[0]) * s.curP), 0);
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
    if (s.A.size() == 1) {
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

int main() {
    State initial{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4}, 0, 5};
    auto probabilities = solveProbabilities(initial);
    std::cout << "Action profile (" << probabilities.size() << "): "
              << vecToString(probabilities) << std::endl;
}
