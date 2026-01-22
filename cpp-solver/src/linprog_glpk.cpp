#include "linprog_glpk.h"

#include <sstream>
#include <glpk.h>

static std::string vecToString(const std::vector<double>& vec) {
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
