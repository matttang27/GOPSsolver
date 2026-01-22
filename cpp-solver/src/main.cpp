#include <iostream>
#include <vector>
#include <glpk.h>

struct State {
    std::vector<int> A;
    std::vector<int> B;
    std::vector<int> P;   // remaining prizes as a set/list
    int diff = 0;
    int curP = 0;         // current prize value (1..N)
};

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
        return cmp(s.diff + (cmp(s.A[0], s.B[0]) * s.P[0]), 0);
    }
    auto M = buildMatrix(s);

}

int main() {
    std::cout << "HI" << std::endl;
}
