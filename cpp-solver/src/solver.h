#pragma once

#include <vector>

#include "mask.h"

struct State {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;   // remaining prizes as a set/list
    int diff = 0;
    int curP = 0;     // current prize value (1..N)
};

extern long long g_solveEVCalls;
extern long long g_guaranteedWins;
extern long long g_guaranteedDetected;

double solveEV(State s);
std::vector<double> solveProbabilities(State s);
std::vector<std::vector<double>> buildMatrix(const State& s);
State full(int n);
void clearEvCache();
