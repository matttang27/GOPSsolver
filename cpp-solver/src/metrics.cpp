#include "solver.h"

long long g_solveEVCalls = 0;
long long g_guaranteedWins = 0;
long long g_guaranteedDetected = 0;
long long g_buildMatrixCalls = 0;
double g_buildMatrixMaeSum = 0.0;
TimingStats g_timing;
bool g_enableGuarantee = true;
bool g_enableCompression = true;
bool g_enableCache = true;
