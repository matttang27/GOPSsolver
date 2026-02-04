#pragma once

#include <string>
#include <vector>

struct StrategyResult {
    bool success = false;
    double expectedValue = 0.0;
    std::vector<double> probabilities;
};

StrategyResult findBestStrategyGlpk(const std::vector<std::vector<double>>& payoffMatrix);
std::string toString(const StrategyResult& result);
