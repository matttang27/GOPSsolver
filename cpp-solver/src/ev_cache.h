#pragma once

#include <cstddef>
#include <cstdint>

#include "mask.h"

struct State;

struct StateKey {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;
    std::int8_t diff = 0;
    std::uint8_t curP = 0;

    bool operator==(const StateKey& other) const;
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const;
};

StateKey makeStateKey(const State& state);
bool evCacheFind(const StateKey& key, double& value);
void evCacheStore(const StateKey& key, double value);
std::size_t evCacheSize();
