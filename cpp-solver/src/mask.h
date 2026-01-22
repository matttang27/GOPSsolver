#pragma once

#include <array>
#include <cstdint>
#include <string>

using CardMask = std::uint16_t;
constexpr int kMaxCards = 16;

int popcount16(CardMask mask);
int listCards(CardMask mask, std::array<std::uint8_t, kMaxCards>& out);
std::uint8_t onlyCard(CardMask mask);
CardMask removeCard(CardMask mask, std::uint8_t card);
std::string maskToString(CardMask mask);
