#include "mask.h"

#include <sstream>

static int lowestBitIndex(CardMask mask) {
    if (!mask) {
        return 0;
    }
    int idx = 0;
    while ((mask & 1u) == 0u) {
        mask = static_cast<CardMask>(mask >> 1);
        ++idx;
    }
    return idx;
}

int popcount16(CardMask mask) {
    int count = 0;
    for (; mask; mask = static_cast<CardMask>(mask & static_cast<CardMask>(mask - 1))) {
        ++count;
    }
    return count;
}

int listCards(CardMask mask, std::array<std::uint8_t, kMaxCards>& out) {
    int count = 0;
    while (mask) {
        const int idx = lowestBitIndex(mask);
        out[count++] = static_cast<std::uint8_t>(idx + 1);
        mask = static_cast<CardMask>(mask & static_cast<CardMask>(mask - 1));
    }
    return count;
}

std::uint8_t onlyCard(CardMask mask) {
    return mask ? static_cast<std::uint8_t>(lowestBitIndex(mask) + 1) : 0;
}

CardMask removeCard(CardMask mask, std::uint8_t card) {
    return static_cast<CardMask>(mask & ~static_cast<CardMask>(1u << (card - 1)));
}

std::string maskToString(CardMask mask) {
    std::array<std::uint8_t, kMaxCards> cards;
    const int count = listCards(mask, cards);
    std::ostringstream out;
    out << "[";
    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << static_cast<int>(cards[i]);
    }
    out << "]";
    return out.str();
}

std::pair<CardMask, CardMask> compressCards(CardMask a, CardMask b) {
    CardMask unionMask = static_cast<CardMask>(a | b);
    CardMask compressedA = 0;
    CardMask compressedB = 0;
    int outBit = 0;
    while (unionMask) {
        if (unionMask & 1u) {
            if (a & 1u) {
                compressedA = static_cast<CardMask>(compressedA | static_cast<CardMask>(1u << outBit));
            }
            if (b & 1u) {
                compressedB = static_cast<CardMask>(compressedB | static_cast<CardMask>(1u << outBit));
            }
            ++outBit;
        }
        unionMask = static_cast<CardMask>(unionMask >> 1);
        a = static_cast<CardMask>(a >> 1);
        b = static_cast<CardMask>(b >> 1);
    }
    return {compressedA, compressedB};
}
