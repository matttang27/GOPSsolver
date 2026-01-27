#include "solver.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "linprog_glpk.h"
#include <glpk.h>

long long g_solveEVCalls = 0;
long long g_guaranteedWins = 0;
long long g_guaranteedDetected = 0;
long long g_buildMatrixCalls = 0;
double g_buildMatrixMaeSum = 0.0;
TimingStats g_timing;
bool g_enableGuarantee = true;
bool g_enableCompression = true;
bool g_enableCache = true;

struct StateKey {
    CardMask A = 0;
    CardMask B = 0;
    CardMask P = 0;
    std::int8_t diff = 0;
    std::uint8_t curP = 0;

    bool operator==(const StateKey& other) const {
        return A == other.A && B == other.B && P == other.P
            && diff == other.diff && curP == other.curP;
    }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const {
        std::size_t h = 0;
        auto mix = [&h](std::size_t v) {
            h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::size_t>(key.A));
        mix(static_cast<std::size_t>(key.B));
        mix(static_cast<std::size_t>(key.P));
        mix(static_cast<std::size_t>(static_cast<std::uint32_t>(key.diff)));
        mix(static_cast<std::size_t>(key.curP));
        return h;
    }
};

static std::unordered_map<StateKey, double, StateKeyHash> g_evCache;
static const std::uint32_t kEvcFormatVersion = 1;

static std::uint64_t packStateKey(const StateKey& key) {
    std::uint64_t packed = 0;
    packed |= static_cast<std::uint64_t>(key.A);
    packed |= static_cast<std::uint64_t>(key.B) << 16;
    packed |= static_cast<std::uint64_t>(key.P) << 32;
    packed |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(key.diff)) << 48;
    packed |= static_cast<std::uint64_t>(key.curP) << 56;
    return packed;
}

static std::string toString(const State& state) {
    std::ostringstream out;
    out << "State{A=" << maskToString(state.A)
        << ", B=" << maskToString(state.B)
        << ", P=" << maskToString(state.P)
        << ", diff=" << state.diff
        << ", curP=" << state.curP
        << "}";
    return out.str();
}

template<typename T>
int cmp(T a, T b) {
    return (a > b) - (a < b);
}

static std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u00" << std::hex << std::uppercase
                        << std::setw(2) << std::setfill('0')
                        << static_cast<int>(c)
                        << std::dec << std::nouppercase;
                } else {
                    out << c;
                }
                break;
        }
    }
    return out.str();
}

static std::string currentUtcIso8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static std::string compilerString() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "";
#endif
}

static std::string buildTypeString() {
#if defined(NDEBUG)
    return "Release";
#else
    return "Debug";
#endif
}

static std::string osString() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "";
#endif
}

static std::string endiannessString() {
    const std::uint16_t test = 1;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&test);
    return bytes[0] == 1 ? "little" : "big";
}

static int highestCard(CardMask mask) {
    for (int card = kMaxCards; card >= 1; --card) {
        if (mask & static_cast<CardMask>(1u << (card - 1))) {
            return card;
        }
    }
    return 0;
}

static int countAbove(CardMask mask, int threshold) {
    if (threshold <= 0) {
        return popcount16(mask);
    }
    if (threshold >= kMaxCards) {
        return 0;
    }
    CardMask higher = static_cast<CardMask>(mask & ~static_cast<CardMask>((1u << threshold) - 1u));
    return popcount16(higher);
}

static int guaranteedOutcome(const State& s) {
    CardMask prizeMask = s.P;
    if (s.curP > 0) {
        prizeMask = static_cast<CardMask>(prizeMask | static_cast<CardMask>(1u << (s.curP - 1)));
    }
    std::array<int, kMaxCards> sortedPrizes;
    int prizeCount = 0;
    int total = 0;
    for (int card = kMaxCards; card >= 1; --card) {
        if (prizeMask & static_cast<CardMask>(1u << (card - 1))) {
            sortedPrizes[prizeCount++] = card;
            total += card;
        }
    }
    if (prizeCount == 0) {
        return 0;
    }
    std::array<int, kMaxCards + 1> guarantee;
    guarantee[0] = -total;
    int sumTop = 0;
    for (int i = 1; i <= prizeCount; ++i) {
        sumTop += sortedPrizes[i - 1];
        guarantee[i] = 2 * sumTop - total;
    }

    const int maxA = highestCard(s.A);
    const int maxB = highestCard(s.B);
    const int guaranteeA = countAbove(s.A, maxB);
    const int guaranteeB = countAbove(s.B, maxA);

    if (guarantee[guaranteeA] + s.diff > 0) {
        return 1;
    }
    if (s.diff - guarantee[guaranteeB] < 0) {
        return -1;
    }
    return 0;
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
    double matrixMaeSum = 0.0;
    long long matrixMaeCount = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::uint8_t cardA = cardsA[i];
            std::uint8_t cardB = cardsB[j];
            auto newA = removeCard(s.A, cardA);
            auto newB = removeCard(s.B, cardB);
            int newDiff = s.diff + cmp(cardA, cardB) * s.curP;
            CardMask nextA = newA;
            CardMask nextB = newB;
            if (g_enableCompression) {
                auto compressed = compressCards(newA, newB);
                nextA = compressed.first;
                nextB = compressed.second;
            }
            double sumEV = 0.0;
            std::array<double, kMaxCards> prizeEvs;
            for (int k = 0; k < countP; k++) {
                std::uint8_t nextPrize = prizes[k];
                auto newRemaining = removeCard(s.P, nextPrize);
                State newState{nextA, nextB, newRemaining, newDiff, nextPrize};
                double ev = solveEV(newState);
                prizeEvs[k] = ev;
                sumEV += ev;
            }
            double avg = sumEV / countP;
            double mae = 0.0;
            for (int k = 0; k < countP; k++) {
                double diff = prizeEvs[k] - avg;
                if (diff < 0.0) {
                    diff = -diff;
                }
                mae += diff;
            }
            mae /= countP;
            matrixMaeSum += mae;
            ++matrixMaeCount;
            mat[i][j] = avg;
        }
    }
    ++g_buildMatrixCalls;
    if (matrixMaeCount > 0) {
        g_buildMatrixMaeSum += matrixMaeSum / matrixMaeCount;
    }
    return mat;
}

double solveEV(State s) {
    StateKey key{s.A, s.B, s.P, s.diff, static_cast<std::uint8_t>(s.curP)};
    if (s.diff < 0 && true) {
        return -solveEV(State{s.B, s.A, s.P, -s.diff, s.curP});
    }
    //Both only reduce states by around 3% each
    if (s.diff == 0) {
        int cmpAB = cmp(s.A, s.B);
        if (cmpAB < 0) {
            return -solveEV(State{s.B, s.A, s.P, -s.diff, s.curP});
        } else if (cmpAB == 0) {
            return 0.0;
        }
    }

    if (g_enableCache) {
        auto cacheStart = std::chrono::steady_clock::now();
        auto cached = g_evCache.find(key);
        auto cacheEnd = std::chrono::steady_clock::now();
        g_timing.cacheNs += std::chrono::duration_cast<std::chrono::nanoseconds>(cacheEnd - cacheStart).count();
        if (cached != g_evCache.end()) {
            ++g_timing.cacheHits;
            return cached->second;
        }
        ++g_timing.cacheMisses;
    }
    ++g_solveEVCalls;
    if (g_enableGuarantee) {
        auto guaranteeStart = std::chrono::steady_clock::now();
        int guaranteed = guaranteedOutcome(s);
        auto guaranteeEnd = std::chrono::steady_clock::now();
        g_timing.guaranteeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(guaranteeEnd - guaranteeStart).count();
        ++g_timing.guaranteeCalls;
        if (guaranteed != 0) {
            ++g_guaranteedDetected;
            ++g_guaranteedWins;
            if (g_enableCache) {
                g_evCache.emplace(key, guaranteed);
            }
            return static_cast<double>(guaranteed);
        }
    }
    if (popcount16(s.A) == 1) {
        std::uint8_t cardA = onlyCard(s.A);
        std::uint8_t cardB = onlyCard(s.B);
        double value = cmp(s.diff + (cmp(cardA, cardB) * s.curP), 0);
        if (value == 1.0 || value == -1.0) {
            ++g_guaranteedWins;
        }
        if (g_enableCache) {
            g_evCache.emplace(key, value);
        }
        return value;
    }
    auto M = buildMatrix(s);
    auto lpStart = std::chrono::steady_clock::now();
    auto result = findBestStrategyGlpk(M);
    auto lpEnd = std::chrono::steady_clock::now();
    g_timing.lpNs += std::chrono::duration_cast<std::chrono::nanoseconds>(lpEnd - lpStart).count();
    ++g_timing.lpCalls;
    if (result.success) {
        if (result.expectedValue >= 1.0 - 1e-10 || result.expectedValue <= -1.0 + 1e-10) {
            ++g_guaranteedWins;
        }
        if (g_enableCache) {
            g_evCache.emplace(key, result.expectedValue);
        }
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

void clearEvCache() {
    g_evCache.clear();
}

bool saveEvCache(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open cache output file: " << path << std::endl;
        return false;
    }
    const char magic[8] = {'G', 'O', 'P', 'S', 'E', 'V', '1', '\0'};
    std::uint32_t version = kEvcFormatVersion;
    std::uint32_t reserved = 0;
    std::uint64_t count = static_cast<std::uint64_t>(g_evCache.size());
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    std::vector<std::pair<std::uint64_t, double>> entries;
    entries.reserve(g_evCache.size());
    for (const auto& entry : g_evCache) {
        entries.emplace_back(packStateKey(entry.first), entry.second);
    }
    std::sort(entries.begin(), entries.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (const auto& entry : entries) {
        out.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        out.write(reinterpret_cast<const char*>(&entry.second), sizeof(entry.second));
    }
    if (!out) {
        std::cerr << "Failed while writing cache output file: " << path << std::endl;
        return false;
    }
    return true;
}

bool saveEvCacheMetadata(const std::string& evcPath,
                         const std::string& args,
                         int minN,
                         int maxN,
                         long long durationMs) {
    std::string path = evcPath + ".json";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open cache metadata file: " << path << std::endl;
        return false;
    }

    double minEv = 0.0;
    double maxEv = 0.0;
    double meanEv = 0.0;
    double m2 = 0.0;
    std::size_t count = 0;
    for (const auto& entry : g_evCache) {
        double v = entry.second;
        if (count == 0) {
            minEv = v;
            maxEv = v;
        } else {
            minEv = std::min(minEv, v);
            maxEv = std::max(maxEv, v);
        }
        ++count;
        double delta = v - meanEv;
        meanEv += delta / static_cast<double>(count);
        m2 += delta * (v - meanEv);
    }
    double stdevEv = 0.0;
    if (count > 1) {
        stdevEv = std::sqrt(m2 / static_cast<double>(count - 1));
    }
    double matrixMaeAvg = g_buildMatrixCalls == 0 ? 0.0 : g_buildMatrixMaeSum / g_buildMatrixCalls;

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"evc_file\": \"" << jsonEscape(evcPath) << "\",\n";
    out << "  \"evc_format_version\": " << kEvcFormatVersion << ",\n";
    out << "  \"created_at_utc\": \"" << jsonEscape(currentUtcIso8601()) << "\",\n";
    out << "  \"notes\": \"\",\n";
    out << "\n";
    out << "  \"build\": {\n";
    out << "    \"git_commit\": \"\",\n";
    out << "    \"build_type\": \"" << jsonEscape(buildTypeString()) << "\",\n";
    out << "    \"compiler\": \"" << jsonEscape(compilerString()) << "\",\n";
    out << "    \"glpk_version\": \"" << jsonEscape(glp_version()) << "\"\n";
    out << "  },\n";
    out << "\n";
    out << "  \"config\": {\n";
    out << "    \"kMaxCards\": " << kMaxCards << ",\n";
    out << "    \"key_packing\": \"A16|B16|P16|diff8|curP8\",\n";
    out << "    \"endianness\": \"" << jsonEscape(endiannessString()) << "\",\n";
    out << "    \"float_format\": \"IEEE754 double\",\n";
    out << "    \"toggles\": {\n";
    out << "      \"cache\": " << (g_enableCache ? "true" : "false") << ",\n";
    out << "      \"guarantee\": " << (g_enableGuarantee ? "true" : "false") << ",\n";
    out << "      \"compression\": " << (g_enableCompression ? "true" : "false") << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "\n";
    out << "  \"run\": {\n";
    out << "    \"minN\": " << minN << ",\n";
    out << "    \"maxN\": " << maxN << ",\n";
    out << "    \"duration_ms\": " << durationMs << ",\n";
    out << "    \"args\": \"" << jsonEscape(args) << "\",\n";
    out << "    \"host\": { \"os\": \"" << jsonEscape(osString())
        << "\", \"cpu\": \"\", \"ram_gb\": null }\n";
    out << "  },\n";
    out << "\n";
    out << "  \"stats\": {\n";
    out << "    \"record_count\": " << static_cast<std::uint64_t>(g_evCache.size()) << ",\n";
    out << "    \"ev_min\": " << minEv << ",\n";
    out << "    \"ev_max\": " << maxEv << ",\n";
    out << "    \"ev_mean\": " << meanEv << ",\n";
    out << "    \"ev_stdev\": " << stdevEv << ",\n";
    out << "    \"guaranteed_wins\": " << g_guaranteedWins << ",\n";
    out << "    \"guaranteed_detected\": " << g_guaranteedDetected << ",\n";
    out << "    \"solveEV_calls\": " << g_solveEVCalls << ",\n";
    out << "    \"matrix_mae_avg\": " << matrixMaeAvg << ",\n";
    out << "    \"cache_hits\": " << g_timing.cacheHits << ",\n";
    out << "    \"cache_misses\": " << g_timing.cacheMisses << "\n";
    out << "  }\n";
    out << "}\n";

    if (!out) {
        std::cerr << "Failed while writing cache metadata file: " << path << std::endl;
        return false;
    }
    return true;
}

void resetTiming() {
    g_timing = TimingStats{};
}
