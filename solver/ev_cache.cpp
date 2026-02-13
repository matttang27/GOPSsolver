#include "ev_cache.h"
#include "solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <glpk.h>

static const std::uint32_t kEvcFormatVersion = 1;

bool StateKey::operator==(const StateKey& other) const {
    return A == other.A && B == other.B && P == other.P
        && diff == other.diff && curP == other.curP;
}

std::size_t StateKeyHash::operator()(const StateKey& key) const {
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

static std::unordered_map<StateKey, double, StateKeyHash> g_evCache;

StateKey makeStateKey(const State& state) {
    return StateKey{
        state.A,
        state.B,
        state.P,
        static_cast<std::int8_t>(state.diff),
        static_cast<std::uint8_t>(state.curP)
    };
}

bool evCacheFind(const StateKey& key, double& value) {
    auto it = g_evCache.find(key);
    if (it == g_evCache.end()) {
        return false;
    }
    value = it->second;
    return true;
}

void evCacheStore(const StateKey& key, double value) {
    g_evCache.emplace(key, value);
}

void clearEvCache() {
    g_evCache.clear();
}

std::size_t evCacheSize() {
    return g_evCache.size();
}

static std::uint64_t packStateKey(const StateKey& key) {
    std::uint64_t packed = 0;
    packed |= static_cast<std::uint64_t>(key.A);
    packed |= static_cast<std::uint64_t>(key.B) << 16;
    packed |= static_cast<std::uint64_t>(key.P) << 32;
    packed |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(key.diff)) << 48;
    packed |= static_cast<std::uint64_t>(key.curP) << 56;
    return packed;
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

    // Stream records directly from the in-memory cache to avoid allocating
    // a second full-size copy during serialization.
    for (const auto& entry : g_evCache) {
        const std::uint64_t packedKey = packStateKey(entry.first);
        const double value = entry.second;
        out.write(reinterpret_cast<const char*>(&packedKey), sizeof(packedKey));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    if (!out) {
        std::cerr << "Failed while writing cache output file: " << path << std::endl;
        return false;
    }
    return true;
}

bool saveEvCacheMetadata(const std::string& evcPath,
                         const std::string& args,
                         int n,
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
    out << "    \"objective\": \"" << objectiveName(g_solveObjective) << "\",\n";
    out << "    \"key_packing\": \"A16|B16|P16|diff8|curP8\",\n";
    out << "    \"key_notes\": \"For objective=points, diff is always 0 and ignored.\",\n";
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
    out << "    \"N\": " << n << ",\n";
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
