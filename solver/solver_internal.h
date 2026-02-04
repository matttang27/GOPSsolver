#pragma once

template<typename T>
inline int cmp(T a, T b) {
    return (a > b) - (a < b);
}
