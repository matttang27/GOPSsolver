import json
import os
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

EPS = 1e-12
_MAGIC = b"GOPSEV1\0"
_HEADER_STRUCT = struct.Struct("<IIQ")
_RECORD_DTYPE = np.dtype([("key", "<u8"), ("ev", "<f8")])


@dataclass(frozen=True)
class State:
    A: int
    B: int
    P: int
    diff: int
    curP: int


class EVCCache(Mapping[int, float]):
    """Read-only sorted cache with binary-search lookups."""

    __slots__ = ("_keys", "_values", "_memo", "_memo_limit")

    def __init__(self, keys: np.ndarray, values: np.ndarray, memo_limit: int = 250_000) -> None:
        self._keys = np.ascontiguousarray(keys, dtype=np.uint64)
        self._values = np.ascontiguousarray(values, dtype=np.float64)
        self._memo: dict[int, float] = {}
        self._memo_limit = int(max(memo_limit, 0))

    def __len__(self) -> int:
        return int(self._keys.size)

    def __iter__(self) -> Iterator[int]:
        return (int(key) for key in self._keys)

    def __getitem__(self, key: int) -> float:
        key_int = int(key)
        try:
            return self._memo[key_int]
        except KeyError:
            pass
        idx = self._find_index(key_int)
        if idx < 0:
            raise KeyError(key)
        value = float(self._values[idx])
        if len(self._memo) < self._memo_limit:
            self._memo[key_int] = value
        return value

    def _find_index(self, key: int) -> int:
        key_int = int(key)
        if key_int < 0:
            return -1
        try:
            key_u64 = np.uint64(key_int)
        except OverflowError:
            return -1
        idx = int(np.searchsorted(self._keys, key_u64, side="left"))
        if idx < len(self._keys) and self._keys[idx] == key_u64:
            return idx
        return -1

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (int, np.integer)):
            return False
        key_int = int(key)
        if key_int in self._memo:
            return True
        return self._find_index(key_int) >= 0


def load_evc(path: str) -> Mapping[int, float]:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError("bad magic")
        _version, _reserved, count = _HEADER_STRUCT.unpack(f.read(_HEADER_STRUCT.size))
        records = np.fromfile(f, dtype=_RECORD_DTYPE, count=count)

    if records.size != count:
        raise ValueError(f"truncated cache: expected {count} records, found {records.size}")
    if records.size == 0:
        return {}

    keys = records["key"]
    values = records["ev"]
    if keys.size > 1 and not bool(np.all(keys[:-1] < keys[1:])):
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        values = values[order]
        # Keep last value for duplicate keys (if any).
        keep = np.empty(keys.size, dtype=bool)
        keep[:-1] = keys[:-1] != keys[1:]
        keep[-1] = True
        keys = keys[keep]
        values = values[keep]

    return EVCCache(keys, values)


def build_matrix(cache: Mapping[int, float],
                 A: int,
                 B: int,
                 P: int,
                 diff: int,
                 curP: int) -> List[List[float]]:

    cardsA = list_cards(A)
    cardsB = list_cards(B)
    prizes = list_cards(P)
    countA = len(cardsA)
    countB = len(cardsB)
    countP = len(prizes)
    #print(cardsA, cardsB, prizes)
    if countA != countB or countP != countA - 1:
        return []
    if countP == 0:
        # Terminal round: value is determined solely by the current prize.
        mat = [[0.0 for _ in range(countA)] for _ in range(countA)]
        for i, cardA in enumerate(cardsA):
            for j, cardB in enumerate(cardsB):
                newDiff = diff + cmp(cardA, cardB) * curP
                mat[i][j] = float(cmp(newDiff, 0))
        return mat
    mat = [[0.0 for _ in range(countA)] for _ in range(countA)]
    for i, cardA in enumerate(cardsA):
        for j, cardB in enumerate(cardsB):
            newA = remove_card(A, cardA)
            newB = remove_card(B, cardB)
            newDiff = diff + cmp(cardA, cardB) * curP
            sum_ev = 0.0
            for prize in prizes:
                newP = remove_card(P, prize)
                try:
                    ev = get_ev(cache, newA, newB, newP, newDiff, prize)
                except KeyError:
                    print(f"Missing state in cache: A={list_cards(newA)} B={list_cards(newB)} P={list_cards(newP)} diff={newDiff} curP={prize}")
                    return []
                sum_ev += ev
            mat[i][j] = sum_ev / countP
    return mat


def load_meta(path: str):
    meta_path = path + ".json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_key(key: int) -> Tuple[int, int, int, int, int]:
    key = int(key)
    A = key & 0xFFFF
    B = (key >> 16) & 0xFFFF
    P = (key >> 32) & 0xFFFF
    diff = (key >> 48) & 0xFF
    if diff >= 128:
        diff -= 256
    curP = (key >> 56) & 0xFF
    return A, B, P, diff, curP


def encode_key(A: int, B: int, P: int, diff: int, curP: int) -> int:
    diff_u8 = diff & 0xFF
    return (A & 0xFFFF) | ((B & 0xFFFF) << 16) | ((P & 0xFFFF) << 32) | (diff_u8 << 48) | ((curP & 0xFF) << 56)


def state_from_key(key: int) -> State:
    A, B, P, diff, curP = decode_key(key)
    return State(A=A, B=B, P=P, diff=diff, curP=curP)


def state_to_key(state: State) -> int:
    return encode_key(state.A, state.B, state.P, state.diff, state.curP)


def canonicalize(A: int, B: int, P: int, diff: int, curP: int) -> Tuple[int, float]:
    sign = 1.0
    if diff < 0:
        A, B = B, A
        diff = -diff
        sign = -sign
    if diff == 0 and A < B:
        A, B = B, A
        sign = -sign
    A, B = compress_cards(A, B)
    return encode_key(A, B, P, diff, curP), sign


def canonicalize_state(state: State) -> Tuple[int, float]:
    return canonicalize(state.A, state.B, state.P, state.diff, state.curP)


def get_ev(cache: Mapping[int, float], A: int, B: int, P: int, diff: int, curP: int) -> float:
    key, sign = canonicalize(A, B, P, diff, curP)
    # Avoid double lookup for non-dict cache backends.
    try:
        ev = cache[key]
    except KeyError:
        if (A == B and diff == 0):
            return 0.0
        if guaranteed(list_cards(A), list_cards(B), diff, list_cards(P)) != 0:
            return sign * cmp(diff, 0)
        
        raise KeyError(f"state not in cache: A={A} B={B} P={P} diff={diff} curP={curP}")
    return sign * ev


def get_ev_state(cache: Mapping[int, float], state: State) -> float:
    return get_ev(cache, state.A, state.B, state.P, state.diff, state.curP)

def list_to_mask(cards: List[int]) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << (card - 1)
    return mask

def list_cards(mask: int) -> List[int]:
    cards = []
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        cards.append(idx + 1)
        mask &= mask - 1
    return cards


def remove_card(mask: int, card: int) -> int:
    return mask & ~(1 << (card - 1))


def popcount(mask: int) -> int:
    return mask.bit_count()


def only_card(mask: int) -> int:
    if mask == 0:
        return 0
    lsb = mask & -mask
    return lsb.bit_length()


def cmp(a: int, b: int) -> int:
    return (a > b) - (a < b)


def compress_cards(a: int, b: int) -> Tuple[int, int]:
    union = a | b
    out_bit = 0
    comp_a = 0
    comp_b = 0
    while union:
        if union & 1:
            if a & 1:
                comp_a |= 1 << out_bit
            if b & 1:
                comp_b |= 1 << out_bit
            out_bit += 1
        union >>= 1
        a >>= 1
        b >>= 1
    return comp_a, comp_b


def guaranteed(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, prizes: tuple[int, ...]) -> int:
    """
    Check if one side has enough cards higher than the other to guarantee a win.
    """
    cardsLeft = len(prizes)
    sorted_prizes = sorted(prizes, reverse=True)
    guarantee = [sum(sorted_prizes[:i]) - sum(sorted_prizes[i:]) for i in range(cardsLeft + 1)]
    
    guaranteeA = sum(1 for card in cardsA if card > cardsB[-1])
    guaranteeB = sum(1 for card in cardsB if card > cardsA[-1])
    if guaranteeA > cardsLeft:
        guaranteeA = cardsLeft
    if guaranteeB > cardsLeft:
        guaranteeB = cardsLeft
    
    if (guarantee[guaranteeA] + pointDiff) > 0:
        return 1
    elif (pointDiff - guarantee[guaranteeB]) < 0:
        return -1
    else:
        return 0
