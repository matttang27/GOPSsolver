import struct, sys, math, itertools, json, os
BASE = "basetest"
MODIFIERS = ["compress", "guarantee"] 
EPS = 1e-12
LP_MAX_N = 4

def load(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"GOPSEV1\0":
            raise ValueError("bad magic")
        version, reserved = struct.unpack("<II", f.read(8))
        count = struct.unpack("<Q", f.read(8))[0]
        data = {}
        for _ in range(count):
            key, ev = struct.unpack("<Qd", f.read(16))
            data[key] = ev
    return data

def load_meta(path):
    meta_path = path + ".json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def decode_key(key):
    A = key & 0xFFFF
    B = (key >> 16) & 0xFFFF
    P = (key >> 32) & 0xFFFF
    diff = (key >> 48) & 0xFF
    if diff >= 128:
        diff -= 256
    curP = (key >> 56) & 0xFF
    return A, B, P, diff, curP

def encode_key(A, B, P, diff, curP):
    diff_u8 = diff & 0xFF
    return (A & 0xFFFF) | ((B & 0xFFFF) << 16) | ((P & 0xFFFF) << 32) | (diff_u8 << 48) | ((curP & 0xFF) << 56)

def cmp(a, b):
    return (a > b) - (a < b)

def canonicalize(A, B, P, diff, curP):
    sign = 1.0
    if diff < 0:
        A, B = B, A
        diff = -diff
        sign = -sign
    if diff == 0:
        if A < B:
            A, B = B, A
            sign = -sign
    return encode_key(A, B, P, diff, curP), sign

def popcount(mask):
    return mask.bit_count()

def only_card(mask):
    if mask == 0:
        return 0
    lsb = mask & -mask
    return lsb.bit_length()

def list_cards(mask):
    cards = []
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        cards.append(idx + 1)
        mask &= mask - 1
    return cards

def remove_card(mask, card):
    return mask & ~(1 << (card - 1))

def compress_cards(a, b):
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

def get_ev(cache, A, B, P, diff, curP):
    key, sign = canonicalize(A, B, P, diff, curP)
    if key not in cache:
        return None
    return sign * cache[key]

def build_matrix(cache, A, B, P, diff, curP, compress_enabled):
    cardsA = list_cards(A)
    cardsB = list_cards(B)
    prizes = list_cards(P)
    countA = len(cardsA)
    countB = len(cardsB)
    countP = len(prizes)
    if countA != countB or countP != countA - 1:
        return None
    if countP == 0:
        return None
    mat = [[0.0 for _ in range(countA)] for _ in range(countA)]
    for i, cardA in enumerate(cardsA):
        for j, cardB in enumerate(cardsB):
            newA = remove_card(A, cardA)
            newB = remove_card(B, cardB)
            newDiff = diff + cmp(cardA, cardB) * curP
            nextA, nextB = newA, newB
            if compress_enabled:
                nextA, nextB = compress_cards(newA, newB)
            sum_ev = 0.0
            for prize in prizes:
                newP = remove_card(P, prize)
                ev = get_ev(cache, nextA, nextB, newP, newDiff, prize)
                if ev is None:
                    return None
                sum_ev += ev
            mat[i][j] = sum_ev / countP
    return mat

def solve_linear(system, rhs):
    n = len(system)
    aug = [list(system[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = None
        max_val = 0.0
        for row in range(col, n):
            val = abs(aug[row][col])
            if val > max_val:
                max_val = val
                pivot = row
        if pivot is None or max_val < EPS:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < EPS:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]

def solve_game_value(mat):
    n = len(mat)
    best_v = -1e9
    found = False
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            for T in itertools.combinations(range(n), k):
                system = []
                rhs = []
                for col in T:
                    row = [mat[i][col] for i in S] + [-1.0]
                    system.append(row)
                    rhs.append(0.0)
                system.append([1.0] * k + [0.0])
                rhs.append(1.0)
                sol = solve_linear(system, rhs)
                if sol is None:
                    continue
                probs = sol[:-1]
                v = sol[-1]
                if any(p < -EPS for p in probs):
                    continue
                min_payoff = None
                for col in range(n):
                    payoff = 0.0
                    for idx, row_idx in enumerate(S):
                        payoff += probs[idx] * mat[row_idx][col]
                    if min_payoff is None or payoff < min_payoff:
                        min_payoff = payoff
                if min_payoff is None or min_payoff + EPS < v:
                    continue
                if v > best_v:
                    best_v = v
                    found = True
    return best_v if found else None

def compare(a, b):
    common = set(a) & set(b)
    max_diff = 0.0
    mean_diff = 0.0
    for k in common:
        d = abs(a[k] - b[k])
        mean_diff += d
        if d > max_diff:
            max_diff = d
    mean_diff = mean_diff / len(common) if common else 0.0
    return len(a), len(b), len(common), max_diff, mean_diff

baseEVC = load(f"{BASE}.evc")
baseMeta = load_meta(f"{BASE}.evc")
baseCompress = False
if baseMeta:
    baseCompress = baseMeta.get("config", {}).get("toggles", {}).get("compression", False)

for mod in MODIFIERS:
    path = f"{mod}.evc"
    modEVC = load(path)
    sa, sb, sc, maxd, meand = compare(baseEVC, modEVC)
    print("Modifier:", mod)
    print("a size:", sa)
    print("b size:", sb)
    print("common:", sc)
    print("max abs diff:", maxd)
    print("mean abs diff:", meand)

print("\nSanity checks for BASE:", BASE)
# 1) Bounds
bound_violations = 0
max_bound = 0.0
for ev in baseEVC.values():
    if ev < -1.0 - EPS or ev > 1.0 + EPS:
        bound_violations += 1
        max_bound = max(max_bound, max(abs(ev + 1.0), abs(ev - 1.0)))
print("bounds violations:", bound_violations, "max violation:", max_bound)

# 2) Symmetry
sym_missing = 0
sym_max = 0.0
for key, ev in baseEVC.items():
    A, B, P, diff, curP = decode_key(key)
    swapped_key, sign = canonicalize(B, A, P, -diff, curP)
    if swapped_key not in baseEVC:
        sym_missing += 1
        continue
    ev_swapped = sign * baseEVC[swapped_key]
    sym_max = max(sym_max, abs(ev + ev_swapped))
print("symmetry missing:", sym_missing, "max abs diff:", sym_max)

# 3) Base-case check
basecase_missing = 0
basecase_max = 0.0
for key, ev in baseEVC.items():
    A, B, P, diff, curP = decode_key(key)
    if popcount(A) != 1 or popcount(B) != 1:
        continue
    cardA = only_card(A)
    cardB = only_card(B)
    expected = cmp(diff + cmp(cardA, cardB) * curP, 0)
    basecase_max = max(basecase_max, abs(ev - expected))
print("basecase max abs diff:", basecase_max, "missing:", basecase_missing)

# 4) LP consistency on small subgames
lp_checked = 0
lp_missing = 0
lp_max = 0.0
for key, ev in baseEVC.items():
    A, B, P, diff, curP = decode_key(key)
    n = popcount(A)
    if n == 0 or n != popcount(B):
        continue
    if n <= 1:
        continue
    if n > LP_MAX_N:
        continue
    if popcount(P) != n - 1:
        continue
    mat = build_matrix(baseEVC, A, B, P, diff, curP, baseCompress)
    if mat is None:
        lp_missing += 1
        continue
    value = solve_game_value(mat)
    if value is None:
        lp_missing += 1
        continue
    lp_max = max(lp_max, abs(ev - value))
    lp_checked += 1
print("lp checked:", lp_checked, "lp missing:", lp_missing, "lp max abs diff:", lp_max)
