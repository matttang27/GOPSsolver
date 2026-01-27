import struct, sys, math
BASE = "basetest"
MODIFIERS = ["compress", "guarantee"] 

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
