from __future__ import annotations

import json
import shutil
import struct
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from tests import test_support  # noqa: F401 - ensures ai/ is on sys.path

from common import cache_objective, encode_key, load_evc


MAGIC = b"GOPSEV1\0"
HEADER = struct.Struct("<IIQ")
TMP_BASE = test_support.ROOT / "tests" / "_tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)


def write_evc(path: Path, version: int, records: list[tuple[int, float]], *, count: int | None = None) -> None:
    if count is None:
        count = len(records)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(HEADER.pack(version, 0, count))
        if version == 1:
            for key, ev in records:
                f.write(struct.pack("<Qd", int(key), float(ev)))
            return
        if version == 2:
            for key, ev in records:
                f.write(struct.pack("<Qf", int(key), float(ev)))
            return
        raise ValueError(f"unsupported test format version: {version}")


@contextmanager
def local_tmp_dir():
    tmp = TMP_BASE / f"tmp_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class TestCommonCacheLoad(unittest.TestCase):
    def test_load_v1_sorts_and_keeps_last_duplicate(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "dup_v1.evc"
            k1 = encode_key(1, 2, 0, 0, 1)
            k2 = encode_key(2, 1, 0, 0, 1)
            records = [
                (k2, 0.2),
                (k1, 0.1),
                (k1, 0.9),
            ]
            write_evc(path, version=1, records=records)

            cache = load_evc(str(path))
            self.assertEqual(len(cache), 2)
            self.assertEqual(set(cache), {k1, k2})
            self.assertAlmostEqual(cache[k1], 0.9)
            self.assertAlmostEqual(cache[k2], 0.2)

    def test_load_v2_and_read_objective_from_metadata(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "points_v2.evc"
            key = encode_key(3, 1, 0, 0, 4)
            write_evc(path, version=2, records=[(key, 1.75)])
            meta = {
                "config": {
                    "objective": "points",
                }
            }
            path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta), encoding="utf-8")

            cache = load_evc(str(path))
            self.assertEqual(len(cache), 1)
            self.assertAlmostEqual(cache[key], 1.75, places=6)
            self.assertEqual(cache_objective(cache), "points")

    def test_bad_magic_raises(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "bad_magic.evc"
            with path.open("wb") as f:
                f.write(b"NOTEVC00")
                f.write(HEADER.pack(1, 0, 0))

            with self.assertRaisesRegex(ValueError, "bad magic"):
                load_evc(str(path))

    def test_unsupported_version_raises(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "bad_ver.evc"
            with path.open("wb") as f:
                f.write(MAGIC)
                f.write(HEADER.pack(99, 0, 0))

            with self.assertRaisesRegex(ValueError, "unsupported cache format version"):
                load_evc(str(path))

    def test_truncated_records_raises(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "truncated.evc"
            key = encode_key(1, 1, 0, 0, 1)
            # Header says count=2, but only one record is written.
            write_evc(path, version=1, records=[(key, 0.0)], count=2)

            with self.assertRaisesRegex(ValueError, "truncated cache"):
                load_evc(str(path))

    def test_contains_and_missing_key_behavior(self) -> None:
        with local_tmp_dir() as tmp:
            path = tmp / "contains.evc"
            key = encode_key(1, 2, 0, 0, 1)
            write_evc(path, version=1, records=[(key, -0.25)])
            cache = load_evc(str(path))

            self.assertIn(key, cache)
            self.assertNotIn(key + 1, cache)
            with self.assertRaises(KeyError):
                _ = cache[key + 1]


if __name__ == "__main__":
    unittest.main()
