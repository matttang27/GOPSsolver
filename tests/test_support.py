from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
ai_path = str(AI_DIR)
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)


def mask(cards: Iterable[int]) -> int:
    out = 0
    for card in cards:
        out |= 1 << (int(card) - 1)
    return out
