from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_DIR = ROOT / "ai"
REPORTS_DIR = ROOT / "reports"
CACHE_FILENAME = "full8.evc"
CACHE_PATH = REPORTS_DIR / CACHE_FILENAME

ai_path = str(AI_DIR)
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

reports_path = str(REPORTS_DIR)
if reports_path not in sys.path:
    sys.path.insert(0, reports_path)
