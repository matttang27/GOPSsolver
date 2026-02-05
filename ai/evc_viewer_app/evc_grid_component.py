from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "evc_grid"
_evc_grid = components.declare_component("evc_grid", path=str(_COMPONENT_PATH))


def evc_grid(
    *,
    cardsA: List[int],
    cardsB: List[int],
    evs: List[List[float]],
    labels: List[List[str]],
    supportA: List[int],
    supportB: List[int],
    disabled: bool,
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return _evc_grid(
        cardsA=cardsA,
        cardsB=cardsB,
        evs=evs,
        labels=labels,
        supportA=supportA,
        supportB=supportB,
        disabled=disabled,
        key=key,
        default=None,
    )
