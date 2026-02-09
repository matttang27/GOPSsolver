from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from common import load_meta
from evc_viewer_app.helpers import build_start_state
from evc_viewer_app.session import trigger_rerun


def get_mode_help(mode: str) -> str:
    if mode == "Manual":
        return "Pick the hands/prizes directly and explore specific states."
    if mode == "Sample":
        return "Browse a random sample of cached states."
    return ""


def render_sidebar(cache_path) -> Tuple[str, Optional[int]]:
    with st.sidebar:
        st.header("Cache")
        st.caption(f"Using reports/{cache_path.name}")
        meta = load_meta(str(cache_path)) if cache_path.exists() else None
        if meta:
            run_n = meta.get("run", {}).get("N")
            record_count = meta.get("stats", {}).get("record_count")
            created = meta.get("created_at_utc")
            if run_n is not None:
                st.caption(f"N = {run_n}")
            if record_count is not None:
                st.caption(f"records: {record_count}")
            if created:
                st.caption(f"created: {created}")
        else:
            run_n = None
            if not cache_path.exists():
                st.error("Cache file not found: reports/full8.evc")

        st.divider()
        st.header("Setup")
        if "max_card" not in st.session_state:
            st.session_state.max_card = int(run_n) if run_n is not None else 9
        st.number_input(
            "Max card value (N)",
            min_value=1,
            max_value=16,
            value=int(st.session_state.max_card),
            key="max_card",
        )
        mode = st.radio(
            "Mode",
            ["Manual", "Sample"],
            horizontal=True,
            key="mode",
            help="Manual: pick hands/prizes. Sample: browse random cached states.",
        )
        st.caption(get_mode_help(mode))
        if st.button("Reset to start state", type="primary"):
            st.session_state.pending_transition = build_start_state(int(st.session_state.max_card))
            trigger_rerun()

    return mode, run_n
