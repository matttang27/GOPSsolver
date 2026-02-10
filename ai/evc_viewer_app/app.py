from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from common import load_evc

from evc_viewer_app.config import CACHE_PATH
from evc_viewer_app.explore import render_explore_tab
from evc_viewer_app.helpers import apply_pending_transition
from evc_viewer_app.play_ui import render_play_tabs
from evc_viewer_app.sidebar import render_sidebar


@st.cache_resource(show_spinner=False)
def load_cache(path: str) -> Mapping[int, float]:
    return load_evc(path)


def main() -> None:
    st.set_page_config(page_title="GOPS EV Cache Explorer", layout="wide")
    st.title("GOPS EV Cache Explorer")
    display_percent = st.toggle(
        "Show win %",
        value=False,
        key="display_percent",
        help="Win % = 50 + 50*EV",
    )
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
            background-color: #d64545;
            border: 1px solid #b43333;
            color: #fff;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #c53b3b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    apply_pending_transition()

    mode, _run_n = render_sidebar(CACHE_PATH)
    if not CACHE_PATH.exists():
        st.error(f"Cache file not found: {CACHE_PATH}")
        return

    with st.spinner("Loading cache..."):
        cache = load_cache(str(CACHE_PATH))

    max_card = int(st.session_state.max_card)
    card_options = list(range(1, max_card + 1))
    st.session_state.card_options = card_options

    explore_tab, play_tab = st.tabs(["Explore State", "Play / Simulate"])
    with explore_tab:
        render_explore_tab(cache, CACHE_PATH, display_percent, mode, max_card, card_options)
    with play_tab:
        render_play_tabs(cache, max_card)


if __name__ == "__main__":
    main()
