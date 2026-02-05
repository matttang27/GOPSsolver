from __future__ import annotations

from typing import List

import streamlit as st


def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def ensure_state_init(card_options: List[int], max_card: int) -> None:
    # Be robust to partial session_state (e.g., after refactors or when keys are cleared).
    if "cardsA" not in st.session_state:
        st.session_state.cardsA = card_options[:]
    if "cardsB" not in st.session_state:
        st.session_state.cardsB = card_options[:]
    if "curP" not in st.session_state:
        st.session_state.curP = card_options[0] if card_options else 1
    if "cardsP" not in st.session_state:
        st.session_state.cardsP = [c for c in card_options if c != int(st.session_state.curP)]
    if "diff" not in st.session_state:
        st.session_state.diff = 0
    if "cardsA_n" not in st.session_state:
        st.session_state.cardsA_n = max_card
    if "curP_prev" not in st.session_state:
        st.session_state.curP_prev = int(st.session_state.curP)

    if st.session_state.get("cardsA_n") != max_card:
        st.session_state.cardsA = [c for c in st.session_state.cardsA if c in card_options] or card_options[:]
        st.session_state.cardsB = [c for c in st.session_state.cardsB if c in card_options] or card_options[:]
        if int(st.session_state.curP) not in card_options:
            st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [
            c for c in st.session_state.cardsP if c in card_options and c != int(st.session_state.curP)
        ]
        if not st.session_state.cardsP:
            st.session_state.cardsP = [c for c in card_options if c != int(st.session_state.curP)]
        st.session_state.cardsA_n = max_card
        st.session_state.curP_prev = int(st.session_state.curP)

