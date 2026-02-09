from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    build_matrix,
    cmp,
    encode_key,
    list_to_mask,
)
from linprog import findBestStrategy

from evc_viewer_app.evc_grid_component import evc_grid
from evc_viewer_app.helpers import (
    decode_key_state,
    reservoir_sample_filtered,
    resolve_next_prize,
    sync_current_prize,
)
from evc_viewer_app.session import ensure_state_init, trigger_rerun
from evc_viewer_app.state_types import ss
from evc_viewer_app.ui import (
    build_strategy_table,
    display_value,
    render_copy_button,
    render_section_header,
    render_state_summary,
    render_strategy_panel,
)


def render_explore_tab(
    cache: Mapping[int, float],
    cache_path,
    display_percent: bool,
    mode: str,
    max_card: int,
    card_options: List[int],
) -> None:
    st.subheader("State Input")
    ensure_state_init(card_options, max_card)
    S = ss()

    def _toggle_list_value(state_key: str, card: int) -> None:
        cur = st.session_state.get(state_key, [])
        cur_set = set(int(x) for x in cur)
        if card in cur_set:
            cur_set.remove(card)
        else:
            cur_set.add(card)
        st.session_state[state_key] = sorted(cur_set)

    def _toggle_button_grid(
        *,
        title: str,
        options: List[int],
        state_key: str,
        disabled: Optional[set[int]] = None,
        cols: int = 8,
    ) -> List[int]:
        disabled = disabled or set()
        st.markdown(f"**{title}**")
        selected_set = set(int(x) for x in st.session_state.get(state_key, []))
        if not options:
            st.session_state[state_key] = []
            return []
        per_row = max(1, min(cols, len(options)))
        for row_start in range(0, len(options), per_row):
            row = options[row_start : row_start + per_row]
            row_cols = st.columns(per_row, gap="small")
            for col, card in zip(row_cols, row):
                with col:
                    if st.button(
                        str(card),
                        key=f"{state_key}_btn_{card}",
                        type="primary" if card in selected_set else "secondary",
                        disabled=card in disabled,
                        use_container_width=True,
                        on_click=_toggle_list_value,
                        args=(state_key, int(card)),
                    ):
                        # on_click handles mutation; this branch is just here to satisfy mypy/linters.
                        pass
        # If on_click fired, Streamlit has already updated st.session_state[state_key] before this render.
        return sorted(int(x) for x in st.session_state.get(state_key, []))

    def _set_current_prize(card: int) -> None:
        st.session_state.curP = int(card)
        sync_current_prize()

    def _current_prize_picker(*, options: List[int]) -> int:
        st.markdown("**Current prize**")
        if not options:
            return 0
        per_row = max(1, min(8, len(options)))
        cur = int(st.session_state.get("curP", options[0]))
        for row_start in range(0, len(options), per_row):
            row = options[row_start : row_start + per_row]
            row_cols = st.columns(per_row, gap="small")
            for col, card in zip(row_cols, row):
                with col:
                    st.button(
                        str(card),
                        key=f"curP_btn_{card}",
                        type="primary" if card == cur else "secondary",
                        use_container_width=True,
                        on_click=_set_current_prize,
                        args=(int(card),),
                    )
        return int(st.session_state.get("curP", cur))

    cardsA: List[int] = []
    cardsB: List[int] = []
    cardsP: List[int] = []
    diff = 0
    curP = 0

    if mode == "Manual":
        col1, col2, col3 = st.columns(3)
        with col1:
            cardsA = _toggle_button_grid(
                title="Player A cards",
                options=card_options,
                state_key="cardsA",
            )
        with col2:
            curP = _current_prize_picker(options=card_options)
            cardsP = _toggle_button_grid(
                title="Remaining prizes (exclude current)",
                options=card_options,
                state_key="cardsP",
                disabled={int(S.curP)},
            )
        with col1:
            cardsB = _toggle_button_grid(
                title="Player B cards",
                options=card_options,
                state_key="cardsB",
            )
        with col3:
            diff = int(
                st.number_input(
                    "Point diff (A - B)",
                    value=int(S.diff),
                    step=1,
                    key="diff",
                )
            )
        # Persist toggle selections back into the canonical session state.
        S.cardsA = cardsA
        S.cardsB = cardsB
        S.cardsP = [c for c in cardsP if c != int(S.curP)]

    elif mode == "Sample":
        if "sample_keys" not in st.session_state:
            S.sample_keys = []
        if "sample_params" not in st.session_state:
            S.sample_params = None
        rng_seed = st.number_input("Sample RNG seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
        if "sample_cards" in st.session_state and S.sample_cards > max_card:
            S.sample_cards = max_card
        sample_cards = st.number_input(
            "Cards per player (N)",
            min_value=1,
            max_value=max_card,
            value=max_card,
            step=1,
            key="sample_cards",
        )
        sample_size = st.number_input("Sample size", min_value=1, max_value=5000, value=200, step=1)
        params = (
            int(rng_seed),
            int(sample_cards),
            int(sample_size),
            str(cache_path),
        )
        if S.sample_params != params:
            rng = random.Random(int(rng_seed))
            with st.spinner("Sampling cache keys..."):
                sample, matched = reservoir_sample_filtered(
                    cache.keys(),
                    int(sample_size),
                    rng,
                    cards_per_player=int(sample_cards),
                )
                S.sample_keys = sample
                S.sample_matched = matched
                S.sample_params = params
        if not S.sample_keys:
            st.info("Build a sample list to browse random states.")
            return
        matched = S.get("sample_matched")
        if matched is not None:
            st.caption(f"Matched states: {matched}")
        key_val = st.selectbox("Sampled state key", S.sample_keys)
        cardsA, cardsB, cardsP, diff, curP = decode_key_state(int(key_val))
        render_state_summary(cardsA, cardsB, cardsP, diff, curP)
    else:
        st.error(f"Unsupported mode: {mode}")
        return

    cardsA = sorted(cardsA)
    cardsB = sorted(cardsB)
    cardsP = sorted(cardsP)

    errors = []
    if not cardsA or not cardsB:
        errors.append("A and B must have at least one card.")
    if len(cardsA) != len(cardsB):
        errors.append("A and B must have the same number of cards.")
    if len(cardsP) != max(len(cardsA) - 1, 0):
        errors.append("P must have exactly |A| - 1 cards.")
    if curP in cardsP:
        errors.append("Current prize must not appear in remaining prizes.")

    if errors:
        for err in errors:
            st.error(err)
        return

    A_mask = list_to_mask(cardsA)
    B_mask = list_to_mask(cardsB)
    P_mask = list_to_mask(cardsP)
    state_key = encode_key(A_mask, B_mask, P_mask, diff, curP)

    # Key + copy controls live with the rest of the state inputs.
    key_col, apply_col, copy_col = st.columns([6, 1, 1], vertical_alignment="bottom")
    with key_col:
        key_text = st.text_input("Key", value=str(state_key), key="state_key_text")
    with apply_col:
        apply_clicked = st.button("Apply", help="Decode the key and switch to Manual mode.")
    with copy_col:
        copy_payload = json.dumps(
            {
                "cardsA": cardsA,
                "cardsB": cardsB,
                "cardsP": cardsP,
                "curP": curP,
                "diff": diff,
                "key": state_key,
            },
            separators=(",", ":"),
        )
        render_copy_button(copy_payload, "Copy state")

    if apply_clicked:
        try:
            key_val = int(key_text.strip(), 0)
        except Exception:
            st.error("Invalid key. Use decimal or 0x-prefixed hex.")
        else:
            a2, b2, p2, diff2, curP2 = decode_key_state(int(key_val))
            max_from_key = max(a2 + b2 + p2 + ([curP2] if curP2 else []), default=1)
            if max_from_key > int(S.max_card):
                S.max_card = int(max_from_key)
            S.cardsA_n = int(S.max_card)
            S.pending_transition = {
                "cardsA": a2,
                "cardsB": b2,
                "cardsP": p2,
                "curP": int(curP2),
                "diff": int(diff2),
            }
            trigger_rerun()

    mat = build_matrix(cache, A_mask, B_mask, P_mask, diff, curP)
    if not mat:
        st.error("Matrix build failed for this state (missing cache entries or invalid state).")
        return

    mat_np = np.array(mat, dtype=np.float64)
    df = pd.DataFrame(mat_np, index=cardsA, columns=cardsB)
    display_df = (50.0 + 50.0 * df).round(2) if display_percent else df.round(4)

    pA, vA = findBestStrategy(mat_np)
    pB, _vB = findBestStrategy(-mat_np.T)
    strategies_ok = pA is not None and pB is not None

    matrix_csv = display_df.to_csv(index=True)
    matrix_title = "Win % Matrix (click a cell to advance)" if display_percent else "EV Matrix (click a cell to advance)"
    render_section_header(
        matrix_title,
        "Click any payoff to advance one round using the next prize below.",
        matrix_csv,
        "Copy matrix CSV",
    )

    advance_next = None
    if cardsP:
        advance_options: List[object] = ["Random"] + cardsP
        if "advance_next_prize" not in st.session_state or S.advance_next_prize not in advance_options:
            S.advance_next_prize = "Random"
        advance_col, _spacer = st.columns([1, 4])
        with advance_col:
            advance_next = st.selectbox(
                "Next prize",
                advance_options,
                key="advance_next_prize",
                help="Used when you click a matrix cell to advance one round.",
            )
    else:
        st.caption("No remaining prizes to advance from this state.")

    if strategies_ok:
        support_a = {card for card, prob in zip(cardsA, pA) if prob > 1e-6}
        support_b = {card for card, prob in zip(cardsB, pB) if prob > 1e-6}
    else:
        support_a = set()
        support_b = set()

    st.markdown("**Clickable Matrix**")
    st.caption("Cell color shows EV for Player A (green=good, red=bad). Outlined cells are in mutual NE support.")

    labels: List[List[str]] = []
    evs: List[List[float]] = []
    for row_idx in range(len(cardsA)):
        row_labels: List[str] = []
        row_evs: List[float] = []
        for col_idx in range(len(cardsB)):
            ev = float(mat_np[row_idx, col_idx])
            value = display_value(ev, display_percent)
            row_labels.append(f"{value:.3f}%" if display_percent else f"{value:+.3f}")
            row_evs.append(ev)
        labels.append(row_labels)
        evs.append(row_evs)

    clicked = evc_grid(
        cardsA=cardsA,
        cardsB=cardsB,
        evs=evs,
        labels=labels,
        supportA=sorted(support_a),
        supportB=sorted(support_b),
        disabled=not bool(cardsP),
        # Keep a stable component identity across state transitions to prevent
        # the iframe from being torn down (which causes a visible flash).
        key="grid",
    )
    if clicked and cardsP:
        try:
            cardA_click = int(clicked.get("a"))
            cardB_click = int(clicked.get("b"))
        except Exception:
            cardA_click = None
            cardB_click = None
        if cardA_click in cardsA and cardB_click in cardsB:
            next_prize = resolve_next_prize(advance_next, cardsP)
            if next_prize is None:
                next_prize = cardsP[0]
            new_diff = diff + cmp(cardA_click, cardB_click) * curP
            S.pending_transition = {
                "cardsA": [c for c in cardsA if c != cardA_click],
                "cardsB": [c for c in cardsB if c != cardB_click],
                "cardsP": [c for c in cardsP if c != next_prize],
                "curP": int(next_prize),
                "diff": int(new_diff),
            }
            trigger_rerun()

    st.subheader("Optimal Mixed Strategies")
    if not strategies_ok:
        st.error("Failed to compute optimal strategies.")
        return

    ev_a = mat_np @ pB
    ev_b = -(pA @ mat_np)
    df_a = build_strategy_table(cardsA, pA, ev_a, display_percent)
    df_b = build_strategy_table(cardsB, pB, ev_b, display_percent)

    colA, colB = st.columns(2)
    with colA:
        title = "Win % (A perspective)" if display_percent else "EV (A perspective)"
        value_label = f"{display_value(float(vA), True):.2f}%" if display_percent else f"{float(vA):+.6f}"
        render_strategy_panel(title, value_label, df_a, "Copy strategy A CSV", support_a)
    with colB:
        title = "Win % (B perspective)" if display_percent else "EV (B perspective)"
        value_label = f"{display_value(float(-vA), True):.2f}%" if display_percent else f"{float(-vA):+.6f}"
        render_strategy_panel(title, value_label, df_b, "Copy strategy B CSV", support_b)
