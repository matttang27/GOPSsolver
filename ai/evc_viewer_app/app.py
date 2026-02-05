from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    State,
    build_matrix,
    cmp,
    encode_key,
    list_cards,
    list_to_mask,
    load_evc,
    load_meta,
    remove_card,
)
from linprog import findBestStrategy
from strategies import build_strategy, strategy_choices

from evc_viewer_app.config import CACHE_PATH
from evc_viewer_app.helpers import (
    apply_pending_transition,
    build_start_state,
    decode_key_state,
    reservoir_sample_filtered,
    resolve_next_prize,
    resolve_rng,
    run_auto_play_series,
    start_play_game,
    sync_current_prize,
)
from evc_viewer_app.ui import (
    build_strategy_table,
    display_value,
    render_section_header,
    render_state_summary,
    render_strategy_panel,
)


@st.cache_resource(show_spinner=False)
def load_cache(path: str) -> Dict[int, float]:
    return load_evc(path)


def get_mode_help(mode: str) -> str:
    if mode == "Manual":
        return "Pick the hands/prizes directly and explore specific states."
    if mode == "Key":
        return "Paste a packed cache key to decode a state."
    if mode == "Sample":
        return "Browse a random sample of cached states."
    return ""


def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def ensure_state_init(card_options: List[int], max_card: int) -> None:
    if "cardsA" not in st.session_state:
        st.session_state.cardsA = card_options[:]
        st.session_state.cardsB = card_options[:]
        st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.diff = 0
        st.session_state.cardsA_n = max_card
        st.session_state.curP_prev = st.session_state.curP
        return
    if st.session_state.get("cardsA_n") != max_card:
        st.session_state.cardsA = [c for c in st.session_state.cardsA if c in card_options] or card_options[:]
        st.session_state.cardsB = [c for c in st.session_state.cardsB if c in card_options] or card_options[:]
        if st.session_state.curP not in card_options:
            st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [
            c for c in st.session_state.cardsP if c in card_options and c != st.session_state.curP
        ]
        if not st.session_state.cardsP:
            st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.cardsA_n = max_card
        st.session_state.curP_prev = st.session_state.curP


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
            ["Manual", "Key", "Sample"],
            horizontal=True,
            key="mode",
            help="Manual: pick hands/prizes. Key: decode a packed cache key. Sample: browse random cached states.",
        )
        st.caption(get_mode_help(mode))
        if st.button("Reset to start state", type="primary"):
            st.session_state.pending_transition = build_start_state(int(st.session_state.max_card))
            trigger_rerun()

    return mode, run_n


def render_play_tabs(cache: Dict[int, float], max_card: int) -> None:
    st.subheader("Play / Simulate")
    bot_choices = strategy_choices()
    play_tab, auto_tab = st.tabs(["Play vs Bot", "Auto-play"])

    with play_tab:
        st.caption("Play a full game with a shuffled prize order.")
        play_n = st.number_input(
            "Cards (N)",
            min_value=1,
            max_value=max_card,
            value=max_card,
            step=1,
            key="play_n",
        )
        play_seed = st.number_input(
            "Game seed (0=random)",
            min_value=0,
            max_value=2**31 - 1,
            value=0,
            step=1,
            key="play_seed",
        )
        bot_strategy = st.selectbox("Bot strategy", bot_choices, key="play_bot_strategy")
        bot_seed = st.number_input(
            "Bot seed (-1=use game RNG, 0=random)",
            min_value=-1,
            max_value=2**31 - 1,
            value=-1,
            step=1,
            key="play_bot_seed",
        )
        start_col, end_col = st.columns(2)
        with start_col:
            if st.button("Start new game", key="play_start"):
                start_play_game(int(play_n), int(play_seed))
                st.session_state.play_bot_rng = random.Random(int(bot_seed)) if int(bot_seed) > 0 else None
                st.session_state.play_bot_rng_seed = int(bot_seed)
        with end_col:
            if st.button("End game", key="play_end"):
                st.session_state.play_active = False
                st.session_state.play_over = False
                st.session_state.play_last_choices = None
                st.session_state.play_final_diff = None

        if st.session_state.get("play_active"):
            cardsA = list_cards(int(st.session_state.play_A_mask))
            cardsB = list_cards(int(st.session_state.play_B_mask))
            cardsP = list_cards(int(st.session_state.play_P_mask))
            curP = int(st.session_state.play_curP)
            diff = int(st.session_state.play_diff)
            render_state_summary(cardsA, cardsB, cardsP, diff, curP)
            if st.session_state.get("play_last_choices"):
                choiceA, choiceB = st.session_state.play_last_choices
                st.caption(f"Last round: you played {choiceA}, bot played {choiceB}.")
            if st.session_state.get("play_over"):
                final_diff = st.session_state.get("play_final_diff", diff)
                if final_diff > 0:
                    result = "You win!"
                elif final_diff < 0:
                    result = "Bot wins!"
                else:
                    result = "Draw."
                st.success(f"Game over. Final diff: {int(final_diff):+d}. {result}")
            elif not cardsA or not cardsB:
                st.info("Start a new game to play.")
            else:
                if "play_choice_a" in st.session_state and st.session_state.play_choice_a not in cardsA:
                    st.session_state.play_choice_a = cardsA[0]
                choiceA = st.selectbox("Your card", cardsA, key="play_choice_a")
                if st.button("Play round", key="play_round"):
                    bot_seed_val = int(st.session_state.play_bot_seed)
                    if bot_seed_val > 0:
                        if (
                            st.session_state.get("play_bot_rng_seed") != bot_seed_val
                            or st.session_state.get("play_bot_rng") is None
                        ):
                            st.session_state.play_bot_rng = random.Random(bot_seed_val)
                            st.session_state.play_bot_rng_seed = bot_seed_val
                        bot_rng = st.session_state.play_bot_rng
                    else:
                        bot_rng = resolve_rng(bot_seed_val, st.session_state.get("play_rng"))

                    bot_strat = build_strategy(bot_strategy, cache=cache, rng=bot_rng)
                    A_mask = int(st.session_state.play_A_mask)
                    B_mask = int(st.session_state.play_B_mask)
                    P_mask = int(st.session_state.play_P_mask)
                    state = State(A=A_mask, B=B_mask, P=P_mask, diff=diff, curP=curP)
                    state_for_b = State(A=B_mask, B=A_mask, P=P_mask, diff=-diff, curP=curP)
                    choiceB = bot_strat(state_for_b)
                    if choiceB not in cardsB:
                        st.error(f"Bot strategy returned invalid card: {choiceB}")
                    else:
                        new_diff = diff + cmp(choiceA, choiceB) * curP
                        new_A = remove_card(A_mask, choiceA)
                        new_B = remove_card(B_mask, choiceB)
                        st.session_state.play_last_choices = (choiceA, choiceB)
                        if P_mask == 0:
                            st.session_state.play_A_mask = new_A
                            st.session_state.play_B_mask = new_B
                            st.session_state.play_curP = 0
                            st.session_state.play_diff = new_diff
                            st.session_state.play_over = True
                            st.session_state.play_final_diff = new_diff
                        else:
                            next_idx = int(st.session_state.play_index) + 1
                            prizes = st.session_state.play_prizes
                            next_prize = prizes[next_idx]
                            st.session_state.play_index = next_idx
                            st.session_state.play_A_mask = new_A
                            st.session_state.play_B_mask = new_B
                            st.session_state.play_P_mask = remove_card(P_mask, next_prize)
                            st.session_state.play_curP = int(next_prize)
                            st.session_state.play_diff = new_diff
                        trigger_rerun()
        else:
            st.info("Start a new game to play against a strategy.")

    with auto_tab:
        st.caption("Simulate multiple games between two strategies.")
        auto_n = st.number_input(
            "Cards (N)",
            min_value=1,
            max_value=max_card,
            value=max_card,
            step=1,
            key="auto_n",
        )
        auto_seed = st.number_input(
            "Series seed (0=random)",
            min_value=0,
            max_value=2**31 - 1,
            value=0,
            step=1,
            key="auto_seed",
        )
        auto_count = st.number_input(
            "Games",
            min_value=1,
            value=50,
            step=1,
            key="auto_count",
        )
        auto_sa = st.selectbox("Strategy A", bot_choices, key="auto_sa")
        auto_sb = st.selectbox("Strategy B", bot_choices, key="auto_sb")
        auto_sa_seed = st.number_input(
            "Strategy A seed (-1=use game RNG, 0=random)",
            min_value=-1,
            max_value=2**31 - 1,
            value=-1,
            step=1,
            key="auto_sa_seed",
        )
        auto_sb_seed = st.number_input(
            "Strategy B seed (-1=use game RNG, 0=random)",
            min_value=-1,
            max_value=2**31 - 1,
            value=-1,
            step=1,
            key="auto_sb_seed",
        )
        show_diffs = st.toggle("Show per-game diffs", value=False, key="auto_show_diffs")
        if st.button("Run auto-play", key="auto_run"):
            st.session_state.auto_results = run_auto_play_series(
                int(auto_n),
                int(auto_seed),
                int(auto_count),
                auto_sa,
                auto_sb,
                int(auto_sa_seed),
                int(auto_sb_seed),
                cache,
            )
        results = st.session_state.get("auto_results")
        if results:
            st.write(f"Strategy A: {results['label_a']}")
            st.write(f"Strategy B: {results['label_b']}")
            st.write(f"A wins: {results['wins_a']} ({results['wins_a'] / results['total_games'] * 100:.1f}%)")
            st.write(f"B wins: {results['wins_b']} ({results['wins_b'] / results['total_games'] * 100:.1f}%)")
            st.write(f"Draws: {results['draws']} ({results['draws'] / results['total_games'] * 100:.1f}%)")
            st.write(f"Average point diff (A-B): {results['avg_diff']:+.3f}")
            st.write(f"Average EV (A=+1, B=-1): {results['avg_ev']:+.3f}")
            if show_diffs:
                df_diffs = pd.DataFrame({"diff": results["diffs"]})
                st.dataframe(df_diffs, width="stretch", hide_index=True)


def render_explore_tab(
    cache: Dict[int, float],
    cache_path,
    display_percent: bool,
    mode: str,
    max_card: int,
    card_options: List[int],
) -> None:
    st.subheader("State Input")
    ensure_state_init(card_options, max_card)

    cardsA: List[int] = []
    cardsB: List[int] = []
    cardsP: List[int] = []
    diff = 0
    curP = 0

    if mode == "Manual":
        col1, col2, col3 = st.columns(3)
        with col1:
            cardsA = st.multiselect(
                "Player A cards",
                card_options,
                default=st.session_state.cardsA,
                key="cardsA",
            )
            cardsB = st.multiselect(
                "Player B cards",
                card_options,
                default=st.session_state.cardsB,
                key="cardsB",
            )
        with col2:
            curP = st.selectbox(
                "Current prize",
                card_options,
                index=card_options.index(st.session_state.curP) if st.session_state.curP in card_options else 0,
                key="curP",
                on_change=sync_current_prize,
            )
            cardsP = st.multiselect(
                "Remaining prizes (exclude current)",
                card_options,
                default=st.session_state.cardsP,
                key="cardsP",
            )
        with col3:
            diff = int(
                st.number_input(
                    "Point diff (A - B)",
                    min_value=-128,
                    max_value=127,
                    value=int(st.session_state.diff),
                    step=1,
                    key="diff",
                )
            )

    elif mode == "Key":
        key_text = st.text_input("Cache key (decimal or 0x...)")
        if key_text.strip():
            try:
                key_val = int(key_text.strip(), 0)
            except ValueError:
                st.error("Invalid key. Use decimal or 0x-prefixed hex.")
                return
            cardsA, cardsB, cardsP, diff, curP = decode_key_state(key_val)
            render_state_summary(cardsA, cardsB, cardsP, diff, curP)
            max_from_key = max(cardsA + cardsB + cardsP + ([curP] if curP else []), default=1)
            if max_from_key > max_card:
                st.session_state.max_card = max_from_key
            if st.button("Use decoded state in Manual mode"):
                st.session_state.cardsA = cardsA
                st.session_state.cardsB = cardsB
                st.session_state.cardsP = cardsP
                st.session_state.curP = curP
                st.session_state.curP_prev = curP
                st.session_state.diff = diff
                st.session_state.cardsA_n = max(st.session_state.max_card, max_from_key)
                st.session_state.mode = "Manual"
        else:
            st.info("Enter a cache key to decode a state.")
            return

    else:  # Sample
        if "sample_keys" not in st.session_state:
            st.session_state.sample_keys = []
        if "sample_params" not in st.session_state:
            st.session_state.sample_params = None
        rng_seed = st.number_input("Sample RNG seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
        if "sample_cards" in st.session_state and st.session_state.sample_cards > max_card:
            st.session_state.sample_cards = max_card
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
        if st.session_state.sample_params != params:
            rng = random.Random(int(rng_seed))
            with st.spinner("Sampling cache keys..."):
                sample, matched = reservoir_sample_filtered(
                    cache.keys(),
                    int(sample_size),
                    rng,
                    cards_per_player=int(sample_cards),
                )
                st.session_state.sample_keys = sample
                st.session_state.sample_matched = matched
                st.session_state.sample_params = params
        if not st.session_state.sample_keys:
            st.info("Build a sample list to browse random states.")
            return
        matched = st.session_state.get("sample_matched")
        if matched is not None:
            st.caption(f"Matched states: {matched}")
        key_val = st.selectbox("Sampled state key", st.session_state.sample_keys)
        cardsA, cardsB, cardsP, diff, curP = decode_key_state(int(key_val))
        render_state_summary(cardsA, cardsB, cardsP, diff, curP)

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

    st.subheader("State")
    render_state_summary(cardsA, cardsB, cardsP, diff, curP, key=state_key)

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
        if "advance_next_prize" not in st.session_state or st.session_state.advance_next_prize not in advance_options:
            st.session_state.advance_next_prize = "Random"
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
    st.caption("Buttons are colored when both actions are optimal (positive probability in the mixed strategy).")
    click_cols = st.columns([1] + [1] * len(cardsB))
    with click_cols[0]:
        st.markdown("**A \\ B**")
    for col_idx, cardB in enumerate(cardsB):
        with click_cols[col_idx + 1]:
            label = f"{cardB} (opt)" if cardB in support_b else f"{cardB}"
            st.markdown(f"**{label}**")

    for row_idx, cardA in enumerate(cardsA):
        row_cols = st.columns([1] + [1] * len(cardsB))
        with row_cols[0]:
            label = f"{cardA} (opt)" if cardA in support_a else f"{cardA}"
            st.markdown(f"**{label}**")
        for col_idx, cardB in enumerate(cardsB):
            value = display_value(float(mat_np[row_idx, col_idx]), display_percent)
            label = f"{value:.3f}%" if display_percent else f"{value:+.3f}"
            is_opt_cell = cardA in support_a and cardB in support_b
            if row_cols[col_idx + 1].button(
                label,
                key=f"cell_{state_key}_{cardA}_{cardB}",
                disabled=not cardsP,
                type="primary" if is_opt_cell else "secondary",
            ):
                if cardsP:
                    next_prize = resolve_next_prize(advance_next, cardsP)
                    if next_prize is None:
                        next_prize = cardsP[0]
                    new_diff = diff + cmp(cardA, cardB) * curP
                    st.session_state.pending_transition = {
                        "cardsA": [c for c in cardsA if c != cardA],
                        "cardsB": [c for c in cardsB if c != cardB],
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
        st.error("Cache file not found: reports/full8.evc")
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
