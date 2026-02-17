from __future__ import annotations

import random
from collections.abc import Mapping
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from common import State, cmp, list_cards, remove_card
from strategies import build_strategy, strategy_choices

from evc_viewer_app.helpers import resolve_rng, run_auto_play_series, start_play_game
from evc_viewer_app.session import trigger_rerun
from evc_viewer_app.state_types import ss
from evc_viewer_app.ui import render_state_summary


def _clamp_int_session_value(key: str, *, default: int, min_value: int, max_value: int) -> None:
    raw_value = st.session_state.get(key, default)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = int(default)
    st.session_state[key] = max(min_value, min(max_value, parsed))


def _init_play_ui_state(max_card: int) -> None:
    S = ss()
    if "play_history" not in st.session_state:
        S.play_history = []
    if "play_reveal_prizes" not in st.session_state:
        S.play_reveal_prizes = False
    # Default opponent: "evc-ne" (best-looking baseline bot for this app).
    if "play_bot_strategy" not in st.session_state:
        S.play_bot_strategy = "evc-ne"
    _clamp_int_session_value("play_n", default=max_card, min_value=1, max_value=max_card)
    _clamp_int_session_value("auto_n", default=max_card, min_value=1, max_value=max_card)


def _render_card_face(
    value: object,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    accent: str = "#0f172a",
) -> None:
    title_html = f"<div style='font-size:12px; opacity:.75; margin-bottom:6px'>{title}</div>" if title else ""
    sub_html = f"<div style='font-size:12px; opacity:.75; margin-top:6px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style="
            width: 100%;
            border: 1px solid rgba(15, 23, 42, 0.20);
            border-radius: 14px;
            padding: 14px 14px;
            background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.98));
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        ">
          {title_html}
          <div style="
            display:flex;
            align-items:center;
            justify-content:center;
            height: 96px;
            font-size: 40px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: {accent};
          ">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hand_button_grid(
    *,
    label: str,
    cards: List[int],
    key_prefix: str,
    per_row: int = 10,
) -> Optional[int]:
    st.markdown(f"**{label}**")
    if not cards:
        st.caption("No cards.")
        return None
    per_row = max(1, min(per_row, len(cards)))
    clicked: Optional[int] = None
    for row_start in range(0, len(cards), per_row):
        row = cards[row_start : row_start + per_row]
        cols = st.columns(len(row), gap="small")
        for col, card in zip(cols, row):
            with col:
                if st.button(
                    str(card),
                    key=f"{key_prefix}_{card}",
                    type="secondary",
                    use_container_width=True,
                ):
                    clicked = card
    return clicked


def render_play_tabs(cache: Mapping[int, float], max_card: int) -> None:
    _init_play_ui_state(max_card)
    S = ss()
    st.subheader("Play / Simulate")
    bot_choices = strategy_choices()
    play_tab, auto_tab = st.tabs(["Play vs Bot", "Auto-play"])

    with play_tab:
        st.caption("Play a full game. Prize order is hidden unless you reveal it.")
        with st.container(border=True):
            left, right = st.columns([2, 1], vertical_alignment="bottom")
            with left:
                c1, c2 = st.columns(2)
                with c1:
                    play_n = st.number_input(
                        "Cards (N)",
                        min_value=1,
                        max_value=max_card,
                        step=1,
                        key="play_n",
                    )
                with c2:
                    play_seed = st.number_input(
                        "Game seed (0=random)",
                        min_value=0,
                        max_value=2**31 - 1,
                        value=int(S.get("play_seed", 0)),
                        step=1,
                        key="play_seed",
                    )
                bot_strategy = st.selectbox("Bot strategy", bot_choices, key="play_bot_strategy")
                bot_seed = st.number_input(
                    "Bot seed (-1=use game RNG, 0=random)",
                    min_value=-1,
                    max_value=2**31 - 1,
                    value=int(S.get("play_bot_seed", -1)),
                    step=1,
                    key="play_bot_seed",
                )
            with right:
                st.toggle("Reveal prize order", value=bool(S.play_reveal_prizes), key="play_reveal_prizes")
                if st.button("Start new game", key="play_start", type="primary", use_container_width=True):
                    start_play_game(int(play_n), int(play_seed))
                    S.play_bot_rng = random.Random(int(bot_seed)) if int(bot_seed) > 0 else None
                    S.play_bot_rng_seed = int(bot_seed)
                    S.play_history = []
                    trigger_rerun()
                if st.button("End game", key="play_end", use_container_width=True):
                    S.play_active = False
                    S.play_over = False
                    S.play_last_choices = None
                    S.play_final_diff = None

        if S.get("play_active"):
            cardsA = list_cards(int(S.play_A_mask))
            cardsB = list_cards(int(S.play_B_mask))
            cardsP = list_cards(int(S.play_P_mask))
            curP = int(S.play_curP)
            diff = int(S.play_diff)

            # Tabletop layout: Score card, Prize card, Last round card.
            top_l, top_c, top_r = st.columns([2, 2, 2], vertical_alignment="top")
            score_accent = "#b91c1c" if diff < 0 else ("#0f766e" if diff > 0 else "#0f172a")
            with top_l:
                _render_card_face(
                    f"{diff:+d}",
                    title="Score (A - B)",
                    subtitle=f"Cards left: {len(cardsA)}",
                    accent=score_accent,
                )
            with top_c:
                _render_card_face(curP, title="Current Prize", subtitle=f"Remaining prizes: {len(cardsP)}", accent="#0f766e")
            with top_r:
                last = (S.get("play_history") or [])[-1] if (S.get("play_history") or []) else None
                if last:
                    _render_card_face(
                        f"{last['you']} vs {last['bot']}",
                        title="Last Round",
                        subtitle=f"Δ {int(last['delta']):+d}  |  Total {int(last['diff']):+d}",
                        accent="#0f172a",
                    )
                else:
                    _render_card_face("—", title="Last Round", subtitle="No rounds played yet.", accent="#0f172a")

            if S.get("play_reveal_prizes") and S.get("play_prizes"):
                prizes = list(S.play_prizes)
                idx = int(S.get("play_index", 0))
                upcoming = prizes[idx + 1 :]
                with st.expander("Prize order (revealed)", expanded=False):
                    st.write(f"Current index: `{idx}`")
                    st.write("Upcoming prizes:", upcoming if upcoming else "None")

            if S.get("play_over"):
                # No extra details needed on game over; the score card is enough.
                pass
            elif not cardsA or not cardsB:
                st.info("Start a new game to play.")
            else:
                with st.container(border=True):
                    st.markdown("**Your move**")
                    clicked_card = _hand_button_grid(
                        label="Select a card to play",
                        cards=cardsA,
                        key_prefix="play_hand",
                        per_row=10,
                    )
                    st.caption("Click a card to play it immediately.")

                if clicked_card is not None:
                    choiceA = int(clicked_card)
                    bot_seed_val = int(S.play_bot_seed)
                    if bot_seed_val > 0:
                        if (
                            S.get("play_bot_rng_seed") != bot_seed_val
                            or S.get("play_bot_rng") is None
                        ):
                            S.play_bot_rng = random.Random(bot_seed_val)
                            S.play_bot_rng_seed = bot_seed_val
                        bot_rng = S.play_bot_rng
                    else:
                        bot_rng = resolve_rng(bot_seed_val, S.get("play_rng"))

                    bot_strat = build_strategy(S.play_bot_strategy, cache=cache, rng=bot_rng)
                    A_mask = int(S.play_A_mask)
                    B_mask = int(S.play_B_mask)
                    P_mask = int(S.play_P_mask)
                    state_for_b = State(A=B_mask, B=A_mask, P=P_mask, diff=-diff, curP=curP)
                    choiceB = int(bot_strat(state_for_b))
                    if choiceB not in cardsB:
                        st.error(f"Bot strategy returned invalid card: {choiceB}")
                    else:
                        delta = int(cmp(choiceA, choiceB) * curP)
                        new_diff = int(diff + delta)
                        new_A = remove_card(A_mask, choiceA)
                        new_B = remove_card(B_mask, choiceB)
                        S.play_last_choices = (choiceA, choiceB)
                        S.play_history.append(
                            {
                                "prize": int(curP),
                                "you": int(choiceA),
                                "bot": int(choiceB),
                                "delta": int(delta),
                                "diff": int(new_diff),
                            }
                        )
                        if P_mask == 0:
                            S.play_A_mask = new_A
                            S.play_B_mask = new_B
                            S.play_curP = 0
                            S.play_diff = new_diff
                            S.play_over = True
                            S.play_final_diff = new_diff
                        else:
                            next_idx = int(S.play_index) + 1
                            prizes = S.play_prizes
                            next_prize = prizes[next_idx]
                            S.play_index = next_idx
                            S.play_A_mask = new_A
                            S.play_B_mask = new_B
                            S.play_P_mask = remove_card(P_mask, next_prize)
                            S.play_curP = int(next_prize)
                            S.play_diff = new_diff
                        trigger_rerun()

            hist = S.get("play_history") or []
            if hist:
                with st.expander("Round log", expanded=False):
                    df_hist = pd.DataFrame(hist)
                    st.dataframe(df_hist, width="stretch", hide_index=True)
        else:
            st.info("Start a new game to play against a strategy.")

    with auto_tab:
        st.caption("Simulate multiple games between two strategies.")
        with st.container(border=True):
            st.markdown("**Setup**")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4, vertical_alignment="bottom")
            with r1c1:
                auto_n = st.number_input(
                    "Cards (N)",
                    min_value=1,
                    max_value=max_card,
                    step=1,
                    key="auto_n",
                )
            with r1c2:
                auto_count = st.number_input(
                    "Games",
                    min_value=1,
                    value=int(S.get("auto_count", 50)),
                    step=1,
                    key="auto_count",
                )
            with r1c3:
                auto_seed = st.number_input(
                    "Series seed (0=random)",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=int(S.get("auto_seed", 0)),
                    step=1,
                    key="auto_seed",
                )
            with r1c4:
                show_diffs = st.toggle("Show per-game diffs", value=False, key="auto_show_diffs")

            st.markdown("**Matchup**")
            r2c1, r2c2, r2c3, r2c4 = st.columns(4, vertical_alignment="bottom")
            with r2c1:
                auto_sa = st.selectbox("Strategy A", bot_choices, key="auto_sa")
            with r2c2:
                auto_sa_seed = st.number_input(
                    "Strategy A seed",
                    min_value=-1,
                    max_value=2**31 - 1,
                    value=int(S.get("auto_sa_seed", -1)),
                    step=1,
                    key="auto_sa_seed",
                    help="-1=use game RNG, 0=random",
                )
            with r2c3:
                auto_sb = st.selectbox("Strategy B", bot_choices, key="auto_sb")
            with r2c4:
                auto_sb_seed = st.number_input(
                    "Strategy B seed",
                    min_value=-1,
                    max_value=2**31 - 1,
                    value=int(S.get("auto_sb_seed", -1)),
                    step=1,
                    key="auto_sb_seed",
                    help="-1=use game RNG, 0=random",
                )

            run_clicked = st.button("Run auto-play", key="auto_run", type="primary", use_container_width=True)

        if run_clicked:
            S.auto_results = run_auto_play_series(
                int(auto_n),
                int(auto_seed),
                int(auto_count),
                auto_sa,
                auto_sb,
                int(auto_sa_seed),
                int(auto_sb_seed),
                cache,
            )
        results = S.get("auto_results")
        if results:
            with st.container(border=True):
                st.markdown("**Results**")
                st.write(f"Strategy A: `{results['label_a']}`")
                st.write(f"Strategy B: `{results['label_b']}`")
                st.write(f"A wins: `{results['wins_a']}` ({results['wins_a'] / results['total_games'] * 100:.1f}%)")
                st.write(f"B wins: `{results['wins_b']}` ({results['wins_b'] / results['total_games'] * 100:.1f}%)")
                st.write(f"Draws: `{results['draws']}` ({results['draws'] / results['total_games'] * 100:.1f}%)")
                st.write(f"Average point diff (A-B): `{results['avg_diff']:+.3f}`")
                st.write(f"Average EV (A=+1, B=-1): `{results['avg_ev']:+.3f}`")
            if show_diffs:
                df_diffs = pd.DataFrame({"diff": results["diffs"]})
                st.dataframe(df_diffs, width="stretch", hide_index=True)
