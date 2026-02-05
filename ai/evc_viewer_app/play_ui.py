from __future__ import annotations

import random
from typing import Dict

import pandas as pd
import streamlit as st

from common import State, cmp, list_cards, remove_card
from strategies import build_strategy, strategy_choices

from evc_viewer_app.helpers import resolve_rng, run_auto_play_series, start_play_game
from evc_viewer_app.session import trigger_rerun
from evc_viewer_app.ui import render_state_summary


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

