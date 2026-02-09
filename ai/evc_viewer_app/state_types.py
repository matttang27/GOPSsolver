from __future__ import annotations

import random
from typing import Any, Protocol, cast

import streamlit as st


class AppState(Protocol):
    # Explore tab
    cardsA: list[int]
    cardsB: list[int]
    cardsP: list[int]
    curP: int
    curP_prev: int
    diff: int
    cardsA_n: int
    max_card: int
    mode: str
    card_options: list[int]
    pending_transition: dict[str, object] | None
    sample_keys: list[int]
    sample_params: tuple[int, int, int, str] | None
    sample_matched: int
    sample_cards: int
    advance_next_prize: object

    # Play tab
    play_history: list[dict[str, int]]
    play_reveal_prizes: bool
    play_bot_strategy: str
    play_bot_seed: int
    play_bot_rng: random.Random | None
    play_bot_rng_seed: int
    play_n: int
    play_seed: int
    play_active: bool
    play_over: bool
    play_prizes: list[int]
    play_index: int
    play_A_mask: int
    play_B_mask: int
    play_curP: int
    play_P_mask: int
    play_diff: int
    play_rng: random.Random | None
    play_last_choices: tuple[int, int] | None
    play_final_diff: int | None

    # Auto-play tab
    auto_n: int
    auto_count: int
    auto_seed: int
    auto_show_diffs: bool
    auto_sa: str
    auto_sb: str
    auto_sa_seed: int
    auto_sb_seed: int
    auto_results: dict[str, object] | None

    def get(self, key: str, default: Any = ...) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...


def ss() -> AppState:
    return cast(AppState, st.session_state)
