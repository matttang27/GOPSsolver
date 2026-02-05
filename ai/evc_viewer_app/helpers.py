from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

from common import State, decode_key, list_cards, list_to_mask, remove_card
from play import run_game
from strategies import build_strategy, strategy_label


def reservoir_sample_filtered(
    keys: Iterable[int],
    sample_size: int,
    rng: random.Random,
    *,
    cards_per_player: Optional[int] = None,
) -> Tuple[List[int], int]:
    sample: List[int] = []
    seen = 0
    for key in keys:
        if cards_per_player is not None:
            A_mask, B_mask, _P_mask, _diff, _curP = decode_key(key)
            if A_mask.bit_count() != cards_per_player or B_mask.bit_count() != cards_per_player:
                continue
        if seen < sample_size:
            sample.append(key)
        else:
            j = rng.randint(0, seen)
            if j < sample_size:
                sample[j] = key
        seen += 1
    return sample, seen


def decode_key_state(key: int) -> Tuple[List[int], List[int], List[int], int, int]:
    A, B, P, diff, curP = decode_key(key)
    return list_cards(A), list_cards(B), list_cards(P), diff, curP


def build_start_state(max_card: int) -> Dict[str, object]:
    cards = list(range(1, max_card + 1))
    curP = cards[0] if cards else 1
    cardsP = [c for c in cards if c != curP]
    return {
        "cardsA": cards,
        "cardsB": cards[:],
        "cardsP": cardsP,
        "curP": curP,
        "diff": 0,
    }


def sync_current_prize() -> None:
    prev = st.session_state.get("curP_prev")
    new = st.session_state.get("curP")
    if new is None:
        return
    if prev is None:
        st.session_state.curP_prev = new
        return
    if prev == new:
        return
    cardsP = list(st.session_state.get("cardsP", []))
    cardsP = [c for c in cardsP if c != new]
    card_options = st.session_state.get("card_options", [])
    if prev in card_options and prev != new and prev not in cardsP:
        cardsP.append(prev)
    st.session_state.cardsP = sorted(cardsP)
    st.session_state.curP_prev = new


def apply_pending_transition() -> None:
    pending = st.session_state.get("pending_transition")
    if not pending:
        return
    st.session_state.cardsA = pending["cardsA"]
    st.session_state.cardsB = pending["cardsB"]
    st.session_state.cardsP = pending["cardsP"]
    st.session_state.curP = pending["curP"]
    st.session_state.curP_prev = pending["curP"]
    st.session_state.diff = pending["diff"]
    st.session_state.mode = "Manual"
    st.session_state.pending_transition = None


def resolve_next_prize(selection: object, options: List[int]) -> Optional[int]:
    if not options:
        return None
    if isinstance(selection, str) and selection.lower() == "random":
        return random.choice(options)
    try:
        return int(selection)
    except (TypeError, ValueError):
        return None


def resolve_rng(seed: int, fallback: Optional[random.Random]) -> random.Random:
    if seed == -1:
        return fallback or random
    if seed == 0:
        return random
    return random.Random(seed)


def start_play_game(n: int, seed: int) -> None:
    rng_game = random.Random(seed) if seed != 0 else random
    prizes = list(range(1, n + 1))
    rng_game.shuffle(prizes)
    st.session_state.play_active = True
    st.session_state.play_over = False
    st.session_state.play_prizes = prizes
    st.session_state.play_index = 0
    st.session_state.play_A_mask = (1 << n) - 1
    st.session_state.play_B_mask = (1 << n) - 1
    st.session_state.play_curP = prizes[0] if prizes else 0
    st.session_state.play_P_mask = list_to_mask(prizes[1:])
    st.session_state.play_diff = 0
    st.session_state.play_rng = rng_game if seed != 0 else None
    st.session_state.play_last_choices = None
    st.session_state.play_final_diff = None


def run_auto_play_series(
    n: int,
    seed: int,
    count: int,
    strat_a: str,
    strat_b: str,
    sa_seed: int,
    sb_seed: int,
    cache: Optional[Dict[int, float]],
) -> Dict[str, object]:
    wins_a = 0
    wins_b = 0
    draws = 0
    total_diff = 0
    diffs: List[int] = []
    for i in range(count):
        seed_i = seed if seed == 0 else seed + i
        rng_game = random.Random(seed_i) if seed_i != 0 else random
        rng_a = resolve_rng(sa_seed, rng_game)
        rng_b = resolve_rng(sb_seed, rng_game)
        stratA = build_strategy(strat_a, cache=cache, rng=rng_a)
        stratB = build_strategy(strat_b, cache=cache, rng=rng_b)

        def choose_actions(state: State, state_for_b: State) -> Tuple[int, int]:
            return stratA(state), stratB(state_for_b)

        diff = run_game(n, seed_i, choose_actions, rng=rng_game)
        diffs.append(diff)
        total_diff += diff
        if diff > 0:
            wins_a += 1
        elif diff < 0:
            wins_b += 1
        else:
            draws += 1
    total_games = max(count, 1)
    avg_diff = total_diff / total_games
    avg_ev = (wins_a - wins_b) / total_games
    label_a = strategy_label(strat_a, None if sa_seed == -1 else sa_seed)
    label_b = strategy_label(strat_b, None if sb_seed == -1 else sb_seed)
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "total_games": total_games,
        "avg_diff": avg_diff,
        "avg_ev": avg_ev,
        "diffs": diffs,
        "label_a": label_a,
        "label_b": label_b,
    }
