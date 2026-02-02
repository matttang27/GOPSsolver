import os
import sys
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

REPORTS_DIR = Path(__file__).resolve().parent
if str(REPORTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTS_DIR))

from common import (
    build_matrix,
    decode_key,
    encode_key,
    list_cards,
    list_to_mask,
    load_evc,
    load_meta,
)
from linprog import findBestStrategy


def resolve_cache_path(raw_path: str) -> Optional[str]:
    if not raw_path:
        return None
    if os.path.exists(raw_path):
        return raw_path
    alt_path = REPORTS_DIR / raw_path
    if alt_path.exists():
        return str(alt_path)
    alt_path = (REPORTS_DIR / ".." / raw_path).resolve()
    if alt_path.exists():
        return str(alt_path)
    return None


@st.cache_resource(show_spinner=False)
def load_cache(path: str) -> Dict[int, float]:
    return load_evc(path)


def build_cache_options() -> List[Path]:
    return sorted(REPORTS_DIR.glob("*.evc"))


def reservoir_sample(keys: Iterable[int], sample_size: int, rng: random.Random) -> List[int]:
    sample: List[int] = []
    for i, key in enumerate(keys):
        if i < sample_size:
            sample.append(key)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                sample[j] = key
    return sample


def decode_key_state(key: int) -> Tuple[List[int], List[int], List[int], int, int]:
    A, B, P, diff, curP = decode_key(key)
    return list_cards(A), list_cards(B), list_cards(P), diff, curP


def build_strategy_table(actions: List[int], probs: np.ndarray, evs: np.ndarray) -> pd.DataFrame:
    rows = []
    for action, p, ev in zip(actions, probs, evs):
        rows.append({"card": action, "prob": float(p), "ev_vs_mix": float(ev)})
    return pd.DataFrame(rows)


def format_cards(cards: List[int]) -> str:
    return "[" + ", ".join(str(c) for c in sorted(cards)) + "]"


def main() -> None:
    st.set_page_config(page_title="GOPS EV Cache Explorer", layout="wide")
    st.title("GOPS EV Cache Explorer")

    cache_files = build_cache_options()
    cache_names = [p.name for p in cache_files]

    with st.sidebar:
        st.header("Cache")
        cache_choice = st.selectbox(
            "Cache file",
            cache_names,
            index=0 if cache_names else None,
            disabled=not cache_names,
        )
        custom_path = st.text_input("Custom cache path (optional)")
        if custom_path.strip():
            cache_path = resolve_cache_path(custom_path.strip())
        elif cache_choice:
            cache_path = str(REPORTS_DIR / cache_choice)
        else:
            cache_path = None

        if cache_path:
            meta = load_meta(cache_path)
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
            meta = None

        if "cache_ready" not in st.session_state:
            st.session_state.cache_ready = False
        if "cache_path_loaded" not in st.session_state:
            st.session_state.cache_path_loaded = None

        if cache_path != st.session_state.cache_path_loaded:
            st.session_state.cache_ready = False

        load_clicked = st.button("Load cache", disabled=not cache_path)
        if load_clicked:
            st.session_state.cache_ready = True
            st.session_state.cache_path_loaded = cache_path

        st.divider()
        st.header("State Input")
        if "max_card" not in st.session_state:
            st.session_state.max_card = int(meta.get("run", {}).get("N", 9)) if meta else 9
        st.number_input(
            "Max card value (N)",
            min_value=1,
            max_value=16,
            value=int(st.session_state.max_card),
            key="max_card",
        )
        mode = st.radio("Mode", ["Manual", "Key", "Sample"], horizontal=True, key="mode")

    if not cache_path:
        st.error("No cache file found. Add a .evc file under cpp-solver/reports or enter a custom path.")
        return

    if not st.session_state.cache_ready:
        st.info("Load a cache to explore EV matrices.")
        return

    with st.spinner("Loading cache..."):
        cache = load_cache(st.session_state.cache_path_loaded)

    max_card = int(st.session_state.max_card)
    card_options = list(range(1, max_card + 1))

    if "cardsA" not in st.session_state:
        st.session_state.cardsA = card_options[:]
        st.session_state.cardsB = card_options[:]
        st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.diff = 0
        st.session_state.cardsA_n = max_card
    elif st.session_state.get("cardsA_n") != max_card:
        st.session_state.cardsA = [c for c in st.session_state.cardsA if c in card_options] or card_options[:]
        st.session_state.cardsB = [c for c in st.session_state.cardsB if c in card_options] or card_options[:]
        if st.session_state.curP not in card_options:
            st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [c for c in st.session_state.cardsP if c in card_options and c != st.session_state.curP]
        if not st.session_state.cardsP:
            st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.cardsA_n = max_card

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
            st.write(
                {
                    "A": format_cards(cardsA),
                    "B": format_cards(cardsB),
                    "P": format_cards(cardsP),
                    "diff": diff,
                    "curP": curP,
                }
            )
            max_from_key = max(cardsA + cardsB + cardsP + ([curP] if curP else []), default=1)
            if max_from_key > max_card:
                st.session_state.max_card = max_from_key
            if st.button("Use decoded state in Manual mode"):
                st.session_state.cardsA = cardsA
                st.session_state.cardsB = cardsB
                st.session_state.cardsP = cardsP
                st.session_state.curP = curP
                st.session_state.diff = diff
                st.session_state.cardsA_n = max(st.session_state.max_card, max_from_key)
                st.session_state.mode = "Manual"
        else:
            st.info("Enter a cache key to decode a state.")
            return

    else:  # Sample
        if "sample_keys" not in st.session_state:
            st.session_state.sample_keys = []
        rng_seed = st.number_input("Sample RNG seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
        sample_size = st.number_input("Sample size", min_value=1, max_value=5000, value=200, step=1)
        if st.button("Build sample list"):
            rng = random.Random(int(rng_seed))
            with st.spinner("Sampling cache keys..."):
                st.session_state.sample_keys = reservoir_sample(cache.keys(), int(sample_size), rng)
        if not st.session_state.sample_keys:
            st.info("Build a sample list to browse random states.")
            return
        key_val = st.selectbox("Sampled state key", st.session_state.sample_keys)
        cardsA, cardsB, cardsP, diff, curP = decode_key_state(int(key_val))
        st.write(
            {
                "A": format_cards(cardsA),
                "B": format_cards(cardsB),
                "P": format_cards(cardsP),
                "diff": diff,
                "curP": curP,
            }
        )

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
    st.write(
        {
            "A": format_cards(cardsA),
            "B": format_cards(cardsB),
            "P": format_cards(cardsP),
            "diff": diff,
            "curP": curP,
            "key": state_key,
        }
    )

    mat = build_matrix(cache, A_mask, B_mask, P_mask, diff, curP)
    if not mat:
        st.error("Matrix build failed for this state (missing cache entries or invalid state).")
        return

    mat_np = np.array(mat, dtype=np.float64)
    df = pd.DataFrame(mat_np, index=cardsA, columns=cardsB)

    st.subheader("EV Matrix (A rows vs B columns)")
    st.dataframe(df, use_container_width=True)

    pA, vA = findBestStrategy(mat_np)
    pB, vB = findBestStrategy(-mat_np.T)

    st.subheader("Optimal Mixed Strategies")
    if pA is None or pB is None:
        st.error("Failed to compute optimal strategies.")
        return

    ev_a = mat_np @ pB
    ev_b = - (pA @ mat_np)

    colA, colB = st.columns(2)
    with colA:
        st.write(f"EV (A perspective): {vA:+.6f}")
        st.dataframe(build_strategy_table(cardsA, pA, ev_a), use_container_width=True)
    with colB:
        st.write(f"EV (B perspective): {-vA:+.6f}")
        st.dataframe(build_strategy_table(cardsB, pB, ev_b), use_container_width=True)


if __name__ == "__main__":
    main()
