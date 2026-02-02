import json
import sys
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPORTS_DIR = Path(__file__).resolve().parent
if str(REPORTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTS_DIR))

from common import (
    build_matrix,
    cmp,
    decode_key,
    encode_key,
    list_cards,
    list_to_mask,
    load_evc,
    load_meta,
)
from linprog import findBestStrategy


@st.cache_resource(show_spinner=False)
def load_cache(path: str) -> Dict[int, float]:
    return load_evc(path)


def build_cache_options() -> List[Path]:
    return sorted(REPORTS_DIR.glob("*.evc"))


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


def format_cards(cards: List[int]) -> str:
    return "[" + ", ".join(str(c) for c in sorted(cards)) + "]"


def render_state_summary(
    cardsA: List[int],
    cardsB: List[int],
    cardsP: List[int],
    diff: int,
    curP: int,
    *,
    key: Optional[int] = None,
) -> None:
    data = {
        "A": format_cards(cardsA),
        "B": format_cards(cardsB),
        "P": format_cards(cardsP),
        "diff": diff,
        "curP": curP,
    }
    if key is not None:
        data["key"] = key
    st.write(data)


def display_value(ev: float, display_percent: bool) -> float:
    return 50.0 + 50.0 * ev if display_percent else ev


def build_strategy_table(actions: List[int], probs: np.ndarray, evs: np.ndarray, display_percent: bool) -> pd.DataFrame:
    display_col = "win_pct_vs_mix" if display_percent else "ev_vs_mix"
    rows = []
    for action, p, ev in zip(actions, probs, evs):
        value = display_value(float(ev), display_percent)
        rows.append(
            {
                "card": action,
                "prob": round(float(p), 4),
                display_col: round(value, 2 if display_percent else 4),
            }
        )
    return pd.DataFrame(rows)


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


def resolve_next_prize(selection: object, options: List[int]) -> Optional[int]:
    if not options:
        return None
    if isinstance(selection, str) and selection.lower() == "random":
        return random.choice(options)
    try:
        return int(selection)
    except (TypeError, ValueError):
        return None


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


def render_copy_button(text: str, label: str) -> None:
    if "copy_counter" not in st.session_state:
        st.session_state.copy_counter = 0
    st.session_state.copy_counter += 1
    block_id = f"copy_btn_{st.session_state.copy_counter}"
    payload = json.dumps(text)
    safe_label = json.dumps(label)
    components.html(
        f"""
        <div style="display:flex;justify-content:flex-end;align-items:center;height:28px;">
          <button id="{block_id}_btn" type="button" title={safe_label}
            style="border:1px solid #ccc;border-radius:6px;padding:2px 6px;background:#fff;cursor:pointer;">
            &#128203;
          </button>
          <span id="{block_id}_status" style="font-size:12px;color:#666;margin-left:6px;"></span>
        </div>
        <script>
          (function() {{
            const btn = document.getElementById("{block_id}_btn");
            const status = document.getElementById("{block_id}_status");
            btn.addEventListener("click", () => {{
              if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText({payload}).then(() => {{
                  status.textContent = "Copied";
                  setTimeout(() => status.textContent = "", 1200);
                }}).catch(() => {{
                  status.textContent = "Copy failed";
                }});
              }} else {{
                status.textContent = "Clipboard unavailable";
              }}
            }});
          }})();
        </script>
        """,
        height=30,
    )


def render_section_header(title: str, caption: Optional[str], copy_text: str, copy_label: str) -> None:
    header_cols = st.columns([8, 1])
    with header_cols[0]:
        st.subheader(title)
        if caption:
            st.caption(caption)
    with header_cols[1]:
        render_copy_button(copy_text, copy_label)


def render_strategy_panel(
    title: str,
    value_label: str,
    df: pd.DataFrame,
    copy_label: str,
    support_cards: Optional[set[int]] = None,
) -> None:
    title_cols = st.columns([6, 1])
    with title_cols[0]:
        st.write(f"{title}: {value_label}")
    with title_cols[1]:
        render_copy_button(df.to_csv(index=False), copy_label)
    if support_cards:
        st.dataframe(style_support_rows(df, support_cards), width="stretch", hide_index=True)
    else:
        st.dataframe(df, width="stretch", hide_index=True)


def get_highlight_colors() -> Dict[str, str]:
    base = st.get_option("theme.base")
    if base == "dark":
        return {
            "row_bg": "#1f3d1f",
            "row_fg": "#e5f5e5",
            "cell_bg": "#4a3b0b",
            "cell_fg": "#fff2cc",
        }
    return {
        "row_bg": "#e8f7e8",
        "row_fg": "#1f3d1f",
        "cell_bg": "#fff2a8",
        "cell_fg": "#5a4b00",
    }


def style_support_rows(df: pd.DataFrame, support_cards: set[int]):
    colors = get_highlight_colors()

    def _row_style(row: pd.Series) -> List[str]:
        if row.get("card") in support_cards:
            return [f"background-color: {colors['row_bg']}; color: {colors['row_fg']}"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


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
        cache_path = str(REPORTS_DIR / cache_choice) if cache_choice else None

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

        run_n = meta.get("run", {}).get("N") if meta else None
        if "cache_path_loaded" not in st.session_state:
            st.session_state.cache_path_loaded = cache_path
            if run_n is not None:
                st.session_state.max_card = int(run_n)
        elif cache_path != st.session_state.cache_path_loaded:
            st.session_state.cache_path_loaded = cache_path
            if run_n is not None:
                st.session_state.max_card = int(run_n)

        st.divider()
        st.header("State Input")
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

    if not cache_path:
        st.error("No cache file found. Add a .evc file under cpp-solver/reports.")
        return

    with st.spinner("Loading cache..."):
        cache = load_cache(st.session_state.cache_path_loaded)

    max_card = int(st.session_state.max_card)
    card_options = list(range(1, max_card + 1))
    st.session_state.card_options = card_options

    if "cardsA" not in st.session_state:
        st.session_state.cardsA = card_options[:]
        st.session_state.cardsB = card_options[:]
        st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.diff = 0
        st.session_state.cardsA_n = max_card
        st.session_state.curP_prev = st.session_state.curP
    elif st.session_state.get("cardsA_n") != max_card:
        st.session_state.cardsA = [c for c in st.session_state.cardsA if c in card_options] or card_options[:]
        st.session_state.cardsB = [c for c in st.session_state.cardsB if c in card_options] or card_options[:]
        if st.session_state.curP not in card_options:
            st.session_state.curP = card_options[0] if card_options else 1
        st.session_state.cardsP = [c for c in st.session_state.cardsP if c in card_options and c != st.session_state.curP]
        if not st.session_state.cardsP:
            st.session_state.cardsP = [c for c in card_options if c != st.session_state.curP]
        st.session_state.cardsA_n = max_card
        st.session_state.curP_prev = st.session_state.curP

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
            st.session_state.cache_path_loaded,
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

    st.session_state.copy_counter = 0
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
        if (
            "advance_next_prize" not in st.session_state
            or st.session_state.advance_next_prize not in advance_options
        ):
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
            label = f"{value:.2f}%" if display_percent else f"{value:+.4f}"
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
    ev_b = - (pA @ mat_np)
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


if __name__ == "__main__":
    main()
