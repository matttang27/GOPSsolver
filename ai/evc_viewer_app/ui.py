from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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


def build_strategy_table(actions: List[int], probs, evs, display_percent: bool) -> pd.DataFrame:
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


def render_copy_button(text: str, label: str) -> None:
    block_id = f"copy_btn_{abs(hash((label, len(text))))}"
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
