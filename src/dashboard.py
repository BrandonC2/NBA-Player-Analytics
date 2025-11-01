# src/dashboard.py
# Streamlit dashboard for PickPrizes — live lines → board → optimal entries
# Run: streamlit run src/dashboard.py

# --- import shim so running from repo root works (and relative imports inside modules keep working)
import os, sys, time, io, json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Local imports (package-style)
from src.prizepicks_connector import fetch_prizepicks_projections
from src.build_board_all import build_board, normalize_prop  # normalize_prop used in diagnostics
from src.optimizer import build_best_entries  # used under the hood by build_entries

st.set_page_config(page_title="PickPrizes Dashboard", layout="wide")

# ---------------------------
# Session slots
# ---------------------------
if "live_df" not in st.session_state:
    st.session_state["live_df"] = pd.DataFrame()
if "live_err" not in st.session_state:
    st.session_state["live_err"] = ""
if "board_df" not in st.session_state:
    st.session_state["board_df"] = pd.DataFrame()

# ---------------------------
# Helpers
# ---------------------------
def _sanitize_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame Arrow/JSON-friendly for st.dataframe."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # column names → strings
    out.columns = [str(c) for c in out.columns]

    # datetimes → ISO strings
    for col in out.columns:
        if is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")

    # object columns: stringify non-scalars and mixed types
    for col in out.columns:
        if is_object_dtype(out[col]):
            if out[col].map(lambda x: isinstance(x, (list, dict, set, tuple))).any():
                out[col] = out[col].map(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict, set, tuple))
                    else ("" if x is None else str(x))
                )
                continue
            types = out[col].map(lambda x: type(x).__name__ if x is not None else "None").unique()
            if len(types) > 2 or (len(types) == 2 and "str" not in types):
                out[col] = out[col].map(lambda x: "" if x is None else str(x))
            else:
                out[col] = out[col].where(out[col].notna(), "")

    # round floats for display
    for col in out.select_dtypes(include=[np.floating]).columns:
        out[col] = out[col].astype(float).round(6)

    return out


def _save_live_to_csv(df: pd.DataFrame, path: str) -> str:
    """Persist live_df to CSV with the columns build_board expects."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    need = ["player", "team", "prop", "line", "source_date", "direction", "side_lock", "pp_tags"]
    for c in need:
        if c not in out.columns:
            out[c] = ""
    out.to_csv(path, index=False)
    return path


@st.cache_data(show_spinner=False, ttl=60)  # auto-refresh after 60s
def _fetch_live(league: str, per_page: int, pages: int, sleep: float, tries: int, _nonce: float | None = None) -> pd.DataFrame:
    df = fetch_prizepicks_projections(
        league_name=league,
        league_id=None,
        per_page=per_page,
        pages=pages,
        base_sleep=sleep,
        max_tries=tries,
    )
    # Normalize expected columns so UI stays stable
    for col in ["player", "team", "prop", "line", "pp_tags", "side_lock", "source_date"]:
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_data(show_spinner=False)
def _build_board_cached(
    date_str: str,
    season: str,
    season_type: str,
    look_back: int,
    use_prizepicks: bool,
    merge_mode: str,
    in_csv: str,
    out_csv: str,
    debug: bool,
) -> pd.DataFrame:
    # build_board writes out_csv; we return the DataFrame for UI
    return build_board(
        date=date_str,
        season=season,
        in_csv=in_csv,
        out_csv=out_csv,
        look_back=look_back,
        season_type=season_type,
        use_prizepicks=use_prizepicks,
        merge_mode=merge_mode,
        debug=debug,
    )


def _to_csv_download(df: pd.DataFrame, filename: str, label: str):
    if df.empty:
        st.info("Nothing to download yet.")
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _locked_badge(row) -> str:
    return "🔒 OVER" if str(row.get("side_lock", "")).upper() == "OVER" else ""


def _chosen_side_badge(row) -> str:
    side = str(row.get("chosen_side", "")).upper()
    return "⬆️ OVER" if side == "OVER" else ("⬇️ UNDER" if side == "UNDER" else "")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Settings")

with st.sidebar.expander("Slate & Season", expanded=True):
    slate_date = st.date_input("Date", value=date.today())
    season = st.text_input("Season (e.g., 2025-26)", value="2025-26")
    season_type = st.selectbox("Season Type", ["Regular Season", "Pre Season", "Playoffs"], index=0)
    look_back = st.number_input("Look-back games (recent)", min_value=3, max_value=30, value=10, step=1)

with st.sidebar.expander("Data Sources", expanded=True):
    use_prizepicks = st.toggle("Use Live PrizePicks", value=True)
    merge_mode = st.selectbox("Merge Mode", ["replace", "union"], index=0)
    manual_csv = st.text_input("Manual CSV (fallback / union)", value="data/prizepicks_lines.csv")

with st.sidebar.expander("PrizePicks Fetch Tuning", expanded=False):
    league = st.text_input("League", value="NBA")
    per_page = st.slider("Per Page", 100, 800, 300, 50)
    pages = st.slider("Pages", 1, 10, 3, 1)
    base_sleep = st.slider("Base Sleep (s)", 0.2, 2.0, 0.6, 0.1)
    max_tries = st.slider("Max Tries", 3, 10, 6, 1)

with st.sidebar.expander("Load Existing Board", expanded=False):
    if st.button("Load board.csv (if exists)"):
        try:
            loaded = pd.read_csv("board.csv")
            st.session_state["board_df"] = loaded
            st.success(f"Loaded {len(loaded)} rows from board.csv")
        except Exception as e:
            st.error(f"Could not load board.csv: {e}")

debug = st.sidebar.toggle("Debug print", value=False)

# ---------------------------
# Header
# ---------------------------
st.title("🏀 PickPrizes — Live Board & Entries")
st.caption("Live PrizePicks → Board with probabilities & EV → Optimal entries. Demon/Goblin legs are OVER-locked automatically.")

# ---------------------------
# Step 1: Fetch PrizePicks Live Lines (robust)
# ---------------------------
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("1) Fetch PrizePicks Live Lines")
    if use_prizepicks:
        force_fresh = st.checkbox("Force fresh fetch (bypass cache)", value=True)
        fetch_clicked = st.button("Fetch Live Now", type="primary", use_container_width=True)
    else:
        force_fresh = False
        fetch_clicked = False
        st.info("Live PrizePicks is off. Turn it on in the sidebar to fetch fresh lines.")
with colB:
    st.subheader("Lock Tags")
    st.write("- **🔒 OVER**: Demon/Goblin tag → side locked to OVER\n- Others remain neutral")

if use_prizepicks and fetch_clicked:
    try:
        nonce = time.time() if force_fresh else None
        with st.spinner("Fetching live projections…"):
            live_df = _fetch_live(league, per_page, pages, base_sleep, max_tries, _nonce=nonce)
        st.session_state["live_df"] = live_df.copy()
        st.session_state["live_err"] = ""
        if live_df.empty:
            st.warning("No live lines fetched. Could be rate-limited or slate not posted yet.")
        else:
            lock_count = (live_df["side_lock"].astype(str).str.upper() == "OVER").sum()
            st.success(f"Fetched {len(live_df)} rows. Demon/Goblin OVER locks: {lock_count}")
    except Exception as e:
        st.session_state["live_err"] = repr(e)
        st.error(f"Live fetch failed: {e!r}")

# Always show last fetch (if any) + any error
if st.session_state["live_err"]:
    with st.expander("Fetch error details", expanded=False):
        st.code(st.session_state["live_err"])
if not st.session_state["live_df"].empty:
    st.dataframe(
        _sanitize_for_streamlit(st.session_state["live_df"].head(300)),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------
# Step 2: Build Board
# ---------------------------
st.subheader("2) Build Daily Board")

# allow forcing board build from cached live lines
use_cached_live_for_board = st.checkbox(
    "Use cached live lines for board build (skip fresh fetch inside build_board)",
    value=True if not st.session_state["live_df"].empty else False,
    help="When ON and live lines are fetched above, the board will be built from those lines directly."
)

build_clicked = st.button("Build Board", type="primary", use_container_width=True)

if build_clicked:
    with st.spinner("Building board with recent stats, probabilities, and EV…"):
        # Decide source routing
        local_in_csv = manual_csv
        local_use_pp = use_prizepicks
        local_merge_mode = merge_mode
        src_note = "[build_board fetches live (if enabled) or uses manual CSV]"

        if use_cached_live_for_board and not st.session_state["live_df"].empty:
            tmp_path = os.path.join(ROOT_DIR, ".dashboard_cache", "live_lines.csv")
            local_in_csv = _save_live_to_csv(st.session_state["live_df"], tmp_path)
            local_use_pp = False           # do not re-fetch inside build_board
            local_merge_mode = "replace"   # rely only on the cached live lines
            src_note = f"[using cached live lines: {os.path.relpath(tmp_path, ROOT_DIR)}]"

        board_df = _build_board_cached(
            date_str=slate_date.isoformat(),
            season=season,
            season_type=season_type,
            look_back=look_back,
            use_prizepicks=local_use_pp,
            merge_mode=local_merge_mode,
            in_csv=local_in_csv,
            out_csv="board.csv",
            debug=debug,
        )

    st.session_state["board_df"] = board_df.copy()
    if board_df.empty:
        st.warning(f"Board came back empty. {src_note}")
    else:
        st.success(f"Board built: {len(board_df)} rows. {src_note}")

# Always use the session board hereafter
board_df = st.session_state["board_df"].copy()

# --- Diagnostics: why the board might be empty/small ---
with st.expander("🔎 Diagnostics (why the board might be empty/small)"):
    diag_lines = pd.DataFrame()
    src_label = ""
    if not st.session_state["live_df"].empty and use_prizepicks:
        diag_lines = st.session_state["live_df"].copy()
        src_label = "live (PrizePicks)"
    else:
        try:
            diag_lines = pd.read_csv(manual_csv) if manual_csv else pd.DataFrame()
            src_label = f"manual CSV ({manual_csv})"
        except Exception as e:
            st.warning(f"Could not read manual CSV: {e}")

    if diag_lines.empty:
        st.write("No input lines available yet for diagnostics.")
    else:
        # Ensure columns
        for c in ["player", "team", "prop", "line", "pp_tags", "side_lock", "source_date"]:
            if c not in diag_lines.columns:
                diag_lines[c] = ""

        tmp = diag_lines.copy()
        tmp["prop_norm"] = tmp["prop"].apply(normalize_prop)

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        tmp["line_num"] = tmp["line"].map(_to_float)

        n_total = len(tmp)
        n_bad_prop = tmp["prop_norm"].isna().sum()
        n_bad_line = tmp["line_num"].isna().sum()
        unknown_props = (
            tmp.loc[tmp["prop_norm"].isna(), "prop"]
            .astype(str).str.upper().str.strip()
            .value_counts().head(20)
        )

        st.write(f"Input source for diagnostics: **{src_label}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input lines", n_total)
        c2.metric("Unknown props (not mapped)", n_bad_prop)
        c3.metric("Non-numeric lines", n_bad_line)
        c4.metric("Board rows built", len(board_df))

        if n_bad_prop > 0:
            st.write("Top unknown prop labels (up to 20):")
            st.dataframe(unknown_props.rename("count").to_frame(), use_container_width=True)
            st.caption("→ Add these labels to _CANON_ALIASES in build_board_all.py → normalize_prop().")

        if n_bad_line > 0:
            st.write("Examples with non-numeric line (first 10):")
            st.dataframe(
                tmp.loc[tmp["line_num"].isna(), ["player", "team", "prop", "line", "pp_tags"]].head(10),
                use_container_width=True, hide_index=True
            )

        st.write("Sample of normalized, numeric rows (first 10):")
        ready = tmp.loc[tmp["prop_norm"].notna() & tmp["line_num"].notna(),
                        ["player", "team", "prop", "prop_norm", "line", "pp_tags", "side_lock"]].head(10)
        st.dataframe(ready, use_container_width=True, hide_index=True)

# ---------------------------
# Board table + filters
# ---------------------------
if board_df.empty:
    st.info("No board yet. Click **Build Board** to generate today’s slate (see Diagnostics above if it stays empty).")
else:
    board_df = board_df.copy()
    board_df.insert(0, "lock", board_df.apply(_locked_badge, axis=1))
    board_df.insert(1, "pick", board_df.apply(_chosen_side_badge, axis=1))

    fcols = st.columns(5)
    with fcols[0]:
        team_filter = st.text_input("Filter Team (abbr)", value="")
    with fcols[1]:
        player_filter = st.text_input("Filter Player (substring)", value="")
    with fcols[2]:
        prop_filter = st.text_input("Filter Prop (e.g., PTS, PRA)", value="")
    with fcols[3]:
        min_p = st.slider("Min p_hit_for_ev", 0.50, 0.90, 0.58, 0.01)
    with fcols[4]:
        hide_locked = st.checkbox("Hide OVER-locked (Demon/Goblin)", value=False)

    filtered = board_df.copy()
    if team_filter.strip():
        filtered = filtered[filtered["team"].astype(str).str.contains(team_filter.strip(), case=False, regex=False)]
    if player_filter.strip():
        filtered = filtered[filtered["player"].astype(str).str.contains(player_filter.strip(), case=False, regex=True)]
    if prop_filter.strip():
        filtered = filtered[filtered["prop"].astype(str).str.contains(prop_filter.strip(), case=False, regex=False)]
    if hide_locked:
        filtered = filtered[filtered["side_lock"].astype(str).str.upper() != "OVER"]
    filtered = filtered[filtered["p_hit_for_ev"] >= min_p]

    filtered = filtered.sort_values(
        by=["ev_flex6_identical", "ev_flex5_identical", "p_hit_for_ev"],
        ascending=False
    ).reset_index(drop=True)

    st.caption(f"Showing {len(filtered)} legs (after filters).")
    st.dataframe(
        _sanitize_for_streamlit(filtered[[
            "lock", "pick", "date", "player", "team", "opponent", "prop", "display_prop", "line",
            "mu", "sigma", "p_over", "p_under", "chosen_side", "p_hit_for_ev",
            "ev_flex3_identical", "ev_flex5_identical", "ev_flex6_identical",
            "pp_tags", "side_lock", "source_date"
        ]]),
        use_container_width=True,
        hide_index=True,
    )
    _to_csv_download(filtered, "board_filtered.csv", "⬇️ Download Filtered Board CSV")

# ---------------------------
# Step 3: Build Entries (Power2 / Flex3 / Flex5 / Flex6)
# ---------------------------
st.subheader("3) Build Optimal Entries")

board_state = st.session_state.get("board_df", pd.DataFrame()).copy()
if board_state.empty:
    st.info("Build the board first to generate entries.")
else:
    ecols = st.columns(4)
    with ecols[0]:
        entry_type = st.selectbox("Entry Type", ["power2", "flex3", "flex5", "flex6"], index=2)
    with ecols[1]:
        ent_min_p = st.slider("Min per-leg p_hit", 0.50, 0.90, 0.58, 0.01)
    with ecols[2]:
        max_candidates = st.slider("Max candidate legs", 10, 100, 36, 2)
    with ecols[3]:
        top_n = st.slider("Top-N combos to output", 10, 250, 100, 10)

    exclude_locked_for_entries = st.checkbox(
        "Exclude OVER-locked legs when building entries (neutral only)",
        value=True
    )

    build_entries_clicked = st.button("Generate Entries", type="primary", use_container_width=True)

    if build_entries_clicked:
        board_for_opt = board_state.copy()
        if exclude_locked_for_entries:
            board_for_opt = board_for_opt[board_for_opt["side_lock"].astype(str).str.upper() != "OVER"]
        board_for_opt = board_for_opt[board_for_opt["p_hit_for_ev"] >= ent_min_p].reset_index(drop=True)

        if board_for_opt.empty:
            st.warning("No legs meet the criteria (after excluding locked and applying min p).")
        else:
            with st.spinner("Searching best combos…"):
                combos = build_best_entries(
                    board=board_for_opt,
                    entry_type=entry_type,
                    min_p=ent_min_p,
                    max_candidates=max_candidates,
                    top_n=top_n,
                )
            if combos.empty:
                st.warning("No combos generated. Try lowering min p or increasing max candidates.")
            else:
                st.success(f"Built {len(combos)} {entry_type} entries.")
                st.dataframe(_sanitize_for_streamlit(combos.head(50)), use_container_width=True, hide_index=True)
                _to_csv_download(combos, f"entries_{entry_type}.csv", f"⬇️ Download {entry_type} Entries CSV")

# Footer
st.markdown("---")
st.caption("PickPrizes — live NBA props analysis • Demon/Goblin legs are OVER-locked automatically.")
