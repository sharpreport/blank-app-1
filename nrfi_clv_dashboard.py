from statistics import mean

import pandas as pd
import streamlit as st

from nrfi_data_logger import (
    read_json_file,
)


LEDGER_PATH = (
    "analytics/final_pregame_edge_ledger.json"
)


def _number(value):
    try:
        return float(value)
    except Exception:
        return None


def _average(rows, key):
    values = [
        _number(
            row.get(key)
        )
        for row in rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    return (
        mean(values)
        if values
        else None
    )


def _pp(value):
    number = _number(value)

    return (
        f"{number:+.2f} pp"
        if number is not None
        else "—"
    )


def _pct(value):
    number = _number(value)

    return (
        f"{number:.1f}%"
        if number is not None
        else "—"
    )


def _eligible(games):
    return [
        row
        for row in games or []
        if (
            row.get(
                "grading_status"
            )
            ==
            "GRADED_FINAL_PREGAME"
            and
            row.get(
                "close_snapshot_time_utc"
            )
            is not None
            and
            _number(
                row.get(
                    "raw_price_clv_pp"
                )
            )
            is not None
        )
    ]


def _summary(rows):
    tracked = len(rows)

    beat = sum(
        1
        for row in rows
        if row.get(
            "beat_close"
        ) is True
    )

    beat_rate = (
        beat
        / tracked
        * 100.0
        if tracked
        else None
    )

    avg_close_minutes = _average(
        rows,
        "close_minutes_before_first_pitch",
    )

    return {
        "Tracked":
            tracked,

        "Beat Close":
            beat,

        "Beat Close %":
            _pct(
                beat_rate
            ),

        "Avg No-Vig CLV":
            _pp(
                _average(
                    rows,
                    "no_vig_clv_pp",
                )
            ),

        "Avg Best-Price CLV":
            _pp(
                _average(
                    rows,
                    "raw_price_clv_pp",
                )
            ),

        "Avg Entry Price Edge":
            _pp(
                _average(
                    rows,
                    "price_edge",
                )
            ),

        "Avg Price Edge at Close":
            _pp(
                _average(
                    rows,
                    "entry_price_edge_at_close",
                )
            ),

        "Avg Close Min Before First Pitch":
            (
                f"{avg_close_minutes:.1f}"
                if avg_close_minutes is not None
                else "—"
            ),
    }


def _band_rows(rows):
    bands = [
        ("2.0–2.9%", 2.0, 3.0),
        ("3.0–3.9%", 3.0, 4.0),
        ("4.0–4.9%", 4.0, 5.0),
        ("5.0–6.9%", 5.0, 7.0),
        ("7.0%+", 7.0, None),
    ]

    output = []

    for label, lower, upper in bands:
        selected = []

        for row in rows:
            edge = _number(
                row.get(
                    "price_edge"
                )
            )

            if edge is None:
                continue

            if edge < lower:
                continue

            if (
                upper is not None
                and
                edge >= upper
            ):
                continue

            selected.append(row)

        output.append({
            "Entry Price Edge":
                label,

            **_summary(
                selected
            ),
        })

    return output


def _side_rows(rows):
    output = []

    for side in (
        "NRFI",
        "YRFI",
    ):
        selected = [
            row
            for row in rows
            if row.get(
                "model_side"
            )
            == side
        ]

        output.append({
            "Entry Side":
                side,

            **_summary(
                selected
            ),
        })

    return output


def render_clv_dashboard(
    token,
    repo,
):
    st.subheader(
        "Near-Close / CLV Tracking"
    )

    st.caption(
        "One additional near-close market snapshot is collected only "
        "for FINAL games whose initial Price Edge was at least +2.0%. "
        "Positive CLV means the market moved toward the original model "
        "side after the entry snapshot."
    )

    loaded = read_json_file(
        token=token,
        repo=repo,
        path=LEDGER_PATH,
    )

    if not loaded:
        st.info(
            "No graded FINAL-game ledger exists yet. "
            "CLV tracking will populate automatically."
        )
        return

    games = (
        loaded
        .get("data", {})
        .get("games", [])
    )

    rows = _eligible(
        games
    )

    if not rows:
        st.info(
            "No completed games have both an initial FINAL entry "
            "snapshot and a qualified near-close snapshot yet."
        )
        return

    st.markdown(
        "**Overall CLV**"
    )

    st.dataframe(
        pd.DataFrame([
            _summary(rows)
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**CLV by Entry Price Edge**"
    )

    st.dataframe(
        pd.DataFrame(
            _band_rows(rows)
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**NRFI vs YRFI CLV**"
    )

    st.dataframe(
        pd.DataFrame(
            _side_rows(rows)
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "No-Vig CLV compares consensus no-vig probability at entry "
        "with the near-close consensus. Best-Price CLV compares the "
        "raw break-even probability of the best available entry price "
        "with the best available near-close price. Positive is better."
    )
