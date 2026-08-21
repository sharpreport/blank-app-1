from statistics import mean

import pandas as pd
import streamlit as st

from nrfi_data_logger import (
    read_json_file,
)


LEDGER_PATH = (
    "analytics/final_pregame_edge_ledger.json"
)


def _number(
    value,
):
    try:
        return float(
            value
        )
    except Exception:
        return None


def _pct(
    value,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:.1f}%"


def _signed_pct(
    value,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:+.1f}%"


def _units(
    value,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:+.2f}"


def _sample_label(
    bets,
):
    if bets < 25:
        return "Very small"

    if bets < 50:
        return "Small"

    if bets < 100:
        return "Developing"

    if bets < 250:
        return "Moderate"

    return "Larger"


def _average(
    rows,
    key,
):
    values = [
        _number(
            row.get(
                key
            )
        )
        for row in rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return mean(
        values
    )


def _summarize(
    rows,
):
    bets = len(
        rows
    )

    wins = sum(
        1
        for row in rows
        if row.get(
            "model_side_won"
        ) is True
    )

    losses = sum(
        1
        for row in rows
        if row.get(
            "model_side_won"
        ) is False
    )

    profit_values = [
        _number(
            row.get(
                "model_side_profit_units_1u_risk"
            )
        )
        for row in rows
    ]

    profit_values = [
        value
        for value in profit_values
        if value is not None
    ]

    units = sum(
        profit_values
    )

    roi = (
        units
        / bets
        * 100.0
        if bets
        else None
    )

    win_rate = (
        wins
        / bets
        * 100.0
        if bets
        else None
    )

    return {
        "Bets":
            bets,

        "Wins":
            wins,

        "Losses":
            losses,

        "Win Rate":
            _pct(
                win_rate
            ),

        "Units":
            _units(
                units
            ),

        "ROI":
            _signed_pct(
                roi
            ),

        "Avg Model":
            _pct(
                _average(
                    rows,
                    "model_probability",
                )
            ),

        "Avg Break-Even":
            _pct(
                _average(
                    rows,
                    "break_even",
                )
            ),

        "Avg Price Edge":
            _signed_pct(
                _average(
                    rows,
                    "price_edge",
                )
            ),

        "Avg Market Edge":
            _signed_pct(
                _average(
                    rows,
                    "market_edge",
                )
            ),

        "Sample":
            _sample_label(
                bets
            ),
    }


def _eligible_rows(
    games,
):
    rows = []

    for row in games or []:
        if (
            row.get(
                "grading_status"
            )
            !=
            "GRADED_FINAL_PREGAME"
        ):
            continue

        if row.get(
            "model_side"
        ) not in {
            "NRFI",
            "YRFI",
        }:
            continue

        if _number(
            row.get(
                "price_edge"
            )
        ) is None:
            continue

        if _number(
            row.get(
                "model_side_profit_units_1u_risk"
            )
        ) is None:
            continue

        rows.append(
            row
        )

    return rows


def _price_edge_bands(
    rows,
):
    bands = [
        (
            "< 0%",
            None,
            0.0,
        ),
        (
            "0.0–1.9%",
            0.0,
            2.0,
        ),
        (
            "2.0–2.9%",
            2.0,
            3.0,
        ),
        (
            "3.0–3.9%",
            3.0,
            4.0,
        ),
        (
            "4.0–4.9%",
            4.0,
            5.0,
        ),
        (
            "5.0–6.9%",
            5.0,
            7.0,
        ),
        (
            "7.0%+",
            7.0,
            None,
        ),
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

            if (
                lower is not None
                and
                edge < lower
            ):
                continue

            if (
                upper is not None
                and
                edge >= upper
            ):
                continue

            selected.append(
                row
            )

        summary = _summarize(
            selected
        )

        output.append({
            "Price Edge Band":
                label,

            **summary,
        })

    return output


def _threshold_rows(
    rows,
):
    thresholds = [
        0.0,
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
    ]

    output = []

    for threshold in thresholds:
        selected = [
            row
            for row in rows
            if (
                _number(
                    row.get(
                        "price_edge"
                    )
                )
                is not None
                and
                _number(
                    row.get(
                        "price_edge"
                    )
                )
                >= threshold
            )
        ]

        summary = _summarize(
            selected
        )

        output.append({
            "Minimum Price Edge":
                f"+{threshold:.0f}%",

            **summary,
        })

    return output


def _side_rows(
    rows,
):
    output = []

    for side in [
        "NRFI",
        "YRFI",
    ]:
        selected = [
            row
            for row in rows
            if row.get(
                "model_side"
            ) == side
        ]

        output.append({
            "Model Side":
                side,

            **_summarize(
                selected
            ),
        })

    return output


def render_edge_performance_dashboard(
    token,
    repo,
):
    st.subheader(
        "Live FINAL Edge Performance"
    )

    st.caption(
        "Forward-tracked performance from saved FINAL pregame "
        "snapshots only. This does not backfill earlier 2026 games. "
        "ROI uses one unit risked at the best price saved before first pitch."
    )


    loaded = read_json_file(
        token=token,
        repo=repo,
        path=LEDGER_PATH,
    )

    if not loaded:
        st.info(
            "No graded FINAL pregame games are in the performance "
            "ledger yet. This section will populate automatically "
            "after saved FINAL games are completed and graded."
        )

        return


    payload = loaded.get(
        "data",
        {}
    )

    rows = _eligible_rows(
        payload.get(
            "games",
            []
        )
    )

    if not rows:
        st.info(
            "The performance ledger exists, but there are no eligible "
            "FINAL pregame observations yet."
        )

        return


    dates = sorted(
        {
            str(
                row.get(
                    "game_date"
                )
            )
            for row in rows
            if row.get(
                "game_date"
            )
        }
    )


    overall = _summarize(
        rows
    )

    overall_row = {
        "Tracked From":
            (
                dates[0]
                if dates
                else "—"
            ),

        "Through":
            (
                dates[-1]
                if dates
                else "—"
            ),

        **overall,
    }


    st.markdown(
        "**Overall FINAL Model-Side Results**"
    )

    st.dataframe(
        pd.DataFrame(
            [
                overall_row
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "**Performance by Price Edge Band**"
    )

    st.dataframe(
        pd.DataFrame(
            _price_edge_bands(
                rows
            )
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "**Cumulative Minimum Price Edge Thresholds**"
    )

    st.caption(
        "This table is the key threshold test: it shows what would "
        "have happened if the minimum required Price Edge had been "
        "+0%, +2%, +3%, +4%, +5%, or +7%."
    )

    st.dataframe(
        pd.DataFrame(
            _threshold_rows(
                rows
            )
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "**NRFI vs YRFI — Model Side**"
    )

    st.dataframe(
        pd.DataFrame(
            _side_rows(
                rows
            )
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "Sample labels are descriptive only: Very small <25 bets, "
        "Small 25–49, Developing 50–99, Moderate 100–249, "
        "Larger 250+. Do not choose an edge cutoff from a tiny sample."
    )
