import math

import numpy as np
import pandas as pd
import streamlit as st

from nrfi_data_logger import (
    read_json_file,
)


LEDGER_PATH = (
    "analytics/final_pregame_edge_ledger.json"
)

WINDOWS = [
    "Season",
    "L30",
    "L20",
    "L10",
]

MIN_TESTABLE_GAMES = 120
TRAIN_FRACTION = 0.70
RIDGE_LAMBDA = 1.0


def _number(
    value,
):
    try:
        number = float(
            value
        )

        if not math.isfinite(
            number
        ):
            return None

        return number

    except Exception:
        return None


def _clip_probability(
    value,
):
    value = float(
        value
    )

    return min(
        max(
            value,
            1e-6,
        ),
        1.0 - 1e-6,
    )


def _logit(
    probability,
):
    probability = _clip_probability(
        probability
    )

    return math.log(
        probability
        /
        (
            1.0
            - probability
        )
    )


def _sigmoid(
    values,
):
    values = np.clip(
        values,
        -35.0,
        35.0,
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(
                -values
            )
        )
    )


def _brier(
    y,
    p,
):
    return float(
        np.mean(
            (
                np.asarray(
                    p,
                    dtype=float,
                )
                -
                np.asarray(
                    y,
                    dtype=float,
                )
            )
            ** 2
        )
    )


def _fit_ridge_logistic(
    x,
    y,
    ridge_lambda=1.0,
    max_iter=100,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n_rows = x.shape[0]

    design = np.column_stack([
        np.ones(
            n_rows
        ),
        x,
    ])

    beta = np.zeros(
        design.shape[1],
        dtype=float,
    )

    penalty = np.eye(
        design.shape[1],
        dtype=float,
    )

    # Never penalize the intercept.
    penalty[
        0,
        0,
    ] = 0.0

    for _ in range(
        max_iter
    ):
        linear = design @ beta

        probability = _sigmoid(
            linear
        )

        weights = (
            probability
            *
            (
                1.0
                - probability
            )
        )

        weights = np.clip(
            weights,
            1e-6,
            None,
        )

        gradient = (
            design.T
            @
            (
                y
                - probability
            )
            -
            ridge_lambda
            *
            (
                penalty
                @ beta
            )
        )

        hessian = (
            -(
                design.T
                @
                (
                    design
                    *
                    weights[
                        :,
                        None
                    ]
                )
            )
            -
            ridge_lambda
            *
            penalty
        )

        try:
            step = np.linalg.solve(
                hessian,
                gradient,
            )
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(
                hessian
            ) @ gradient

        new_beta = (
            beta
            - step
        )

        if np.max(
            np.abs(
                new_beta
                - beta
            )
        ) < 1e-8:
            beta = new_beta
            break

        beta = new_beta

    return beta


def _predict_logistic(
    beta,
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    design = np.column_stack([
        np.ones(
            x.shape[0]
        ),
        x,
    ])

    return _sigmoid(
        design
        @ beta
    )


def _standardize_train_test(
    train,
    test,
):
    train = np.asarray(
        train,
        dtype=float,
    )

    test = np.asarray(
        test,
        dtype=float,
    )

    means = np.mean(
        train,
        axis=0,
    )

    stds = np.std(
        train,
        axis=0,
    )

    stds = np.where(
        stds < 1e-8,
        1.0,
        stds,
    )

    return (
        (
            train
            - means
        )
        / stds,
        (
            test
            - means
        )
        / stds,
    )


def _eligible_rows(
    games,
    window,
):
    rows = []

    required = [
        "nrfi_probability",
        "nrfi",
        f"Away Pitcher FI {window} Starts",
        f"Home Pitcher FI {window} Starts",
        f"Away Pitcher FI {window} Scoreless Opp 1st %",
        f"Home Pitcher FI {window} Scoreless Opp 1st %",
        f"Away Pitcher FI {window} Runs/Start",
        f"Home Pitcher FI {window} Runs/Start",
    ]

    for row in games or []:
        if (
            row.get(
                "grading_status"
            )
            !=
            "GRADED_FINAL_PREGAME"
        ):
            continue

        values = {
            key:
                _number(
                    row.get(
                        key
                    )
                )
            for key in required
            if key
            not in {
                "nrfi",
            }
        }

        if any(
            value is None
            for value in values.values()
        ):
            continue

        if min(
            values[
                f"Away Pitcher FI {window} Starts"
            ],
            values[
                f"Home Pitcher FI {window} Starts"
            ],
        ) < 5:
            continue

        nrfi_probability = (
            values[
                "nrfi_probability"
            ]
            / 100.0
        )

        rows.append({
            "game_date":
                str(
                    row.get(
                        "game_date",
                        "",
                    )
                ),

            "game_start_utc":
                str(
                    row.get(
                        "game_start_utc",
                        "",
                    )
                ),

            "game_id":
                row.get(
                    "game_id"
                ),

            "y":
                1.0
                if row.get(
                    "nrfi"
                ) is True
                else 0.0,

            "model_p":
                _clip_probability(
                    nrfi_probability
                ),

            "away_scoreless":
                (
                    values[
                        f"Away Pitcher FI {window} Scoreless Opp 1st %"
                    ]
                    / 100.0
                ),

            "home_scoreless":
                (
                    values[
                        f"Home Pitcher FI {window} Scoreless Opp 1st %"
                    ]
                    / 100.0
                ),

            "away_runs":
                values[
                    f"Away Pitcher FI {window} Runs/Start"
                ],

            "home_runs":
                values[
                    f"Home Pitcher FI {window} Runs/Start"
                ],

            "away_starts":
                values[
                    f"Away Pitcher FI {window} Starts"
                ],

            "home_starts":
                values[
                    f"Home Pitcher FI {window} Starts"
                ],
        })

    rows.sort(
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_start_utc"
            ],
            str(
                row[
                    "game_id"
                ]
            ),
        )
    )

    return rows


def _build_feature_matrix(
    rows,
    feature_set,
):
    output = []

    for row in rows:
        base_logit = _logit(
            row[
                "model_p"
            ]
        )

        features = [
            base_logit
        ]

        if feature_set in {
            "Scoreless %",
            "Scoreless % + Runs/Start",
        }:
            features.extend([
                row[
                    "away_scoreless"
                ],
                row[
                    "home_scoreless"
                ],
            ])

        if feature_set in {
            "Runs/Start",
            "Scoreless % + Runs/Start",
        }:
            features.extend([
                row[
                    "away_runs"
                ],
                row[
                    "home_runs"
                ],
            ])

        # Sample controls help prevent small rolling windows from
        # being treated as equally reliable as deeper histories.
        features.append(
            math.log1p(
                min(
                    row[
                        "away_starts"
                    ],
                    row[
                        "home_starts"
                    ],
                )
            )
        )

        output.append(
            features
        )

    return np.asarray(
        output,
        dtype=float,
    )


def _evaluate_candidate(
    rows,
    feature_set,
):
    n_games = len(
        rows
    )

    if n_games < MIN_TESTABLE_GAMES:
        return {
            "games":
                n_games,

            "train":
                None,

            "test":
                None,

            "raw_model_brier":
                None,

            "baseline_brier":
                None,

            "candidate_brier":
                None,

            "candidate_gain":
                None,

            "status":
                "COLLECTING",
        }

    split_index = int(
        n_games
        * TRAIN_FRACTION
    )

    split_index = max(
        split_index,
        60,
    )

    if (
        n_games
        - split_index
    ) < 30:
        split_index = (
            n_games
            - 30
        )

    train_rows = rows[
        :split_index
    ]

    test_rows = rows[
        split_index:
    ]

    y_train = np.asarray(
        [
            row[
                "y"
            ]
            for row in train_rows
        ],
        dtype=float,
    )

    y_test = np.asarray(
        [
            row[
                "y"
            ]
            for row in test_rows
        ],
        dtype=float,
    )

    raw_test_p = np.asarray(
        [
            row[
                "model_p"
            ]
            for row in test_rows
        ],
        dtype=float,
    )

    # Baseline challenger: training-only logistic recalibration
    # of the production model probability.
    baseline_train = np.asarray(
        [
            [
                _logit(
                    row[
                        "model_p"
                    ]
                )
            ]
            for row in train_rows
        ],
        dtype=float,
    )

    baseline_test = np.asarray(
        [
            [
                _logit(
                    row[
                        "model_p"
                    ]
                )
            ]
            for row in test_rows
        ],
        dtype=float,
    )

    (
        baseline_train_std,
        baseline_test_std,
    ) = _standardize_train_test(
        baseline_train,
        baseline_test,
    )

    baseline_beta = (
        _fit_ridge_logistic(
            baseline_train_std,
            y_train,
            ridge_lambda=
                RIDGE_LAMBDA,
        )
    )

    baseline_p = (
        _predict_logistic(
            baseline_beta,
            baseline_test_std,
        )
    )

    candidate_train = (
        _build_feature_matrix(
            train_rows,
            feature_set,
        )
    )

    candidate_test = (
        _build_feature_matrix(
            test_rows,
            feature_set,
        )
    )

    (
        candidate_train_std,
        candidate_test_std,
    ) = _standardize_train_test(
        candidate_train,
        candidate_test,
    )

    candidate_beta = (
        _fit_ridge_logistic(
            candidate_train_std,
            y_train,
            ridge_lambda=
                RIDGE_LAMBDA,
        )
    )

    candidate_p = (
        _predict_logistic(
            candidate_beta,
            candidate_test_std,
        )
    )

    raw_brier = _brier(
        y_test,
        raw_test_p,
    )

    baseline_brier = _brier(
        y_test,
        baseline_p,
    )

    candidate_brier = _brier(
        y_test,
        candidate_p,
    )

    candidate_gain = (
        baseline_brier
        - candidate_brier
    )

    return {
        "games":
            n_games,

        "train":
            len(
                train_rows
            ),

        "test":
            len(
                test_rows
            ),

        "raw_model_brier":
            raw_brier,

        "baseline_brier":
            baseline_brier,

        "candidate_brier":
            candidate_brier,

        "candidate_gain":
            candidate_gain,

        "status":
            (
                "POSITIVE"
                if candidate_gain > 0
                else "NEGATIVE"
            ),
    }


def _fmt_brier(
    value,
):
    if value is None:
        return "—"

    return f"{value:.6f}"


def _fmt_gain(
    value,
):
    if value is None:
        return "—"

    return f"{value:+.6f}"


def _sample_label(
    games,
):
    if games < MIN_TESTABLE_GAMES:
        return (
            f"Collecting "
            f"({games}/{MIN_TESTABLE_GAMES})"
        )

    if games < 250:
        return "Early"

    if games < 500:
        return "Developing"

    return "Stronger"


def _descriptive_spread(
    rows,
):
    if len(
        rows
    ) < 40:
        return None

    scored = []

    for row in rows:
        average_scoreless = (
            row[
                "away_scoreless"
            ]
            +
            row[
                "home_scoreless"
            ]
        ) / 2.0

        scored.append(
            (
                average_scoreless,
                row[
                    "y"
                ],
            )
        )

    scored.sort(
        key=lambda item:
            item[0]
    )

    quartile = max(
        len(
            scored
        )
        // 4,
        1,
    )

    low = scored[
        :quartile
    ]

    high = scored[
        -quartile:
    ]

    low_nrfi = (
        sum(
            item[1]
            for item in low
        )
        / len(
            low
        )
        * 100.0
    )

    high_nrfi = (
        sum(
            item[1]
            for item in high
        )
        / len(
            high
        )
        * 100.0
    )

    return {
        "low":
            low_nrfi,

        "high":
            high_nrfi,

        "spread":
            high_nrfi
            - low_nrfi,
    }


def render_pitcher_history_research(
    token,
    repo,
):
    st.subheader(
        "Rolling Pitcher 1st-Inning Research"
    )

    st.caption(
        "Season/L30/L20/L10 pitcher first-inning history is "
        "research-only and is not a Model v1 weight. This monitor "
        "tests whether those logged features add predictive value "
        "to the existing production probability on later unseen games."
    )

    loaded = read_json_file(
        token=token,
        repo=repo,
        path=LEDGER_PATH,
    )

    if not loaded:
        st.info(
            "No graded FINAL-game ledger exists yet. "
            "This research monitor will populate automatically."
        )

        return

    games = (
        loaded
        .get(
            "data",
            {},
        )
        .get(
            "games",
            [],
        )
    )

    summary_rows = []

    for window in WINDOWS:
        rows = _eligible_rows(
            games,
            window,
        )

        spread = _descriptive_spread(
            rows
        )

        for feature_set in [
            "Scoreless %",
            "Runs/Start",
            "Scoreless % + Runs/Start",
        ]:
            result = _evaluate_candidate(
                rows,
                feature_set,
            )

            summary_rows.append({
                "Window":
                    window,

                "Candidate":
                    feature_set,

                "Eligible Games":
                    result[
                        "games"
                    ],

                "Train":
                    (
                        result[
                            "train"
                        ]
                        if result[
                            "train"
                        ] is not None
                        else "—"
                    ),

                "Test":
                    (
                        result[
                            "test"
                        ]
                        if result[
                            "test"
                        ] is not None
                        else "—"
                    ),

                "Raw Model Brier":
                    _fmt_brier(
                        result[
                            "raw_model_brier"
                        ]
                    ),

                "Baseline Brier":
                    _fmt_brier(
                        result[
                            "baseline_brier"
                        ]
                    ),

                "Candidate Brier":
                    _fmt_brier(
                        result[
                            "candidate_brier"
                        ]
                    ),

                "Incremental Brier Gain":
                    _fmt_gain(
                        result[
                            "candidate_gain"
                        ]
                    ),

                "High-vs-Low Scoreless Quartile NRFI Spread":
                    (
                        f"{spread['spread']:+.1f} pp"
                        if spread is not None
                        else "—"
                    ),

                "Research Status":
                    _sample_label(
                        result[
                            "games"
                        ]
                    ),
            })

    st.dataframe(
        pd.DataFrame(
            summary_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "How to read this: positive Incremental Brier Gain means "
        "the rolling-history challenger beat a training-only baseline "
        "that already knows the production model probability. "
        "The test is chronological: earlier games train the challenger "
        "and later games remain unseen. No promotion should be considered "
        "from a small sample or from one isolated positive window."
    )

    current_counts = {}

    for window in WINDOWS:
        current_counts[
            window
        ] = len(
            _eligible_rows(
                games,
                window,
            )
        )

    if max(
        current_counts.values(),
        default=0,
    ) < MIN_TESTABLE_GAMES:
        count_text = ", ".join(
            f"{window} {count}"
            for window, count
            in current_counts.items()
        )

        st.info(
            "The challenger test is intentionally locked until "
            f"at least {MIN_TESTABLE_GAMES} eligible FINAL games "
            "exist for a window. Current eligible counts: "
            f"{count_text}."
        )
