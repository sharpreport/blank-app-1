import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nrfi_data_logger import (
    read_json_file,
    upsert_json_file,
)

from nrfi_probability_model import (
    build_half_features,
)


LEDGER_PATH = (
    "analytics/final_pregame_edge_ledger.json"
)

GOVERNANCE_PATH = (
    "analytics/model_governance.json"
)

CHALLENGER_PATH = (
    "models/challenger_recent_core.json"
)

INCUMBENT_MODEL_PATH = Path(
    "sharpreport_nrfi_model_v1.json"
)

FEATURES = [
    "p_xwoba",
    "p_k_pct",
    "o_xwoba",
    "o_k_pct",
    "o_bb_pct",
    "o_barrel_pct",
    "p_log_pa",
    "p_no_prior_pa",
    "o_log_combined_pa",
    "o_log_min_pa",
    "o_missing_core",
    "park_runs",
    "park_missing",
]

MIN_ELIGIBLE_GAMES = 300

RIDGE_LAMBDA = 1.0

# Promotion review requires all of these. The production model is
# NEVER changed automatically.
MIN_BRIER_GAIN = 0.00010
MIN_LOGLOSS_GAIN = 0.00020
MAX_ECE_WORSENING = 0.005
MIN_FOLD_WINS = 2


def _number(
    value,
):
    if value is None:
        return None

    try:
        number = float(
            value
        )
    except Exception:
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


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
    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return float(
        np.mean(
            (
                p
                - y
            )
            ** 2
        )
    )


def _log_loss(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        [
            _clip_probability(
                value
            )
            for value in p
        ],
        dtype=float,
    )

    return float(
        -np.mean(
            y
            * np.log(
                p
            )
            +
            (
                1.0
                - y
            )
            * np.log(
                1.0
                - p
            )
        )
    )


def _calibration_ece(
    y,
    p,
    bins=5,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if len(
        y
    ) == 0:
        return None

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(
        y
    )

    ece = 0.0

    for index in range(
        bins
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == (
            bins - 1
        ):
            mask = (
                (p >= lower)
                &
                (p <= upper)
            )
        else:
            mask = (
                (p >= lower)
                &
                (p < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        predicted = float(
            p[
                mask
            ].mean()
        )

        actual = float(
            y[
                mask
            ].mean()
        )

        ece += (
            count
            / total
            *
            abs(
                predicted
                - actual
            )
        )

    return float(
        ece
    )


def _american_profit(
    price,
    won,
):
    price = _number(
        price
    )

    if price is None or price == 0:
        return None

    if not won:
        return -1.0

    if price > 0:
        return (
            price
            / 100.0
        )

    return (
        100.0
        / abs(
            price
        )
    )


def _model_file_metadata():
    if not INCUMBENT_MODEL_PATH.exists():
        return {
            "model_name":
                "SharpReport NRFI/YRFI Model v1",

            "version":
                "1.0",

            "training_end_date":
                "2026-08-20",

            "sha256":
                None,
        }

    raw = INCUMBENT_MODEL_PATH.read_bytes()

    try:
        payload = json.loads(
            raw.decode(
                "utf-8"
            )
        )
    except Exception:
        payload = {}

    return {
        "model_name":
            payload.get(
                "model_name",
                "SharpReport NRFI/YRFI Model v1",
            ),

        "version":
            payload.get(
                "version",
                "1.0",
            ),

        "training_end_date":
            payload.get(
                "training_end_date",
                "2026-08-20",
            ),

        "training_games":
            payload.get(
                "training_games"
            ),

        "sha256":
            hashlib.sha256(
                raw
            ).hexdigest(),
    }


def _has_logged_core_inputs(
    row,
):
    required_keys = [
        "Away Pitcher xwOBA",
        "Away Pitcher K%",
        "Away Pitcher PA",
        "Home Pitcher xwOBA",
        "Home Pitcher K%",
        "Home Pitcher PA",
        "Away Top4 xwOBA",
        "Away Top4 K%",
        "Away Top4 BB%",
        "Away Top4 Barrel%",
        "Away Top4 Combined PA",
        "Away Top4 Min PA",
        "Away Top4 Complete Core",
        "Home Top4 xwOBA",
        "Home Top4 K%",
        "Home Top4 BB%",
        "Home Top4 Barrel%",
        "Home Top4 Combined PA",
        "Home Top4 Min PA",
        "Home Top4 Complete Core",
        "Model Run Factor",
    ]

    return all(
        key in row
        for key in required_keys
    )


def _eligible_games(
    ledger_games,
):
    rows = []

    for row in ledger_games or []:
        if (
            row.get(
                "grading_status"
            )
            !=
            "GRADED_FINAL_PREGAME"
        ):
            continue

        if not _has_logged_core_inputs(
            row
        ):
            continue

        baseline_nrfi = _number(
            row.get(
                "nrfi_probability"
            )
        )

        if baseline_nrfi is None:
            continue

        if row.get(
            "nrfi"
        ) not in {
            True,
            False,
        }:
            continue

        rows.append(
            row
        )

    rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "game_date",
                    ""
                )
            ),
            str(
                row.get(
                    "game_start_utc",
                    ""
                )
            ),
            str(
                row.get(
                    "game_id",
                    ""
                )
            ),
        )
    )

    return rows


def _half_features(
    row,
    half,
):
    park_runs = row.get(
        "Model Run Factor"
    )

    if half == "Top":
        return build_half_features(
            pitcher_xwoba=
                row.get(
                    "Home Pitcher xwOBA"
                ),

            pitcher_k_pct=
                row.get(
                    "Home Pitcher K%"
                ),

            pitcher_pa=
                row.get(
                    "Home Pitcher PA"
                ),

            offense_xwoba=
                row.get(
                    "Away Top4 xwOBA"
                ),

            offense_k_pct=
                row.get(
                    "Away Top4 K%"
                ),

            offense_bb_pct=
                row.get(
                    "Away Top4 BB%"
                ),

            offense_barrel_pct=
                row.get(
                    "Away Top4 Barrel%"
                ),

            offense_combined_pa=
                row.get(
                    "Away Top4 Combined PA"
                ),

            offense_min_pa=
                row.get(
                    "Away Top4 Min PA"
                ),

            offense_complete_core=
                row.get(
                    "Away Top4 Complete Core"
                ),

            park_runs=
                park_runs,
        )

    return build_half_features(
        pitcher_xwoba=
            row.get(
                "Away Pitcher xwOBA"
            ),

        pitcher_k_pct=
            row.get(
                "Away Pitcher K%"
            ),

        pitcher_pa=
            row.get(
                "Away Pitcher PA"
            ),

        offense_xwoba=
            row.get(
                "Home Top4 xwOBA"
            ),

        offense_k_pct=
            row.get(
                "Home Top4 K%"
            ),

        offense_bb_pct=
            row.get(
                "Home Top4 BB%"
            ),

        offense_barrel_pct=
            row.get(
                "Home Top4 Barrel%"
            ),

        offense_combined_pa=
            row.get(
                "Home Top4 Combined PA"
            ),

        offense_min_pa=
            row.get(
                "Home Top4 Min PA"
            ),

        offense_complete_core=
            row.get(
                "Home Top4 Complete Core"
            ),

        park_runs=
            park_runs,
    )


def _half_outcome(
    row,
    half,
):
    if half == "Top":
        runs = _number(
            row.get(
                "away_first_inning_runs"
            )
        )
    else:
        runs = _number(
            row.get(
                "home_first_inning_runs"
            )
        )

    if runs is None:
        return None

    return (
        1.0
        if runs > 0
        else 0.0
    )


def _fit_half_model(
    games,
    half,
):
    feature_rows = []
    outcomes = []

    for row in games:
        outcome = _half_outcome(
            row,
            half,
        )

        if outcome is None:
            continue

        feature_rows.append(
            _half_features(
                row,
                half,
            )
        )

        outcomes.append(
            outcome
        )

    if not feature_rows:
        raise ValueError(
            "No half-inning training rows."
        )

    raw = pd.DataFrame(
        feature_rows,
        columns=FEATURES,
    )

    means = {}
    stds = {}

    standardized = raw.copy()

    for feature in FEATURES:
        standardized[
            feature
        ] = pd.to_numeric(
            standardized[
                feature
            ],
            errors="coerce",
        )

        if standardized[
            feature
        ].notna().any():
            mean_value = float(
                standardized[
                    feature
                ].mean()
            )
        else:
            mean_value = 0.0

        standardized[
            feature
        ] = (
            standardized[
                feature
            ]
            .fillna(
                mean_value
            )
        )

        std_value = float(
            standardized[
                feature
            ].std(
                ddof=0
            )
        )

        if (
            not math.isfinite(
                std_value
            )
            or
            std_value < 1e-12
        ):
            std_value = 1.0

        standardized[
            feature
        ] = (
            standardized[
                feature
            ]
            - mean_value
        ) / std_value

        means[
            feature
        ] = mean_value

        stds[
            feature
        ] = std_value

    x = standardized[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y = np.asarray(
        outcomes,
        dtype=float,
    )

    design = np.column_stack([
        np.ones(
            len(
                x
            )
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

    penalty[
        0,
        0
    ] = 0.0

    for _ in range(
        100
    ):
        linear = (
            design
            @ beta
        )

        probability = _sigmoid(
            linear
        )

        weights = np.clip(
            probability
            *
            (
                1.0
                - probability
            ),
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
            RIDGE_LAMBDA
            *
            (
                penalty
                @ beta
            )
        )

        hessian = (
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
            +
            RIDGE_LAMBDA
            *
            penalty
        )

        try:
            step = np.linalg.solve(
                hessian,
                gradient,
            )
        except np.linalg.LinAlgError:
            step = (
                np.linalg.pinv(
                    hessian
                )
                @ gradient
            )

        new_beta = (
            beta
            + step
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

    return {
        "rows":
            int(
                len(
                    y
                )
            ),

        "base_scoring_rate":
            float(
                y.mean()
            ),

        "intercept":
            float(
                beta[
                    0
                ]
            ),

        "coefficients":
            {
                feature:
                    float(
                        beta[
                            index + 1
                        ]
                    )
                for index, feature
                in enumerate(
                    FEATURES
                )
            },

        "means":
            means,

        "stds":
            stds,
    }


def _predict_half(
    model,
    features,
):
    linear = float(
        model[
            "intercept"
        ]
    )

    for feature in FEATURES:
        raw = _number(
            features.get(
                feature
            )
        )

        mean_value = float(
            model[
                "means"
            ][
                feature
            ]
        )

        std_value = float(
            model[
                "stds"
            ][
                feature
            ]
        )

        if raw is None:
            raw = mean_value

        if abs(
            std_value
        ) < 1e-12:
            std_value = 1.0

        z_value = (
            raw
            - mean_value
        ) / std_value

        linear += (
            float(
                model[
                    "coefficients"
                ][
                    feature
                ]
            )
            * z_value
        )

    return float(
        _sigmoid(
            np.asarray([
                linear
            ])
        )[
            0
        ]
    )


def _fit_candidate(
    train_games,
):
    return {
        "Top":
            _fit_half_model(
                train_games,
                "Top",
            ),

        "Bottom":
            _fit_half_model(
                train_games,
                "Bottom",
            ),
    }


def _candidate_nrfi(
    candidate,
    row,
):
    top_p = _predict_half(
        candidate[
            "Top"
        ],
        _half_features(
            row,
            "Top",
        ),
    )

    bottom_p = _predict_half(
        candidate[
            "Bottom"
        ],
        _half_features(
            row,
            "Bottom",
        ),
    )

    return (
        1.0
        - top_p
    ) * (
        1.0
        - bottom_p
    )


def _roi_for_probabilities(
    rows,
    probabilities,
):
    profits = []

    for row, nrfi_probability in zip(
        rows,
        probabilities,
    ):
        if (
            nrfi_probability
            >= 0.5
        ):
            side = "NRFI"
            price = row.get(
                "best_nrfi_price"
            )
            won = (
                row.get(
                    "nrfi"
                )
                is True
            )
        else:
            side = "YRFI"
            price = row.get(
                "best_yrfi_price"
            )
            won = (
                row.get(
                    "yrfi"
                )
                is True
            )

        profit = _american_profit(
            price,
            won,
        )

        if profit is not None:
            profits.append(
                profit
            )

    if not profits:
        return {
            "bets":
                0,

            "units":
                None,

            "roi":
                None,
        }

    units = float(
        sum(
            profits
        )
    )

    return {
        "bets":
            len(
                profits
            ),

        "units":
            units,

        "roi":
            units
            / len(
                profits
            )
            * 100.0,
    }


def _evaluate_walk_forward(
    games,
):
    n_games = len(
        games
    )

    fold_ranges = [
        (
            0.50,
            2.0 / 3.0,
        ),
        (
            2.0 / 3.0,
            5.0 / 6.0,
        ),
        (
            5.0 / 6.0,
            1.0,
        ),
    ]

    all_y = []
    all_incumbent = []
    all_candidate = []
    all_test_rows = []
    fold_rows = []

    for fold_number, (
        train_fraction,
        test_end_fraction,
    ) in enumerate(
        fold_ranges,
        start=1,
    ):
        train_end = int(
            n_games
            * train_fraction
        )

        test_end = int(
            n_games
            * test_end_fraction
        )

        if fold_number == 3:
            test_end = n_games

        train_rows = games[
            :train_end
        ]

        test_rows = games[
            train_end:
            test_end
        ]

        if (
            len(
                train_rows
            ) < 120
            or
            len(
                test_rows
            ) < 40
        ):
            continue

        candidate = _fit_candidate(
            train_rows
        )

        y = np.asarray(
            [
                1.0
                if row.get(
                    "nrfi"
                )
                is True
                else 0.0
                for row in test_rows
            ],
            dtype=float,
        )

        incumbent_p = np.asarray(
            [
                _clip_probability(
                    _number(
                        row.get(
                            "nrfi_probability"
                        )
                    )
                    / 100.0
                )
                for row in test_rows
            ],
            dtype=float,
        )

        candidate_p = np.asarray(
            [
                _clip_probability(
                    _candidate_nrfi(
                        candidate,
                        row,
                    )
                )
                for row in test_rows
            ],
            dtype=float,
        )

        incumbent_brier = _brier(
            y,
            incumbent_p,
        )

        candidate_brier = _brier(
            y,
            candidate_p,
        )

        incumbent_logloss = _log_loss(
            y,
            incumbent_p,
        )

        candidate_logloss = _log_loss(
            y,
            candidate_p,
        )

        fold_win = (
            candidate_brier
            < incumbent_brier
            and
            candidate_logloss
            < incumbent_logloss
        )

        fold_rows.append({
            "fold":
                fold_number,

            "train_games":
                len(
                    train_rows
                ),

            "test_games":
                len(
                    test_rows
                ),

            "incumbent_brier":
                incumbent_brier,

            "candidate_brier":
                candidate_brier,

            "brier_gain":
                incumbent_brier
                - candidate_brier,

            "incumbent_logloss":
                incumbent_logloss,

            "candidate_logloss":
                candidate_logloss,

            "logloss_gain":
                incumbent_logloss
                - candidate_logloss,

            "win":
                bool(
                    fold_win
                ),
        })

        all_y.extend(
            y.tolist()
        )

        all_incumbent.extend(
            incumbent_p.tolist()
        )

        all_candidate.extend(
            candidate_p.tolist()
        )

        all_test_rows.extend(
            test_rows
        )

    if not all_y:
        return None

    y = np.asarray(
        all_y,
        dtype=float,
    )

    incumbent_p = np.asarray(
        all_incumbent,
        dtype=float,
    )

    candidate_p = np.asarray(
        all_candidate,
        dtype=float,
    )

    incumbent_brier = _brier(
        y,
        incumbent_p,
    )

    candidate_brier = _brier(
        y,
        candidate_p,
    )

    incumbent_logloss = _log_loss(
        y,
        incumbent_p,
    )

    candidate_logloss = _log_loss(
        y,
        candidate_p,
    )

    incumbent_ece = _calibration_ece(
        y,
        incumbent_p,
    )

    candidate_ece = _calibration_ece(
        y,
        candidate_p,
    )

    incumbent_roi = _roi_for_probabilities(
        all_test_rows,
        incumbent_p,
    )

    candidate_roi = _roi_for_probabilities(
        all_test_rows,
        candidate_p,
    )

    return {
        "oos_test_games":
            len(
                all_y
            ),

        "folds":
            fold_rows,

        "fold_wins":
            sum(
                1
                for row in fold_rows
                if row[
                    "win"
                ]
            ),

        "incumbent":
            {
                "brier":
                    incumbent_brier,

                "logloss":
                    incumbent_logloss,

                "ece":
                    incumbent_ece,

                "roi":
                    incumbent_roi,
            },

        "candidate":
            {
                "brier":
                    candidate_brier,

                "logloss":
                    candidate_logloss,

                "ece":
                    candidate_ece,

                "roi":
                    candidate_roi,
            },

        "brier_gain":
            incumbent_brier
            - candidate_brier,

        "logloss_gain":
            incumbent_logloss
            - candidate_logloss,

        "ece_change":
            candidate_ece
            - incumbent_ece,
    }


def _build_candidate_payload(
    games,
    incumbent_metadata,
):
    models = _fit_candidate(
        games
    )

    start_date = (
        str(
            games[
                0
            ].get(
                "game_date",
                ""
            )
        )
        if games
        else None
    )

    end_date = (
        str(
            games[
                -1
            ].get(
                "game_date",
                ""
            )
        )
        if games
        else None
    )

    return {
        "model_name":
            "SharpReport NRFI/YRFI Recent-Core Challenger",

        "version":
            (
                "shadow-"
                f"{end_date or 'unknown'}-"
                f"n{len(games)}"
            ),

        "status":
            "SHADOW_ONLY_NOT_PRODUCTION",

        "training_scope":
            (
                "Forward-tracked FINAL pregame games only. "
                "This challenger is used for drift/retraining governance "
                "and is not an automatic replacement for the historically "
                "validated incumbent."
            ),

        "incumbent_reference":
            incumbent_metadata,

        "probability_type":
            "RAW_LOGISTIC_NO_POST_CALIBRATION",

        "training_games":
            len(
                games
            ),

        "training_half_innings":
            len(
                games
            )
            * 2,

        "training_start_date":
            start_date,

        "training_end_date":
            end_date,

        "features":
            FEATURES,

        "ridge_lambda":
            RIDGE_LAMBDA,

        "park_rule":
            (
                "Use logged prior-completed-season 3-year "
                "Run Factor from the exact FINAL pregame snapshot."
            ),

        "game_probability_formula":
            (
                "NRFI=(1-P_TOP_SCORES)*(1-P_BOTTOM_SCORES); "
                "YRFI=1-NRFI"
            ),

        "models":
            models,
    }


def update_model_governance(
    token,
    repo,
):
    incumbent = _model_file_metadata()

    loaded = read_json_file(
        token=token,
        repo=repo,
        path=LEDGER_PATH,
    )

    ledger_games = (
        loaded
        .get(
            "data",
            {},
        )
        .get(
            "games",
            [],
        )
        if loaded
        else []
    )

    eligible = _eligible_games(
        ledger_games
    )

    eligible_count = len(
        eligible
    )

    evaluated_through = (
        str(
            eligible[
                -1
            ].get(
                "game_date",
                ""
            )
        )
        if eligible
        else None
    )

    governance = {
        "governance_version":
            "1.0",

        "incumbent":
            incumbent,

        "production_model_change":
            "NEVER_AUTOMATIC",

        "eligible_forward_games":
            eligible_count,

        "minimum_games_for_challenger_test":
            MIN_ELIGIBLE_GAMES,

        "evaluated_through":
            evaluated_through,

        "candidate_type":
            (
                "Recent-data retrain of the existing v1 core "
                "feature set, Top/Bottom separately."
            ),

        "validation_method":
            (
                "Three chronological expanding-window folds. "
                "Every candidate test game occurs after its training data."
            ),

        "promotion_review_gate":
            {
                "minimum_brier_gain":
                    MIN_BRIER_GAIN,

                "minimum_logloss_gain":
                    MIN_LOGLOSS_GAIN,

                "maximum_ece_worsening":
                    MAX_ECE_WORSENING,

                "minimum_fold_wins":
                    MIN_FOLD_WINS,

                "automatic_promotion":
                    False,
            },

        "recommendation":
            "COLLECTING",

        "recommendation_reason":
            (
                f"Need at least {MIN_ELIGIBLE_GAMES} eligible "
                "forward FINAL games before challenger evaluation."
            ),

        "walk_forward":
            None,

        "challenger_model_path":
            None,
    }


    if eligible_count >= MIN_ELIGIBLE_GAMES:

        evaluation = _evaluate_walk_forward(
            eligible
        )

        governance[
            "walk_forward"
        ] = evaluation

        if evaluation is not None:

            criteria = {
                "brier":
                    evaluation[
                        "brier_gain"
                    ]
                    >=
                    MIN_BRIER_GAIN,

                "logloss":
                    evaluation[
                        "logloss_gain"
                    ]
                    >=
                    MIN_LOGLOSS_GAIN,

                "calibration":
                    evaluation[
                        "ece_change"
                    ]
                    <=
                    MAX_ECE_WORSENING,

                "folds":
                    evaluation[
                        "fold_wins"
                    ]
                    >=
                    MIN_FOLD_WINS,
            }

            governance[
                "promotion_review_criteria"
            ] = criteria

            candidate_payload = (
                _build_candidate_payload(
                    eligible,
                    incumbent,
                )
            )

            candidate_write = (
                upsert_json_file(
                    token=token,
                    repo=repo,
                    path=CHALLENGER_PATH,
                    payload=candidate_payload,
                    commit_message=(
                        "Update shadow NRFI challenger model"
                    ),
                )
            )

            governance[
                "challenger_model_path"
            ] = CHALLENGER_PATH

            governance[
                "challenger_model_updated"
            ] = bool(
                candidate_write.get(
                    "changed"
                )
            )

            if all(
                criteria.values()
            ):
                governance[
                    "recommendation"
                ] = "REVIEW_FOR_PROMOTION"

                governance[
                    "recommendation_reason"
                ] = (
                    "The shadow challenger passed every configured "
                    "out-of-sample review gate. Production remains v1 "
                    "until an explicit manual review and promotion."
                )

            else:
                governance[
                    "recommendation"
                ] = "KEEP_V1"

                failed = [
                    name
                    for name, passed
                    in criteria.items()
                    if not passed
                ]

                governance[
                    "recommendation_reason"
                ] = (
                    "The challenger did not pass every review gate. "
                    "Failed criteria: "
                    + ", ".join(
                        failed
                    )
                    + "."
                )


    write_result = upsert_json_file(
        token=token,
        repo=repo,
        path=GOVERNANCE_PATH,
        payload=governance,
        commit_message=(
            "Update NRFI model governance status"
        ),
    )

    return {
        **governance,

        "governance_path":
            GOVERNANCE_PATH,

        "governance_updated":
            bool(
                write_result.get(
                    "changed"
                )
            ),
    }


def _fmt_metric(
    value,
    digits=6,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:.{digits}f}"


def _fmt_gain(
    value,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:+.6f}"


def _fmt_pct(
    value,
):
    number = _number(
        value
    )

    if number is None:
        return "—"

    return f"{number:+.1f}%"


def render_model_governance_dashboard(
    token,
    repo,
):
    st.subheader(
        "Model Governance / Retraining Control"
    )

    st.caption(
        "The production model is never replaced automatically. "
        "A recent-data shadow challenger is periodically retrained "
        "and evaluated only on later unseen games. Promotion requires "
        "an explicit manual review."
    )

    loaded = read_json_file(
        token=token,
        repo=repo,
        path=GOVERNANCE_PATH,
    )

    if not loaded:
        st.info(
            "Model governance has not been evaluated yet. "
            "The scheduled collector will initialize it automatically."
        )

        return

    data = loaded.get(
        "data",
        {}
    )

    incumbent = data.get(
        "incumbent",
        {}
    )

    st.markdown(
        "**Incumbent Production Model**"
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Model":
                    incumbent.get(
                        "model_name",
                        "SharpReport NRFI/YRFI Model v1",
                    ),

                "Version":
                    incumbent.get(
                        "version",
                        "1.0",
                    ),

                "Historical Training Through":
                    incumbent.get(
                        "training_end_date",
                        "—",
                    ),

                "Historical Training Games":
                    incumbent.get(
                        "training_games",
                        "—",
                    ),

                "Production Status":
                    "ACTIVE — FROZEN UNTIL MANUAL PROMOTION",
            }
        ]),
        use_container_width=True,
        hide_index=True,
    )


    eligible_count = int(
        data.get(
            "eligible_forward_games",
            0,
        )
        or 0
    )

    minimum = int(
        data.get(
            "minimum_games_for_challenger_test",
            MIN_ELIGIBLE_GAMES,
        )
        or MIN_ELIGIBLE_GAMES
    )

    recommendation = data.get(
        "recommendation",
        "COLLECTING",
    )

    st.markdown(
        "**Current Governance Status**"
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Eligible Forward Games":
                    eligible_count,

                "Minimum for Challenger Test":
                    minimum,

                "Evaluated Through":
                    data.get(
                        "evaluated_through",
                        "—",
                    ),

                "Status":
                    recommendation,

                "Automatic Promotion":
                    "NO",
            }
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        data.get(
            "recommendation_reason",
            "",
        )
    )


    evaluation = data.get(
        "walk_forward"
    )

    if not evaluation:

        st.info(
            f"Collecting forward data: {eligible_count}/{minimum} "
            "eligible FINAL games. The challenger remains locked."
        )

        return


    incumbent_eval = evaluation.get(
        "incumbent",
        {}
    )

    candidate_eval = evaluation.get(
        "candidate",
        {}
    )


    st.markdown(
        "**Out-of-Sample Incumbent vs Shadow Challenger**"
    )

    comparison = pd.DataFrame([
        {
            "Model":
                "Production v1",

            "Brier":
                _fmt_metric(
                    incumbent_eval.get(
                        "brier"
                    )
                ),

            "Log Loss":
                _fmt_metric(
                    incumbent_eval.get(
                        "logloss"
                    )
                ),

            "Calibration ECE":
                _fmt_metric(
                    incumbent_eval.get(
                        "ece"
                    ),
                    4,
                ),

            "ROI on Same OOS Games":
                _fmt_pct(
                    (
                        incumbent_eval
                        .get("roi", {})
                        .get("roi")
                    )
                ),
        },
        {
            "Model":
                "Recent-Core Challenger",

            "Brier":
                _fmt_metric(
                    candidate_eval.get(
                        "brier"
                    )
                ),

            "Log Loss":
                _fmt_metric(
                    candidate_eval.get(
                        "logloss"
                    )
                ),

            "Calibration ECE":
                _fmt_metric(
                    candidate_eval.get(
                        "ece"
                    ),
                    4,
                ),

            "ROI on Same OOS Games":
                _fmt_pct(
                    (
                        candidate_eval
                        .get("roi", {})
                        .get("roi")
                    )
                ),
        },
    ])

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "**Promotion Review Gates**"
    )

    criteria = data.get(
        "promotion_review_criteria",
        {}
    )

    gate_rows = [
        {
            "Gate":
                "Brier improvement",

            "Observed":
                _fmt_gain(
                    evaluation.get(
                        "brier_gain"
                    )
                ),

            "Required":
                f">= +{MIN_BRIER_GAIN:.5f}",

            "Pass":
                "YES"
                if criteria.get(
                    "brier"
                )
                else "NO",
        },
        {
            "Gate":
                "Log-loss improvement",

            "Observed":
                _fmt_gain(
                    evaluation.get(
                        "logloss_gain"
                    )
                ),

            "Required":
                f">= +{MIN_LOGLOSS_GAIN:.5f}",

            "Pass":
                "YES"
                if criteria.get(
                    "logloss"
                )
                else "NO",
        },
        {
            "Gate":
                "Calibration change",

            "Observed":
                _fmt_gain(
                    evaluation.get(
                        "ece_change"
                    )
                ),

            "Required":
                f"<= +{MAX_ECE_WORSENING:.3f}",

            "Pass":
                "YES"
                if criteria.get(
                    "calibration"
                )
                else "NO",
        },
        {
            "Gate":
                "Chronological fold wins",

            "Observed":
                (
                    f"{evaluation.get('fold_wins', 0)}/"
                    f"{len(evaluation.get('folds', []))}"
                ),

            "Required":
                f">= {MIN_FOLD_WINS}",

            "Pass":
                "YES"
                if criteria.get(
                    "folds"
                )
                else "NO",
        },
    ]

    st.dataframe(
        pd.DataFrame(
            gate_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    fold_rows = []

    for fold in evaluation.get(
        "folds",
        []
    ):
        fold_rows.append({
            "Fold":
                fold.get(
                    "fold"
                ),

            "Train Games":
                fold.get(
                    "train_games"
                ),

            "Unseen Test Games":
                fold.get(
                    "test_games"
                ),

            "Brier Gain":
                _fmt_gain(
                    fold.get(
                        "brier_gain"
                    )
                ),

            "Log-Loss Gain":
                _fmt_gain(
                    fold.get(
                        "logloss_gain"
                    )
                ),

            "Win":
                "YES"
                if fold.get(
                    "win"
                )
                else "NO",
        })

    if fold_rows:

        st.markdown(
            "**Walk-Forward Fold Detail**"
        )

        st.dataframe(
            pd.DataFrame(
                fold_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


    if recommendation == "REVIEW_FOR_PROMOTION":

        st.success(
            "The shadow challenger passed every configured review gate. "
            "This is a REVIEW signal only — Model v1 remains production "
            "until we manually inspect and approve a version change."
        )

    else:

        st.info(
            "Model v1 remains the production model. "
            "The challenger has not earned promotion."
        )
