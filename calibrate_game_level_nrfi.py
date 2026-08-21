
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7H2
#
# WALK-FORWARD GAME-LEVEL PROBABILITY CALIBRATION
#
# We already have fully out-of-sample NRFI probabilities for:
#   2023, 2024, 2025, 2026
#
# This stage tests whether a simple 2-parameter logistic
# recalibration improves the final game-level probabilities.
#
# Calibration is fit ONLY on PRIOR out-of-sample predictions:
#   2024 calibration <- 2023 OOS predictions
#   2025 calibration <- 2023-2024 OOS predictions
#   2026 calibration <- 2023-2025 OOS predictions
#
# 2023 is not recalibrated because no earlier OOS prediction
# season exists in this file.
#
# Candidates:
#   1) Core + Offense BB + Run Factor  (current multi-year champion)
#   2) Core + Run Factor               (simpler runner-up)
#
# No sklearn dependency.
# =========================================================


PREDICTIONS_FILE = Path(
    "historical_game_level_nrfi_predictions.csv"
)

SUMMARY_FILE = Path(
    "historical_game_level_calibration_summary.txt"
)

RESULTS_FILE = Path(
    "historical_game_level_calibration_results.csv"
)

CALIBRATED_FILE = Path(
    "historical_game_level_calibrated_predictions.csv"
)


CANDIDATE_MODELS = [
    "Core + Offense BB + Run Factor",
    "Core + Run Factor",
]

CALIBRATION_YEARS = [
    2024,
    2025,
    2026,
]

MAX_ITER = 100
TOL = 1e-10
RIDGE = 1e-6


# =========================================================
# HELPERS
# =========================================================

def sigmoid(z):
    z = np.clip(
        z,
        -30,
        30,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-z)
        )
    )


def logit(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )

    return np.log(
        p / (1 - p)
    )


def brier_score(y, p):
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
            (p - y) ** 2
        )
    )


def log_loss(y, p):
    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1 - y)
            * np.log(1 - p)
        )
    )


def fit_logistic_calibrator(
    raw_prob,
    y,
):
    x = logit(
        raw_prob
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.column_stack([
        np.ones(
            len(x)
        ),
        x,
    ])

    beta = np.array(
        [0.0, 1.0],
        dtype=float,
    )

    penalty = np.eye(
        2,
        dtype=float,
    )

    # Do not materially penalize intercept.
    penalty[
        0,
        0,
    ] = 0.0

    for _ in range(
        MAX_ITER
    ):
        eta = (
            X @ beta
        )

        p = sigmoid(
            eta
        )

        w = np.clip(
            p * (1 - p),
            1e-6,
            None,
        )

        gradient = (
            X.T
            @ (y - p)
            - RIDGE
            * (
                penalty
                @ beta
            )
        )

        hessian = (
            X.T
            @ (
                w[:, None]
                * X
            )
            + RIDGE
            * penalty
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

        beta_new = (
            beta
            + step
        )

        if np.max(
            np.abs(
                beta_new
                - beta
            )
        ) < TOL:
            beta = beta_new
            break

        beta = beta_new

    return (
        float(beta[0]),
        float(beta[1]),
    )


def apply_calibrator(
    raw_prob,
    intercept,
    slope,
):
    return sigmoid(
        intercept
        + slope
        * logit(
            raw_prob
        )
    )


# =========================================================
# LOAD
# =========================================================

if not PREDICTIONS_FILE.exists():
    raise SystemExit(
        "ERROR: "
        "historical_game_level_nrfi_predictions.csv "
        "was not found."
    )


predictions = pd.read_csv(
    PREDICTIONS_FILE
)


required = [
    "holdout_year",
    "model",
    "game_pk",
    "nrfi_actual",
    "nrfi_probability",
]

for column in required:
    if column not in predictions.columns:
        raise SystemExit(
            f"ERROR: Missing required column: "
            f"{column}"
        )


predictions[
    "holdout_year"
] = pd.to_numeric(
    predictions[
        "holdout_year"
    ],
    errors="raise",
).astype(int)


predictions[
    "nrfi_actual"
] = pd.to_numeric(
    predictions[
        "nrfi_actual"
    ],
    errors="raise",
).astype(int)


predictions[
    "nrfi_probability"
] = pd.to_numeric(
    predictions[
        "nrfi_probability"
    ],
    errors="raise",
)


# =========================================================
# WALK-FORWARD CALIBRATION
# =========================================================

result_rows = []
calibrated_rows = []


for model_name in CANDIDATE_MODELS:
    model_data = (
        predictions[
            predictions[
                "model"
            ] == model_name
        ]
        .copy()
        .sort_values(
            [
                "holdout_year",
                "game_pk",
            ]
        )
    )

    if model_data.empty:
        raise SystemExit(
            f"ERROR: Model not found in "
            f"prediction file: {model_name}"
        )

    print()
    print(
        f"CALIBRATING: "
        f"{model_name}"
    )
    print(
        "-" * 72
    )

    # Keep 2023 raw as reference.
    first_year = int(
        model_data[
            "holdout_year"
        ].min()
    )

    first_rows = model_data[
        model_data[
            "holdout_year"
        ] == first_year
    ].copy()

    for _, row in first_rows.iterrows():
        calibrated_rows.append({
            "holdout_year":
                first_year,
            "model":
                model_name,
            "game_pk":
                int(
                    row[
                        "game_pk"
                    ]
                ),
            "nrfi_actual":
                int(
                    row[
                        "nrfi_actual"
                    ]
                ),
            "raw_nrfi_probability":
                float(
                    row[
                        "nrfi_probability"
                    ]
                ),
            "calibrated_nrfi_probability":
                float(
                    row[
                        "nrfi_probability"
                    ]
                ),
            "calibration_intercept":
                np.nan,
            "calibration_slope":
                np.nan,
            "calibration_training_games":
                0,
            "calibration_status":
                "RAW - NO PRIOR OOS CALIBRATION SEASON",
        })

    for holdout_year in CALIBRATION_YEARS:
        calibration_train = model_data[
            model_data[
                "holdout_year"
            ] < holdout_year
        ].copy()

        test = model_data[
            model_data[
                "holdout_year"
            ] == holdout_year
        ].copy()

        if (
            calibration_train.empty
            or test.empty
        ):
            continue

        intercept, slope = (
            fit_logistic_calibrator(
                calibration_train[
                    "nrfi_probability"
                ].to_numpy(
                    dtype=float
                ),
                calibration_train[
                    "nrfi_actual"
                ].to_numpy(
                    dtype=float
                ),
            )
        )

        raw_p = test[
            "nrfi_probability"
        ].to_numpy(
            dtype=float
        )

        y = test[
            "nrfi_actual"
        ].to_numpy(
            dtype=float
        )

        calibrated_p = (
            apply_calibrator(
                raw_p,
                intercept,
                slope,
            )
        )

        raw_brier = (
            brier_score(
                y,
                raw_p,
            )
        )

        calibrated_brier = (
            brier_score(
                y,
                calibrated_p,
            )
        )

        raw_logloss = (
            log_loss(
                y,
                raw_p,
            )
        )

        calibrated_logloss = (
            log_loss(
                y,
                calibrated_p,
            )
        )

        result_rows.append({
            "holdout_year":
                holdout_year,
            "model":
                model_name,
            "games":
                len(test),
            "calibration_training_games":
                len(
                    calibration_train
                ),
            "calibration_intercept":
                intercept,
            "calibration_slope":
                slope,
            "raw_brier":
                raw_brier,
            "calibrated_brier":
                calibrated_brier,
            "brier_improvement_from_calibration":
                raw_brier
                - calibrated_brier,
            "raw_logloss":
                raw_logloss,
            "calibrated_logloss":
                calibrated_logloss,
            "logloss_improvement_from_calibration":
                raw_logloss
                - calibrated_logloss,
        })

        for row_index, (
            (_, row),
            calibrated_value,
        ) in enumerate(
            zip(
                test.iterrows(),
                calibrated_p,
            )
        ):
            calibrated_rows.append({
                "holdout_year":
                    holdout_year,
                "model":
                    model_name,
                "game_pk":
                    int(
                        row[
                            "game_pk"
                        ]
                    ),
                "nrfi_actual":
                    int(
                        row[
                            "nrfi_actual"
                        ]
                    ),
                "raw_nrfi_probability":
                    float(
                        row[
                            "nrfi_probability"
                        ]
                    ),
                "calibrated_nrfi_probability":
                    float(
                        calibrated_value
                    ),
                "calibration_intercept":
                    intercept,
                "calibration_slope":
                    slope,
                "calibration_training_games":
                    len(
                        calibration_train
                    ),
                "calibration_status":
                    (
                        "CALIBRATED USING PRIOR "
                        "OUT-OF-SAMPLE SEASONS"
                    ),
            })

        print(
            f"{holdout_year}: "
            f"training games="
            f"{len(calibration_train)} | "
            f"intercept="
            f"{intercept:+.4f} | "
            f"slope="
            f"{slope:.4f} | "
            f"Brier improvement="
            f"{raw_brier - calibrated_brier:+.6f}"
        )


results = pd.DataFrame(
    result_rows
)

calibrated = pd.DataFrame(
    calibrated_rows
)


results.to_csv(
    RESULTS_FILE,
    index=False,
)

calibrated.to_csv(
    CALIBRATED_FILE,
    index=False,
)


# =========================================================
# 2026 CALIBRATION BUCKETS
# =========================================================

bucket_lines = []


for model_name in CANDIDATE_MODELS:
    subset = calibrated[
        (
            calibrated[
                "model"
            ] == model_name
        )
        &
        (
            calibrated[
                "holdout_year"
            ] == 2026
        )
    ].copy()

    if subset.empty:
        continue

    for probability_column, label in [
        (
            "raw_nrfi_probability",
            "RAW",
        ),
        (
            "calibrated_nrfi_probability",
            "CALIBRATED",
        ),
    ]:
        subset[
            "bucket"
        ] = pd.cut(
            subset[
                probability_column
            ],
            bins=[
                0.00,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                1.00,
            ],
            include_lowest=True,
        )

        grouped = (
            subset
            .groupby(
                "bucket",
                observed=True,
            )
            .agg(
                games=(
                    "game_pk",
                    "count",
                ),
                avg_predicted=(
                    probability_column,
                    "mean",
                ),
                actual_nrfi_rate=(
                    "nrfi_actual",
                    "mean",
                ),
            )
            .reset_index()
        )

        bucket_lines.extend([
            "",
            (
                f"2026 {label} CALIBRATION — "
                f"{model_name}"
            ),
            "-" * 88,
        ])

        for _, row in grouped.iterrows():
            bucket_lines.append(
                f"{row['bucket']} | "
                f"games "
                f"{int(row['games'])} | "
                f"pred "
                f"{row['avg_predicted']:.3f} | "
                f"actual "
                f"{row['actual_nrfi_rate']:.3f}"
            )


# =========================================================
# SUMMARY
# =========================================================

lines = [
    "SHARPREPORT GAME-LEVEL NRFI PROBABILITY CALIBRATION TEST",
    "=" * 88,
    "",
    "METHOD:",
    (
        "Logistic recalibration is trained ONLY on prior "
        "out-of-sample game predictions."
    ),
    (
        "This tests whether the model is systematically "
        "too aggressive or too conservative."
    ),
    (
        "Slope below 1.0 generally means the raw probabilities "
        "should be pulled toward the middle."
    ),
    (
        "Positive Brier / Log Loss improvement means "
        "calibration improved future probability accuracy."
    ),
    "",
]


ranking_rows = []


for model_name in CANDIDATE_MODELS:
    subset = results[
        results[
            "model"
        ] == model_name
    ].copy()

    if subset.empty:
        continue

    brier_wins = int(
        (
            subset[
                "brier_improvement_from_calibration"
            ] > 0
        ).sum()
    )

    logloss_wins = int(
        (
            subset[
                "logloss_improvement_from_calibration"
            ] > 0
        ).sum()
    )

    avg_brier = float(
        subset[
            "brier_improvement_from_calibration"
        ].mean()
    )

    avg_logloss = float(
        subset[
            "logloss_improvement_from_calibration"
        ].mean()
    )

    worst_brier = float(
        subset[
            "brier_improvement_from_calibration"
        ].min()
    )

    lines.extend([
        model_name,
        "-" * 88,
        (
            f"Calibration improved Brier: "
            f"{brier_wins}/{len(subset)} "
            f"holdouts"
        ),
        (
            f"Calibration improved Log Loss: "
            f"{logloss_wins}/{len(subset)} "
            f"holdouts"
        ),
        (
            f"Average Brier improvement from calibration: "
            f"{avg_brier:+.6f}"
        ),
        (
            f"Average Log Loss improvement from calibration: "
            f"{avg_logloss:+.6f}"
        ),
        (
            f"Worst Brier calibration result: "
            f"{worst_brier:+.6f}"
        ),
        "",
    ])

    for _, row in subset.iterrows():
        lines.append(
            f"{int(row['holdout_year'])}: "
            f"train games "
            f"{int(row['calibration_training_games'])} | "
            f"intercept "
            f"{row['calibration_intercept']:+.4f} | "
            f"slope "
            f"{row['calibration_slope']:.4f} | "
            f"raw Brier "
            f"{row['raw_brier']:.6f} | "
            f"cal "
            f"{row['calibrated_brier']:.6f} | "
            f"gain "
            f"{row['brier_improvement_from_calibration']:+.6f}"
        )

    lines.extend([
        "",
        "=" * 88,
        "",
    ])

    ranking_rows.append({
        "model":
            model_name,
        "brier_wins":
            brier_wins,
        "logloss_wins":
            logloss_wins,
        "avg_brier_gain":
            avg_brier,
        "avg_logloss_gain":
            avg_logloss,
        "worst_brier_gain":
            worst_brier,
    })


ranking = (
    pd.DataFrame(
        ranking_rows
    )
    .sort_values(
        [
            "brier_wins",
            "avg_brier_gain",
            "worst_brier_gain",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)


lines.extend([
    "CALIBRATION RANKING",
    "-" * 88,
])

for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    lines.append(
        f"#{rank} {row['model']} | "
        f"Brier wins "
        f"{int(row['brier_wins'])}/3 | "
        f"avg gain "
        f"{row['avg_brier_gain']:+.6f} | "
        f"worst "
        f"{row['worst_brier_gain']:+.6f}"
    )


lines.extend(
    bucket_lines
)


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print(
    "GAME-LEVEL CALIBRATION TEST COMPLETE"
)
print()
print(
    f"Created: {SUMMARY_FILE}"
)
print(
    f"Created: {RESULTS_FILE}"
)
print(
    f"Created: {CALIBRATED_FILE}"
)
