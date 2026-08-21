
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7E4
#
# Walk-forward test:
# Do pregame Top-4 hitter xwOBA / K% / BB% / Barrel%
# improve prediction of future 1st-inning scoring?
#
# Training / holdout:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom 1st modeled separately.
# Positive Brier improvement = better probability forecast.
#
# Every skill model includes Top-4 sample-size controls and
# is also compared against a sample-only control.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
TOP4_FILE = Path("historical_top4_skill_context.csv")

SUMMARY_FILE = Path(
    "historical_top4_skill_predictive_summary.txt"
)

RESULTS_FILE = Path(
    "historical_top4_skill_predictive_results.csv"
)

COEFFICIENTS_FILE = Path(
    "historical_top4_skill_coefficients.csv"
)

HOLDOUT_YEARS = [2023, 2024, 2025, 2026]

RIDGE_LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-8


# =========================================================
# HELPERS
# =========================================================

def find_column(df, candidates, label, required=True):
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise SystemExit(
            f"ERROR: Could not find {label}. "
            f"Available columns:\n"
            + ", ".join(map(str, df.columns))
        )

    return None


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def prepare_matrix(train_df, test_df, feature_cols):
    train = train_df[feature_cols].copy()
    test = test_df[feature_cols].copy()

    for col in feature_cols:
        train[col] = pd.to_numeric(
            train[col],
            errors="coerce",
        )
        test[col] = pd.to_numeric(
            test[col],
            errors="coerce",
        )

        mean_value = (
            float(train[col].mean())
            if train[col].notna().any()
            else 0.0
        )

        train[col] = train[col].fillna(mean_value)
        test[col] = test[col].fillna(mean_value)

        std_value = float(
            train[col].std(ddof=0)
        )

        if (
            not np.isfinite(std_value)
            or std_value < 1e-12
        ):
            std_value = 1.0

        train[col] = (
            train[col] - mean_value
        ) / std_value

        test[col] = (
            test[col] - mean_value
        ) / std_value

    return (
        train.to_numpy(dtype=float),
        test.to_numpy(dtype=float),
    )


def fit_ridge_logistic(X, y, lam=1.0):
    y = np.asarray(y, dtype=float)

    Xd = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    beta = np.zeros(
        Xd.shape[1],
        dtype=float,
    )

    penalty = np.eye(
        Xd.shape[1],
        dtype=float,
    )
    penalty[0, 0] = 0.0

    for _ in range(MAX_ITER):
        eta = Xd @ beta
        p = sigmoid(eta)

        w = np.clip(
            p * (1.0 - p),
            1e-6,
            None,
        )

        gradient = (
            Xd.T @ (y - p)
            - lam * (penalty @ beta)
        )

        hessian = (
            Xd.T
            @ (w[:, None] * Xd)
            + lam * penalty
        )

        try:
            step = np.linalg.solve(
                hessian,
                gradient,
            )
        except np.linalg.LinAlgError:
            step = (
                np.linalg.pinv(hessian)
                @ gradient
            )

        beta_new = beta + step

        if np.max(
            np.abs(beta_new - beta)
        ) < TOL:
            beta = beta_new
            break

        beta = beta_new

    return beta


def predict_logistic(beta, X):
    Xd = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    return sigmoid(
        Xd @ beta
    )


# =========================================================
# LOAD DATA
# =========================================================

if not OUTCOMES_FILE.exists():
    raise SystemExit(
        "ERROR: historical_first_innings.csv was not found."
    )

if not TOP4_FILE.exists():
    raise SystemExit(
        "ERROR: historical_top4_skill_context.csv was not found."
    )

outcomes = pd.read_csv(
    OUTCOMES_FILE
)

top4 = pd.read_csv(
    TOP4_FILE
)


game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in historical outcomes",
)

game_pk_top4 = find_column(
    top4,
    ["game_pk", "gamepk", "game_id"],
    "game ID in Top-4 skill context",
)

season_out = find_column(
    outcomes,
    ["season", "year"],
    "season in historical outcomes",
)

date_out = find_column(
    outcomes,
    ["date", "game_date"],
    "date in historical outcomes",
)


top_scored_col = find_column(
    outcomes,
    [
        "top_scored",
        "top_1_scored",
        "top1_scored",
        "away_scored",
        "away_scored_first",
        "away_scored_1st",
        "away_first_scored",
    ],
    "Top 1st scored flag",
    required=False,
)

bottom_scored_col = find_column(
    outcomes,
    [
        "bottom_scored",
        "bottom_1_scored",
        "bottom1_scored",
        "home_scored",
        "home_scored_first",
        "home_scored_1st",
        "home_first_scored",
    ],
    "Bottom 1st scored flag",
    required=False,
)


if top_scored_col is None:
    away_runs_col = find_column(
        outcomes,
        [
            "away_first_inning_runs",
            "away_1st_inning_runs",
            "away_first_runs",
            "away_1st_runs",
            "top_first_inning_runs",
            "top_1_runs",
            "top1_runs",
        ],
        "away / Top 1st runs",
    )
else:
    away_runs_col = None


if bottom_scored_col is None:
    home_runs_col = find_column(
        outcomes,
        [
            "home_first_inning_runs",
            "home_1st_inning_runs",
            "home_first_runs",
            "home_1st_runs",
            "bottom_first_inning_runs",
            "bottom_1_runs",
            "bottom1_runs",
        ],
        "home / Bottom 1st runs",
    )
else:
    home_runs_col = None


# =========================================================
# MERGE
# =========================================================

merged = outcomes.merge(
    top4,
    left_on=game_pk_out,
    right_on=game_pk_top4,
    how="inner",
    suffixes=("_out", "_top4"),
)

if len(merged) != len(outcomes):
    raise SystemExit(
        "ERROR: Outcomes and Top-4 context game counts "
        f"did not match after merge. "
        f"Outcomes={len(outcomes)}, merged={len(merged)}"
    )


season_merged = find_column(
    merged,
    [
        f"{season_out}_out",
        season_out,
        "season_out",
        "season",
    ],
    "season after merge",
)

date_merged = find_column(
    merged,
    [
        f"{date_out}_out",
        date_out,
        "date_out",
        "date",
        "game_date_out",
        "game_date",
    ],
    "date after merge",
)


season_series = pd.to_numeric(
    merged[season_merged],
    errors="raise",
).astype(int)

date_series = pd.to_datetime(
    merged[date_merged],
    errors="raise",
)


def scored_values(which):
    if which == "top":
        if top_scored_col is not None:
            return pd.to_numeric(
                merged[top_scored_col],
                errors="raise",
            ).astype(int)

        return (
            pd.to_numeric(
                merged[away_runs_col],
                errors="raise",
            ) > 0
        ).astype(int)

    if bottom_scored_col is not None:
        return pd.to_numeric(
            merged[bottom_scored_col],
            errors="raise",
        ).astype(int)

    return (
        pd.to_numeric(
            merged[home_runs_col],
            errors="raise",
        ) > 0
    ).astype(int)


# =========================================================
# BUILD HALF-INNING ROWS
# =========================================================

# Top 1st = AWAY Top-4 offense.
top = pd.DataFrame({
    "game_pk":
        merged[game_pk_out],
    "date":
        date_series,
    "season":
        season_series,
    "half":
        "Top",
    "scored":
        scored_values("top"),

    "xwoba":
        pd.to_numeric(
            merged["away_top4_xwoba"],
            errors="coerce",
        ),
    "k_pct":
        pd.to_numeric(
            merged["away_top4_k_pct"],
            errors="coerce",
        ),
    "bb_pct":
        pd.to_numeric(
            merged["away_top4_bb_pct"],
            errors="coerce",
        ),
    "barrel_pct":
        pd.to_numeric(
            merged["away_top4_barrel_pct"],
            errors="coerce",
        ),
    "combined_pa":
        pd.to_numeric(
            merged["away_top4_combined_pa"],
            errors="coerce",
        ),
    "min_pa":
        pd.to_numeric(
            merged["away_top4_min_pa"],
            errors="coerce",
        ),
    "complete_core":
        pd.to_numeric(
            merged["away_top4_complete_core"],
            errors="coerce",
        ),
})


# Bottom 1st = HOME Top-4 offense.
bottom = pd.DataFrame({
    "game_pk":
        merged[game_pk_out],
    "date":
        date_series,
    "season":
        season_series,
    "half":
        "Bottom",
    "scored":
        scored_values("bottom"),

    "xwoba":
        pd.to_numeric(
            merged["home_top4_xwoba"],
            errors="coerce",
        ),
    "k_pct":
        pd.to_numeric(
            merged["home_top4_k_pct"],
            errors="coerce",
        ),
    "bb_pct":
        pd.to_numeric(
            merged["home_top4_bb_pct"],
            errors="coerce",
        ),
    "barrel_pct":
        pd.to_numeric(
            merged["home_top4_barrel_pct"],
            errors="coerce",
        ),
    "combined_pa":
        pd.to_numeric(
            merged["home_top4_combined_pa"],
            errors="coerce",
        ),
    "min_pa":
        pd.to_numeric(
            merged["home_top4_min_pa"],
            errors="coerce",
        ),
    "complete_core":
        pd.to_numeric(
            merged["home_top4_complete_core"],
            errors="coerce",
        ),
})


data = pd.concat(
    [top, bottom],
    ignore_index=True,
)

data["combined_pa"] = (
    data["combined_pa"]
    .fillna(0)
    .clip(lower=0)
)

data["min_pa"] = (
    data["min_pa"]
    .fillna(0)
    .clip(lower=0)
)

data["complete_core"] = (
    data["complete_core"]
    .fillna(0)
)

data["log_combined_pa"] = np.log1p(
    data["combined_pa"]
)

data["log_min_pa"] = np.log1p(
    data["min_pa"]
)

data["missing_core"] = (
    data["complete_core"] < 1
).astype(float)


if len(data) != 2 * len(outcomes):
    raise SystemExit(
        "ERROR: Half-inning row count is wrong."
    )


# =========================================================
# MODELS TO TEST
# =========================================================

sample_controls = [
    "log_combined_pa",
    "log_min_pa",
    "missing_core",
]


models = {
    "Sample only control":
        sample_controls,

    "Top4 xwOBA + sample":
        [
            "xwoba",
            *sample_controls,
        ],

    "Top4 K% + sample":
        [
            "k_pct",
            *sample_controls,
        ],

    "Top4 BB% + sample":
        [
            "bb_pct",
            *sample_controls,
        ],

    "Top4 Barrel% + sample":
        [
            "barrel_pct",
            *sample_controls,
        ],

    "Top4 xwOBA + K + sample":
        [
            "xwoba",
            "k_pct",
            *sample_controls,
        ],

    "Top4 xwOBA + K + Barrel + sample":
        [
            "xwoba",
            "k_pct",
            "barrel_pct",
            *sample_controls,
        ],

    "All 4 Top4 skill + sample":
        [
            "xwoba",
            "k_pct",
            "bb_pct",
            "barrel_pct",
            *sample_controls,
        ],
}


# =========================================================
# WALK-FORWARD TEST
# =========================================================

results = []
coefficient_rows = []


for holdout_year in HOLDOUT_YEARS:
    train = data[
        data["season"] < holdout_year
    ].copy()

    test = (
        data[
            data["season"] == holdout_year
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train.empty or test.empty:
        continue

    print()
    print(
        f"Testing {holdout_year} "
        f"using seasons before {holdout_year}"
    )

    year_predictions = {
        name: np.full(
            len(test),
            np.nan,
            dtype=float,
        )
        for name in models
    }

    baseline_predictions = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    for half in ["Top", "Bottom"]:
        train_half = train[
            train["half"] == half
        ].copy()

        test_mask = (
            test["half"] == half
        )

        test_half = test[
            test_mask
        ].copy()

        if train_half.empty or test_half.empty:
            continue

        y_train = train_half[
            "scored"
        ].to_numpy(dtype=float)

        baseline_rate = float(
            np.mean(y_train)
        )

        baseline_predictions[
            test_mask.to_numpy()
        ] = baseline_rate

        for model_name, feature_cols in models.items():
            (
                X_train,
                X_test,
            ) = prepare_matrix(
                train_half,
                test_half,
                feature_cols,
            )

            beta = fit_ridge_logistic(
                X_train,
                y_train,
                lam=RIDGE_LAMBDA,
            )

            pred = predict_logistic(
                beta,
                X_test,
            )

            year_predictions[
                model_name
            ][
                test_mask.to_numpy()
            ] = pred

            for i, feature in enumerate(feature_cols):
                coefficient_rows.append({
                    "holdout_year":
                        holdout_year,
                    "half":
                        half,
                    "model":
                        model_name,
                    "feature":
                        feature,
                    "standardized_coefficient":
                        float(beta[i + 1]),
                })

    y_test = test[
        "scored"
    ].to_numpy(dtype=float)

    baseline_brier = brier_score(
        y_test,
        baseline_predictions,
    )

    sample_control_brier = None

    year_rows = []

    for model_name in models:
        model_brier = brier_score(
            y_test,
            year_predictions[model_name],
        )

        if model_name == "Sample only control":
            sample_control_brier = model_brier

        year_rows.append({
            "holdout_year":
                holdout_year,
            "model":
                model_name,
            "rows":
                len(test),
            "baseline_brier":
                baseline_brier,
            "model_brier":
                model_brier,
            "brier_improvement_vs_baseline":
                baseline_brier - model_brier,
        })

    for row in year_rows:
        row[
            "brier_improvement_vs_sample_control"
        ] = (
            sample_control_brier
            - row["model_brier"]
        )

        results.append(row)


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)

coeff_df = pd.DataFrame(
    coefficient_rows
)

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

coeff_df.to_csv(
    COEFFICIENTS_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

lines = [
    "SHARPREPORT HISTORICAL TOP-4 OFFENSE PREDICTIVE TEST",
    "=" * 78,
    "",
    f"Historical games: {len(outcomes)}",
    f"Half-inning rows: {len(data)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample testing.",
    "Top 1st and Bottom 1st modeled separately.",
    "Positive Brier improvement = better probability forecast.",
    (
        "Every offensive skill model includes Top-4 sample-size "
        "controls and is also compared against a sample-only control."
    ),
    "",
]


ranking_rows = []


for model_name in models:
    subset = results_df[
        results_df["model"] == model_name
    ].copy()

    if subset.empty:
        continue

    wins_baseline = int(
        (
            subset[
                "brier_improvement_vs_baseline"
            ] > 0
        ).sum()
    )

    wins_sample = int(
        (
            subset[
                "brier_improvement_vs_sample_control"
            ] > 0
        ).sum()
    )

    avg_baseline = float(
        subset[
            "brier_improvement_vs_baseline"
        ].mean()
    )

    avg_sample = float(
        subset[
            "brier_improvement_vs_sample_control"
        ].mean()
    )

    worst = float(
        subset[
            "brier_improvement_vs_baseline"
        ].min()
    )

    best = float(
        subset[
            "brier_improvement_vs_baseline"
        ].max()
    )

    lines.extend([
        model_name,
        "-" * 78,
        (
            f"Beat half-specific baseline: "
            f"{wins_baseline}/{len(subset)} holdouts"
        ),
        (
            f"Beat sample-only control: "
            f"{wins_sample}/{len(subset)} holdouts"
        ),
        (
            f"Average Brier improvement vs baseline: "
            f"{avg_baseline:+.6f}"
        ),
        (
            f"Average Brier improvement vs sample-only: "
            f"{avg_sample:+.6f}"
        ),
        (
            f"Worst holdout improvement vs baseline: "
            f"{worst:+.6f}"
        ),
        (
            f"Best holdout improvement vs baseline: "
            f"{best:+.6f}"
        ),
        "",
    ])

    for _, row in subset.iterrows():
        lines.append(
            f"{int(row['holdout_year'])}: "
            f"baseline {row['baseline_brier']:.6f} | "
            f"model {row['model_brier']:.6f} | "
            f"vs baseline "
            f"{row['brier_improvement_vs_baseline']:+.6f} | "
            f"vs sample "
            f"{row['brier_improvement_vs_sample_control']:+.6f}"
        )

    lines.extend([
        "",
        "=" * 78,
        "",
    ])

    if model_name != "Sample only control":
        ranking_rows.append({
            "model":
                model_name,
            "wins_vs_baseline":
                wins_baseline,
            "wins_vs_sample":
                wins_sample,
            "avg_vs_baseline":
                avg_baseline,
            "avg_vs_sample":
                avg_sample,
            "worst_vs_baseline":
                worst,
        })


ranking = (
    pd.DataFrame(
        ranking_rows
    )
    .sort_values(
        [
            "wins_vs_baseline",
            "avg_vs_baseline",
            "worst_vs_baseline",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)


lines.extend([
    "FINAL RANKING",
    "-" * 78,
])


for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    lines.append(
        f"#{rank} {row['model']} | "
        f"baseline wins "
        f"{int(row['wins_vs_baseline'])}/4 | "
        f"sample wins "
        f"{int(row['wins_vs_sample'])}/4 | "
        f"avg baseline "
        f"{row['avg_vs_baseline']:+.6f} | "
        f"avg sample "
        f"{row['avg_vs_sample']:+.6f} | "
        f"worst "
        f"{row['worst_vs_baseline']:+.6f}"
    )


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print(
    "TOP-4 OFFENSE PREDICTIVE TEST COMPLETE"
)
print()
print(
    f"Created: {SUMMARY_FILE}"
)
print(
    f"Created: {RESULTS_FILE}"
)
print(
    f"Created: {COEFFICIENTS_FILE}"
)
