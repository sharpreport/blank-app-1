
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7D3
#
# Walk-forward test:
# Do pregame pitcher xwOBA / K% / BB% / Barrel%
# improve prediction of future 1st-inning scoring?
#
# Training / holdout structure:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom 1st are modeled separately.
# Positive "Brier improvement" = better than baseline.
#
# Also compares every skill model against a PA-only control,
# so we can see whether the metrics add signal beyond sample size.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
SKILL_FILE = Path("historical_pitcher_skill_context.csv")

SUMMARY_FILE = Path(
    "historical_pitcher_skill_predictive_summary.txt"
)

RESULTS_FILE = Path(
    "historical_pitcher_skill_predictive_results.csv"
)

COEFFICIENTS_FILE = Path(
    "historical_pitcher_skill_coefficients.csv"
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

    means = {}
    stds = {}

    for col in feature_cols:
        train[col] = pd.to_numeric(
            train[col],
            errors="coerce",
        )
        test[col] = pd.to_numeric(
            test[col],
            errors="coerce",
        )

        mean_value = float(
            train[col].mean()
        ) if train[col].notna().any() else 0.0

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

        means[col] = mean_value
        stds[col] = std_value

    return (
        train.to_numpy(dtype=float),
        test.to_numpy(dtype=float),
        means,
        stds,
    )


def fit_ridge_logistic(X, y, lam=1.0):
    y = np.asarray(y, dtype=float)

    X_design = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    beta = np.zeros(
        X_design.shape[1],
        dtype=float,
    )

    penalty = np.eye(
        X_design.shape[1],
        dtype=float,
    )
    penalty[0, 0] = 0.0

    for _ in range(MAX_ITER):
        eta = X_design @ beta
        p = sigmoid(eta)

        w = np.clip(
            p * (1.0 - p),
            1e-6,
            None,
        )

        gradient = (
            X_design.T @ (y - p)
            - lam * (penalty @ beta)
        )

        hessian = (
            X_design.T
            @ (w[:, None] * X_design)
            + lam * penalty
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

        beta_new = beta + step

        if np.max(
            np.abs(beta_new - beta)
        ) < TOL:
            beta = beta_new
            break

        beta = beta_new

    return beta


def predict_logistic(beta, X):
    X_design = np.column_stack([
        np.ones(len(X)),
        X,
    ])
    return sigmoid(
        X_design @ beta
    )


# =========================================================
# LOAD DATA
# =========================================================

if not OUTCOMES_FILE.exists():
    raise SystemExit(
        "ERROR: historical_first_innings.csv was not found."
    )

if not SKILL_FILE.exists():
    raise SystemExit(
        "ERROR: historical_pitcher_skill_context.csv was not found."
    )

outcomes = pd.read_csv(OUTCOMES_FILE)
skill = pd.read_csv(SKILL_FILE)


game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in historical outcomes",
)

game_pk_skill = find_column(
    skill,
    ["game_pk", "gamepk", "game_id"],
    "game ID in pitcher skill context",
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


# Try direct scored flags first.
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


# If scored flags are absent, derive from first-inning runs.
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
# CONSTRUCT HALF-INNING DATASET
# =========================================================

merged = outcomes.merge(
    skill,
    left_on=game_pk_out,
    right_on=game_pk_skill,
    how="inner",
    suffixes=("_out", "_skill"),
)

if len(merged) != len(outcomes):
    raise SystemExit(
        "ERROR: Outcomes and pitcher-skill game counts "
        f"did not match after merge. "
        f"Outcomes={len(outcomes)}, merged={len(merged)}"
    )


# After the merge, columns that exist in BOTH source files
# (such as season/date) receive the _out / _skill suffixes.
# Resolve the actual merged column names instead of assuming
# the original unsuffixed names still exist.

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
            values = pd.to_numeric(
                merged[top_scored_col],
                errors="coerce",
            )
        else:
            values = (
                pd.to_numeric(
                    merged[away_runs_col],
                    errors="coerce",
                ) > 0
            ).astype(int)
    else:
        if bottom_scored_col is not None:
            values = pd.to_numeric(
                merged[bottom_scored_col],
                errors="coerce",
            )
        else:
            values = (
                pd.to_numeric(
                    merged[home_runs_col],
                    errors="coerce",
                ) > 0
            ).astype(int)

    return values.astype(int)


# Top 1st: away offense faces HOME starter.
top = pd.DataFrame({
    "game_pk": merged[game_pk_out],
    "date": date_series,
    "season": season_series,
    "half": "Top",
    "scored": scored_values("top"),

    "pa": pd.to_numeric(
        merged["home_sp_pa_before_game"],
        errors="coerce",
    ),
    "xwoba": pd.to_numeric(
        merged["home_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "k_pct": pd.to_numeric(
        merged["home_sp_k_pct"],
        errors="coerce",
    ),
    "bb_pct": pd.to_numeric(
        merged["home_sp_bb_pct"],
        errors="coerce",
    ),
    "barrel_pct": pd.to_numeric(
        merged["home_sp_barrel_pct"],
        errors="coerce",
    ),
})


# Bottom 1st: home offense faces AWAY starter.
bottom = pd.DataFrame({
    "game_pk": merged[game_pk_out],
    "date": date_series,
    "season": season_series,
    "half": "Bottom",
    "scored": scored_values("bottom"),

    "pa": pd.to_numeric(
        merged["away_sp_pa_before_game"],
        errors="coerce",
    ),
    "xwoba": pd.to_numeric(
        merged["away_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "k_pct": pd.to_numeric(
        merged["away_sp_k_pct"],
        errors="coerce",
    ),
    "bb_pct": pd.to_numeric(
        merged["away_sp_bb_pct"],
        errors="coerce",
    ),
    "barrel_pct": pd.to_numeric(
        merged["away_sp_barrel_pct"],
        errors="coerce",
    ),
})


data = pd.concat(
    [top, bottom],
    ignore_index=True,
)

data["pa"] = data["pa"].fillna(0.0)
data["log_pa"] = np.log1p(
    data["pa"].clip(lower=0)
)
data["no_prior_pa"] = (
    data["pa"] <= 0
).astype(float)


if len(data) != 2 * len(outcomes):
    raise SystemExit(
        "ERROR: Half-inning row count is wrong."
    )


# =========================================================
# MODELS TO TEST
# =========================================================

models = {
    "PA only control": [
        "log_pa",
        "no_prior_pa",
    ],

    "xwOBA + PA": [
        "xwoba",
        "log_pa",
        "no_prior_pa",
    ],

    "K% + PA": [
        "k_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "BB% + PA": [
        "bb_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "Barrel% + PA": [
        "barrel_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "All 4 pitcher skill + PA": [
        "xwoba",
        "k_pct",
        "bb_pct",
        "barrel_pct",
        "log_pa",
        "no_prior_pa",
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

    test = data[
        data["season"] == holdout_year
    ].copy()

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

    test = test.reset_index(drop=True)

    # Separate Top and Bottom models because their
    # historical base rates differ materially.
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
                means,
                stds,
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
                    "holdout_year": holdout_year,
                    "half": half,
                    "model": model_name,
                    "feature": feature,
                    "standardized_coefficient": float(
                        beta[i + 1]
                    ),
                })


    y_test = test[
        "scored"
    ].to_numpy(dtype=float)

    baseline_brier = brier_score(
        y_test,
        baseline_predictions,
    )

    pa_control_brier = None

    for model_name in models:
        model_brier = brier_score(
            y_test,
            year_predictions[model_name],
        )

        if model_name == "PA only control":
            pa_control_brier = model_brier

        results.append({
            "holdout_year": holdout_year,
            "model": model_name,
            "rows": len(test),
            "baseline_brier": baseline_brier,
            "model_brier": model_brier,
            "brier_improvement_vs_baseline":
                baseline_brier - model_brier,
            "brier_improvement_vs_pa_control":
                np.nan,
        })

    # Fill incremental improvement vs PA control.
    for row in results:
        if (
            row["holdout_year"] == holdout_year
            and pa_control_brier is not None
        ):
            row[
                "brier_improvement_vs_pa_control"
            ] = (
                pa_control_brier
                - row["model_brier"]
            )


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)
coeff_df = pd.DataFrame(coefficient_rows)

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
    "SHARPREPORT HISTORICAL PITCHER SKILL PREDICTIVE TEST",
    "=" * 76,
    "",
    f"Historical games: {len(outcomes)}",
    f"Half-inning rows: {len(data)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample testing.",
    "Top 1st and Bottom 1st modeled separately.",
    "Positive Brier improvement = better probability forecast.",
    (
        "Every skill model includes prior-PA sample size, and is "
        "also compared against a PA-only control."
    ),
    "",
]


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

    wins_pa = int(
        (
            subset[
                "brier_improvement_vs_pa_control"
            ] > 0
        ).sum()
    )

    avg_baseline = float(
        subset[
            "brier_improvement_vs_baseline"
        ].mean()
    )

    avg_pa = float(
        subset[
            "brier_improvement_vs_pa_control"
        ].mean()
    )

    worst_baseline = float(
        subset[
            "brier_improvement_vs_baseline"
        ].min()
    )

    best_baseline = float(
        subset[
            "brier_improvement_vs_baseline"
        ].max()
    )

    lines.extend([
        model_name,
        "-" * 76,
        (
            f"Beat half-specific baseline: "
            f"{wins_baseline}/{len(subset)} holdouts"
        ),
    ])

    if model_name != "PA only control":
        lines.append(
            f"Beat PA-only control: "
            f"{wins_pa}/{len(subset)} holdouts"
        )

    lines.extend([
        (
            f"Average Brier improvement vs baseline: "
            f"{avg_baseline:+.6f}"
        ),
    ])

    if model_name != "PA only control":
        lines.append(
            f"Average Brier improvement vs PA-only: "
            f"{avg_pa:+.6f}"
        )

    lines.extend([
        (
            f"Worst holdout improvement vs baseline: "
            f"{worst_baseline:+.6f}"
        ),
        (
            f"Best holdout improvement vs baseline: "
            f"{best_baseline:+.6f}"
        ),
        "",
    ])

    for _, row in subset.iterrows():
        extra = ""

        if model_name != "PA only control":
            extra = (
                f" | vs PA "
                f"{row['brier_improvement_vs_pa_control']:+.6f}"
            )

        lines.append(
            f"{int(row['holdout_year'])}: "
            f"baseline {row['baseline_brier']:.6f} | "
            f"model {row['model_brier']:.6f} | "
            f"improvement "
            f"{row['brier_improvement_vs_baseline']:+.6f}"
            f"{extra}"
        )

    lines.extend([
        "",
        "=" * 76,
        "",
    ])


# Rank non-control models by average incremental value over PA only.
ranking = (
    results_df[
        results_df["model"] != "PA only control"
    ]
    .groupby("model", as_index=False)
    .agg(
        holdouts=(
            "holdout_year",
            "count",
        ),
        wins_vs_pa=(
            "brier_improvement_vs_pa_control",
            lambda x: int((x > 0).sum()),
        ),
        avg_vs_pa=(
            "brier_improvement_vs_pa_control",
            "mean",
        ),
        avg_vs_baseline=(
            "brier_improvement_vs_baseline",
            "mean",
        ),
    )
    .sort_values(
        [
            "avg_vs_pa",
            "avg_vs_baseline",
        ],
        ascending=False,
    )
)


lines.extend([
    "RANKING BY INCREMENTAL VALUE OVER PA-ONLY CONTROL",
    "-" * 76,
])

for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    lines.append(
        f"#{rank} {row['model']} | "
        f"wins vs PA {int(row['wins_vs_pa'])}/"
        f"{int(row['holdouts'])} | "
        f"avg vs PA {row['avg_vs_pa']:+.6f} | "
        f"avg vs baseline {row['avg_vs_baseline']:+.6f}"
    )


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print("PITCHER SKILL PREDICTIVE TEST COMPLETE")
print()
print(f"Created: {SUMMARY_FILE}")
print(f"Created: {RESULTS_FILE}")
print(f"Created: {COEFFICIENTS_FILE}")
