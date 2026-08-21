
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7F1
#
# COMBINED CORE MODEL TEST
#
# Question:
# Do the validated pitcher and Top-4 offense signals improve
# first-inning probability forecasts when used together?
#
# Pitcher core:
#   xwOBA allowed
#   K%
#   prior PA sample controls
#
# Offense core:
#   Top-4 xwOBA
#   Top-4 K%
#   Top-4 Barrel%
#   Top-4 sample controls
#
# Walk-forward:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom 1st are modeled separately.
# Positive Brier improvement = better forecast.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
PITCHER_FILE = Path("historical_pitcher_skill_context.csv")
OFFENSE_FILE = Path("historical_top4_skill_context.csv")

SUMMARY_FILE = Path(
    "historical_combined_core_model_summary.txt"
)

RESULTS_FILE = Path(
    "historical_combined_core_model_results.csv"
)

COEFFICIENTS_FILE = Path(
    "historical_combined_core_model_coefficients.csv"
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


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


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
# LOAD
# =========================================================

for path in [
    OUTCOMES_FILE,
    PITCHER_FILE,
    OFFENSE_FILE,
]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} was not found."
        )

outcomes = pd.read_csv(
    OUTCOMES_FILE
)

pitcher = pd.read_csv(
    PITCHER_FILE
)

offense = pd.read_csv(
    OFFENSE_FILE
)


game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in historical outcomes",
)

game_pk_pitcher = find_column(
    pitcher,
    ["game_pk", "gamepk", "game_id"],
    "game ID in pitcher skill",
)

game_pk_offense = find_column(
    offense,
    ["game_pk", "gamepk", "game_id"],
    "game ID in Top-4 offense skill",
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
# KEEP ONLY REQUIRED FEATURE COLUMNS BEFORE MERGING
# =========================================================

pitcher_keep = [
    game_pk_pitcher,

    "away_sp_pa_before_game",
    "away_sp_xwoba_allowed",
    "away_sp_k_pct",
    "away_sp_bb_pct",
    "away_sp_barrel_pct",

    "home_sp_pa_before_game",
    "home_sp_xwoba_allowed",
    "home_sp_k_pct",
    "home_sp_bb_pct",
    "home_sp_barrel_pct",
]

offense_keep = [
    game_pk_offense,

    "away_top4_xwoba",
    "away_top4_k_pct",
    "away_top4_bb_pct",
    "away_top4_barrel_pct",
    "away_top4_combined_pa",
    "away_top4_min_pa",
    "away_top4_complete_core",

    "home_top4_xwoba",
    "home_top4_k_pct",
    "home_top4_bb_pct",
    "home_top4_barrel_pct",
    "home_top4_combined_pa",
    "home_top4_min_pa",
    "home_top4_complete_core",
]

for column in pitcher_keep:
    if column not in pitcher.columns:
        raise SystemExit(
            f"ERROR: Missing pitcher column: {column}"
        )

for column in offense_keep:
    if column not in offense.columns:
        raise SystemExit(
            f"ERROR: Missing offense column: {column}"
        )


merged = outcomes.merge(
    pitcher[pitcher_keep],
    left_on=game_pk_out,
    right_on=game_pk_pitcher,
    how="inner",
)

merged = merged.merge(
    offense[offense_keep],
    left_on=game_pk_out,
    right_on=game_pk_offense,
    how="inner",
    suffixes=("", "_offense"),
)

if len(merged) != len(outcomes):
    raise SystemExit(
        "ERROR: Combined merge did not preserve "
        f"all historical games. "
        f"Outcomes={len(outcomes)}, merged={len(merged)}"
    )


season_series = pd.to_numeric(
    merged[season_out],
    errors="raise",
).astype(int)

date_series = pd.to_datetime(
    merged[date_out],
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
# BUILD HALF-INNING DATA
# =========================================================

# Top 1st:
# Away Top-4 offense vs Home starting pitcher.
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

    "p_xwoba":
        pd.to_numeric(
            merged["home_sp_xwoba_allowed"],
            errors="coerce",
        ),
    "p_k_pct":
        pd.to_numeric(
            merged["home_sp_k_pct"],
            errors="coerce",
        ),
    "p_bb_pct":
        pd.to_numeric(
            merged["home_sp_bb_pct"],
            errors="coerce",
        ),
    "p_barrel_pct":
        pd.to_numeric(
            merged["home_sp_barrel_pct"],
            errors="coerce",
        ),
    "p_pa":
        pd.to_numeric(
            merged["home_sp_pa_before_game"],
            errors="coerce",
        ),

    "o_xwoba":
        pd.to_numeric(
            merged["away_top4_xwoba"],
            errors="coerce",
        ),
    "o_k_pct":
        pd.to_numeric(
            merged["away_top4_k_pct"],
            errors="coerce",
        ),
    "o_bb_pct":
        pd.to_numeric(
            merged["away_top4_bb_pct"],
            errors="coerce",
        ),
    "o_barrel_pct":
        pd.to_numeric(
            merged["away_top4_barrel_pct"],
            errors="coerce",
        ),
    "o_combined_pa":
        pd.to_numeric(
            merged["away_top4_combined_pa"],
            errors="coerce",
        ),
    "o_min_pa":
        pd.to_numeric(
            merged["away_top4_min_pa"],
            errors="coerce",
        ),
    "o_complete_core":
        pd.to_numeric(
            merged["away_top4_complete_core"],
            errors="coerce",
        ),
})


# Bottom 1st:
# Home Top-4 offense vs Away starting pitcher.
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

    "p_xwoba":
        pd.to_numeric(
            merged["away_sp_xwoba_allowed"],
            errors="coerce",
        ),
    "p_k_pct":
        pd.to_numeric(
            merged["away_sp_k_pct"],
            errors="coerce",
        ),
    "p_bb_pct":
        pd.to_numeric(
            merged["away_sp_bb_pct"],
            errors="coerce",
        ),
    "p_barrel_pct":
        pd.to_numeric(
            merged["away_sp_barrel_pct"],
            errors="coerce",
        ),
    "p_pa":
        pd.to_numeric(
            merged["away_sp_pa_before_game"],
            errors="coerce",
        ),

    "o_xwoba":
        pd.to_numeric(
            merged["home_top4_xwoba"],
            errors="coerce",
        ),
    "o_k_pct":
        pd.to_numeric(
            merged["home_top4_k_pct"],
            errors="coerce",
        ),
    "o_bb_pct":
        pd.to_numeric(
            merged["home_top4_bb_pct"],
            errors="coerce",
        ),
    "o_barrel_pct":
        pd.to_numeric(
            merged["home_top4_barrel_pct"],
            errors="coerce",
        ),
    "o_combined_pa":
        pd.to_numeric(
            merged["home_top4_combined_pa"],
            errors="coerce",
        ),
    "o_min_pa":
        pd.to_numeric(
            merged["home_top4_min_pa"],
            errors="coerce",
        ),
    "o_complete_core":
        pd.to_numeric(
            merged["home_top4_complete_core"],
            errors="coerce",
        ),
})


data = pd.concat(
    [top, bottom],
    ignore_index=True,
)


# =========================================================
# SAMPLE CONTROLS / MISSINGNESS
# =========================================================

data["p_pa"] = (
    data["p_pa"]
    .fillna(0)
    .clip(lower=0)
)

data["o_combined_pa"] = (
    data["o_combined_pa"]
    .fillna(0)
    .clip(lower=0)
)

data["o_min_pa"] = (
    data["o_min_pa"]
    .fillna(0)
    .clip(lower=0)
)

data["o_complete_core"] = (
    data["o_complete_core"]
    .fillna(0)
)

data["p_log_pa"] = np.log1p(
    data["p_pa"]
)

data["p_no_prior_pa"] = (
    data["p_pa"] <= 0
).astype(float)

data["o_log_combined_pa"] = np.log1p(
    data["o_combined_pa"]
)

data["o_log_min_pa"] = np.log1p(
    data["o_min_pa"]
)

data["o_missing_core"] = (
    data["o_complete_core"] < 1
).astype(float)


if len(data) != 2 * len(outcomes):
    raise SystemExit(
        "ERROR: Half-inning row count is wrong."
    )


# =========================================================
# MODEL SETS
# =========================================================

sample_controls = [
    "p_log_pa",
    "p_no_prior_pa",
    "o_log_combined_pa",
    "o_log_min_pa",
    "o_missing_core",
]


models = {
    "Combined sample control":
        sample_controls,

    "Pitcher core only":
        [
            "p_xwoba",
            "p_k_pct",
            *sample_controls,
        ],

    "Offense core only":
        [
            "o_xwoba",
            "o_k_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "Pitcher + offense core":
        [
            "p_xwoba",
            "p_k_pct",
            "o_xwoba",
            "o_k_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "Core + pitcher Barrel":
        [
            "p_xwoba",
            "p_k_pct",
            "p_barrel_pct",
            "o_xwoba",
            "o_k_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "Core + offense BB":
        [
            "p_xwoba",
            "p_k_pct",
            "o_xwoba",
            "o_k_pct",
            "o_bb_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "Core + pitcher BB":
        [
            "p_xwoba",
            "p_k_pct",
            "p_bb_pct",
            "o_xwoba",
            "o_k_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "All pitcher + all offense skill":
        [
            "p_xwoba",
            "p_k_pct",
            "p_bb_pct",
            "p_barrel_pct",
            "o_xwoba",
            "o_k_pct",
            "o_bb_pct",
            "o_barrel_pct",
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

    baseline_predictions = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    model_predictions = {
        name: np.full(
            len(test),
            np.nan,
            dtype=float,
        )
        for name in models
    }

    for half in ["Top", "Bottom"]:
        train_half = train[
            train["half"] == half
        ].copy()

        mask = (
            test["half"] == half
        )

        test_half = test[
            mask
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
            mask.to_numpy()
        ] = baseline_rate

        for model_name, features in models.items():
            X_train, X_test = prepare_matrix(
                train_half,
                test_half,
                features,
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

            model_predictions[
                model_name
            ][
                mask.to_numpy()
            ] = pred

            for i, feature in enumerate(features):
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

    control_brier = None
    year_rows = []

    for model_name in models:
        model_brier = brier_score(
            y_test,
            model_predictions[model_name],
        )

        if model_name == "Combined sample control":
            control_brier = model_brier

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
            "improvement_vs_baseline":
                baseline_brier - model_brier,
        })

    for row in year_rows:
        row["improvement_vs_sample_control"] = (
            control_brier
            - row["model_brier"]
        )

        results.append(row)


# =========================================================
# SAVE
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
    "SHARPREPORT COMBINED PITCHER + OFFENSE CORE MODEL TEST",
    "=" * 82,
    "",
    f"Historical games: {len(outcomes)}",
    f"Half-inning rows: {len(data)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample testing.",
    "Top and Bottom 1st modeled separately.",
    "Positive Brier improvement = better probability forecast.",
    "",
    "VALIDATED CORE INPUTS:",
    "Pitcher: xwOBA allowed + K% + prior PA",
    "Offense: Top-4 xwOBA + K% + Barrel% + sample controls",
    "",
]


ranking_rows = []


for model_name in models:
    subset = results_df[
        results_df["model"] == model_name
    ].copy()

    wins_base = int(
        (
            subset[
                "improvement_vs_baseline"
            ] > 0
        ).sum()
    )

    wins_control = int(
        (
            subset[
                "improvement_vs_sample_control"
            ] > 0
        ).sum()
    )

    avg_base = float(
        subset[
            "improvement_vs_baseline"
        ].mean()
    )

    avg_control = float(
        subset[
            "improvement_vs_sample_control"
        ].mean()
    )

    worst = float(
        subset[
            "improvement_vs_baseline"
        ].min()
    )

    best = float(
        subset[
            "improvement_vs_baseline"
        ].max()
    )

    lines.extend([
        model_name,
        "-" * 82,
        (
            f"Beat half-specific baseline: "
            f"{wins_base}/{len(subset)}"
        ),
        (
            f"Beat combined sample control: "
            f"{wins_control}/{len(subset)}"
        ),
        (
            f"Average improvement vs baseline: "
            f"{avg_base:+.6f}"
        ),
        (
            f"Average improvement vs sample control: "
            f"{avg_control:+.6f}"
        ),
        (
            f"Worst holdout vs baseline: "
            f"{worst:+.6f}"
        ),
        (
            f"Best holdout vs baseline: "
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
            f"{row['improvement_vs_baseline']:+.6f} | "
            f"vs sample "
            f"{row['improvement_vs_sample_control']:+.6f}"
        )

    lines.extend([
        "",
        "=" * 82,
        "",
    ])

    if model_name != "Combined sample control":
        ranking_rows.append({
            "model":
                model_name,
            "wins_vs_baseline":
                wins_base,
            "wins_vs_control":
                wins_control,
            "avg_vs_baseline":
                avg_base,
            "avg_vs_control":
                avg_control,
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
    "-" * 82,
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
        f"{int(row['wins_vs_control'])}/4 | "
        f"avg baseline "
        f"{row['avg_vs_baseline']:+.6f} | "
        f"avg sample "
        f"{row['avg_vs_control']:+.6f} | "
        f"worst "
        f"{row['worst_vs_baseline']:+.6f}"
    )


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print(
    "COMBINED CORE MODEL TEST COMPLETE"
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
