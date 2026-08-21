
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7H1
#
# GAME-LEVEL NRFI / YRFI VALIDATION
#
# We have been validating Top and Bottom 1st separately.
# This stage answers the question the final product actually
# cares about:
#
#   Does the model improve prediction of the full-game
#   NRFI/YRFI outcome?
#
# Candidate models:
#   1) Combined core
#   2) Core + Run Factor
#   3) Core + Offense BB%
#   4) Core + Offense BB% + Run Factor
#
# NRFI probability:
#   P(NRFI) = (1 - P(Top scores)) * (1 - P(Bottom scores))
#
# Walk-forward:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom scoring models are fit separately.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
PITCHER_FILE = Path("historical_pitcher_skill_context.csv")
OFFENSE_FILE = Path("historical_top4_skill_context.csv")
PARK_FILE = Path("historical_park_context.csv")

SUMMARY_FILE = Path(
    "historical_game_level_nrfi_validation_summary.txt"
)

RESULTS_FILE = Path(
    "historical_game_level_nrfi_validation_results.csv"
)

PREDICTIONS_FILE = Path(
    "historical_game_level_nrfi_predictions.csv"
)

HOLDOUT_YEARS = [2023, 2024, 2025, 2026]

RIDGE_LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-8


def find_column(df, candidates, label, required=True):
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

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


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    return float(
        -np.mean(
            y * np.log(p)
            + (1 - y) * np.log(1 - p)
        )
    )


def prepare_matrix(train_df, test_df, features):
    train = train_df[features].copy()
    test = test_df[features].copy()

    for col in features:
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
    PARK_FILE,
]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} was not found."
        )

outcomes = pd.read_csv(OUTCOMES_FILE)
pitcher = pd.read_csv(PITCHER_FILE)
offense = pd.read_csv(OFFENSE_FILE)
park = pd.read_csv(PARK_FILE)

game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in outcomes",
)

season_col = find_column(
    outcomes,
    ["season", "year"],
    "season",
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
    "Top 1st scored",
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
    "Bottom 1st scored",
    required=False,
)

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
    "Top 1st runs",
    required=False,
)

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
    "Bottom 1st runs",
    required=False,
)

if top_scored_col is None and away_runs_col is None:
    raise SystemExit(
        "ERROR: Could not resolve Top 1st outcome."
    )

if bottom_scored_col is None and home_runs_col is None:
    raise SystemExit(
        "ERROR: Could not resolve Bottom 1st outcome."
    )


game_pk_pitcher = find_column(
    pitcher,
    ["game_pk", "gamepk", "game_id"],
    "game ID in pitcher skill",
)

game_pk_offense = find_column(
    offense,
    ["game_pk", "gamepk", "game_id"],
    "game ID in offense skill",
)

game_pk_park = find_column(
    park,
    ["game_pk", "gamepk", "game_id"],
    "game ID in park context",
)


pitcher_keep = [
    game_pk_pitcher,
    "away_sp_pa_before_game",
    "away_sp_xwoba_allowed",
    "away_sp_k_pct",
    "home_sp_pa_before_game",
    "home_sp_xwoba_allowed",
    "home_sp_k_pct",
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

park_keep = [
    game_pk_park,
    "park_runs",
]

for col in pitcher_keep:
    if col not in pitcher.columns:
        raise SystemExit(
            f"ERROR: Missing pitcher column: {col}"
        )

for col in offense_keep:
    if col not in offense.columns:
        raise SystemExit(
            f"ERROR: Missing offense column: {col}"
        )

for col in park_keep:
    if col not in park.columns:
        raise SystemExit(
            f"ERROR: Missing park column: {col}"
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
)

merged = merged.merge(
    park[park_keep],
    left_on=game_pk_out,
    right_on=game_pk_park,
    how="inner",
)

if len(merged) != len(outcomes):
    raise SystemExit(
        "ERROR: Combined merge did not preserve "
        f"all historical games. "
        f"Outcomes={len(outcomes)}, merged={len(merged)}"
    )


season_series = pd.to_numeric(
    merged[season_col],
    errors="raise",
).astype(int)


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


top_scored = scored_values("top")
bottom_scored = scored_values("bottom")

park_runs = pd.to_numeric(
    merged["park_runs"],
    errors="coerce",
)

park_missing = park_runs.isna().astype(float)

# Missing special-venue park values are neutral.
park_runs = park_runs.fillna(100.0)


# =========================================================
# HALF-INNING FEATURE DATA
# =========================================================

top = pd.DataFrame({
    "game_pk":
        merged[game_pk_out],
    "season":
        season_series,
    "half":
        "Top",
    "scored":
        top_scored,

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

    "park_runs":
        park_runs,
    "park_missing":
        park_missing,
})


bottom = pd.DataFrame({
    "game_pk":
        merged[game_pk_out],
    "season":
        season_series,
    "half":
        "Bottom",
    "scored":
        bottom_scored,

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

    "park_runs":
        park_runs,
    "park_missing":
        park_missing,
})


data = pd.concat(
    [top, bottom],
    ignore_index=True,
)


for col in [
    "p_pa",
    "o_combined_pa",
    "o_min_pa",
]:
    data[col] = (
        data[col]
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


sample_controls = [
    "p_log_pa",
    "p_no_prior_pa",
    "o_log_combined_pa",
    "o_log_min_pa",
    "o_missing_core",
]

core_features = [
    "p_xwoba",
    "p_k_pct",
    "o_xwoba",
    "o_k_pct",
    "o_barrel_pct",
    *sample_controls,
]

models = {
    "Combined core":
        core_features,

    "Core + Run Factor":
        [
            *core_features,
            "park_runs",
            "park_missing",
        ],

    "Core + Offense BB":
        [
            "p_xwoba",
            "p_k_pct",
            "o_xwoba",
            "o_k_pct",
            "o_bb_pct",
            "o_barrel_pct",
            *sample_controls,
        ],

    "Core + Offense BB + Run Factor":
        [
            "p_xwoba",
            "p_k_pct",
            "o_xwoba",
            "o_k_pct",
            "o_bb_pct",
            "o_barrel_pct",
            *sample_controls,
            "park_runs",
            "park_missing",
        ],
}


# =========================================================
# WALK-FORWARD GAME-LEVEL PREDICTIONS
# =========================================================

prediction_rows = []


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

    print()
    print(
        f"Testing game-level NRFI {holdout_year}"
    )

    # Half-specific historical baseline rates.
    baseline_half = {}

    for half in ["Top", "Bottom"]:
        train_half = train[
            train["half"] == half
        ]

        baseline_half[half] = float(
            train_half["scored"].mean()
        )

    baseline_nrfi = (
        (1.0 - baseline_half["Top"])
        * (1.0 - baseline_half["Bottom"])
    )

    # Actual game-level outcomes for holdout.
    game_actual = (
        test
        .pivot(
            index="game_pk",
            columns="half",
            values="scored",
        )
        .reset_index()
    )

    game_actual["nrfi_actual"] = (
        (
            game_actual["Top"] == 0
        )
        &
        (
            game_actual["Bottom"] == 0
        )
    ).astype(int)

    for model_name, features in models.items():
        half_predictions = []

        for half in ["Top", "Bottom"]:
            train_half = train[
                train["half"] == half
            ].copy()

            test_half = test[
                test["half"] == half
            ].copy()

            X_train, X_test = prepare_matrix(
                train_half,
                test_half,
                features,
            )

            beta = fit_ridge_logistic(
                X_train,
                train_half["scored"].to_numpy(
                    dtype=float
                ),
                lam=RIDGE_LAMBDA,
            )

            pred = predict_logistic(
                beta,
                X_test,
            )

            temp = pd.DataFrame({
                "game_pk":
                    test_half["game_pk"].to_numpy(),
                "half":
                    half,
                "p_scores":
                    pred,
            })

            half_predictions.append(
                temp
            )

        half_df = pd.concat(
            half_predictions,
            ignore_index=True,
        )

        wide = (
            half_df
            .pivot(
                index="game_pk",
                columns="half",
                values="p_scores",
            )
            .reset_index()
        )

        wide["nrfi_probability"] = (
            (1.0 - wide["Top"])
            * (1.0 - wide["Bottom"])
        )

        combined = wide.merge(
            game_actual[
                [
                    "game_pk",
                    "nrfi_actual",
                ]
            ],
            on="game_pk",
            how="inner",
        )

        for _, row in combined.iterrows():
            prediction_rows.append({
                "holdout_year":
                    holdout_year,
                "model":
                    model_name,
                "game_pk":
                    int(row["game_pk"]),
                "nrfi_actual":
                    int(row["nrfi_actual"]),
                "nrfi_probability":
                    float(row["nrfi_probability"]),
                "baseline_nrfi_probability":
                    float(baseline_nrfi),
            })


predictions = pd.DataFrame(
    prediction_rows
)

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# =========================================================
# RESULTS
# =========================================================

result_rows = []


for holdout_year in HOLDOUT_YEARS:
    year_preds = predictions[
        predictions["holdout_year"] == holdout_year
    ]

    # The actual rows are identical across candidate models.
    base_sample = year_preds[
        year_preds["model"] == "Combined core"
    ]

    y = base_sample[
        "nrfi_actual"
    ].to_numpy(dtype=float)

    baseline_prob = base_sample[
        "baseline_nrfi_probability"
    ].to_numpy(dtype=float)

    baseline_brier = brier_score(
        y,
        baseline_prob,
    )

    baseline_logloss = log_loss(
        y,
        baseline_prob,
    )

    core_brier = None
    core_logloss = None

    year_model_rows = []

    for model_name in models:
        subset = year_preds[
            year_preds["model"] == model_name
        ]

        p = subset[
            "nrfi_probability"
        ].to_numpy(dtype=float)

        model_brier = brier_score(
            y,
            p,
        )

        model_logloss = log_loss(
            y,
            p,
        )

        if model_name == "Combined core":
            core_brier = model_brier
            core_logloss = model_logloss

        year_model_rows.append({
            "holdout_year":
                holdout_year,
            "model":
                model_name,
            "games":
                len(subset),
            "baseline_brier":
                baseline_brier,
            "model_brier":
                model_brier,
            "improvement_vs_baseline_brier":
                baseline_brier - model_brier,
            "baseline_logloss":
                baseline_logloss,
            "model_logloss":
                model_logloss,
            "improvement_vs_baseline_logloss":
                baseline_logloss - model_logloss,
        })

    for row in year_model_rows:
        row["improvement_vs_core_brier"] = (
            core_brier
            - row["model_brier"]
        )

        row["improvement_vs_core_logloss"] = (
            core_logloss
            - row["model_logloss"]
        )

        result_rows.append(row)


results = pd.DataFrame(
    result_rows
)

results.to_csv(
    RESULTS_FILE,
    index=False,
)


# =========================================================
# CALIBRATION TABLES
# =========================================================

# Use 2026 as the cleanest current holdout for a simple
# probability-bucket sanity check.
calibration_lines = []

holdout_2026 = predictions[
    predictions["holdout_year"] == 2026
].copy()

for model_name in models:
    subset = holdout_2026[
        holdout_2026["model"] == model_name
    ].copy()

    if subset.empty:
        continue

    subset["bucket"] = pd.cut(
        subset["nrfi_probability"],
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
            games=("game_pk", "count"),
            avg_predicted=(
                "nrfi_probability",
                "mean",
            ),
            actual_nrfi_rate=(
                "nrfi_actual",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration_lines.extend([
        "",
        f"2026 CALIBRATION — {model_name}",
        "-" * 86,
    ])

    for _, row in grouped.iterrows():
        calibration_lines.append(
            f"{row['bucket']} | "
            f"games {int(row['games'])} | "
            f"pred {row['avg_predicted']:.3f} | "
            f"actual {row['actual_nrfi_rate']:.3f}"
        )


# =========================================================
# SUMMARY
# =========================================================

lines = [
    "SHARPREPORT GAME-LEVEL NRFI / YRFI VALIDATION",
    "=" * 86,
    "",
    f"Historical games: {len(outcomes)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample validation.",
    "Top and Bottom scoring probabilities are fit separately.",
    (
        "Game NRFI probability = "
        "(1 - Top scoring probability) × "
        "(1 - Bottom scoring probability)."
    ),
    (
        "Positive Brier/log-loss improvement means "
        "better full-game NRFI probability forecasts."
    ),
    "",
]


ranking_rows = []


for model_name in models:
    subset = results[
        results["model"] == model_name
    ].copy()

    wins_base_brier = int(
        (
            subset[
                "improvement_vs_baseline_brier"
            ] > 0
        ).sum()
    )

    wins_core_brier = int(
        (
            subset[
                "improvement_vs_core_brier"
            ] > 0
        ).sum()
    )

    wins_base_log = int(
        (
            subset[
                "improvement_vs_baseline_logloss"
            ] > 0
        ).sum()
    )

    avg_base_brier = float(
        subset[
            "improvement_vs_baseline_brier"
        ].mean()
    )

    avg_core_brier = float(
        subset[
            "improvement_vs_core_brier"
        ].mean()
    )

    avg_base_log = float(
        subset[
            "improvement_vs_baseline_logloss"
        ].mean()
    )

    worst_base_brier = float(
        subset[
            "improvement_vs_baseline_brier"
        ].min()
    )

    lines.extend([
        model_name,
        "-" * 86,
        (
            f"Beat NRFI baseline by Brier: "
            f"{wins_base_brier}/{len(subset)}"
        ),
        (
            f"Beat current core by Brier: "
            f"{wins_core_brier}/{len(subset)}"
        ),
        (
            f"Beat NRFI baseline by Log Loss: "
            f"{wins_base_log}/{len(subset)}"
        ),
        (
            f"Average Brier improvement vs baseline: "
            f"{avg_base_brier:+.6f}"
        ),
        (
            f"Average Brier improvement vs core: "
            f"{avg_core_brier:+.6f}"
        ),
        (
            f"Average Log Loss improvement vs baseline: "
            f"{avg_base_log:+.6f}"
        ),
        (
            f"Worst Brier holdout vs baseline: "
            f"{worst_base_brier:+.6f}"
        ),
        "",
    ])

    for _, row in subset.iterrows():
        lines.append(
            f"{int(row['holdout_year'])}: "
            f"Brier baseline {row['baseline_brier']:.6f} | "
            f"model {row['model_brier']:.6f} | "
            f"vs base "
            f"{row['improvement_vs_baseline_brier']:+.6f} | "
            f"vs core "
            f"{row['improvement_vs_core_brier']:+.6f} | "
            f"LogLoss "
            f"{row['model_logloss']:.6f}"
        )

    lines.extend([
        "",
        "=" * 86,
        "",
    ])

    ranking_rows.append({
        "model":
            model_name,
        "wins_vs_baseline_brier":
            wins_base_brier,
        "wins_vs_core_brier":
            wins_core_brier,
        "avg_vs_baseline_brier":
            avg_base_brier,
        "avg_vs_core_brier":
            avg_core_brier,
        "avg_vs_baseline_logloss":
            avg_base_log,
        "worst_vs_baseline_brier":
            worst_base_brier,
    })


ranking = (
    pd.DataFrame(
        ranking_rows
    )
    .sort_values(
        [
            "wins_vs_baseline_brier",
            "avg_vs_baseline_brier",
            "worst_vs_baseline_brier",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)


lines.extend([
    "FINAL GAME-LEVEL RANKING",
    "-" * 86,
])

for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    lines.append(
        f"#{rank} {row['model']} | "
        f"Brier baseline wins "
        f"{int(row['wins_vs_baseline_brier'])}/4 | "
        f"core wins "
        f"{int(row['wins_vs_core_brier'])}/4 | "
        f"avg base "
        f"{row['avg_vs_baseline_brier']:+.6f} | "
        f"avg core "
        f"{row['avg_vs_core_brier']:+.6f} | "
        f"worst "
        f"{row['worst_vs_baseline_brier']:+.6f}"
    )


lines.extend(
    calibration_lines
)


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print(
    "GAME-LEVEL NRFI VALIDATION COMPLETE"
)
print()
print(
    f"Created: {SUMMARY_FILE}"
)
print(
    f"Created: {RESULTS_FILE}"
)
print(
    f"Created: {PREDICTIONS_FILE}"
)
