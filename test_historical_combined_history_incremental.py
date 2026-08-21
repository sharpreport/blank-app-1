
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7F2
#
# Incremental test:
# Does pitcher first-inning history add value AFTER the
# validated pitcher + Top-4 offense skill core?
#
# Champion core:
#   Pitcher xwOBA allowed
#   Pitcher K%
#   Top-4 xwOBA
#   Top-4 K%
#   Top-4 Barrel%
#   sample-size controls
#
# Challengers:
#   + offense BB%
#   + pitcher YTD 1st-inning runs/start + starts
#   + both
#
# Walk-forward:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom 1st modeled separately.
# Same-day history leakage is prevented.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
STARTERS_FILE = Path("historical_actual_starters.csv")
PITCHER_FILE = Path("historical_pitcher_skill_context.csv")
OFFENSE_FILE = Path("historical_top4_skill_context.csv")

SUMMARY_FILE = Path(
    "historical_combined_history_incremental_summary.txt"
)

RESULTS_FILE = Path(
    "historical_combined_history_incremental_results.csv"
)

COEFFICIENTS_FILE = Path(
    "historical_combined_history_incremental_coefficients.csv"
)

HOLDOUT_YEARS = [2023, 2024, 2025, 2026]

RIDGE_LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-8


def find_column(df, candidates, label, required=True):
    lower_map = {str(c).lower(): c for c in df.columns}

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
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

        mean_value = (
            float(train[col].mean())
            if train[col].notna().any()
            else 0.0
        )

        train[col] = train[col].fillna(mean_value)
        test[col] = test[col].fillna(mean_value)

        std_value = float(train[col].std(ddof=0))

        if not np.isfinite(std_value) or std_value < 1e-12:
            std_value = 1.0

        train[col] = (train[col] - mean_value) / std_value
        test[col] = (test[col] - mean_value) / std_value

    return train.to_numpy(dtype=float), test.to_numpy(dtype=float)


def fit_ridge_logistic(X, y, lam=1.0):
    y = np.asarray(y, dtype=float)

    Xd = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    beta = np.zeros(Xd.shape[1], dtype=float)

    penalty = np.eye(Xd.shape[1], dtype=float)
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
            Xd.T @ (w[:, None] * Xd)
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

        if np.max(np.abs(beta_new - beta)) < TOL:
            beta = beta_new
            break

        beta = beta_new

    return beta


def predict_logistic(beta, X):
    Xd = np.column_stack([
        np.ones(len(X)),
        X,
    ])
    return sigmoid(Xd @ beta)


# =========================================================
# LOAD
# =========================================================

for path in [
    OUTCOMES_FILE,
    STARTERS_FILE,
    PITCHER_FILE,
    OFFENSE_FILE,
]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} was not found."
        )

outcomes = pd.read_csv(OUTCOMES_FILE)
starters = pd.read_csv(STARTERS_FILE)
pitcher = pd.read_csv(PITCHER_FILE)
offense = pd.read_csv(OFFENSE_FILE)


game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in outcomes",
)

season_col = find_column(
    outcomes,
    ["season", "year"],
    "season in outcomes",
)

date_col = find_column(
    outcomes,
    ["date", "game_date"],
    "date in outcomes",
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
    raise SystemExit("ERROR: Could not resolve Top 1st outcome.")

if bottom_scored_col is None and home_runs_col is None:
    raise SystemExit("ERROR: Could not resolve Bottom 1st outcome.")


# =========================================================
# MERGE VERIFIED ACTUAL STARTERS
# =========================================================

game_pk_st = find_column(
    starters,
    ["game_pk", "gamepk", "game_id"],
    "game ID in starters",
)

away_sp_id_col = find_column(
    starters,
    ["away_sp_id", "away_starter_id"],
    "away starter ID",
)

home_sp_id_col = find_column(
    starters,
    ["home_sp_id", "home_starter_id"],
    "home starter ID",
)

base = outcomes.merge(
    starters[
        [
            game_pk_st,
            away_sp_id_col,
            home_sp_id_col,
        ]
    ],
    left_on=game_pk_out,
    right_on=game_pk_st,
    how="inner",
)

if len(base) != len(outcomes):
    raise SystemExit(
        "ERROR: Outcomes/starter merge did not preserve all games."
    )

base["season_model"] = pd.to_numeric(
    base[season_col],
    errors="raise",
).astype(int)

base["date_model"] = pd.to_datetime(
    base[date_col],
    errors="raise",
).dt.date

base["away_sp_model"] = pd.to_numeric(
    base[away_sp_id_col],
    errors="raise",
).astype(int)

base["home_sp_model"] = pd.to_numeric(
    base[home_sp_id_col],
    errors="raise",
).astype(int)

if top_scored_col is not None:
    base["top_scored_model"] = pd.to_numeric(
        base[top_scored_col],
        errors="raise",
    ).astype(int)
else:
    base["top_scored_model"] = (
        pd.to_numeric(
            base[away_runs_col],
            errors="raise",
        ) > 0
    ).astype(int)

if bottom_scored_col is not None:
    base["bottom_scored_model"] = pd.to_numeric(
        base[bottom_scored_col],
        errors="raise",
    ).astype(int)
else:
    base["bottom_scored_model"] = (
        pd.to_numeric(
            base[home_runs_col],
            errors="raise",
        ) > 0
    ).astype(int)

if away_runs_col is not None:
    base["top_runs_model"] = pd.to_numeric(
        base[away_runs_col],
        errors="coerce",
    ).fillna(0.0)
else:
    base["top_runs_model"] = base[
        "top_scored_model"
    ].astype(float)

if home_runs_col is not None:
    base["bottom_runs_model"] = pd.to_numeric(
        base[home_runs_col],
        errors="coerce",
    ).fillna(0.0)
else:
    base["bottom_runs_model"] = base[
        "bottom_scored_model"
    ].astype(float)


# =========================================================
# REBUILD PITCHER FIRST-INNING HISTORY
# WITH NO SAME-DAY LEAKAGE
# =========================================================

history_rows = []

for season in sorted(base["season_model"].unique()):
    season_games = (
        base[
            base["season_model"] == season
        ]
        .copy()
        .sort_values(
            ["date_model", game_pk_out]
        )
    )

    hist = {}

    def get_hist(pid):
        if pid not in hist:
            hist[pid] = {
                "starts": 0,
                "runs": 0.0,
            }
        return hist[pid]

    for day, day_games in season_games.groupby(
        "date_model",
        sort=True,
    ):
        # Snapshot all games on date first.
        for _, game in day_games.iterrows():
            away_pid = int(game["away_sp_model"])
            home_pid = int(game["home_sp_model"])

            away_h = get_hist(away_pid)
            home_h = get_hist(home_pid)

            history_rows.append({
                "game_pk_history":
                    int(game[game_pk_out]),

                "away_hist_starts":
                    away_h["starts"],

                "away_hist_runs_per_start":
                    (
                        away_h["runs"]
                        / away_h["starts"]
                        if away_h["starts"] > 0
                        else np.nan
                    ),

                "home_hist_starts":
                    home_h["starts"],

                "home_hist_runs_per_start":
                    (
                        home_h["runs"]
                        / home_h["starts"]
                        if home_h["starts"] > 0
                        else np.nan
                    ),
            })

        # Update after all snapshots for that date.
        for _, game in day_games.iterrows():
            away_pid = int(game["away_sp_model"])
            home_pid = int(game["home_sp_model"])

            away_h = get_hist(away_pid)
            home_h = get_hist(home_pid)

            away_h["starts"] += 1
            away_h["runs"] += float(
                game["bottom_runs_model"]
            )

            home_h["starts"] += 1
            home_h["runs"] += float(
                game["top_runs_model"]
            )


history_df = pd.DataFrame(history_rows)

base = base.merge(
    history_df,
    left_on=game_pk_out,
    right_on="game_pk_history",
    how="left",
)

if len(base) != len(outcomes):
    raise SystemExit(
        "ERROR: History merge did not preserve all games."
    )


# =========================================================
# MERGE PITCHER / OFFENSE SKILL
# =========================================================

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

base = base.merge(
    pitcher[pitcher_keep],
    left_on=game_pk_out,
    right_on=game_pk_pitcher,
    how="inner",
)

base = base.merge(
    offense[offense_keep],
    left_on=game_pk_out,
    right_on=game_pk_offense,
    how="inner",
)

if len(base) != len(outcomes):
    raise SystemExit(
        "ERROR: Skill merge did not preserve all games."
    )


# =========================================================
# HALF-INNING DATA
# =========================================================

top = pd.DataFrame({
    "game_pk": base[game_pk_out],
    "season": base["season_model"],
    "half": "Top",
    "scored": base["top_scored_model"],

    "p_xwoba": pd.to_numeric(
        base["home_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "p_k_pct": pd.to_numeric(
        base["home_sp_k_pct"],
        errors="coerce",
    ),
    "p_pa": pd.to_numeric(
        base["home_sp_pa_before_game"],
        errors="coerce",
    ),

    "o_xwoba": pd.to_numeric(
        base["away_top4_xwoba"],
        errors="coerce",
    ),
    "o_k_pct": pd.to_numeric(
        base["away_top4_k_pct"],
        errors="coerce",
    ),
    "o_bb_pct": pd.to_numeric(
        base["away_top4_bb_pct"],
        errors="coerce",
    ),
    "o_barrel_pct": pd.to_numeric(
        base["away_top4_barrel_pct"],
        errors="coerce",
    ),
    "o_combined_pa": pd.to_numeric(
        base["away_top4_combined_pa"],
        errors="coerce",
    ),
    "o_min_pa": pd.to_numeric(
        base["away_top4_min_pa"],
        errors="coerce",
    ),
    "o_complete_core": pd.to_numeric(
        base["away_top4_complete_core"],
        errors="coerce",
    ),

    "hist_starts": pd.to_numeric(
        base["home_hist_starts"],
        errors="coerce",
    ),
    "hist_runs_per_start": pd.to_numeric(
        base["home_hist_runs_per_start"],
        errors="coerce",
    ),
})


bottom = pd.DataFrame({
    "game_pk": base[game_pk_out],
    "season": base["season_model"],
    "half": "Bottom",
    "scored": base["bottom_scored_model"],

    "p_xwoba": pd.to_numeric(
        base["away_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "p_k_pct": pd.to_numeric(
        base["away_sp_k_pct"],
        errors="coerce",
    ),
    "p_pa": pd.to_numeric(
        base["away_sp_pa_before_game"],
        errors="coerce",
    ),

    "o_xwoba": pd.to_numeric(
        base["home_top4_xwoba"],
        errors="coerce",
    ),
    "o_k_pct": pd.to_numeric(
        base["home_top4_k_pct"],
        errors="coerce",
    ),
    "o_bb_pct": pd.to_numeric(
        base["home_top4_bb_pct"],
        errors="coerce",
    ),
    "o_barrel_pct": pd.to_numeric(
        base["home_top4_barrel_pct"],
        errors="coerce",
    ),
    "o_combined_pa": pd.to_numeric(
        base["home_top4_combined_pa"],
        errors="coerce",
    ),
    "o_min_pa": pd.to_numeric(
        base["home_top4_min_pa"],
        errors="coerce",
    ),
    "o_complete_core": pd.to_numeric(
        base["home_top4_complete_core"],
        errors="coerce",
    ),

    "hist_starts": pd.to_numeric(
        base["away_hist_starts"],
        errors="coerce",
    ),
    "hist_runs_per_start": pd.to_numeric(
        base["away_hist_runs_per_start"],
        errors="coerce",
    ),
})


data = pd.concat(
    [top, bottom],
    ignore_index=True,
)

for col in [
    "p_pa",
    "o_combined_pa",
    "o_min_pa",
    "hist_starts",
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

data["hist_log_starts"] = np.log1p(
    data["hist_starts"]
)

data["hist_no_starts"] = (
    data["hist_starts"] <= 0
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

history_features = [
    "hist_runs_per_start",
    "hist_log_starts",
    "hist_no_starts",
]


models = {
    "Combined core":
        core_features,

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

    "Core + pitcher 1st-inning history":
        [
            *core_features,
            *history_features,
        ],

    "Core + offense BB + pitcher history":
        [
            "p_xwoba",
            "p_k_pct",
            "o_xwoba",
            "o_k_pct",
            "o_bb_pct",
            "o_barrel_pct",
            *sample_controls,
            *history_features,
        ],
}


# =========================================================
# WALK-FORWARD
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

    print()
    print(
        f"Testing {holdout_year} "
        f"using seasons before {holdout_year}"
    )

    baseline_predictions = np.full(
        len(test),
        np.nan,
    )

    model_predictions = {
        name: np.full(
            len(test),
            np.nan,
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
                    "holdout_year": holdout_year,
                    "half": half,
                    "model": model_name,
                    "feature": feature,
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

    core_brier = None
    year_rows = []

    for model_name in models:
        model_brier = brier_score(
            y_test,
            model_predictions[model_name],
        )

        if model_name == "Combined core":
            core_brier = model_brier

        year_rows.append({
            "holdout_year": holdout_year,
            "model": model_name,
            "baseline_brier":
                baseline_brier,
            "model_brier":
                model_brier,
            "improvement_vs_baseline":
                baseline_brier - model_brier,
        })

    for row in year_rows:
        row["improvement_vs_core"] = (
            core_brier
            - row["model_brier"]
        )
        results.append(row)


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
    "SHARPREPORT COMBINED MODEL — FIRST-INNING HISTORY INCREMENTAL TEST",
    "=" * 86,
    "",
    f"Historical games: {len(outcomes)}",
    f"Half-inning rows: {len(data)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample testing.",
    "Top and Bottom 1st modeled separately.",
    "Pitcher history rebuilt with no same-day leakage.",
    "Positive improvement = lower (better) Brier score.",
    "",
]


ranking_rows = []

for model_name in models:
    subset = results_df[
        results_df["model"] == model_name
    ].copy()

    wins_baseline = int(
        (
            subset[
                "improvement_vs_baseline"
            ] > 0
        ).sum()
    )

    wins_core = int(
        (
            subset[
                "improvement_vs_core"
            ] > 0
        ).sum()
    )

    avg_base = float(
        subset[
            "improvement_vs_baseline"
        ].mean()
    )

    avg_core = float(
        subset[
            "improvement_vs_core"
        ].mean()
    )

    worst_base = float(
        subset[
            "improvement_vs_baseline"
        ].min()
    )

    lines.extend([
        model_name,
        "-" * 86,
        (
            f"Beat baseline: "
            f"{wins_baseline}/{len(subset)}"
        ),
        (
            f"Beat current combined core: "
            f"{wins_core}/{len(subset)}"
        ),
        (
            f"Average improvement vs baseline: "
            f"{avg_base:+.6f}"
        ),
        (
            f"Average incremental improvement vs core: "
            f"{avg_core:+.6f}"
        ),
        (
            f"Worst holdout vs baseline: "
            f"{worst_base:+.6f}"
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
            f"vs core "
            f"{row['improvement_vs_core']:+.6f}"
        )

    lines.extend([
        "",
        "=" * 86,
        "",
    ])

    ranking_rows.append({
        "model": model_name,
        "wins_vs_baseline":
            wins_baseline,
        "wins_vs_core":
            wins_core,
        "avg_vs_baseline":
            avg_base,
        "avg_vs_core":
            avg_core,
        "worst_vs_baseline":
            worst_base,
    })


ranking = (
    pd.DataFrame(ranking_rows)
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
    "-" * 86,
])

for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    lines.append(
        f"#{rank} {row['model']} | "
        f"baseline wins "
        f"{int(row['wins_vs_baseline'])}/4 | "
        f"core wins "
        f"{int(row['wins_vs_core'])}/4 | "
        f"avg baseline "
        f"{row['avg_vs_baseline']:+.6f} | "
        f"avg core "
        f"{row['avg_vs_core']:+.6f} | "
        f"worst "
        f"{row['worst_vs_baseline']:+.6f}"
    )


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print(
    "COMBINED HISTORY INCREMENTAL TEST COMPLETE"
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
