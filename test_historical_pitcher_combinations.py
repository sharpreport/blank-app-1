
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# SHARPREPORT STAGE 7D4
#
# Nested pitcher model test:
# Which combination of validated pitcher skill + validated
# first-inning history produces the most stable future
# first-inning probability forecasts?
#
# Walk-forward:
#   2023 <- train 2022
#   2024 <- train 2022-2023
#   2025 <- train 2022-2024
#   2026 <- train 2022-2025
#
# Top and Bottom 1st are modeled separately.
# No same-day leakage in first-inning history.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
STARTERS_FILE = Path("historical_actual_starters.csv")
SKILL_FILE = Path("historical_pitcher_skill_context.csv")

SUMMARY_FILE = Path(
    "historical_pitcher_combination_summary.txt"
)

RESULTS_FILE = Path(
    "historical_pitcher_combination_results.csv"
)

COEFFICIENTS_FILE = Path(
    "historical_pitcher_combination_coefficients.csv"
)

HOLDOUT_YEARS = [2023, 2024, 2025, 2026]

RIDGE_LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-8


# =========================================================
# HELPERS
# =========================================================

def find_column(df, candidates, label, required=True):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

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

        train[col] = train[col].fillna(
            mean_value
        )
        test[col] = test[col].fillna(
            mean_value
        )

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
    return sigmoid(Xd @ beta)


# =========================================================
# LOAD
# =========================================================

for path in [
    OUTCOMES_FILE,
    STARTERS_FILE,
    SKILL_FILE,
]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} was not found."
        )

outcomes = pd.read_csv(OUTCOMES_FILE)
starters = pd.read_csv(STARTERS_FILE)
skill = pd.read_csv(SKILL_FILE)


# =========================================================
# RESOLVE OUTCOME COLUMNS
# =========================================================

game_pk_out = find_column(
    outcomes,
    ["game_pk", "gamepk", "game_id"],
    "game ID in outcomes",
)

season_out = find_column(
    outcomes,
    ["season", "year"],
    "season in outcomes",
)

date_out = find_column(
    outcomes,
    ["date", "game_date"],
    "date in outcomes",
)

away_team_out = find_column(
    outcomes,
    ["away_team", "away_name"],
    "away team",
    required=False,
)

home_team_out = find_column(
    outcomes,
    ["home_team", "home_name"],
    "home team",
    required=False,
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
    raise SystemExit(
        "ERROR: Could not resolve Top 1st outcome."
    )

if bottom_scored_col is None and home_runs_col is None:
    raise SystemExit(
        "ERROR: Could not resolve Bottom 1st outcome."
    )


# =========================================================
# RESOLVE STARTER COLUMNS
# =========================================================

game_pk_st = find_column(
    starters,
    ["game_pk", "gamepk", "game_id"],
    "game ID in starters",
)

away_sp_col = find_column(
    starters,
    ["away_sp_id", "away_starter_id"],
    "away starter ID",
)

home_sp_col = find_column(
    starters,
    ["home_sp_id", "home_starter_id"],
    "home starter ID",
)


# =========================================================
# MAKE GAME-LEVEL BASE
# =========================================================

base = outcomes.merge(
    starters[
        [
            game_pk_st,
            away_sp_col,
            home_sp_col,
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
    base[season_out],
    errors="raise",
).astype(int)

base["date_model"] = pd.to_datetime(
    base[date_out],
    errors="raise",
).dt.date

base["away_sp_model"] = pd.to_numeric(
    base[away_sp_col],
    errors="raise",
).astype(int)

base["home_sp_model"] = pd.to_numeric(
    base[home_sp_col],
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


# We need runs, not only binary scored flags, to recreate
# pitcher first-inning runs/start. If explicit run columns
# are unavailable, binary scored is a weaker fallback.
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
# REBUILD NO-LOOKAHEAD PITCHER 1ST-INNING HISTORY
# =========================================================

history_rows = []

for season in sorted(
    base["season_model"].unique()
):
    season_games = (
        base[
            base["season_model"] == season
        ]
        .copy()
        .sort_values(
            [
                "date_model",
                game_pk_out,
            ]
        )
    )

    pitcher_history = {}

    def get_hist(pid):
        if pid not in pitcher_history:
            pitcher_history[pid] = {
                "starts": 0,
                "runs": 0.0,
            }
        return pitcher_history[pid]

    for day, day_games in season_games.groupby(
        "date_model",
        sort=True,
    ):
        # Snapshot FIRST for all games that day.
        for _, game in day_games.iterrows():
            away_pid = int(
                game["away_sp_model"]
            )
            home_pid = int(
                game["home_sp_model"]
            )

            away_h = get_hist(away_pid)
            home_h = get_hist(home_pid)

            history_rows.append({
                "game_pk_model": int(
                    game[game_pk_out]
                ),
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

        # Update only AFTER all snapshots for that date.
        for _, game in day_games.iterrows():
            away_pid = int(
                game["away_sp_model"]
            )
            home_pid = int(
                game["home_sp_model"]
            )

            away_h = get_hist(away_pid)
            home_h = get_hist(home_pid)

            # Away starter allows Bottom-1st home runs.
            away_h["starts"] += 1
            away_h["runs"] += float(
                game["bottom_runs_model"]
            )

            # Home starter allows Top-1st away runs.
            home_h["starts"] += 1
            home_h["runs"] += float(
                game["top_runs_model"]
            )

history_df = pd.DataFrame(
    history_rows
)

if len(history_df) != len(base):
    raise SystemExit(
        "ERROR: History rebuild row count mismatch."
    )

base = base.merge(
    history_df,
    left_on=game_pk_out,
    right_on="game_pk_model",
    how="left",
)


# =========================================================
# MERGE VALIDATED PITCHER SKILL
# =========================================================

game_pk_skill = find_column(
    skill,
    ["game_pk", "gamepk", "game_id"],
    "game ID in pitcher skill",
)

skill_cols = [
    game_pk_skill,
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

for col in skill_cols:
    if col not in skill.columns:
        raise SystemExit(
            f"ERROR: Missing pitcher skill column: {col}"
        )

base = base.merge(
    skill[skill_cols],
    left_on=game_pk_out,
    right_on=game_pk_skill,
    how="inner",
    suffixes=("", "_skill"),
)

if len(base) != len(outcomes):
    raise SystemExit(
        "ERROR: Skill merge did not preserve all historical games."
    )


# =========================================================
# BUILD HALF-INNING ROWS
# =========================================================

top = pd.DataFrame({
    "game_pk": base[game_pk_out],
    "season": base["season_model"],
    "half": "Top",
    "scored": base["top_scored_model"],

    # Top 1st faces HOME starter.
    "pa": pd.to_numeric(
        base["home_sp_pa_before_game"],
        errors="coerce",
    ),
    "xwoba": pd.to_numeric(
        base["home_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "k_pct": pd.to_numeric(
        base["home_sp_k_pct"],
        errors="coerce",
    ),
    "bb_pct": pd.to_numeric(
        base["home_sp_bb_pct"],
        errors="coerce",
    ),
    "barrel_pct": pd.to_numeric(
        base["home_sp_barrel_pct"],
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

    # Bottom 1st faces AWAY starter.
    "pa": pd.to_numeric(
        base["away_sp_pa_before_game"],
        errors="coerce",
    ),
    "xwoba": pd.to_numeric(
        base["away_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "k_pct": pd.to_numeric(
        base["away_sp_k_pct"],
        errors="coerce",
    ),
    "bb_pct": pd.to_numeric(
        base["away_sp_bb_pct"],
        errors="coerce",
    ),
    "barrel_pct": pd.to_numeric(
        base["away_sp_barrel_pct"],
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

data["pa"] = data["pa"].fillna(0.0)
data["hist_starts"] = data[
    "hist_starts"
].fillna(0.0)

data["log_pa"] = np.log1p(
    data["pa"].clip(lower=0)
)

data["no_prior_pa"] = (
    data["pa"] <= 0
).astype(float)

data["log_hist_starts"] = np.log1p(
    data["hist_starts"].clip(lower=0)
)

data["no_hist_starts"] = (
    data["hist_starts"] <= 0
).astype(float)


# =========================================================
# NESTED MODELS
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

    "xwOBA + K + PA": [
        "xwoba",
        "k_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "xwOBA + K + Barrel + PA": [
        "xwoba",
        "k_pct",
        "barrel_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "All 4 skill + PA": [
        "xwoba",
        "k_pct",
        "bb_pct",
        "barrel_pct",
        "log_pa",
        "no_prior_pa",
    ],

    "History + PA": [
        "hist_runs_per_start",
        "log_hist_starts",
        "no_hist_starts",
        "log_pa",
        "no_prior_pa",
    ],

    "xwOBA + K + History + PA": [
        "xwoba",
        "k_pct",
        "hist_runs_per_start",
        "log_hist_starts",
        "no_hist_starts",
        "log_pa",
        "no_prior_pa",
    ],

    "xwOBA + K + Barrel + History + PA": [
        "xwoba",
        "k_pct",
        "barrel_pct",
        "hist_runs_per_start",
        "log_hist_starts",
        "no_hist_starts",
        "log_pa",
        "no_prior_pa",
    ],

    "All 4 skill + History + PA": [
        "xwoba",
        "k_pct",
        "bb_pct",
        "barrel_pct",
        "hist_runs_per_start",
        "log_hist_starts",
        "no_hist_starts",
        "log_pa",
        "no_prior_pa",
    ],
}


# =========================================================
# WALK-FORWARD
# =========================================================

results = []
coefficients = []

for holdout_year in HOLDOUT_YEARS:
    train = data[
        data["season"] < holdout_year
    ].copy()

    test = data[
        data["season"] == holdout_year
    ].copy().reset_index(drop=True)

    if train.empty or test.empty:
        continue

    baseline_pred = np.full(
        len(test),
        np.nan,
    )

    model_preds = {
        name: np.full(
            len(test),
            np.nan,
        )
        for name in models
    }

    print()
    print(
        f"Testing holdout {holdout_year}"
    )

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

        baseline_pred[
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

            model_preds[
                model_name
            ][
                mask.to_numpy()
            ] = pred

            for i, feature in enumerate(features):
                coefficients.append({
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
        baseline_pred,
    )

    pa_brier = None

    year_rows = []

    for model_name in models:
        score = brier_score(
            y_test,
            model_preds[model_name],
        )

        if model_name == "PA only control":
            pa_brier = score

        row = {
            "holdout_year": holdout_year,
            "model": model_name,
            "rows": len(test),
            "baseline_brier":
                baseline_brier,
            "model_brier":
                score,
            "improvement_vs_baseline":
                baseline_brier - score,
        }

        year_rows.append(row)

    for row in year_rows:
        row["improvement_vs_pa"] = (
            pa_brier
            - row["model_brier"]
        )

        results.append(row)


results_df = pd.DataFrame(
    results
)

coeff_df = pd.DataFrame(
    coefficients
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

summary = [
    "SHARPREPORT HISTORICAL PITCHER COMBINATION TEST",
    "=" * 82,
    "",
    f"Historical games: {len(outcomes)}",
    f"Half-inning rows: {len(data)}",
    "",
    "METHOD:",
    "Walk-forward out-of-sample testing.",
    "Top and Bottom 1st modeled separately.",
    "First-inning pitcher history rebuilt with no same-day leakage.",
    "Positive Brier improvement = better probability forecast.",
    "",
]


rank_rows = []

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

    wins_pa = int(
        (
            subset[
                "improvement_vs_pa"
            ] > 0
        ).sum()
    )

    avg_base = float(
        subset[
            "improvement_vs_baseline"
        ].mean()
    )

    avg_pa = float(
        subset[
            "improvement_vs_pa"
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

    summary.extend([
        model_name,
        "-" * 82,
        (
            f"Beat baseline: "
            f"{wins_baseline}/{len(subset)}"
        ),
        (
            f"Beat PA-only control: "
            f"{wins_pa}/{len(subset)}"
        ),
        (
            f"Average improvement vs baseline: "
            f"{avg_base:+.6f}"
        ),
        (
            f"Average improvement vs PA-only: "
            f"{avg_pa:+.6f}"
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
        summary.append(
            f"{int(row['holdout_year'])}: "
            f"baseline {row['baseline_brier']:.6f} | "
            f"model {row['model_brier']:.6f} | "
            f"vs baseline "
            f"{row['improvement_vs_baseline']:+.6f} | "
            f"vs PA "
            f"{row['improvement_vs_pa']:+.6f}"
        )

    summary.extend([
        "",
        "=" * 82,
        "",
    ])

    if model_name != "PA only control":
        rank_rows.append({
            "model": model_name,
            "wins_vs_baseline":
                wins_baseline,
            "wins_vs_pa":
                wins_pa,
            "avg_vs_baseline":
                avg_base,
            "avg_vs_pa":
                avg_pa,
            "worst_vs_baseline":
                worst,
        })


ranking = pd.DataFrame(
    rank_rows
).sort_values(
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


summary.extend([
    "FINAL RANKING",
    "-" * 82,
])

for rank, (_, row) in enumerate(
    ranking.iterrows(),
    start=1,
):
    summary.append(
        f"#{rank} {row['model']} | "
        f"baseline wins "
        f"{int(row['wins_vs_baseline'])}/4 | "
        f"PA wins "
        f"{int(row['wins_vs_pa'])}/4 | "
        f"avg baseline "
        f"{row['avg_vs_baseline']:+.6f} | "
        f"avg PA "
        f"{row['avg_vs_pa']:+.6f} | "
        f"worst "
        f"{row['worst_vs_baseline']:+.6f}"
    )


SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


print()
print(
    "PITCHER COMBINATION TEST COMPLETE"
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
