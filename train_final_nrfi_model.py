
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7H3
#
# TRAIN FINAL LIVE NRFI/YRFI MODEL
#
# Champion model selected from walk-forward testing:
#
# Pitcher:
#   xwOBA allowed
#   K%
#   prior PA / sample controls
#
# Top-4 offense:
#   xwOBA
#   K%
#   BB%
#   Barrel%
#   sample controls
#
# Environment:
#   prior-completed-season 3-year Run Factor
#
# IMPORTANT:
# Logistic recalibration is NOT applied because it worsened
# future Brier / Log Loss in walk-forward testing.
#
# Fits separate Top-1st and Bottom-1st scoring models using
# all validated historical data currently available.
# =========================================================


OUTCOMES_FILE = Path("historical_first_innings.csv")
PITCHER_FILE = Path("historical_pitcher_skill_context.csv")
OFFENSE_FILE = Path("historical_top4_skill_context.csv")
PARK_FILE = Path("historical_park_context.csv")

MODEL_FILE = Path("sharpreport_nrfi_model_v1.json")
SUMMARY_FILE = Path("sharpreport_nrfi_model_v1_summary.txt")

RIDGE_LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-8


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


def fit_standardization(df, features):
    means = {}
    stds = {}

    X = df[features].copy()

    for col in features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        mean_value = (
            float(X[col].mean())
            if X[col].notna().any()
            else 0.0
        )

        X[col] = X[col].fillna(mean_value)

        std_value = float(
            X[col].std(ddof=0)
        )

        if (
            not np.isfinite(std_value)
            or std_value < 1e-12
        ):
            std_value = 1.0

        X[col] = (
            X[col] - mean_value
        ) / std_value

        means[col] = mean_value
        stds[col] = std_value

    return (
        X.to_numpy(dtype=float),
        means,
        stds,
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

date_col = find_column(
    outcomes,
    ["date", "game_date"],
    "date",
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
    "source_year",
]


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

date_series = pd.to_datetime(
    merged[date_col],
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


park_runs = pd.to_numeric(
    merged["park_runs"],
    errors="coerce",
)

park_missing = (
    park_runs.isna()
).astype(float)

park_runs = (
    park_runs
    .fillna(100.0)
)


top = pd.DataFrame({
    "game_pk": merged[game_pk_out],
    "season": season_series,
    "date": date_series,
    "half": "Top",
    "scored": scored_values("top"),

    "p_xwoba": pd.to_numeric(
        merged["home_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "p_k_pct": pd.to_numeric(
        merged["home_sp_k_pct"],
        errors="coerce",
    ),
    "p_pa": pd.to_numeric(
        merged["home_sp_pa_before_game"],
        errors="coerce",
    ),

    "o_xwoba": pd.to_numeric(
        merged["away_top4_xwoba"],
        errors="coerce",
    ),
    "o_k_pct": pd.to_numeric(
        merged["away_top4_k_pct"],
        errors="coerce",
    ),
    "o_bb_pct": pd.to_numeric(
        merged["away_top4_bb_pct"],
        errors="coerce",
    ),
    "o_barrel_pct": pd.to_numeric(
        merged["away_top4_barrel_pct"],
        errors="coerce",
    ),
    "o_combined_pa": pd.to_numeric(
        merged["away_top4_combined_pa"],
        errors="coerce",
    ),
    "o_min_pa": pd.to_numeric(
        merged["away_top4_min_pa"],
        errors="coerce",
    ),
    "o_complete_core": pd.to_numeric(
        merged["away_top4_complete_core"],
        errors="coerce",
    ),

    "park_runs": park_runs,
    "park_missing": park_missing,
})


bottom = pd.DataFrame({
    "game_pk": merged[game_pk_out],
    "season": season_series,
    "date": date_series,
    "half": "Bottom",
    "scored": scored_values("bottom"),

    "p_xwoba": pd.to_numeric(
        merged["away_sp_xwoba_allowed"],
        errors="coerce",
    ),
    "p_k_pct": pd.to_numeric(
        merged["away_sp_k_pct"],
        errors="coerce",
    ),
    "p_pa": pd.to_numeric(
        merged["away_sp_pa_before_game"],
        errors="coerce",
    ),

    "o_xwoba": pd.to_numeric(
        merged["home_top4_xwoba"],
        errors="coerce",
    ),
    "o_k_pct": pd.to_numeric(
        merged["home_top4_k_pct"],
        errors="coerce",
    ),
    "o_bb_pct": pd.to_numeric(
        merged["home_top4_bb_pct"],
        errors="coerce",
    ),
    "o_barrel_pct": pd.to_numeric(
        merged["home_top4_barrel_pct"],
        errors="coerce",
    ),
    "o_combined_pa": pd.to_numeric(
        merged["home_top4_combined_pa"],
        errors="coerce",
    ),
    "o_min_pa": pd.to_numeric(
        merged["home_top4_min_pa"],
        errors="coerce",
    ),
    "o_complete_core": pd.to_numeric(
        merged["home_top4_complete_core"],
        errors="coerce",
    ),

    "park_runs": park_runs,
    "park_missing": park_missing,
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


# =========================================================
# TRAIN TOP / BOTTOM FINAL MODELS
# =========================================================

model_payload = {
    "model_name":
        "SharpReport NRFI/YRFI Model v1",
    "version":
        "1.0",
    "probability_type":
        "RAW_LOGISTIC_NO_POST_CALIBRATION",
    "training_games":
        int(len(outcomes)),
    "training_half_innings":
        int(len(data)),
    "training_start_date":
        str(
            date_series.min().date()
        ),
    "training_end_date":
        str(
            date_series.max().date()
        ),
    "features":
        FEATURES,
    "ridge_lambda":
        RIDGE_LAMBDA,
    "park_rule":
        (
            "Use prior completed season's "
            "3-year rolling Run Factor for live model input."
        ),
    "game_probability_formula":
        (
            "NRFI=(1-P_TOP_SCORES)*(1-P_BOTTOM_SCORES); "
            "YRFI=1-NRFI"
        ),
    "models": {},
}


summary_lines = [
    "SHARPREPORT NRFI/YRFI MODEL v1",
    "=" * 78,
    "",
    f"Training games: {len(outcomes)}",
    f"Training half-innings: {len(data)}",
    (
        f"Training dates: "
        f"{date_series.min().date()} "
        f"through {date_series.max().date()}"
    ),
    "",
    "CHAMPION FEATURE SET:",
    "Pitcher xwOBA allowed",
    "Pitcher K%",
    "Pitcher prior PA / sample controls",
    "Top-4 xwOBA",
    "Top-4 K%",
    "Top-4 BB%",
    "Top-4 Barrel%",
    "Top-4 sample controls",
    "Prior-completed-season 3-year Run Factor",
    "",
    "POST-CALIBRATION:",
    "None. Raw logistic probabilities retained.",
    (
        "Reason: walk-forward logistic recalibration "
        "reduced future Brier and Log Loss performance."
    ),
    "",
]


for half in ["Top", "Bottom"]:
    half_data = data[
        data["half"] == half
    ].copy()

    X, means, stds = fit_standardization(
        half_data,
        FEATURES,
    )

    y = half_data[
        "scored"
    ].to_numpy(dtype=float)

    beta = fit_ridge_logistic(
        X,
        y,
        lam=RIDGE_LAMBDA,
    )

    feature_coefficients = {
        feature:
            float(
                beta[index + 1]
            )
        for index, feature in enumerate(
            FEATURES
        )
    }

    model_payload[
        "models"
    ][half] = {
        "rows":
            int(len(half_data)),
        "base_scoring_rate":
            float(y.mean()),
        "intercept":
            float(beta[0]),
        "coefficients":
            feature_coefficients,
        "means":
            {
                key: float(value)
                for key, value in means.items()
            },
        "stds":
            {
                key: float(value)
                for key, value in stds.items()
            },
    }

    summary_lines.extend([
        f"{half.upper()} 1ST MODEL",
        "-" * 78,
        f"Rows: {len(half_data)}",
        (
            f"Historical scoring rate: "
            f"{y.mean() * 100:.2f}%"
        ),
        f"Intercept: {beta[0]:+.6f}",
        "",
        "Standardized coefficients:",
    ])

    for feature in FEATURES:
        summary_lines.append(
            f"{feature}: "
            f"{feature_coefficients[feature]:+.6f}"
        )

    summary_lines.extend([
        "",
        "=" * 78,
        "",
    ])


MODEL_FILE.write_text(
    json.dumps(
        model_payload,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

SUMMARY_FILE.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


print()
print(
    "FINAL SHARPREPORT NRFI MODEL TRAINED"
)
print()
print(
    f"Created: {MODEL_FILE}"
)
print(
    f"Created: {SUMMARY_FILE}"
)
