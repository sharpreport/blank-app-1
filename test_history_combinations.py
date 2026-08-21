
import math
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("historical_half_inning_context.csv")
SUMMARY_FILE = Path("historical_history_combo_summary.txt")
RESULTS_FILE = Path("historical_history_combo_results.csv")

HOLDOUT_SEASONS = [2023, 2024, 2025, 2026]
RIDGE = 1.0


FEATURE_SETS = {
    "Pitcher scoreless only": [
        "pitcher_scoring_pct",
        "log_pitcher_starts",
    ],

    "Pitcher runs only": [
        "pregame_pitcher_1st_runs_per_start",
        "log_pitcher_starts",
    ],

    "Pitcher history combo": [
        "pitcher_scoring_pct",
        "pregame_pitcher_1st_runs_per_start",
        "log_pitcher_starts",
    ],

    "Offense history combo": [
        "pregame_offense_scoring_pct",
        "pregame_offense_runs_per_game",
        "log_offense_games",
    ],

    "Home/Away split combo": [
        "pregame_split_scoring_pct",
        "pregame_split_runs_per_game",
        "log_split_games",
    ],

    "Pitcher + offense": [
        "pitcher_scoring_pct",
        "pregame_pitcher_1st_runs_per_start",
        "log_pitcher_starts",
        "pregame_offense_scoring_pct",
        "pregame_offense_runs_per_game",
        "log_offense_games",
    ],

    "All first-inning history": [
        "pitcher_scoring_pct",
        "pregame_pitcher_1st_runs_per_start",
        "log_pitcher_starts",
        "pregame_offense_scoring_pct",
        "pregame_offense_runs_per_game",
        "log_offense_games",
        "pregame_split_scoring_pct",
        "pregame_split_runs_per_game",
        "log_split_games",
    ],
}


# =========================================================
# HELPERS
# =========================================================

def sigmoid(values):

    values = np.clip(
        values,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -values
            )
        )
    )


def brier_score(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return float(
        np.mean(
            (
                actual
                - predicted
            )
            ** 2
        )
    )


def prepare_matrix(
    train,
    test,
    features
):

    train_x = train[
        features
    ].copy()

    test_x = test[
        features
    ].copy()


    # Impute using ONLY the training data.

    medians = (
        train_x
        .median(
            numeric_only=True
        )
    )


    train_x = (
        train_x
        .fillna(
            medians
        )
        .fillna(
            0.0
        )
    )

    test_x = (
        test_x
        .fillna(
            medians
        )
        .fillna(
            0.0
        )
    )


    means = (
        train_x
        .mean()
    )

    stds = (
        train_x
        .std(
            ddof=0
        )
        .replace(
            0,
            1.0
        )
        .fillna(
            1.0
        )
    )


    train_x = (
        train_x
        - means
    ) / stds


    test_x = (
        test_x
        - means
    ) / stds


    train_matrix = np.column_stack(
        [
            np.ones(
                len(train_x)
            ),
            train_x.to_numpy(
                dtype=float
            ),
        ]
    )


    test_matrix = np.column_stack(
        [
            np.ones(
                len(test_x)
            ),
            test_x.to_numpy(
                dtype=float
            ),
        ]
    )


    return (
        train_matrix,
        test_matrix,
    )


def fit_logistic_ridge(
    x,
    y,
    ridge=RIDGE,
    max_iter=60,
):

    y = np.asarray(
        y,
        dtype=float
    )


    beta = np.zeros(
        x.shape[1],
        dtype=float
    )


    # Start the intercept near the observed base rate.

    base_rate = float(
        np.clip(
            y.mean(),
            0.001,
            0.999,
        )
    )


    beta[0] = math.log(
        base_rate
        / (
            1.0
            - base_rate
        )
    )


    penalty = np.full(
        x.shape[1],
        ridge,
        dtype=float
    )

    # Do not materially penalize the intercept.

    penalty[0] = 1e-8


    for _ in range(
        max_iter
    ):

        probabilities = sigmoid(
            x @ beta
        )


        weights = (
            probabilities
            * (
                1.0
                - probabilities
            )
        )


        weights = np.clip(
            weights,
            1e-6,
            None,
        )


        gradient = (
            x.T
            @ (
                probabilities
                - y
            )
            + penalty * beta
        )


        hessian = (
            x.T
            @ (
                weights[:, None]
                * x
            )
            + np.diag(
                penalty
            )
        )


        try:

            step = np.linalg.solve(
                hessian,
                gradient
            )

        except np.linalg.LinAlgError:

            step = np.linalg.pinv(
                hessian
            ) @ gradient


        beta = (
            beta
            - step
        )


        if np.max(
            np.abs(
                step
            )
        ) < 1e-7:

            break


    return beta


# =========================================================
# LOAD DATA
# =========================================================

if not INPUT_FILE.exists():

    raise SystemExit(
        "ERROR: historical_half_inning_context.csv "
        "was not found."
    )


data = pd.read_csv(
    INPUT_FILE
)


numeric_columns = [
    "season",
    "game_pk",
    "scored",
    "pregame_pitcher_starts",
    "pregame_pitcher_scoreless_pct",
    "pregame_pitcher_1st_runs_per_start",
    "pregame_offense_games",
    "pregame_offense_scoring_pct",
    "pregame_offense_runs_per_game",
    "pregame_split_games",
    "pregame_split_scoring_pct",
    "pregame_split_runs_per_game",
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# Convert scoreless percentage into the more intuitive
# "pitcher allowed a run in the 1st" percentage.

data[
    "pitcher_scoring_pct"
] = (
    100.0
    - data[
        "pregame_pitcher_scoreless_pct"
    ]
)


# Sample-size signals.

data[
    "log_pitcher_starts"
] = np.log1p(
    data[
        "pregame_pitcher_starts"
    ].fillna(
        0
    )
)


data[
    "log_offense_games"
] = np.log1p(
    data[
        "pregame_offense_games"
    ].fillna(
        0
    )
)


data[
    "log_split_games"
] = np.log1p(
    data[
        "pregame_split_games"
    ].fillna(
        0
    )
)


# =========================================================
# WALK-FORWARD COMBINATION TEST
# =========================================================

result_rows = []


for model_name, features in FEATURE_SETS.items():

    for holdout in HOLDOUT_SEASONS:

        train_seasons = [
            season
            for season in sorted(
                data[
                    "season"
                ]
                .dropna()
                .astype(int)
                .unique()
            )
            if season < holdout
        ]


        if not train_seasons:
            continue


        all_actual = []

        all_baseline = []

        all_model = []


        for half in [
            "Top 1st",
            "Bottom 1st",
        ]:

            train = data[
                (
                    data[
                        "half"
                    ] == half
                )
                &
                (
                    data[
                        "season"
                    ].isin(
                        train_seasons
                    )
                )
            ].copy()


            test = data[
                (
                    data[
                        "half"
                    ] == half
                )
                &
                (
                    data[
                        "season"
                    ] == holdout
                )
            ].copy()


            if (
                len(train) < 500
                or len(test) < 100
            ):

                continue


            baseline_probability = float(
                train[
                    "scored"
                ].mean()
            )


            (
                train_x,
                test_x,
            ) = prepare_matrix(
                train,
                test,
                features,
            )


            train_y = (
                train[
                    "scored"
                ]
                .astype(float)
                .to_numpy()
            )


            beta = fit_logistic_ridge(
                train_x,
                train_y,
            )


            model_probability = sigmoid(
                test_x @ beta
            )


            all_actual.extend(
                test[
                    "scored"
                ]
                .astype(float)
                .tolist()
            )


            all_baseline.extend(
                [
                    baseline_probability
                ]
                * len(test)
            )


            all_model.extend(
                model_probability.tolist()
            )


        if not all_actual:
            continue


        baseline_brier = brier_score(
            all_actual,
            all_baseline,
        )


        model_brier = brier_score(
            all_actual,
            all_model,
        )


        improvement = (
            baseline_brier
            - model_brier
        )


        result_rows.append({

            "Model":
                model_name,

            "Holdout season":
                holdout,

            "Train seasons":
                ",".join(
                    str(x)
                    for x in train_seasons
                ),

            "Test rows":
                len(
                    all_actual
                ),

            "Baseline Brier":
                round(
                    baseline_brier,
                    6
                ),

            "Model Brier":
                round(
                    model_brier,
                    6
                ),

            "Brier improvement":
                round(
                    improvement,
                    6
                ),

            "Beat baseline":
                int(
                    improvement > 0
                ),
        })


results = pd.DataFrame(
    result_rows
)


if results.empty:

    raise SystemExit(
        "ERROR: No combination tests were created."
    )


results = results.sort_values(
    [
        "Model",
        "Holdout season",
    ]
).reset_index(
    drop=True
)


results.to_csv(
    RESULTS_FILE,
    index=False
)


# =========================================================
# SUMMARIZE MODELS
# =========================================================

summary_rows = []


for (
    model_name,
    group
) in results.groupby(
    "Model"
):

    positive = int(
        group[
            "Beat baseline"
        ].sum()
    )


    average_improvement = float(
        group[
            "Brier improvement"
        ].mean()
    )


    worst_improvement = float(
        group[
            "Brier improvement"
        ].min()
    )


    best_improvement = float(
        group[
            "Brier improvement"
        ].max()
    )


    season_2026 = group[
        group[
            "Holdout season"
        ] == 2026
    ]


    if len(
        season_2026
    ):

        improvement_2026 = float(
            season_2026.iloc[
                0
            ][
                "Brier improvement"
            ]
        )

    else:

        improvement_2026 = None


    if (
        positive >= 3
        and average_improvement > 0
        and worst_improvement > -0.00025
    ):

        decision = (
            "PROMISING"
        )

    elif (
        positive >= 2
        and average_improvement > 0
    ):

        decision = (
            "MIXED"
        )

    else:

        decision = (
            "WEAK"
        )


    summary_rows.append({

        "Model":
            model_name,

        "Positive holdouts":
            positive,

        "Holdouts tested":
            len(
                group
            ),

        "Average Brier improvement":
            round(
                average_improvement,
                6
            ),

        "Worst Brier improvement":
            round(
                worst_improvement,
                6
            ),

        "Best Brier improvement":
            round(
                best_improvement,
                6
            ),

        "2026 improvement":
            (
                round(
                    improvement_2026,
                    6
                )
                if improvement_2026
                is not None
                else None
            ),

        "Decision":
            decision,
    })


summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(
    [
        "Positive holdouts",
        "Average Brier improvement",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(
    drop=True
)


summary.insert(
    0,
    "Rank",
    range(
        1,
        len(summary) + 1
    )
)


# =========================================================
# WRITE SUMMARY
# =========================================================

lines = [
    "SHARPREPORT HISTORICAL COMBINATION TEST",
    "=" * 76,
    "",
    "METHOD:",
    "Walk-forward logistic regression with ridge regularization.",
    "Top 1st and Bottom 1st are fit separately.",
    "Missing early-season history is imputed using TRAINING data only.",
    "Sample size is included as a model input.",
    "",
    "2023 holdout: train on 2022",
    "2024 holdout: train on 2022-2023",
    "2025 holdout: train on 2022-2024",
    "2026 holdout: train on 2022-2025",
    "",
    "Positive Brier improvement means the combination beat",
    "the Top/Bottom historical base-rate forecast.",
    "",
    "MODEL RANKING",
    "-" * 76,
]


for _, row in summary.iterrows():

    lines.extend(
        [
            f"#{int(row['Rank'])} {row['Model']}",
            (
                f"Beat baseline: "
                f"{int(row['Positive holdouts'])} of "
                f"{int(row['Holdouts tested'])} holdouts"
            ),
            (
                f"Average improvement: "
                f"{row['Average Brier improvement']:+.6f}"
            ),
            (
                f"Worst / Best: "
                f"{row['Worst Brier improvement']:+.6f} / "
                f"{row['Best Brier improvement']:+.6f}"
            ),
            (
                f"2026 improvement: "
                f"{row['2026 improvement']:+.6f}"
            ),
            f"Decision: {row['Decision']}",
            "-" * 76,
        ]
    )


lines.extend(
    [
        "",
        "SEASON DETAIL",
        "-" * 76,
    ]
)


for model_name in summary[
    "Model"
]:

    lines.append("")
    lines.append(
        model_name
    )


    model_rows = results[
        results[
            "Model"
        ] == model_name
    ].sort_values(
        "Holdout season"
    )


    for _, row in model_rows.iterrows():

        lines.append(
            f"{int(row['Holdout season'])} | "
            f"improvement "
            f"{row['Brier improvement']:+.6f} | "
            f"model Brier "
            f"{row['Model Brier']:.6f}"
        )


lines.extend(
    [
        "",
        "INTERPRETATION RULE:",
        "If the history combinations remain weak or unstable,",
        "we will NOT force them into the final model.",
        "",
        "The next major build is historical Statcast skill:",
        "pitcher xwOBA/K%/BB%/Barrel% and offense quality.",
        "Those are expected to be more informative than",
        "simple first-inning result history.",
    ]
)


SUMMARY_FILE.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


print()
print(
    "HISTORY COMBINATION TEST COMPLETE"
)
print()

print(
    "Top combination:"
)

print(
    summary.iloc[
        0
    ][
        "Model"
    ]
)

print()

print(
    f"Created: {SUMMARY_FILE}"
)

print(
    f"Created: {RESULTS_FILE}"
)
