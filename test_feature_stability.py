
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("historical_half_inning_context.csv")
SUMMARY_FILE = Path("historical_feature_stability_summary.txt")
RESULTS_FILE = Path("historical_feature_stability_results.csv")

BUCKETS = 5
HOLDOUT_SEASONS = [2023, 2024, 2025, 2026]

FEATURES = [
    {
        "name": "Pitcher scoreless 1st %",
        "column": "pregame_pitcher_scoreless_pct",
        "sample_column": "pregame_pitcher_starts",
        "min_sample": 5,
    },
    {
        "name": "Pitcher 1st-inning runs/start",
        "column": "pregame_pitcher_1st_runs_per_start",
        "sample_column": "pregame_pitcher_starts",
        "min_sample": 5,
    },
    {
        "name": "Offense 1st-inning scoring %",
        "column": "pregame_offense_scoring_pct",
        "sample_column": "pregame_offense_games",
        "min_sample": 10,
    },
    {
        "name": "Offense 1st-inning runs/game",
        "column": "pregame_offense_runs_per_game",
        "sample_column": "pregame_offense_games",
        "min_sample": 10,
    },
    {
        "name": "Home/Away split scoring %",
        "column": "pregame_split_scoring_pct",
        "sample_column": "pregame_split_games",
        "min_sample": 5,
    },
    {
        "name": "Home/Away split runs/game",
        "column": "pregame_split_runs_per_game",
        "sample_column": "pregame_split_games",
        "min_sample": 5,
    },
]


def brier(actual, predicted):
    return float(((actual.astype(float) - predicted.astype(float)) ** 2).mean())


def evaluate_feature_for_holdout(data, feature, holdout):
    train_seasons = [
        season
        for season in sorted(data["season"].dropna().astype(int).unique())
        if season < holdout
    ]

    if not train_seasons:
        return None

    name = feature["name"]
    column = feature["column"]
    sample_column = feature["sample_column"]
    min_sample = feature["min_sample"]

    prediction_parts = []

    for half in ["Top 1st", "Bottom 1st"]:
        half_data = data[data["half"] == half].copy()

        train = half_data[
            half_data["season"].isin(train_seasons)
            & half_data[column].notna()
            & half_data[sample_column].notna()
            & (half_data[sample_column] >= min_sample)
        ].copy()

        test = half_data[
            (half_data["season"] == holdout)
            & half_data[column].notna()
            & half_data[sample_column].notna()
            & (half_data[sample_column] >= min_sample)
        ].copy()

        if len(train) < 100 or len(test) < 25:
            continue

        baseline_probability = float(train["scored"].mean())

        try:
            _, bins = pd.qcut(
                train[column],
                q=BUCKETS,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            continue

        bins = list(bins)

        if len(bins) < 3:
            continue

        bins[0] = float("-inf")
        bins[-1] = float("inf")

        train["bucket"] = pd.cut(
            train[column],
            bins=bins,
            labels=False,
            include_lowest=True,
        )

        test["bucket"] = pd.cut(
            test[column],
            bins=bins,
            labels=False,
            include_lowest=True,
        )

        bucket_rates = (
            train.groupby("bucket", observed=True)["scored"]
            .mean()
            .to_dict()
        )

        test["baseline_prediction"] = baseline_probability
        test["feature_prediction"] = (
            test["bucket"]
            .map(bucket_rates)
            .fillna(baseline_probability)
            .astype(float)
        )

        prediction_parts.append(
            test[
                [
                    "game_pk",
                    "half",
                    "scored",
                    "bucket",
                    "baseline_prediction",
                    "feature_prediction",
                ]
            ].copy()
        )

    if not prediction_parts:
        return None

    predictions = pd.concat(prediction_parts, ignore_index=True)

    baseline_brier = brier(
        predictions["scored"],
        predictions["baseline_prediction"],
    )

    feature_brier = brier(
        predictions["scored"],
        predictions["feature_prediction"],
    )

    improvement = baseline_brier - feature_brier

    total_holdout_rows = len(data[data["season"] == holdout])
    coverage = (
        len(predictions) / total_holdout_rows
        if total_holdout_rows
        else 0.0
    )

    bucket_summary = (
        predictions.groupby("bucket", observed=True)["scored"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("bucket")
    )

    if len(bucket_summary) >= 2:
        low_rate = float(bucket_summary.iloc[0]["mean"])
        high_rate = float(bucket_summary.iloc[-1]["mean"])
        spread = high_rate - low_rate
    else:
        low_rate = None
        high_rate = None
        spread = None

    return {
        "Feature": name,
        "Holdout season": holdout,
        "Train seasons": ",".join(str(x) for x in train_seasons),
        "Minimum prior sample": min_sample,
        "Test rows": len(predictions),
        "Coverage %": round(coverage * 100, 2),
        "Baseline Brier": round(baseline_brier, 6),
        "Feature Brier": round(feature_brier, 6),
        "Brier improvement": round(improvement, 6),
        "Low bucket scoring %": (
            round(low_rate * 100, 2) if low_rate is not None else None
        ),
        "High bucket scoring %": (
            round(high_rate * 100, 2) if high_rate is not None else None
        ),
        "High-low spread pp": (
            round(spread * 100, 2) if spread is not None else None
        ),
        "Beat baseline": int(improvement > 0),
    }


if not INPUT_FILE.exists():
    raise SystemExit(
        "ERROR: historical_half_inning_context.csv was not found."
    )

data = pd.read_csv(INPUT_FILE)

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
    data[column] = pd.to_numeric(data[column], errors="coerce")

rows = []

for feature in FEATURES:
    for holdout in HOLDOUT_SEASONS:
        result = evaluate_feature_for_holdout(
            data,
            feature,
            holdout,
        )
        if result is not None:
            rows.append(result)

results = pd.DataFrame(rows)

if results.empty:
    raise SystemExit("ERROR: No stability tests were created.")

results = results.sort_values(
    ["Feature", "Holdout season"]
).reset_index(drop=True)

results.to_csv(
    RESULTS_FILE,
    index=False,
)

summary_rows = []

for feature_name, group in results.groupby("Feature"):
    positive = int(group["Beat baseline"].sum())
    seasons_tested = len(group)
    avg_improvement = float(group["Brier improvement"].mean())
    worst_improvement = float(group["Brier improvement"].min())
    best_improvement = float(group["Brier improvement"].max())
    avg_coverage = float(group["Coverage %"].mean())

    valid_spreads = group["High-low spread pp"].dropna()
    avg_abs_spread = (
        float(valid_spreads.abs().mean())
        if len(valid_spreads)
        else 0.0
    )

    if positive >= 3 and avg_improvement > 0:
        screen = "STABLE CANDIDATE"
    elif positive == 2 and avg_improvement > 0:
        screen = "MIXED / KEEP TESTING"
    else:
        screen = "WEAK / DO NOT WEIGHT YET"

    summary_rows.append(
        {
            "Feature": feature_name,
            "Positive holdouts": positive,
            "Seasons tested": seasons_tested,
            "Average Brier improvement": round(avg_improvement, 6),
            "Worst Brier improvement": round(worst_improvement, 6),
            "Best Brier improvement": round(best_improvement, 6),
            "Average coverage %": round(avg_coverage, 2),
            "Average absolute bucket spread pp": round(avg_abs_spread, 2),
            "Screen": screen,
        }
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    [
        "Positive holdouts",
        "Average Brier improvement",
    ],
    ascending=[False, False],
).reset_index(drop=True)

summary_df.insert(
    0,
    "Rank",
    range(1, len(summary_df) + 1),
)

lines = [
    "SHARPREPORT WALK-FORWARD FEATURE STABILITY TEST",
    "=" * 74,
    "",
    "METHOD:",
    "2023 holdout: train on 2022",
    "2024 holdout: train on 2022-2023",
    "2025 holdout: train on 2022-2024",
    "2026 holdout: train on 2022-2025",
    "",
    "Each holdout season is completely unseen when its bucket rates are built.",
    "Positive Brier improvement means the feature beat the Top/Bottom base rate.",
    "",
    "STABILITY RANKING",
    "-" * 74,
]

for _, row in summary_df.iterrows():
    lines.extend(
        [
            f"#{int(row['Rank'])} {row['Feature']}",
            (
                f"Beat baseline: {int(row['Positive holdouts'])} of "
                f"{int(row['Seasons tested'])} holdout seasons"
            ),
            (
                f"Average Brier improvement: "
                f"{row['Average Brier improvement']:+.6f}"
            ),
            (
                f"Worst / Best improvement: "
                f"{row['Worst Brier improvement']:+.6f} / "
                f"{row['Best Brier improvement']:+.6f}"
            ),
            (
                f"Average coverage: "
                f"{row['Average coverage %']:.2f}%"
            ),
            (
                f"Average absolute bucket spread: "
                f"{row['Average absolute bucket spread pp']:.2f} pp"
            ),
            f"Screen: {row['Screen']}",
            "-" * 74,
        ]
    )

lines.extend(
    [
        "",
        "SEASON-BY-SEASON DETAIL",
        "-" * 74,
    ]
)

for feature_name in summary_df["Feature"]:
    lines.append("")
    lines.append(feature_name)

    feature_rows = results[
        results["Feature"] == feature_name
    ].sort_values("Holdout season")

    for _, row in feature_rows.iterrows():
        lines.append(
            f"{int(row['Holdout season'])} | "
            f"Brier improvement {row['Brier improvement']:+.6f} | "
            f"coverage {row['Coverage %']:.2f}% | "
            f"spread {row['High-low spread pp']:+.2f} pp"
        )

lines.extend(
    [
        "",
        "IMPORTANT:",
        "This is still screening, not the final model.",
        "Correlated features should not automatically receive separate weights.",
        "The next stage will test combinations of the stable candidates.",
    ]
)

SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("FEATURE STABILITY TEST COMPLETE")
print()
print(f"Created: {SUMMARY_FILE}")
print(f"Created: {RESULTS_FILE}")
print()
print("Top stability feature:")
print(summary_df.iloc[0]["Feature"])
