
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("historical_half_inning_context.csv")
SUMMARY_FILE = Path("historical_feature_test_summary.txt")
RESULTS_FILE = Path("historical_feature_test_results.csv")
DETAIL_FILE = Path("historical_feature_bucket_details.csv")

TRAIN_SEASONS = [2022, 2023, 2024, 2025]
TEST_SEASON = 2026
BUCKETS = 5

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


def evaluate_feature(data, feature):
    name = feature["name"]
    column = feature["column"]
    sample_column = feature["sample_column"]
    min_sample = feature["min_sample"]

    prediction_parts = []
    detail_rows = []

    for half in ["Top 1st", "Bottom 1st"]:
        half_data = data[data["half"] == half].copy()

        train = half_data[
            half_data["season"].isin(TRAIN_SEASONS)
            & half_data[column].notna()
            & half_data[sample_column].notna()
            & (half_data[sample_column] >= min_sample)
        ].copy()

        test = half_data[
            (half_data["season"] == TEST_SEASON)
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

        bucket_numbers = sorted(
            set(train["bucket"].dropna().astype(int).tolist())
            | set(test["bucket"].dropna().astype(int).tolist())
        )

        for bucket in bucket_numbers:
            tr = train[train["bucket"] == bucket]
            te = test[test["bucket"] == bucket]

            detail_rows.append(
                {
                    "feature": name,
                    "half": half,
                    "bucket": bucket + 1,
                    "train_rows": len(tr),
                    "train_scoring_rate": (
                        float(tr["scored"].mean()) if len(tr) else None
                    ),
                    "test_rows": len(te),
                    "test_scoring_rate": (
                        float(te["scored"].mean()) if len(te) else None
                    ),
                    "feature_min": (
                        float(tr[column].min()) if len(tr) else None
                    ),
                    "feature_max": (
                        float(tr[column].max()) if len(tr) else None
                    ),
                }
            )

    if not prediction_parts:
        return None, detail_rows

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

    test_total = len(data[data["season"] == TEST_SEASON])
    coverage = len(predictions) / test_total if test_total else 0.0

    holdout_buckets = (
        predictions.groupby("bucket", observed=True)["scored"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("bucket")
    )

    if len(holdout_buckets) >= 2:
        low_rate = float(holdout_buckets.iloc[0]["mean"])
        high_rate = float(holdout_buckets.iloc[-1]["mean"])
        spread = high_rate - low_rate
    else:
        low_rate = None
        high_rate = None
        spread = None

    if spread is None:
        direction = "Insufficient buckets"
    elif spread > 0:
        direction = "Higher value = more scoring"
    elif spread < 0:
        direction = "Higher value = less scoring"
    else:
        direction = "No holdout separation"

    result = {
        "Feature": name,
        "Minimum prior sample": min_sample,
        "2026 test rows": len(predictions),
        "2026 coverage %": round(coverage * 100, 2),
        "Baseline Brier": round(baseline_brier, 6),
        "Feature Brier": round(feature_brier, 6),
        "Brier improvement": round(improvement, 6),
        "Lowest bucket scoring %": (
            round(low_rate * 100, 2) if low_rate is not None else None
        ),
        "Highest bucket scoring %": (
            round(high_rate * 100, 2) if high_rate is not None else None
        ),
        "High-low spread pp": (
            round(spread * 100, 2) if spread is not None else None
        ),
        "Holdout direction": direction,
    }

    return result, detail_rows


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

train_data = data[data["season"].isin(TRAIN_SEASONS)]
test_data = data[data["season"] == TEST_SEASON]

base_rates = []
for half in ["Top 1st", "Bottom 1st"]:
    train_half = train_data[train_data["half"] == half]
    test_half = test_data[test_data["half"] == half]
    base_rates.append(
        {
            "half": half,
            "train_rows": len(train_half),
            "train_rate": float(train_half["scored"].mean()),
            "test_rows": len(test_half),
            "test_rate": float(test_half["scored"].mean()),
        }
    )

results = []
details = []

for feature in FEATURES:
    result, feature_details = evaluate_feature(data, feature)
    if result is not None:
        results.append(result)
    details.extend(feature_details)

results_df = pd.DataFrame(results)

if results_df.empty:
    raise SystemExit("ERROR: No feature tests were created.")

results_df = results_df.sort_values(
    "Brier improvement",
    ascending=False,
).reset_index(drop=True)

results_df.insert(
    0,
    "Rank",
    range(1, len(results_df) + 1),
)

results_df.to_csv(RESULTS_FILE, index=False)
pd.DataFrame(details).to_csv(DETAIL_FILE, index=False)

lines = [
    "SHARPREPORT HISTORICAL FEATURE TEST",
    "=" * 70,
    "",
    "METHOD:",
    "Training seasons: 2022-2025",
    "True holdout season: 2026",
    "Target: Did the offense score in its half of the 1st inning?",
    "",
    "Each feature is divided into five buckets using ONLY 2022-2025.",
    "The 2026 holdout is then scored using the historical bucket rates.",
    "Positive Brier improvement means the feature beat the simple",
    "Top/Bottom 1st historical base-rate forecast on 2026.",
    "",
    "BASE RATES",
    "-" * 70,
]

for row in base_rates:
    lines.append(
        f"{row['half']} | "
        f"2022-25: {row['train_rate'] * 100:.2f}% "
        f"({row['train_rows']} rows) | "
        f"2026: {row['test_rate'] * 100:.2f}% "
        f"({row['test_rows']} rows)"
    )

lines.extend(
    [
        "",
        "FEATURE RANKING",
        "-" * 70,
    ]
)

for _, row in results_df.iterrows():
    lines.extend(
        [
            f"#{int(row['Rank'])} {row['Feature']}",
            (
                f"2026 rows: {int(row['2026 test rows'])} "
                f"({row['2026 coverage %']:.2f}% coverage)"
            ),
            f"Baseline Brier: {row['Baseline Brier']:.6f}",
            f"Feature Brier: {row['Feature Brier']:.6f}",
            f"Brier improvement: {row['Brier improvement']:+.6f}",
            (
                f"Lowest bucket scoring: "
                f"{row['Lowest bucket scoring %']:.2f}%"
            ),
            (
                f"Highest bucket scoring: "
                f"{row['Highest bucket scoring %']:.2f}%"
            ),
            (
                f"High-low spread: "
                f"{row['High-low spread pp']:+.2f} pp"
            ),
            f"Direction: {row['Holdout direction']}",
            "-" * 70,
        ]
    )

positive_count = int(
    (results_df["Brier improvement"] > 0).sum()
)

lines.extend(
    [
        "",
        "INITIAL SCREEN",
        "-" * 70,
        (
            f"Features beating baseline on 2026: "
            f"{positive_count} of {len(results_df)}"
        ),
        "",
        "IMPORTANT:",
        "This is a feature-screening step, not the final prediction model.",
        "The next stage will test combinations and season-to-season stability",
        "before we convert anything into NRFI/YRFI probabilities.",
    ]
)

SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("HISTORICAL FEATURE TEST COMPLETE")
print()
print(f"Training rows: {len(train_data)}")
print(f"2026 holdout rows: {len(test_data)}")
print()
print("Top feature:")
print(results_df.iloc[0]["Feature"])
print()
print(f"Created: {SUMMARY_FILE}")
print(f"Created: {RESULTS_FILE}")
print(f"Created: {DETAIL_FILE}")
