from pathlib import Path
import math
import pandas as pd

INPUT_FILE = Path("historical_game_level_nrfi_predictions.csv")
OUT_BUCKETS = Path("nrfi_probability_bucket_analysis.csv")
OUT_THRESHOLDS = Path("nrfi_probability_threshold_analysis.csv")
OUT_YEARLY = Path("nrfi_probability_threshold_by_year.csv")
OUT_SUMMARY = Path("nrfi_probability_threshold_summary.txt")

MODEL_NAME = "Core + Offense BB + Run Factor"

BUCKETS = [
    0.00, 0.50, 0.52, 0.54, 0.56, 0.58,
    0.60, 0.62, 0.65, 0.70, 1.00,
]

CUTOFFS = [
    0.50, 0.52, 0.54, 0.55, 0.56, 0.57,
    0.58, 0.59, 0.60, 0.61, 0.62, 0.65,
]

FIXED_PRICES = [-110, -120, -130, -140, -150]


def wilson_interval(wins, n, z=1.96):
    if n <= 0:
        return (None, None)

    p = wins / n
    denom = 1 + (z * z / n)
    center = (
        p + z * z / (2 * n)
    ) / denom
    half = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + z * z / (4 * n * n)
        )
        / denom
    )
    return center - half, center + half


def american_from_probability(p):
    if p is None or not 0 < p < 1:
        return None

    if p >= 0.5:
        return -100 * p / (1 - p)

    return 100 * (1 - p) / p


def profit_per_win(odds):
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


def roi_at_fixed_odds(wins, losses, odds):
    bets = wins + losses
    if bets == 0:
        return None

    pnl = wins * profit_per_win(odds) - losses
    return pnl / bets


if not INPUT_FILE.exists():
    raise SystemExit(
        f"ERROR: {INPUT_FILE} was not found in this folder."
    )

df = pd.read_csv(INPUT_FILE)

required = {
    "holdout_year",
    "model",
    "game_pk",
    "nrfi_actual",
    "nrfi_probability",
}

missing = required - set(df.columns)
if missing:
    raise SystemExit(
        "ERROR: Missing required columns: "
        + ", ".join(sorted(missing))
    )

df["holdout_year"] = pd.to_numeric(
    df["holdout_year"],
    errors="raise",
).astype(int)

df["nrfi_actual"] = pd.to_numeric(
    df["nrfi_actual"],
    errors="raise",
).astype(int)

df["nrfi_probability"] = pd.to_numeric(
    df["nrfi_probability"],
    errors="raise",
)

data = df[
    df["model"] == MODEL_NAME
].copy()

if data.empty:
    available = sorted(
        df["model"].dropna().astype(str).unique()
    )
    raise SystemExit(
        "ERROR: Champion model name not found.\n"
        f"Expected: {MODEL_NAME}\n"
        "Available models:\n- "
        + "\n- ".join(available)
    )

# ---------------------------------------------------------
# 1) Probability buckets
# ---------------------------------------------------------
data["bucket"] = pd.cut(
    data["nrfi_probability"],
    bins=BUCKETS,
    include_lowest=True,
    right=False,
)

bucket_rows = []

for bucket, group in data.groupby(
    "bucket",
    observed=True,
):
    n = len(group)
    wins = int(group["nrfi_actual"].sum())
    losses = n - wins
    hit = wins / n
    pred = float(group["nrfi_probability"].mean())
    lo, hi = wilson_interval(wins, n)

    bucket_rows.append({
        "Probability Bucket": str(bucket),
        "Games": n,
        "NRFI Wins": wins,
        "NRFI Losses": losses,
        "Average Model Probability": pred,
        "Actual NRFI Rate": hit,
        "Calibration Gap Actual-Minus-Predicted": hit - pred,
        "95% CI Low": lo,
        "95% CI High": hi,
        "Observed Break-Even American Odds": american_from_probability(hit),
    })

bucket_df = pd.DataFrame(bucket_rows)
bucket_df.to_csv(OUT_BUCKETS, index=False)

# ---------------------------------------------------------
# 2) Cumulative thresholds
# ---------------------------------------------------------
threshold_rows = []

for cutoff in CUTOFFS:
    group = data[
        data["nrfi_probability"] >= cutoff
    ].copy()

    if group.empty:
        continue

    n = len(group)
    wins = int(group["nrfi_actual"].sum())
    losses = n - wins
    hit = wins / n
    pred = float(group["nrfi_probability"].mean())
    lo, hi = wilson_interval(wins, n)

    row = {
        "Minimum Model Probability": cutoff,
        "Games": n,
        "NRFI Wins": wins,
        "NRFI Losses": losses,
        "Average Model Probability": pred,
        "Actual NRFI Rate": hit,
        "Calibration Gap Actual-Minus-Predicted": hit - pred,
        "95% CI Low": lo,
        "95% CI High": hi,
        "Observed Break-Even American Odds": american_from_probability(hit),
    }

    for odds in FIXED_PRICES:
        row[f"Hypothetical ROI at {odds}"] = roi_at_fixed_odds(
            wins,
            losses,
            odds,
        )

    threshold_rows.append(row)

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUT_THRESHOLDS, index=False)

# ---------------------------------------------------------
# 3) Year-by-year threshold stability
# ---------------------------------------------------------
year_rows = []

for cutoff in CUTOFFS:
    group = data[
        data["nrfi_probability"] >= cutoff
    ]

    for year, year_group in group.groupby(
        "holdout_year"
    ):
        n = len(year_group)
        if n == 0:
            continue

        wins = int(
            year_group["nrfi_actual"].sum()
        )
        losses = n - wins
        hit = wins / n

        year_rows.append({
            "Minimum Model Probability": cutoff,
            "Holdout Year": int(year),
            "Games": n,
            "NRFI Wins": wins,
            "NRFI Losses": losses,
            "Actual NRFI Rate": hit,
            "Observed Break-Even American Odds": american_from_probability(hit),
            "ROI at -110": roi_at_fixed_odds(wins, losses, -110),
            "ROI at -120": roi_at_fixed_odds(wins, losses, -120),
            "ROI at -130": roi_at_fixed_odds(wins, losses, -130),
        })

year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT_YEARLY, index=False)

# ---------------------------------------------------------
# 4) Summary
# ---------------------------------------------------------
lines = [
    "SHARPREPORT HISTORICAL NRFI PROBABILITY THRESHOLD ANALYSIS",
    "=" * 78,
    "",
    f"Model: {MODEL_NAME}",
    f"Out-of-sample games analyzed: {len(data):,}",
    f"Holdout seasons: {', '.join(map(str, sorted(data['holdout_year'].unique())))}",
    "",
    "IMPORTANT:",
    "These are out-of-sample historical MODEL probabilities and actual outcomes.",
    "They are NOT actual historical betting ROI because historical sportsbook",
    "first-inning prices were not available for every game.",
    "",
    "The fixed-price ROI columns answer a hypothetical question:",
    "'If every qualifying NRFI could have been bet at this same price,",
    "what would the historical ROI have been?'",
    "",
    "CUMULATIVE PROBABILITY THRESHOLDS",
    "-" * 78,
]

for _, row in threshold_df.iterrows():
    lines.append(
        f">= {row['Minimum Model Probability']:.0%} | "
        f"games {int(row['Games']):4d} | "
        f"pred {row['Average Model Probability']:.1%} | "
        f"actual {row['Actual NRFI Rate']:.1%} | "
        f"95% CI {row['95% CI Low']:.1%}-{row['95% CI High']:.1%} | "
        f"BE odds {row['Observed Break-Even American Odds']:+.0f} | "
        f"ROI -110 {row['Hypothetical ROI at -110']:+.1%} | "
        f"-120 {row['Hypothetical ROI at -120']:+.1%} | "
        f"-130 {row['Hypothetical ROI at -130']:+.1%}"
    )

lines.extend([
    "",
    "HOW TO USE THIS",
    "-" * 78,
    "1. Look for thresholds with a materially higher actual NRFI rate.",
    "2. Prefer thresholds that remain reasonably stable across holdout seasons.",
    "3. Do not choose a cutoff only because one small bucket had a hot result.",
    "4. Compare the eventual live sportsbook price with the observed hit rate.",
    "5. Forward results should confirm any threshold before it becomes permanent.",
])

OUT_SUMMARY.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(f"Analyzed {len(data):,} out-of-sample games for:")
print(MODEL_NAME)
print()
print("Created:")
print(f"- {OUT_BUCKETS}")
print(f"- {OUT_THRESHOLDS}")
print(f"- {OUT_YEARLY}")
print(f"- {OUT_SUMMARY}")
print()
print("Run complete.")
