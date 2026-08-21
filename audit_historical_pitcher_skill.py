
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("historical_pitcher_skill_context.csv")
SUMMARY_FILE = Path("historical_pitcher_skill_audit_summary.txt")

if not INPUT_FILE.exists():
    raise SystemExit(
        "ERROR: historical_pitcher_skill_context.csv was not found."
    )

games = pd.read_csv(INPUT_FILE)

required_game_columns = [
    "game_pk",
    "date",
    "season",
]

for column in required_game_columns:
    if column not in games.columns:
        raise SystemExit(
            f"ERROR: Missing required column: {column}"
        )

side_rows = []

for side in ["away", "home"]:
    needed = {
        "sp_id": f"{side}_sp_id",
        "pa": f"{side}_sp_pa_before_game",
        "bbe": f"{side}_sp_bbe_before_game",
        "xwoba": f"{side}_sp_xwoba_allowed",
        "k_pct": f"{side}_sp_k_pct",
        "bb_pct": f"{side}_sp_bb_pct",
        "barrel_pct": f"{side}_sp_barrel_pct",
    }

    for label, column in needed.items():
        if column not in games.columns:
            raise SystemExit(
                f"ERROR: Missing required column: {column}"
            )

    temp = pd.DataFrame({
        "game_pk": games["game_pk"],
        "date": games["date"],
        "season": games["season"],
        "side": side,
        "sp_id": games[needed["sp_id"]],
        "pa": games[needed["pa"]],
        "bbe": games[needed["bbe"]],
        "xwoba": games[needed["xwoba"]],
        "k_pct": games[needed["k_pct"]],
        "bb_pct": games[needed["bb_pct"]],
        "barrel_pct": games[needed["barrel_pct"]],
    })

    side_rows.append(temp)

data = pd.concat(side_rows, ignore_index=True)

numeric_columns = [
    "season",
    "sp_id",
    "pa",
    "bbe",
    "xwoba",
    "k_pct",
    "bb_pct",
    "barrel_pct",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data["any_prior_pa"] = data["pa"].fillna(0) > 0
data["pa_50"] = data["pa"].fillna(0) >= 50

data["xwoba_available"] = data["xwoba"].notna()
data["k_available"] = data["k_pct"].notna()
data["bb_available"] = data["bb_pct"].notna()
data["barrel_available"] = data["barrel_pct"].notna()

data["all_four_available"] = (
    data["xwoba_available"]
    & data["k_available"]
    & data["bb_available"]
    & data["barrel_available"]
)

data["all_four_with_prior_pa"] = (
    data["any_prior_pa"]
    & data["all_four_available"]
)

data["all_four_at_50_pa"] = (
    data["pa_50"]
    & data["all_four_available"]
)

total = len(data)

def pct(count, denom):
    if denom == 0:
        return 0.0
    return count / denom * 100.0

lines = [
    "SHARPREPORT HISTORICAL PITCHER SKILL AUDIT",
    "=" * 68,
    "",
    f"Historical games: {len(games)}",
    f"Starter sides: {total}",
    "",
]

metrics = [
    ("Any prior Statcast PA", "any_prior_pa"),
    ("At least 50 prior PA", "pa_50"),
    ("xwOBA available", "xwoba_available"),
    ("K% available", "k_available"),
    ("BB% available", "bb_available"),
    ("Barrel% available", "barrel_available"),
    ("All four core metrics available", "all_four_available"),
    ("All four + any prior PA", "all_four_with_prior_pa"),
    ("All four + at least 50 prior PA", "all_four_at_50_pa"),
]

for label, column in metrics:
    count = int(data[column].sum())
    lines.append(
        f"{label}: {count} ({pct(count, total):.2f}%)"
    )

lines.extend([
    "",
    "SANITY CHECKS",
    "-" * 68,
])

zero_pa_with_metrics = int(
    (
        (~data["any_prior_pa"])
        & data["all_four_available"]
    ).sum()
)

lines.append(
    "Rows with zero prior PA but all four metrics populated: "
    f"{zero_pa_with_metrics}"
)

invalid_xwoba = int(
    (
        data["xwoba"].notna()
        & (
            (data["xwoba"] < 0)
            | (data["xwoba"] > 1.5)
        )
    ).sum()
)

invalid_pct = int(
    (
        (
            data["k_pct"].notna()
            & (
                (data["k_pct"] < 0)
                | (data["k_pct"] > 100)
            )
        )
        |
        (
            data["bb_pct"].notna()
            & (
                (data["bb_pct"] < 0)
                | (data["bb_pct"] > 100)
            )
        )
        |
        (
            data["barrel_pct"].notna()
            & (
                (data["barrel_pct"] < 0)
                | (data["barrel_pct"] > 100)
            )
        )
    ).sum()
)

lines.append(
    f"Rows with impossible numeric ranges: {invalid_xwoba + invalid_pct}"
)

for year in sorted(data["season"].dropna().astype(int).unique()):
    year_data = data[data["season"] == year]
    denom = len(year_data)

    lines.extend([
        "",
        str(year),
        f"Starter sides: {denom}",
    ])

    for label, column in [
        ("Any prior PA", "any_prior_pa"),
        ("At least 50 prior PA", "pa_50"),
        ("All four core metrics", "all_four_available"),
        ("All four + 50 PA", "all_four_at_50_pa"),
    ]:
        count = int(year_data[column].sum())
        lines.append(
            f"{label}: {count} ({pct(count, denom):.2f}%)"
        )

    lines.append("-" * 68)

SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("PITCHER SKILL AUDIT COMPLETE")
print()
print(f"Created: {SUMMARY_FILE}")
