
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7E2
#
# Build historical PRE-GAME Top-4 hitter skill from
# Baseball Savant Statcast data with no same-day leakage.
#
# Input:
#   historical_top4_lineups.csv
#
# Outputs:
#   statcast_daily_batter_2022.csv ... 2026.csv
#   statcast_batter_progress.json
#   historical_top4_skill_context.csv
#   historical_top4_skill_summary.txt
#
# The daily files are compact summaries, not raw pitch files.
# Resumable by season/date chunk.
# =========================================================


LINEUPS_FILE = Path("historical_top4_lineups.csv")

OUTPUT_FILE = Path(
    "historical_top4_skill_context.csv"
)

SUMMARY_FILE = Path(
    "historical_top4_skill_summary.txt"
)

PROGRESS_FILE = Path(
    "statcast_batter_progress.json"
)

CHUNK_DAYS = 5

BASE_URL = (
    "https://baseballsavant.mlb.com/"
    "statcast_search/csv"
)


# =========================================================
# HELPERS
# =========================================================

def safe_rate(
    numerator,
    denominator,
    multiplier=100.0,
):
    if not denominator:
        return None

    return (
        float(numerator)
        / float(denominator)
        * multiplier
    )


def fmt(value, digits=3):
    if value is None:
        return np.nan

    if pd.isna(value):
        return np.nan

    return round(
        float(value),
        digits,
    )


def load_progress():
    if not PROGRESS_FILE.exists():
        return {}

    try:
        return json.loads(
            PROGRESS_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def save_progress(progress):
    PROGRESS_FILE.write_text(
        json.dumps(
            progress,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# =========================================================
# BASEBALL SAVANT DOWNLOAD
# =========================================================

def build_statcast_url(
    start_date,
    end_date,
    season,
):
    params = [
        ("all", "true"),
        ("hfPT", ""),
        ("hfAB", ""),
        ("hfBBT", ""),
        ("hfPR", ""),
        ("hfZ", ""),
        ("stadium", ""),
        ("hfBBL", ""),
        ("hfNewZones", ""),
        ("hfGT", "R|"),
        ("hfSea", f"{season}|"),
        ("hfSit", ""),
        ("player_type", "batter"),
        ("hfOuts", ""),
        ("opponent", ""),
        ("pitcher_throws", ""),
        ("batter_stands", ""),
        ("hfSA", ""),
        ("game_date_gt", start_date),
        ("game_date_lt", end_date),
        ("team", ""),
        ("position", ""),
        ("hfRO", ""),
        ("home_road", ""),
        ("hfFlag", ""),
        ("metric_1", ""),
        ("hfInn", ""),
        ("min_pitches", "0"),
        ("min_results", "0"),
        ("group_by", "name"),
        ("sort_col", "pitches"),
        ("player_event_sort", "h_launch_speed"),
        ("sort_order", "desc"),
        ("min_pas", "0"),
        ("type", "details"),
    ]

    return (
        BASE_URL
        + "?"
        + urlencode(params)
    )


def download_statcast(
    start_date,
    end_date,
    season,
):
    url = build_statcast_url(
        start_date,
        end_date,
        season,
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent":
                        "Mozilla/5.0",
                    "Accept":
                        "text/csv,text/plain,*/*",
                },
            )

            with urlopen(
                request,
                timeout=120,
            ) as response:
                text = (
                    response
                    .read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            if not text.strip():
                return pd.DataFrame()

            if "game_date" not in text[:5000]:
                raise RuntimeError(
                    "Baseball Savant did not return "
                    "a normal Statcast CSV."
                )

            return pd.read_csv(
                io.StringIO(text),
                low_memory=False,
            )

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt} failed: "
                f"{error}"
            )

            if attempt < 3:
                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        f"Statcast download failed: "
        f"{last_error}"
    )


# =========================================================
# TURN PITCH DATA INTO COMPACT DAILY BATTER TOTALS
# =========================================================

def summarize_chunk(df):
    wanted_columns = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "batter",
        "events",
        "woba_value",
        "woba_denom",
        "estimated_woba_using_speedangle",
        "launch_speed_angle",
    ]

    missing = [
        column
        for column in wanted_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Statcast CSV is missing columns: "
            + ", ".join(missing)
        )

    work = df[
        wanted_columns
    ].copy()

    for column in [
        "batter",
        "game_pk",
        "at_bat_number",
    ]:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    # Final event row = one completed plate appearance.
    pa = work[
        work["events"].notna()
        &
        work["batter"].notna()
    ].copy()

    if pa.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "batter",
                "pa",
                "strikeouts",
                "walks",
                "xwoba_num",
                "xwoba_denom",
                "bbe",
                "barrels",
            ]
        )

    pa = pa.drop_duplicates(
        subset=[
            "game_pk",
            "at_bat_number",
            "batter",
        ],
        keep="first",
    )

    pa["events"] = (
        pa["events"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in [
        "woba_value",
        "woba_denom",
        "estimated_woba_using_speedangle",
        "launch_speed_angle",
    ]:
        pa[column] = pd.to_numeric(
            pa[column],
            errors="coerce",
        )

    pa["pa"] = 1

    pa["strikeouts"] = pa[
        "events"
    ].isin(
        [
            "strikeout",
            "strikeout_double_play",
        ]
    ).astype(int)

    pa["walks"] = pa[
        "events"
    ].isin(
        [
            "walk",
            "intent_walk",
            "intentional_walk",
        ]
    ).astype(int)

    # Reconstruct expected-wOBA style contribution:
    # expected value for tracked contact; actual event value
    # for non-batted-ball outcomes.
    contribution = (
        pa[
            "estimated_woba_using_speedangle"
        ]
        .where(
            pa[
                "estimated_woba_using_speedangle"
            ].notna(),
            pa["woba_value"],
        )
    )

    valid_woba = (
        contribution.notna()
        &
        pa["woba_denom"].notna()
        &
        (pa["woba_denom"] > 0)
    )

    pa["xwoba_num"] = 0.0
    pa.loc[
        valid_woba,
        "xwoba_num",
    ] = (
        contribution[valid_woba]
        *
        pa.loc[
            valid_woba,
            "woba_denom",
        ]
    )

    pa["xwoba_denom_valid"] = 0.0
    pa.loc[
        valid_woba,
        "xwoba_denom_valid",
    ] = pa.loc[
        valid_woba,
        "woba_denom",
    ]

    pa["bbe"] = (
        pa[
            "launch_speed_angle"
        ].notna()
    ).astype(int)

    pa["barrels"] = (
        pa[
            "launch_speed_angle"
        ] == 6
    ).astype(int)

    daily = (
        pa
        .groupby(
            [
                "game_date",
                "batter",
            ],
            as_index=False,
        )
        .agg(
            {
                "pa": "sum",
                "strikeouts": "sum",
                "walks": "sum",
                "xwoba_num": "sum",
                "xwoba_denom_valid": "sum",
                "bbe": "sum",
                "barrels": "sum",
            }
        )
        .rename(
            columns={
                "xwoba_denom_valid":
                    "xwoba_denom",
            }
        )
    )

    daily["batter"] = (
        daily["batter"]
        .astype(int)
    )

    return daily


# =========================================================
# LOAD VERIFIED TOP-4 LINEUPS
# =========================================================

if not LINEUPS_FILE.exists():
    raise SystemExit(
        "ERROR: historical_top4_lineups.csv "
        "was not found."
    )

lineups = pd.read_csv(
    LINEUPS_FILE
)

required = [
    "game_pk",
    "date",
    "season",
    "away_team",
    "home_team",
]

for side in ["away", "home"]:
    for slot in range(1, 5):
        required.append(
            f"{side}_b{slot}_id"
        )
        required.append(
            f"{side}_b{slot}_name"
        )

for column in required:
    if column not in lineups.columns:
        raise SystemExit(
            f"ERROR: Missing required column: "
            f"{column}"
        )

lineups["game_pk"] = pd.to_numeric(
    lineups["game_pk"],
    errors="raise",
).astype(int)

lineups["season"] = pd.to_numeric(
    lineups["season"],
    errors="raise",
).astype(int)

lineups["date"] = pd.to_datetime(
    lineups["date"]
).dt.date

for side in ["away", "home"]:
    for slot in range(1, 5):
        column = f"{side}_b{slot}_id"

        lineups[column] = pd.to_numeric(
            lineups[column],
            errors="raise",
        ).astype(int)

seasons = sorted(
    lineups["season"].unique()
)


# =========================================================
# DOWNLOAD / RESUME DAILY BATTER STATCAST
# =========================================================

progress = load_progress()

print()
print(
    "SHARPREPORT HISTORICAL "
    "TOP-4 OFFENSE SKILL BUILD"
)
print("=" * 70)
print()

for season in seasons:
    season_games = lineups[
        lineups["season"] == season
    ].copy()

    season_start = min(
        season_games["date"]
    )

    season_end = max(
        season_games["date"]
    )

    daily_file = Path(
        f"statcast_daily_batter_{season}.csv"
    )

    if daily_file.exists():
        existing_daily = pd.read_csv(
            daily_file
        )
    else:
        existing_daily = pd.DataFrame(
            columns=[
                "game_date",
                "batter",
                "pa",
                "strikeouts",
                "walks",
                "xwoba_num",
                "xwoba_denom",
                "bbe",
                "barrels",
            ]
        )

    progress_date = (
        progress
        .get(str(season), {})
        .get("through_date")
    )

    if progress_date:
        current = (
            date.fromisoformat(
                progress_date
            )
            + timedelta(days=1)
        )
    else:
        current = season_start

    if current > season_end:
        print(
            f"{season}: already downloaded "
            f"through {season_end}"
        )
        continue

    print(
        f"{season}: downloading "
        f"{current} through {season_end}"
    )

    while current <= season_end:
        chunk_end = min(
            current
            + timedelta(
                days=CHUNK_DAYS - 1
            ),
            season_end,
        )

        start_text = current.isoformat()
        end_text = chunk_end.isoformat()

        print(
            f"  {start_text} "
            f"through {end_text}"
        )

        raw = download_statcast(
            start_text,
            end_text,
            season,
        )

        if raw.empty:
            daily_chunk = pd.DataFrame(
                columns=[
                    "game_date",
                    "batter",
                    "pa",
                    "strikeouts",
                    "walks",
                    "xwoba_num",
                    "xwoba_denom",
                    "bbe",
                    "barrels",
                ]
            )
        else:
            daily_chunk = summarize_chunk(
                raw
            )

        # Safe rerun replacement for this date chunk.
        if not existing_daily.empty:
            existing_dates = (
                pd.to_datetime(
                    existing_daily["game_date"],
                    errors="coerce",
                )
                .dt.date
            )

            keep = ~(
                (existing_dates >= current)
                &
                (existing_dates <= chunk_end)
            )

            existing_daily = (
                existing_daily[
                    keep
                ].copy()
            )

        existing_daily = pd.concat(
            [
                existing_daily,
                daily_chunk,
            ],
            ignore_index=True,
        )

        if not existing_daily.empty:
            existing_daily[
                "game_date"
            ] = pd.to_datetime(
                existing_daily["game_date"],
                errors="coerce",
            ).dt.strftime(
                "%Y-%m-%d"
            )

            existing_daily = (
                existing_daily
                .sort_values(
                    [
                        "game_date",
                        "batter",
                    ]
                )
                .reset_index(drop=True)
            )

        existing_daily.to_csv(
            daily_file,
            index=False,
        )

        progress[str(season)] = {
            "through_date": end_text
        }

        save_progress(progress)

        current = (
            chunk_end
            + timedelta(days=1)
        )

        time.sleep(0.35)


# =========================================================
# BUILD PRE-GAME HITTER SNAPSHOTS
#
# For each calendar date:
# 1) snapshot every historical Top-4 hitter
# 2) only then add that date's Statcast results
#
# This prevents same-day / doubleheader leakage.
# =========================================================

output_rows = []


for season in seasons:
    print()
    print(
        f"Building pregame Top-4 "
        f"snapshots for {season}..."
    )

    season_games = (
        lineups[
            lineups["season"] == season
        ]
        .copy()
        .sort_values(
            [
                "date",
                "game_pk",
            ]
        )
    )

    daily_file = Path(
        f"statcast_daily_batter_{season}.csv"
    )

    if not daily_file.exists():
        raise RuntimeError(
            f"Missing {daily_file}"
        )

    daily = pd.read_csv(
        daily_file
    )

    if not daily.empty:
        daily["game_date"] = pd.to_datetime(
            daily["game_date"]
        ).dt.date

        for column in [
            "batter",
            "pa",
            "strikeouts",
            "walks",
            "xwoba_num",
            "xwoba_denom",
            "bbe",
            "barrels",
        ]:
            daily[column] = pd.to_numeric(
                daily[column],
                errors="coerce",
            ).fillna(0)

    daily_by_date = (
        {
            day: group
            for day, group in daily.groupby(
                "game_date"
            )
        }
        if not daily.empty
        else {}
    )

    games_by_date = {
        day: group
        for day, group in season_games.groupby(
            "date"
        )
    }

    history = {}

    calendar_dates = sorted(
        set(games_by_date.keys())
        |
        set(daily_by_date.keys())
    )

    def get_history(batter_id):
        if batter_id not in history:
            history[batter_id] = {
                "pa": 0.0,
                "strikeouts": 0.0,
                "walks": 0.0,
                "xwoba_num": 0.0,
                "xwoba_denom": 0.0,
                "bbe": 0.0,
                "barrels": 0.0,
            }

        return history[batter_id]

    def snapshot(batter_id):
        h = get_history(batter_id)

        return {
            "pa": int(h["pa"]),
            "bbe": int(h["bbe"]),
            "xwoba": fmt(
                safe_rate(
                    h["xwoba_num"],
                    h["xwoba_denom"],
                    multiplier=1.0,
                ),
                4,
            ),
            "k_pct": fmt(
                safe_rate(
                    h["strikeouts"],
                    h["pa"],
                ),
                2,
            ),
            "bb_pct": fmt(
                safe_rate(
                    h["walks"],
                    h["pa"],
                ),
                2,
            ),
            "barrel_pct": fmt(
                safe_rate(
                    h["barrels"],
                    h["bbe"],
                ),
                2,
            ),
        }

    for day in calendar_dates:
        game_group = games_by_date.get(day)

        if game_group is not None:
            for _, game in (
                game_group
                .sort_values("game_pk")
                .iterrows()
            ):
                row = {
                    "game_pk":
                        int(game["game_pk"]),
                    "date":
                        day.isoformat(),
                    "season":
                        int(season),
                    "away_team":
                        game["away_team"],
                    "home_team":
                        game["home_team"],
                }

                for side in [
                    "away",
                    "home",
                ]:
                    hitter_snapshots = []

                    for slot in range(1, 5):
                        pid = int(
                            game[
                                f"{side}_b{slot}_id"
                            ]
                        )

                        name = game[
                            f"{side}_b{slot}_name"
                        ]

                        snap = snapshot(pid)

                        hitter_snapshots.append(
                            snap
                        )

                        row[
                            f"{side}_b{slot}_id"
                        ] = pid

                        row[
                            f"{side}_b{slot}_name"
                        ] = name

                        row[
                            f"{side}_b{slot}_pa_before_game"
                        ] = snap["pa"]

                        row[
                            f"{side}_b{slot}_bbe_before_game"
                        ] = snap["bbe"]

                        row[
                            f"{side}_b{slot}_xwoba"
                        ] = snap["xwoba"]

                        row[
                            f"{side}_b{slot}_k_pct"
                        ] = snap["k_pct"]

                        row[
                            f"{side}_b{slot}_bb_pct"
                        ] = snap["bb_pct"]

                        row[
                            f"{side}_b{slot}_barrel_pct"
                        ] = snap["barrel_pct"]

                    prior_count = sum(
                        1
                        for s in hitter_snapshots
                        if s["pa"] > 0
                    )

                    pa25_count = sum(
                        1
                        for s in hitter_snapshots
                        if s["pa"] >= 25
                    )

                    pa50_count = sum(
                        1
                        for s in hitter_snapshots
                        if s["pa"] >= 50
                    )

                    complete_core = all(
                        (
                            pd.notna(s["xwoba"])
                            and pd.notna(s["k_pct"])
                            and pd.notna(s["bb_pct"])
                            and pd.notna(s["barrel_pct"])
                        )
                        for s in hitter_snapshots
                    )

                    row[
                        f"{side}_top4_hitters_with_prior_pa"
                    ] = prior_count

                    row[
                        f"{side}_top4_hitters_25_pa"
                    ] = pa25_count

                    row[
                        f"{side}_top4_hitters_50_pa"
                    ] = pa50_count

                    row[
                        f"{side}_top4_combined_pa"
                    ] = sum(
                        s["pa"]
                        for s in hitter_snapshots
                    )

                    row[
                        f"{side}_top4_min_pa"
                    ] = min(
                        s["pa"]
                        for s in hitter_snapshots
                    )

                    row[
                        f"{side}_top4_complete_core"
                    ] = int(complete_core)

                    if complete_core:
                        row[
                            f"{side}_top4_xwoba"
                        ] = fmt(
                            np.mean(
                                [
                                    s["xwoba"]
                                    for s in hitter_snapshots
                                ]
                            ),
                            4,
                        )

                        row[
                            f"{side}_top4_k_pct"
                        ] = fmt(
                            np.mean(
                                [
                                    s["k_pct"]
                                    for s in hitter_snapshots
                                ]
                            ),
                            2,
                        )

                        row[
                            f"{side}_top4_bb_pct"
                        ] = fmt(
                            np.mean(
                                [
                                    s["bb_pct"]
                                    for s in hitter_snapshots
                                ]
                            ),
                            2,
                        )

                        row[
                            f"{side}_top4_barrel_pct"
                        ] = fmt(
                            np.mean(
                                [
                                    s["barrel_pct"]
                                    for s in hitter_snapshots
                                ]
                            ),
                            2,
                        )
                    else:
                        row[
                            f"{side}_top4_xwoba"
                        ] = np.nan

                        row[
                            f"{side}_top4_k_pct"
                        ] = np.nan

                        row[
                            f"{side}_top4_bb_pct"
                        ] = np.nan

                        row[
                            f"{side}_top4_barrel_pct"
                        ] = np.nan

                output_rows.append(row)

        # Update every batter with all Statcast PA from the day
        # only after every game snapshot for the date is created.
        day_stats = daily_by_date.get(day)

        if day_stats is not None:
            for _, stat_row in (
                day_stats.iterrows()
            ):
                batter_id = int(
                    stat_row["batter"]
                )

                h = get_history(
                    batter_id
                )

                for field in [
                    "pa",
                    "strikeouts",
                    "walks",
                    "xwoba_num",
                    "xwoba_denom",
                    "bbe",
                    "barrels",
                ]:
                    h[field] += float(
                        stat_row[field]
                    )


# =========================================================
# SAVE
# =========================================================

output = pd.DataFrame(
    output_rows
)

output = output.sort_values(
    [
        "date",
        "game_pk",
    ]
).reset_index(drop=True)

if len(output) != len(lineups):
    raise RuntimeError(
        "ERROR: Top-4 skill context "
        f"has {len(output)} games but "
        f"expected {len(lineups)}."
    )

output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY / COVERAGE
# =========================================================

team_sides = len(output) * 2

all4_prior = 0
all4_25 = 0
all4_50 = 0
core_available = 0

season_lines = []

for season in seasons:
    year_rows = output[
        output["season"] == season
    ]

    year_sides = len(
        year_rows
    ) * 2

    year_prior = 0
    year_25 = 0
    year_50 = 0
    year_core = 0

    for side in [
        "away",
        "home",
    ]:
        year_prior += int(
            (
                year_rows[
                    f"{side}_top4_hitters_with_prior_pa"
                ] == 4
            ).sum()
        )

        year_25 += int(
            (
                year_rows[
                    f"{side}_top4_hitters_25_pa"
                ] == 4
            ).sum()
        )

        year_50 += int(
            (
                year_rows[
                    f"{side}_top4_hitters_50_pa"
                ] == 4
            ).sum()
        )

        year_core += int(
            (
                year_rows[
                    f"{side}_top4_complete_core"
                ] == 1
            ).sum()
        )

    all4_prior += year_prior
    all4_25 += year_25
    all4_50 += year_50
    core_available += year_core

    season_lines.extend(
        [
            "",
            str(season),
            f"Team Top-4 sides: {year_sides}",
            (
                f"All 4 hitters with prior PA: "
                f"{year_prior} "
                f"({year_prior / year_sides * 100:.2f}%)"
            ),
            (
                f"All 4 hitters with at least 25 prior PA: "
                f"{year_25} "
                f"({year_25 / year_sides * 100:.2f}%)"
            ),
            (
                f"All 4 hitters with at least 50 prior PA: "
                f"{year_50} "
                f"({year_50 / year_sides * 100:.2f}%)"
            ),
            (
                f"All 4 core aggregate metrics available: "
                f"{year_core} "
                f"({year_core / year_sides * 100:.2f}%)"
            ),
            "-" * 70,
        ]
    )


summary_lines = [
    "SHARPREPORT HISTORICAL TOP-4 OFFENSE SKILL CONTEXT",
    "=" * 70,
    "",
    f"Historical games: {len(output)}",
    f"Team Top-4 sides: {team_sides}",
    "",
    (
        f"All 4 hitters with prior Statcast PA: "
        f"{all4_prior} "
        f"({all4_prior / team_sides * 100:.2f}%)"
    ),
    (
        f"All 4 hitters with at least 25 prior PA: "
        f"{all4_25} "
        f"({all4_25 / team_sides * 100:.2f}%)"
    ),
    (
        f"All 4 hitters with at least 50 prior PA: "
        f"{all4_50} "
        f"({all4_50 / team_sides * 100:.2f}%)"
    ),
    (
        f"All 4 core aggregate metrics available: "
        f"{core_available} "
        f"({core_available / team_sides * 100:.2f}%)"
    ),
    "",
    "CORE PRE-GAME TOP-4 METRICS:",
    "Top-4 xwOBA",
    "Top-4 K%",
    "Top-4 BB%",
    "Top-4 Barrel%",
    "Combined prior PA",
    "Minimum prior PA among the four hitters",
    "",
    "NO-LOOKAHEAD CONTROL:",
    (
        "Every historical Top-4 snapshot is created BEFORE "
        "that calendar date's Statcast results are added."
    ),
    (
        "Same-day games and doubleheaders therefore cannot "
        "leak into one another."
    ),
    "",
    "AGGREGATION:",
    (
        "The four starting hitters are averaged equally, matching "
        "the live SharpReport Top-4 aggregate approach."
    ),
    (
        "Aggregate metrics are left missing when any of the four "
        "hitters has no legitimate prior-season Statcast sample."
    ),
]

summary_lines.extend(
    season_lines
)

SUMMARY_FILE.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)

print()
print(
    "HISTORICAL TOP-4 OFFENSE SKILL BUILD COMPLETE"
)
print()
print(
    f"Games: {len(output)}"
)
print(
    f"Team Top-4 sides: {team_sides}"
)
print(
    f"All 4 hitters with prior PA: "
    f"{all4_prior} "
    f"({all4_prior / team_sides * 100:.2f}%)"
)
print(
    f"All 4 hitters with 50+ prior PA: "
    f"{all4_50} "
    f"({all4_50 / team_sides * 100:.2f}%)"
)
print()
print(
    f"Created: {OUTPUT_FILE}"
)
print(
    f"Created: {SUMMARY_FILE}"
)
