
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7D1
#
# Build historical PRE-GAME starting-pitcher skill from
# Baseball Savant Statcast data with no same-day leakage.
#
# Outputs:
#   statcast_daily_pitcher_2022.csv ... 2026.csv
#   statcast_pitcher_progress.json
#   historical_pitcher_skill_context.csv
#   historical_pitcher_skill_summary.txt
#
# The daily files are compact summaries, not raw pitch files.
# =========================================================


STARTERS_FILE = Path(
    "historical_actual_starters.csv"
)

OUTPUT_FILE = Path(
    "historical_pitcher_skill_context.csv"
)

SUMMARY_FILE = Path(
    "historical_pitcher_skill_summary.txt"
)

PROGRESS_FILE = Path(
    "statcast_pitcher_progress.json"
)

CHUNK_DAYS = 5

BASE_URL = (
    "https://baseballsavant.mlb.com/"
    "statcast_search/csv"
)


# =========================================================
# HELPERS
# =========================================================

def safe_int(value):

    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    text = str(value).strip()

    if text in (
        "",
        "None",
        "nan",
        "NaN",
    ):
        return None

    return int(
        float(text)
    )


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
        return ""

    if pd.isna(value):
        return ""

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
        ("player_type", "pitcher"),
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

    for attempt in range(
        1,
        4,
    ):

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

            if (
                "game_date"
                not in text[:5000]
            ):

                raise RuntimeError(
                    "Baseball Savant did not "
                    "return a normal Statcast CSV."
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
# TURN PITCH DATA INTO COMPACT DAILY PITCHER TOTALS
# =========================================================

def summarize_chunk(df):

    wanted_columns = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitcher",
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


    work[
        "pitcher"
    ] = pd.to_numeric(
        work["pitcher"],
        errors="coerce",
    )


    work[
        "game_pk"
    ] = pd.to_numeric(
        work["game_pk"],
        errors="coerce",
    )


    work[
        "at_bat_number"
    ] = pd.to_numeric(
        work["at_bat_number"],
        errors="coerce",
    )


    # One Statcast row carries the final event
    # for each completed plate appearance.

    pa = work[
        work["events"].notna()
        &
        work["pitcher"].notna()
    ].copy()


    if pa.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "pitcher",
                "pa",
                "strikeouts",
                "walks",
                "xwoba_num",
                "xwoba_denom",
                "bbe",
                "barrels",
            ]
        )


    # Defensive de-duplication in case an API response
    # ever repeats a plate appearance.

    pa = pa.drop_duplicates(
        subset=[
            "game_pk",
            "at_bat_number",
            "pitcher",
        ],
        keep="first",
    )


    pa[
        "events"
    ] = (
        pa["events"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    pa[
        "woba_value"
    ] = pd.to_numeric(
        pa["woba_value"],
        errors="coerce",
    )


    pa[
        "woba_denom"
    ] = pd.to_numeric(
        pa["woba_denom"],
        errors="coerce",
    )


    pa[
        "estimated_woba_using_speedangle"
    ] = pd.to_numeric(
        pa[
            "estimated_woba_using_speedangle"
        ],
        errors="coerce",
    )


    pa[
        "launch_speed_angle"
    ] = pd.to_numeric(
        pa["launch_speed_angle"],
        errors="coerce",
    )


    pa[
        "pa"
    ] = 1


    pa[
        "strikeouts"
    ] = pa[
        "events"
    ].isin(
        [
            "strikeout",
            "strikeout_double_play",
        ]
    ).astype(int)


    pa[
        "walks"
    ] = pa[
        "events"
    ].isin(
        [
            "walk",
            "intent_walk",
            "intentional_walk",
        ]
    ).astype(int)


    # For batted balls, use Statcast's expected wOBA.
    # For non-batted-ball PA outcomes such as BB/K/HBP,
    # fall back to the recorded wOBA value.
    #
    # This reconstructs a pregame expected-wOBA style
    # measure from the detailed Statcast feed.

    contribution = (
        pa[
            "estimated_woba_using_speedangle"
        ]
        .where(
            pa[
                "estimated_woba_using_speedangle"
            ].notna(),
            pa[
                "woba_value"
            ],
        )
    )


    valid_woba = (
        contribution.notna()
        &
        pa[
            "woba_denom"
        ].notna()
        &
        (
            pa[
                "woba_denom"
            ] > 0
        )
    )


    pa[
        "xwoba_num"
    ] = 0.0

    pa.loc[
        valid_woba,
        "xwoba_num",
    ] = (
        contribution[
            valid_woba
        ]
        *
        pa.loc[
            valid_woba,
            "woba_denom",
        ]
    )


    pa[
        "xwoba_denom_valid"
    ] = 0.0

    pa.loc[
        valid_woba,
        "xwoba_denom_valid",
    ] = pa.loc[
        valid_woba,
        "woba_denom",
    ]


    pa[
        "bbe"
    ] = (
        pa[
            "launch_speed_angle"
        ].notna()
    ).astype(int)


    pa[
        "barrels"
    ] = (
        pa[
            "launch_speed_angle"
        ] == 6
    ).astype(int)


    daily = (
        pa
        .groupby(
            [
                "game_date",
                "pitcher",
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


    daily[
        "pitcher"
    ] = (
        daily["pitcher"]
        .astype(int)
    )


    return daily


# =========================================================
# LOAD VERIFIED ACTUAL STARTERS
# =========================================================

if not STARTERS_FILE.exists():

    raise SystemExit(
        "ERROR: historical_actual_starters.csv "
        "was not found."
    )


starters = pd.read_csv(
    STARTERS_FILE
)


starters[
    "game_pk"
] = pd.to_numeric(
    starters["game_pk"],
    errors="raise",
).astype(int)


starters[
    "season"
] = pd.to_numeric(
    starters["season"],
    errors="raise",
).astype(int)


starters[
    "away_sp_id"
] = pd.to_numeric(
    starters["away_sp_id"],
    errors="raise",
).astype(int)


starters[
    "home_sp_id"
] = pd.to_numeric(
    starters["home_sp_id"],
    errors="raise",
).astype(int)


starters[
    "date"
] = pd.to_datetime(
    starters["date"]
).dt.date


seasons = sorted(
    starters[
        "season"
    ].unique()
)


# =========================================================
# DOWNLOAD / RESUME COMPACT DAILY STATCAST FILES
# =========================================================

progress = load_progress()


print()
print(
    "SHARPREPORT HISTORICAL "
    "PITCHER SKILL BUILD"
)
print(
    "=" * 64
)
print()


for season in seasons:

    season_games = starters[
        starters[
            "season"
        ] == season
    ].copy()


    season_start = min(
        season_games[
            "date"
        ]
    )


    season_end = max(
        season_games[
            "date"
        ]
    )


    daily_file = Path(
        f"statcast_daily_pitcher_"
        f"{season}.csv"
    )


    if daily_file.exists():

        existing_daily = pd.read_csv(
            daily_file
        )

    else:

        existing_daily = pd.DataFrame(
            columns=[
                "game_date",
                "pitcher",
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
        .get(
            str(season),
            {}
        )
        .get(
            "through_date"
        )
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
        f"{current} through "
        f"{season_end}"
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
                    "pitcher",
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

            daily_chunk = (
                summarize_chunk(
                    raw
                )
            )


        # Reruns are safe:
        # remove any previously stored dates in the
        # chunk, then replace them with fresh totals.

        if not existing_daily.empty:

            existing_dates = (
                pd.to_datetime(
                    existing_daily[
                        "game_date"
                    ],
                    errors="coerce",
                )
                .dt.date
            )

            keep = ~(
                (
                    existing_dates
                    >= current
                )
                &
                (
                    existing_dates
                    <= chunk_end
                )
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
                existing_daily[
                    "game_date"
                ],
                errors="coerce",
            ).dt.strftime(
                "%Y-%m-%d"
            )


            existing_daily = (
                existing_daily
                .sort_values(
                    [
                        "game_date",
                        "pitcher",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


        existing_daily.to_csv(
            daily_file,
            index=False,
        )


        progress[
            str(season)
        ] = {
            "through_date":
                end_text
        }


        save_progress(
            progress
        )


        current = (
            chunk_end
            + timedelta(days=1)
        )


        time.sleep(
            0.35
        )


# =========================================================
# BUILD PRE-GAME STARTER SNAPSHOTS
#
# IMPORTANT:
# For each date, snapshots are created BEFORE that date's
# Statcast data is added to the pitcher's season history.
# =========================================================

output_rows = []


for season in seasons:

    print()
    print(
        f"Building pregame pitcher "
        f"snapshots for {season}..."
    )


    season_games = (
        starters[
            starters[
                "season"
            ] == season
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
        f"statcast_daily_pitcher_"
        f"{season}.csv"
    )


    if not daily_file.exists():

        raise RuntimeError(
            f"Missing {daily_file}"
        )


    daily = pd.read_csv(
        daily_file
    )


    if not daily.empty:

        daily[
            "game_date"
        ] = pd.to_datetime(
            daily[
                "game_date"
            ]
        ).dt.date


        numeric_daily = [
            "pitcher",
            "pa",
            "strikeouts",
            "walks",
            "xwoba_num",
            "xwoba_denom",
            "bbe",
            "barrels",
        ]


        for column in numeric_daily:

            daily[
                column
            ] = pd.to_numeric(
                daily[
                    column
                ],
                errors="coerce",
            ).fillna(
                0
            )


    daily_by_date = {
        day: group
        for (
            day,
            group
        ) in daily.groupby(
            "game_date"
        )
    } if not daily.empty else {}


    games_by_date = {
        day: group
        for (
            day,
            group
        ) in season_games.groupby(
            "date"
        )
    }


    history = {}


    calendar_dates = sorted(
        set(
            games_by_date.keys()
        )
        |
        set(
            daily_by_date.keys()
        )
    )


    def get_history(
        pitcher_id
    ):

        if pitcher_id not in history:

            history[
                pitcher_id
            ] = {
                "pa": 0.0,
                "strikeouts": 0.0,
                "walks": 0.0,
                "xwoba_num": 0.0,
                "xwoba_denom": 0.0,
                "bbe": 0.0,
                "barrels": 0.0,
            }

        return history[
            pitcher_id
        ]


    def snapshot(
        pitcher_id
    ):

        h = get_history(
            pitcher_id
        )


        xwoba = safe_rate(
            h[
                "xwoba_num"
            ],
            h[
                "xwoba_denom"
            ],
            multiplier=1.0,
        )


        k_pct = safe_rate(
            h[
                "strikeouts"
            ],
            h[
                "pa"
            ],
        )


        bb_pct = safe_rate(
            h[
                "walks"
            ],
            h[
                "pa"
            ],
        )


        barrel_pct = safe_rate(
            h[
                "barrels"
            ],
            h[
                "bbe"
            ],
        )


        return {
            "pa":
                int(
                    h[
                        "pa"
                    ]
                ),

            "bbe":
                int(
                    h[
                        "bbe"
                    ]
                ),

            "xwoba":
                fmt(
                    xwoba,
                    4,
                ),

            "k_pct":
                fmt(
                    k_pct,
                    2,
                ),

            "bb_pct":
                fmt(
                    bb_pct,
                    2,
                ),

            "barrel_pct":
                fmt(
                    barrel_pct,
                    2,
                ),
        }


    for day in calendar_dates:

        # ---------------------------------------------
        # FIRST: snapshot every starter for this date.
        # ---------------------------------------------

        game_group = (
            games_by_date
            .get(day)
        )


        if game_group is not None:

            for _, game in (
                game_group
                .sort_values(
                    "game_pk"
                )
                .iterrows()
            ):

                away_id = int(
                    game[
                        "away_sp_id"
                    ]
                )

                home_id = int(
                    game[
                        "home_sp_id"
                    ]
                )


                away_skill = snapshot(
                    away_id
                )

                home_skill = snapshot(
                    home_id
                )


                output_rows.append({

                    "game_pk":
                        int(
                            game[
                                "game_pk"
                            ]
                        ),

                    "date":
                        day.isoformat(),

                    "season":
                        int(season),

                    "away_team":
                        game[
                            "away_team"
                        ],

                    "home_team":
                        game[
                            "home_team"
                        ],

                    "away_sp_id":
                        away_id,

                    "away_sp_name":
                        game[
                            "away_sp_name"
                        ],

                    "away_sp_pa_before_game":
                        away_skill[
                            "pa"
                        ],

                    "away_sp_bbe_before_game":
                        away_skill[
                            "bbe"
                        ],

                    "away_sp_xwoba_allowed":
                        away_skill[
                            "xwoba"
                        ],

                    "away_sp_k_pct":
                        away_skill[
                            "k_pct"
                        ],

                    "away_sp_bb_pct":
                        away_skill[
                            "bb_pct"
                        ],

                    "away_sp_barrel_pct":
                        away_skill[
                            "barrel_pct"
                        ],

                    "home_sp_id":
                        home_id,

                    "home_sp_name":
                        game[
                            "home_sp_name"
                        ],

                    "home_sp_pa_before_game":
                        home_skill[
                            "pa"
                        ],

                    "home_sp_bbe_before_game":
                        home_skill[
                            "bbe"
                        ],

                    "home_sp_xwoba_allowed":
                        home_skill[
                            "xwoba"
                        ],

                    "home_sp_k_pct":
                        home_skill[
                            "k_pct"
                        ],

                    "home_sp_bb_pct":
                        home_skill[
                            "bb_pct"
                        ],

                    "home_sp_barrel_pct":
                        home_skill[
                            "barrel_pct"
                        ],
                })


        # ---------------------------------------------
        # SECOND: update with all Statcast from the date.
        # ---------------------------------------------

        day_stats = (
            daily_by_date
            .get(day)
        )


        if day_stats is not None:

            for _, row in (
                day_stats.iterrows()
            ):

                pitcher_id = int(
                    row[
                        "pitcher"
                    ]
                )


                h = get_history(
                    pitcher_id
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

                    h[
                        field
                    ] += float(
                        row[
                            field
                        ]
                    )


# =========================================================
# SAVE PREGAME CONTEXT
# =========================================================

output = pd.DataFrame(
    output_rows
)


output = output.sort_values(
    [
        "date",
        "game_pk",
    ]
).reset_index(
    drop=True
)


if len(output) != len(
    starters
):

    raise RuntimeError(
        "ERROR: Pregame pitcher context "
        f"has {len(output)} games but "
        f"expected {len(starters)}."
    )


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY / COVERAGE
# =========================================================

starter_sides = len(
    output
) * 2


any_prior = 0
pa_50 = 0
complete_core = 0


season_lines = []


for season in seasons:

    season_rows = output[
        output[
            "season"
        ] == season
    ]


    season_sides = len(
        season_rows
    ) * 2


    season_any = 0
    season_50 = 0
    season_core = 0


    for side in [
        "away",
        "home",
    ]:

        pa_col = (
            f"{side}_sp_pa_before_game"
        )

        x_col = (
            f"{side}_sp_xwoba_allowed"
        )

        k_col = (
            f"{side}_sp_k_pct"
        )

        bb_col = (
            f"{side}_sp_bb_pct"
        )

        brl_col = (
            f"{side}_sp_barrel_pct"
        )


        season_any += int(
            (
                season_rows[
                    pa_col
                ] > 0
            ).sum()
        )


        season_50 += int(
            (
                season_rows[
                    pa_col
                ] >= 50
            ).sum()
        )


        season_core += int(
            (
                season_rows[
                    x_col
                ].notna()
                &
                season_rows[
                    k_col
                ].notna()
                &
                season_rows[
                    bb_col
                ].notna()
                &
                season_rows[
                    brl_col
                ].notna()
            ).sum()
        )


    any_prior += season_any
    pa_50 += season_50
    complete_core += season_core


    season_lines.extend(
        [
            "",
            str(season),
            (
                f"Starter sides: "
                f"{season_sides}"
            ),
            (
                f"Any prior Statcast PA: "
                f"{season_any} "
                f"({season_any / season_sides * 100:.2f}%)"
            ),
            (
                f"At least 50 prior PA: "
                f"{season_50} "
                f"({season_50 / season_sides * 100:.2f}%)"
            ),
            (
                f"All four core skill metrics available: "
                f"{season_core} "
                f"({season_core / season_sides * 100:.2f}%)"
            ),
            "-" * 64,
        ]
    )


summary_lines = [
    "SHARPREPORT HISTORICAL PITCHER SKILL CONTEXT",
    "=" * 64,
    "",
    f"Historical games: {len(output)}",
    f"Starter sides: {starter_sides}",
    "",
    (
        f"Starter sides with any prior Statcast PA: "
        f"{any_prior} "
        f"({any_prior / starter_sides * 100:.2f}%)"
    ),
    (
        f"Starter sides with at least 50 prior PA: "
        f"{pa_50} "
        f"({pa_50 / starter_sides * 100:.2f}%)"
    ),
    (
        f"Starter sides with all four core skill metrics: "
        f"{complete_core} "
        f"({complete_core / starter_sides * 100:.2f}%)"
    ),
    "",
    "CORE PRE-GAME METRICS:",
    "Reconstructed xwOBA allowed",
    "K%",
    "BB%",
    "Barrel%",
    "",
    "NO-LOOKAHEAD CONTROL:",
    (
        "Every starter snapshot is created BEFORE "
        "that calendar date's Statcast results are added."
    ),
    (
        "Therefore same-day games and doubleheaders "
        "cannot leak into one another."
    ),
    "",
    "NOTE:",
    (
        "The reconstructed xwOBA measure uses Statcast "
        "expected wOBA on tracked batted balls and the "
        "recorded wOBA value for non-batted-ball PA outcomes."
    ),
    (
        "We will validate this reconstruction against "
        "Baseball Savant leaderboards before model training."
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
    "HISTORICAL PITCHER SKILL BUILD COMPLETE"
)
print()
print(
    f"Games: {len(output)}"
)
print(
    f"Starter sides: {starter_sides}"
)
print(
    f"At least 50 prior PA: "
    f"{pa_50} "
    f"({pa_50 / starter_sides * 100:.2f}%)"
)
print()
print(
    f"Created: {OUTPUT_FILE}"
)
print(
    f"Created: {SUMMARY_FILE}"
)
