import csv

from collections import defaultdict


# =========================================================
# FILES
# =========================================================

OUTCOMES_FILE = (
    "historical_first_innings.csv"
)

STARTERS_FILE = (
    "historical_actual_starters.csv"
)

OUTPUT_FILE = (
    "historical_half_inning_context.csv"
)

SUMMARY_FILE = (
    "historical_context_summary.txt"
)


# =========================================================
# HELPERS
# =========================================================

def to_int(value):

    if value in (
        None,
        "",
        "None",
    ):
        return None

    return int(value)


def percentage(
    numerator,
    denominator
):

    if not denominator:
        return None

    return round(
        numerator
        / denominator
        * 100,
        2
    )


def average(
    numerator,
    denominator
):

    if not denominator:
        return None

    return round(
        numerator
        / denominator,
        3
    )


# =========================================================
# LOAD OUTCOMES
# =========================================================

outcomes = []


with open(
    OUTCOMES_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        outcomes.append({

            "game_pk":
                int(row["game_pk"]),

            "date":
                row["date"],

            "season":
                int(row["season"]),

            "away_team_id":
                int(row["away_team_id"]),

            "away_team":
                row["away_team"],

            "home_team_id":
                int(row["home_team_id"]),

            "home_team":
                row["home_team"],

            "venue_id":
                to_int(
                    row["venue_id"]
                ),

            "venue_name":
                row["venue_name"],

            "away_1st_runs":
                int(
                    row["away_1st_runs"]
                ),

            "home_1st_runs":
                int(
                    row["home_1st_runs"]
                ),

            "nrfi":
                int(row["nrfi"]),

            "yrfi":
                int(row["yrfi"]),
        })


# =========================================================
# LOAD VERIFIED ACTUAL STARTERS
# =========================================================

starter_lookup = {}


with open(
    STARTERS_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        starter_lookup[
            int(row["game_pk"])
        ] = {

            "away_sp_id":
                to_int(
                    row["away_sp_id"]
                ),

            "away_sp_name":
                row["away_sp_name"],

            "home_sp_id":
                to_int(
                    row["home_sp_id"]
                ),

            "home_sp_name":
                row["home_sp_name"],
        }


# =========================================================
# VERIFY STARTER DATABASE
# =========================================================

missing_games = [

    game["game_pk"]

    for game in outcomes

    if game["game_pk"]
    not in starter_lookup
]


if missing_games:

    raise RuntimeError(
        "ERROR: Actual starter database "
        f"is missing {len(missing_games)} games."
    )


incomplete_starters = []


for game in outcomes:

    starters = starter_lookup[
        game["game_pk"]
    ]

    if (
        not starters["away_sp_id"]
        or not starters["home_sp_id"]
    ):

        incomplete_starters.append(
            game["game_pk"]
        )


if incomplete_starters:

    raise RuntimeError(
        "ERROR: Actual starter database "
        f"contains {len(incomplete_starters)} "
        "games with missing starters."
    )


# =========================================================
# HISTORY STORAGE
# =========================================================

pitcher_history = defaultdict(
    lambda: {
        "starts": 0,
        "scoreless": 0,
        "runs": 0,
    }
)


team_history = defaultdict(
    lambda: {
        "games": 0,
        "scored": 0,
        "runs": 0,
    }
)


away_history = defaultdict(
    lambda: {
        "games": 0,
        "scored": 0,
        "runs": 0,
    }
)


home_history = defaultdict(
    lambda: {
        "games": 0,
        "scored": 0,
        "runs": 0,
    }
)


# =========================================================
# GROUP GAMES BY SEASON + DATE
#
# We process an entire date BEFORE adding
# any outcomes from that date to history.
#
# This prevents same-day / doubleheader leakage.
# =========================================================

games_by_date = defaultdict(
    list
)


for game in outcomes:

    key = (
        game["season"],
        game["date"],
    )

    games_by_date[
        key
    ].append(
        game
    )


date_keys = sorted(
    games_by_date.keys()
)


context_rows = []

current_season = None


# =========================================================
# BUILD NO-LOOKAHEAD FEATURES
# =========================================================

for (
    season,
    game_date
) in date_keys:


    if season != current_season:

        pitcher_history.clear()

        team_history.clear()

        away_history.clear()

        home_history.clear()

        current_season = season


    day_games = games_by_date[
        (
            season,
            game_date,
        )
    ]


    # =====================================================
    # FIRST PASS
    #
    # Build every pregame feature row for the date.
    # NO histories are updated yet.
    # =====================================================

    for game in day_games:

        game_pk = game[
            "game_pk"
        ]

        starters = starter_lookup[
            game_pk
        ]


        away_sp_id = starters[
            "away_sp_id"
        ]

        away_sp_name = starters[
            "away_sp_name"
        ]

        home_sp_id = starters[
            "home_sp_id"
        ]

        home_sp_name = starters[
            "home_sp_name"
        ]


        away_team_id = game[
            "away_team_id"
        ]

        home_team_id = game[
            "home_team_id"
        ]


        away_runs = game[
            "away_1st_runs"
        ]

        home_runs = game[
            "home_1st_runs"
        ]


        # =================================================
        # TOP 1ST
        #
        # Away offense vs actual Home starter
        # =================================================

        pitcher_prior = (
            pitcher_history[
                home_sp_id
            ]
        )

        offense_prior = (
            team_history[
                away_team_id
            ]
        )

        split_prior = (
            away_history[
                away_team_id
            ]
        )


        context_rows.append({

            "game_pk":
                game_pk,

            "date":
                game_date,

            "season":
                season,

            "game":
                (
                    f"{game['away_team']} @ "
                    f"{game['home_team']}"
                ),

            "half":
                "Top 1st",

            "offense_team_id":
                away_team_id,

            "offense_team":
                game["away_team"],

            "defense_team_id":
                home_team_id,

            "defense_team":
                game["home_team"],

            "starter_id":
                home_sp_id,

            "starter_name":
                home_sp_name,

            "starter_source":
                "Actual completed-game boxscore",

            "venue_id":
                game["venue_id"],

            "venue_name":
                game["venue_name"],

            "pregame_pitcher_starts":
                pitcher_prior[
                    "starts"
                ],

            "pregame_pitcher_scoreless":
                pitcher_prior[
                    "scoreless"
                ],

            "pregame_pitcher_scoreless_pct":
                percentage(
                    pitcher_prior[
                        "scoreless"
                    ],
                    pitcher_prior[
                        "starts"
                    ],
                ),

            "pregame_pitcher_1st_runs_per_start":
                average(
                    pitcher_prior[
                        "runs"
                    ],
                    pitcher_prior[
                        "starts"
                    ],
                ),

            "pregame_offense_games":
                offense_prior[
                    "games"
                ],

            "pregame_offense_scored":
                offense_prior[
                    "scored"
                ],

            "pregame_offense_scoring_pct":
                percentage(
                    offense_prior[
                        "scored"
                    ],
                    offense_prior[
                        "games"
                    ],
                ),

            "pregame_offense_runs_per_game":
                average(
                    offense_prior[
                        "runs"
                    ],
                    offense_prior[
                        "games"
                    ],
                ),

            "pregame_split_games":
                split_prior[
                    "games"
                ],

            "pregame_split_scored":
                split_prior[
                    "scored"
                ],

            "pregame_split_scoring_pct":
                percentage(
                    split_prior[
                        "scored"
                    ],
                    split_prior[
                        "games"
                    ],
                ),

            "pregame_split_runs_per_game":
                average(
                    split_prior[
                        "runs"
                    ],
                    split_prior[
                        "games"
                    ],
                ),

            "runs_scored":
                away_runs,

            "scored":
                int(
                    away_runs > 0
                ),

            "game_nrfi":
                game["nrfi"],

            "game_yrfi":
                game["yrfi"],
        })


        # =================================================
        # BOTTOM 1ST
        #
        # Home offense vs actual Away starter
        # =================================================

        pitcher_prior = (
            pitcher_history[
                away_sp_id
            ]
        )

        offense_prior = (
            team_history[
                home_team_id
            ]
        )

        split_prior = (
            home_history[
                home_team_id
            ]
        )


        context_rows.append({

            "game_pk":
                game_pk,

            "date":
                game_date,

            "season":
                season,

            "game":
                (
                    f"{game['away_team']} @ "
                    f"{game['home_team']}"
                ),

            "half":
                "Bottom 1st",

            "offense_team_id":
                home_team_id,

            "offense_team":
                game["home_team"],

            "defense_team_id":
                away_team_id,

            "defense_team":
                game["away_team"],

            "starter_id":
                away_sp_id,

            "starter_name":
                away_sp_name,

            "starter_source":
                "Actual completed-game boxscore",

            "venue_id":
                game["venue_id"],

            "venue_name":
                game["venue_name"],

            "pregame_pitcher_starts":
                pitcher_prior[
                    "starts"
                ],

            "pregame_pitcher_scoreless":
                pitcher_prior[
                    "scoreless"
                ],

            "pregame_pitcher_scoreless_pct":
                percentage(
                    pitcher_prior[
                        "scoreless"
                    ],
                    pitcher_prior[
                        "starts"
                    ],
                ),

            "pregame_pitcher_1st_runs_per_start":
                average(
                    pitcher_prior[
                        "runs"
                    ],
                    pitcher_prior[
                        "starts"
                    ],
                ),

            "pregame_offense_games":
                offense_prior[
                    "games"
                ],

            "pregame_offense_scored":
                offense_prior[
                    "scored"
                ],

            "pregame_offense_scoring_pct":
                percentage(
                    offense_prior[
                        "scored"
                    ],
                    offense_prior[
                        "games"
                    ],
                ),

            "pregame_offense_runs_per_game":
                average(
                    offense_prior[
                        "runs"
                    ],
                    offense_prior[
                        "games"
                    ],
                ),

            "pregame_split_games":
                split_prior[
                    "games"
                ],

            "pregame_split_scored":
                split_prior[
                    "scored"
                ],

            "pregame_split_scoring_pct":
                percentage(
                    split_prior[
                        "scored"
                    ],
                    split_prior[
                        "games"
                    ],
                ),

            "pregame_split_runs_per_game":
                average(
                    split_prior[
                        "runs"
                    ],
                    split_prior[
                        "games"
                    ],
                ),

            "runs_scored":
                home_runs,

            "scored":
                int(
                    home_runs > 0
                ),

            "game_nrfi":
                game["nrfi"],

            "game_yrfi":
                game["yrfi"],
        })


    # =====================================================
    # SECOND PASS
    #
    # Only AFTER every game on the date has its
    # pregame features do we update histories.
    # =====================================================

    for game in day_games:

        game_pk = game[
            "game_pk"
        ]

        starters = starter_lookup[
            game_pk
        ]


        away_sp_id = starters[
            "away_sp_id"
        ]

        home_sp_id = starters[
            "home_sp_id"
        ]


        away_team_id = game[
            "away_team_id"
        ]

        home_team_id = game[
            "home_team_id"
        ]


        away_runs = game[
            "away_1st_runs"
        ]

        home_runs = game[
            "home_1st_runs"
        ]


        # Home pitcher faced away offense.

        pitcher_history[
            home_sp_id
        ]["starts"] += 1

        pitcher_history[
            home_sp_id
        ]["runs"] += away_runs

        if away_runs == 0:

            pitcher_history[
                home_sp_id
            ]["scoreless"] += 1


        # Away pitcher faced home offense.

        pitcher_history[
            away_sp_id
        ]["starts"] += 1

        pitcher_history[
            away_sp_id
        ]["runs"] += home_runs

        if home_runs == 0:

            pitcher_history[
                away_sp_id
            ]["scoreless"] += 1


        # Away offense

        team_history[
            away_team_id
        ]["games"] += 1

        team_history[
            away_team_id
        ]["runs"] += away_runs


        away_history[
            away_team_id
        ]["games"] += 1

        away_history[
            away_team_id
        ]["runs"] += away_runs


        if away_runs > 0:

            team_history[
                away_team_id
            ]["scored"] += 1

            away_history[
                away_team_id
            ]["scored"] += 1


        # Home offense

        team_history[
            home_team_id
        ]["games"] += 1

        team_history[
            home_team_id
        ]["runs"] += home_runs


        home_history[
            home_team_id
        ]["games"] += 1

        home_history[
            home_team_id
        ]["runs"] += home_runs


        if home_runs > 0:

            team_history[
                home_team_id
            ]["scored"] += 1

            home_history[
                home_team_id
            ]["scored"] += 1


# =========================================================
# SAVE CONTEXT
# =========================================================

if len(context_rows) != (
    len(outcomes) * 2
):

    raise RuntimeError(
        "ERROR: Incorrect number "
        "of half-inning rows."
    )


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = list(
        context_rows[0].keys()
    )

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        context_rows
    )


# =========================================================
# VALIDATION
# =========================================================

starter_rows = sum(

    bool(row["starter_id"])

    for row in context_rows
)


starter_coverage = (
    starter_rows
    / len(context_rows)
    * 100
)


pitcher_prior_rows = sum(

    row[
        "pregame_pitcher_starts"
    ] > 0

    for row in context_rows
)


offense_prior_rows = sum(

    row[
        "pregame_offense_games"
    ] > 0

    for row in context_rows
)


# =========================================================
# SUMMARY
# =========================================================

lines = [

    "SHARPREPORT VERIFIED HISTORICAL "
    "PREGAME CONTEXT",

    "=" * 60,

    "",

    f"Historical games: "
    f"{len(outcomes)}",

    f"Half-inning rows: "
    f"{len(context_rows)}",

    "",

    "Starter source: "
    "Actual completed-game boxscore",

    f"Rows with verified starter: "
    f"{starter_rows}",

    f"Starter coverage: "
    f"{starter_coverage:.2f}%",

    "",

    f"Rows with prior pitcher history: "
    f"{pitcher_prior_rows}",

    f"Rows with prior offense history: "
    f"{offense_prior_rows}",

    "",

    "NO-LOOKAHEAD CONTROLS:",

    "1. Every feature uses only games "
    "from PRIOR calendar dates.",

    "2. All games on the same date are "
    "scored before that date updates history.",

    "3. This prevents doubleheader and "
    "same-day outcome leakage.",

    "4. Histories reset at the beginning "
    "of each MLB season.",

    "5. Actual starting pitchers are used, "
    "not historical probable pitchers.",

]


for year in sorted(
    set(
        row["season"]
        for row in context_rows
    )
):

    year_rows = [

        row

        for row in context_rows

        if row["season"] == year
    ]


    lines.extend([

        "",

        f"{year}",

        f"Half-inning rows: "
        f"{len(year_rows)}",

        "-" * 60,

    ])


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(lines)
    )


print()
print(
    "VERIFIED HISTORICAL CONTEXT COMPLETE"
)
print()

print(
    f"Games: {len(outcomes)}"
)

print(
    f"Half-inning rows: "
    f"{len(context_rows)}"
)

print(
    f"Starter coverage: "
    f"{starter_coverage:.2f}%"
)

print()

print(
    f"Created: {OUTPUT_FILE}"
)

print(
    f"Created: {SUMMARY_FILE}"
)
