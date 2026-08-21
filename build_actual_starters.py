import csv
import json
import time

from pathlib import Path
from urllib.request import urlopen, Request


# =========================================================
# SETTINGS
# =========================================================

OUTCOMES_FILE = Path(
    "historical_first_innings.csv"
)

OUTPUT_FILE = Path(
    "historical_actual_starters.csv"
)

SUMMARY_FILE = Path(
    "historical_actual_starters_summary.txt"
)


# =========================================================
# WEB REQUEST
# =========================================================

def get_json(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(
        request,
        timeout=45
    ) as response:

        return json.load(response)


# =========================================================
# LOAD HISTORICAL GAMES
# =========================================================

games = []

with open(
    OUTCOMES_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        games.append({

            "game_pk":
                int(row["game_pk"]),

            "date":
                row["date"],

            "season":
                int(row["season"]),

            "away_team":
                row["away_team"],

            "home_team":
                row["home_team"],
        })


# =========================================================
# LOAD EXISTING PROGRESS
#
# This lets the script resume if it is interrupted.
# =========================================================

existing = {}


if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            existing[
                int(row["game_pk"])
            ] = row


print()
print(
    "SHARPREPORT ACTUAL STARTER DATABASE"
)
print(
    "=" * 60
)
print()

print(
    f"Historical games: {len(games)}"
)

print(
    f"Already completed: {len(existing)}"
)

print()


# =========================================================
# STARTER NAME
# =========================================================

def get_pitcher_name(
    team_box,
    pitcher_id
):

    if not pitcher_id:
        return ""

    player = (
        team_box
        .get("players", {})
        .get(
            f"ID{pitcher_id}",
            {}
        )
    )

    return (
        player
        .get("person", {})
        .get("fullName", "")
    )


# =========================================================
# ACTUAL STARTERS FROM BOXSCORE
# =========================================================

def get_actual_starters(
    game_pk
):

    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/game/{game_pk}/boxscore"
    )


    for attempt in range(3):

        try:

            data = get_json(url)

            break

        except Exception as error:

            if attempt == 2:

                print(
                    f"FAILED game "
                    f"{game_pk}: {error}"
                )

                return (
                    None,
                    "",
                    None,
                    "",
                )

            time.sleep(
                1.5
            )


    teams = data.get(
        "teams",
        {}
    )

    away_box = teams.get(
        "away",
        {}
    )

    home_box = teams.get(
        "home",
        {}
    )


    away_pitchers = away_box.get(
        "pitchers",
        []
    )

    home_pitchers = home_box.get(
        "pitchers",
        []
    )


    away_id = (
        int(away_pitchers[0])
        if away_pitchers
        else None
    )

    home_id = (
        int(home_pitchers[0])
        if home_pitchers
        else None
    )


    away_name = get_pitcher_name(
        away_box,
        away_id
    )

    home_name = get_pitcher_name(
        home_box,
        home_id
    )


    return (
        away_id,
        away_name,
        home_id,
        home_name,
    )


# =========================================================
# SAVE FUNCTION
# =========================================================

FIELDNAMES = [
    "game_pk",
    "date",
    "season",
    "away_team",
    "home_team",
    "away_sp_id",
    "away_sp_name",
    "home_sp_id",
    "home_sp_name",
]


def save_progress():

    rows = sorted(
        existing.values(),
        key=lambda row: (
            row["date"],
            int(row["game_pk"]),
        )
    )


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        writer.writerows(rows)


# =========================================================
# BUILD DATABASE
# =========================================================

remaining = [
    game
    for game in games
    if game["game_pk"] not in existing
]


total_remaining = len(
    remaining
)


for index, game in enumerate(
    remaining,
    start=1
):

    game_pk = game[
        "game_pk"
    ]


    (
        away_sp_id,
        away_sp_name,
        home_sp_id,
        home_sp_name,
    ) = get_actual_starters(
        game_pk
    )


    existing[
        game_pk
    ] = {

        "game_pk":
            game_pk,

        "date":
            game["date"],

        "season":
            game["season"],

        "away_team":
            game["away_team"],

        "home_team":
            game["home_team"],

        "away_sp_id":
            away_sp_id or "",

        "away_sp_name":
            away_sp_name,

        "home_sp_id":
            home_sp_id or "",

        "home_sp_name":
            home_sp_name,
    }


    if (
        index == 1
        or index % 100 == 0
        or index == total_remaining
    ):

        print(
            f"Completed "
            f"{index} of "
            f"{total_remaining} "
            f"remaining games"
        )


    # Save every 100 games so
    # progress is never lost.

    if (
        index % 100 == 0
    ):

        save_progress()


    time.sleep(
        0.05
    )


save_progress()


# =========================================================
# SUMMARY
# =========================================================

rows = list(
    existing.values()
)


missing_away = sum(
    not row[
        "away_sp_id"
    ]
    for row in rows
)

missing_home = sum(
    not row[
        "home_sp_id"
    ]
    for row in rows
)

complete_games = sum(
    bool(row["away_sp_id"])
    and bool(row["home_sp_id"])
    for row in rows
)


coverage = (
    complete_games
    / len(rows)
    * 100
    if rows
    else 0
)


summary = [

    "SHARPREPORT HISTORICAL "
    "ACTUAL STARTERS",

    "=" * 60,

    "",

    f"Historical games: "
    f"{len(games)}",

    f"Starter database games: "
    f"{len(rows)}",

    "",

    f"Games with both "
    f"actual starters: "
    f"{complete_games}",

    f"Games missing away "
    f"starter: "
    f"{missing_away}",

    f"Games missing home "
    f"starter: "
    f"{missing_home}",

    f"Complete-game starter "
    f"coverage: "
    f"{coverage:.2f}%",

]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary)
    )


print()
print(
    "ACTUAL STARTER DATABASE COMPLETE"
)
print()

print(
    f"Created: {OUTPUT_FILE}"
)

print(
    f"Created: {SUMMARY_FILE}"
)