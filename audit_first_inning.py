import json

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from first_inning_history import (
    get_season_first_innings,
    get_pitcher_starts,
    get_pitcher_first_inning_history,
)


ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).date()
YEAR = TODAY.year


def get_json(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return json.load(response)


# ---------------------------------------------------------
# GET TODAY'S STARTERS WITH CORRECT MLB IDS
# ---------------------------------------------------------

date_string = TODAY.strftime("%Y-%m-%d")

url = (
    "https://statsapi.mlb.com/api/v1/schedule"
    f"?sportId=1&date={date_string}"
    "&hydrate=probablePitcher"
)

data = get_json(url)

targets = {
    "Anthony Kay",
    "George Kirby",
    "Jacob deGrom",
    "Andrew Alvarez",
}

pitchers = []


for date_data in data.get("dates", []):

    for game in date_data.get("games", []):

        teams = game.get("teams", {})

        for side in ["away", "home"]:

            probable = (
                teams
                .get(side, {})
                .get("probablePitcher", {})
            )

            name = probable.get("fullName")
            player_id = probable.get("id")

            if (
                name in targets
                and player_id
            ):

                pitchers.append(
                    (
                        int(player_id),
                        name
                    )
                )


# ---------------------------------------------------------
# LOAD FIRST-INNING HISTORY
# ---------------------------------------------------------

history_end_date = (
    TODAY
    - timedelta(days=1)
)

season_games = (
    get_season_first_innings(
        YEAR,
        history_end_date.isoformat()
    )
)


# ---------------------------------------------------------
# AUDIT EACH PITCHER
# ---------------------------------------------------------

for player_id, name in pitchers:

    print()
    print("=" * 60)

    print(
        f"{name} | MLB ID {player_id}"
    )

    print("=" * 60)


    starts = get_pitcher_starts(
        player_id,
        YEAR
    )

    history = (
        get_pitcher_first_inning_history(
            player_id,
            YEAR,
            season_games
        )
    )


    print(
        f"Starts returned: "
        f"{len(starts)}"
    )

    print(
        f"Starts evaluated: "
        f"{history['starts']}"
    )

    print(
        f"Scoreless firsts: "
        f"{history['scoreless']}"
    )

    print(
        f"Scoreless 1st %: "
        f"{history['scoreless_percent']}"
    )

    print(
        f"Runs/start: "
        f"{history['runs_per_start']}"
    )

    print(
        f"NRFI record: "
        f"{history['nrfi']}-"
        f"{history['yrfi']}"
    )

    print(
        f"NRFI %: "
        f"{history['nrfi_percent']}"
    )