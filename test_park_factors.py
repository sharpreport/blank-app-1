import json
import re

from urllib.request import urlopen, Request
from datetime import datetime
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).date()
YEAR = TODAY.year


# =========================================================
# REQUEST HELPERS
# =========================================================

def get_text(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return response.read().decode(
            "utf-8"
        )


def get_json(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return json.load(response)


# =========================================================
# BASEBALL SAVANT PARK FACTORS
# =========================================================

def get_park_factors(year):

    url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast-park-factors"
        "?type=year"
        f"&year={year}"
        "&batSide="
        "&stat=index_wOBA"
        "&condition=All"
        "&rolling=3"
        "&parks=mlb"
    )

    html = get_text(url)

    match = re.search(
        r"\bdata\s*=\s*(\[.*?\]);",
        html,
        re.DOTALL
    )

    if not match:

        raise Exception(
            "Could not locate Baseball Savant "
            "park-factor data."
        )


    raw_data = json.loads(
        match.group(1)
    )


    park_lookup = {}


    for park in raw_data:

        venue_id = park.get(
            "venue_id"
        )

        if not venue_id:
            continue


        park_lookup[
            int(venue_id)
        ] = {

            "venue_name":
                park.get(
                    "venue_name",
                    ""
                ),

            "team":
                park.get(
                    "name_display_club",
                    ""
                ),

            "runs":
                park.get(
                    "index_runs"
                ),

            "woba":
                park.get(
                    "index_woba"
                ),

            "hr":
                park.get(
                    "index_hr"
                ),
        }


    return park_lookup


# =========================================================
# TODAY'S MLB GAMES
# =========================================================

def get_today_games():

    date_string = TODAY.strftime(
        "%Y-%m-%d"
    )

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_string}"
    )

    data = get_json(url)

    games = []


    for date_data in data.get(
        "dates",
        []
    ):

        for game in date_data.get(
            "games",
            []
        ):

            away = (
                game
                .get("teams", {})
                .get("away", {})
                .get("team", {})
                .get("name", "Away")
            )

            home = (
                game
                .get("teams", {})
                .get("home", {})
                .get("team", {})
                .get("name", "Home")
            )

            venue = game.get(
                "venue",
                {}
            )


            games.append({

                "game":
                    f"{away} @ {home}",

                "venue_id":
                    venue.get("id"),

                "venue_name":
                    venue.get(
                        "name",
                        ""
                    ),
            })


    return games


# =========================================================
# RUN TEST
# =========================================================

park_factors = get_park_factors(
    YEAR
)

games = get_today_games()


output = []

output.append(
    "SHARPREPORT TODAY PARK FACTOR TEST"
)

output.append(
    "=" * 70
)

output.append("")


missing = 0


for game in games:

    venue_id = game[
        "venue_id"
    ]

    park = park_factors.get(
        int(venue_id)
        if venue_id
        else -1
    )


    output.append(
        f"Game: {game['game']}"
    )

    output.append(
        f"MLB Venue: {game['venue_name']}"
    )

    output.append(
        f"MLB Venue ID: {venue_id}"
    )


    if not park:

        missing += 1

        output.append(
            "PARK FACTOR: MISSING"
        )

    else:

        output.append(
            f"Savant Venue: "
            f"{park['venue_name']}"
        )

        output.append(
            f"Run Factor: "
            f"{park['runs']}"
        )

        output.append(
            f"wOBA Factor: "
            f"{park['woba']}"
        )

        output.append(
            f"HR Factor: "
            f"{park['hr']}"
        )


    output.append(
        "-" * 70
    )


output.append("")

output.append(
    f"Games checked: {len(games)}"
)

output.append(
    f"Missing park factors: {missing}"
)


with open(
    "today_park_factors_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )


print("")
print(
    "TODAY PARK FACTOR TEST COMPLETE"
)

print("")

print(
    "Open today_park_factors_output.txt"
)