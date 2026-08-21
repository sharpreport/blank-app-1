import json
from urllib.request import urlopen, Request
from datetime import datetime
from zoneinfo import ZoneInfo

from pybaseball import (
    statcast_batter_expected_stats,
    statcast_batter_exitvelo_barrels,
)


ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).date()
YEAR = TODAY.year


def get_json(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=20) as response:
        return json.load(response)


output = []

output.append("SHARPREPORT HITTER METRICS TEST")
output.append("=" * 50)
output.append("")


# --------------------------------------------------
# FIND FIRST CONFIRMED HITTER FROM TODAY'S SLATE
# --------------------------------------------------

date_string = TODAY.strftime("%Y-%m-%d")

schedule_url = (
    "https://statsapi.mlb.com/api/v1/schedule"
    f"?sportId=1&date={date_string}"
)

schedule = get_json(schedule_url)

player_id = None
player_name = None


for date_data in schedule.get("dates", []):

    for game in date_data.get("games", []):

        game_pk = game.get("gamePk")

        boxscore_url = (
            "https://statsapi.mlb.com/api/v1/"
            f"game/{game_pk}/boxscore"
        )

        try:

            boxscore = get_json(boxscore_url)

            away = (
                boxscore
                .get("teams", {})
                .get("away", {})
            )

            batting_order = away.get(
                "battingOrder",
                []
            )

            players = away.get(
                "players",
                {}
            )

            if batting_order:

                player_id = batting_order[0]

                player_name = (
                    players
                    .get(
                        f"ID{player_id}",
                        {}
                    )
                    .get("person", {})
                    .get(
                        "fullName",
                        "Unknown"
                    )
                )

                break

        except Exception:
            pass

    if player_id:
        break


if not player_id:
    raise Exception(
        "No confirmed hitter was found."
    )


output.append(
    f"Test Hitter: {player_name}"
)

output.append(
    f"MLB Player ID: {player_id}"
)

output.append("")


# --------------------------------------------------
# MLB HITTING STATS
# --------------------------------------------------

stats_url = (
    "https://statsapi.mlb.com/api/v1/"
    f"people/{player_id}/stats"
    "?stats=season"
    "&group=hitting"
    f"&season={YEAR}"
)

stats_data = get_json(stats_url)

stat = (
    stats_data["stats"][0]
    ["splits"][0]
    ["stat"]
)

plate_appearances = stat.get(
    "plateAppearances",
    0
)

strikeouts = stat.get(
    "strikeOuts",
    0
)

walks = stat.get(
    "baseOnBalls",
    0
)


if plate_appearances:

    k_percent = (
        strikeouts
        / plate_appearances
        * 100
    )

    bb_percent = (
        walks
        / plate_appearances
        * 100
    )

else:

    k_percent = None
    bb_percent = None


# --------------------------------------------------
# BASEBALL SAVANT EXPECTED STATS
# --------------------------------------------------

expected = (
    statcast_batter_expected_stats(
        YEAR,
        minPA=1
    )
)

expected_row = expected[
    expected["player_id"] == player_id
]


if not expected_row.empty:

    xwoba = float(
        expected_row.iloc[0]["est_woba"]
    )

else:

    xwoba = None


# --------------------------------------------------
# BASEBALL SAVANT BARREL DATA
# --------------------------------------------------

barrels = (
    statcast_batter_exitvelo_barrels(
        YEAR,
        minBBE=1
    )
)

barrel_row = barrels[
    barrels["player_id"] == player_id
]


if not barrel_row.empty:

    barrel_percent = float(
        barrel_row.iloc[0]["brl_percent"]
    )

else:

    barrel_percent = None


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

output.append(
    f"Plate Appearances: {plate_appearances}"
)

output.append(
    f"Strikeouts: {strikeouts}"
)

output.append(
    f"Walks: {walks}"
)

output.append("")

output.append(
    f"xwOBA: "
    f"{xwoba:.3f}"
    if xwoba is not None
    else "xwOBA: MISSING"
)

output.append(
    f"K%: "
    f"{k_percent:.1f}%"
    if k_percent is not None
    else "K%: MISSING"
)

output.append(
    f"BB%: "
    f"{bb_percent:.1f}%"
    if bb_percent is not None
    else "BB%: MISSING"
)

output.append(
    f"Barrel%: "
    f"{barrel_percent:.1f}%"
    if barrel_percent is not None
    else "Barrel%: MISSING"
)


with open(
    "hitter_metrics_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(output))


print("")
print("HITTER TEST COMPLETE")
print("")
print(
    "Open hitter_metrics_output.txt"
)