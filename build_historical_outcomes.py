import csv
import json
import time

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# =========================================================
# SETTINGS
# =========================================================

ET = ZoneInfo("America/New_York")

TODAY = datetime.now(ET).date()

CURRENT_YEAR = TODAY.year

START_YEAR = 2022

END_YEAR = CURRENT_YEAR


# =========================================================
# REQUEST
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
# FIRST-INNING FALLBACK
# =========================================================

def get_first_inning_fallback(
    game_pk
):

    url = (
        "https://statsapi.mlb.com/api/v1.1/"
        f"game/{game_pk}/feed/live"
    )

    try:

        data = get_json(url)

        innings = (
            data
            .get("liveData", {})
            .get("linescore", {})
            .get("innings", [])
        )

        for inning in innings:

            if inning.get("num") == 1:

                away_runs = (
                    inning
                    .get("away", {})
                    .get("runs")
                )

                home_runs = (
                    inning
                    .get("home", {})
                    .get("runs")
                )

                if (
                    away_runs is not None
                    and home_runs is not None
                ):

                    return (
                        int(away_runs),
                        int(home_runs),
                    )

    except Exception:

        pass


    return None


# =========================================================
# SEASON DATE RANGE
# =========================================================

def season_dates(year):

    start_date = datetime(
        year,
        3,
        1
    ).date()

    if year == CURRENT_YEAR:

        end_date = (
            TODAY
            - timedelta(days=1)
        )

    else:

        end_date = datetime(
            year,
            11,
            15
        ).date()

    return (
        start_date,
        end_date
    )


# =========================================================
# DOWNLOAD ONE SEASON
# =========================================================

def get_season_games(year):

    start_date, end_date = (
        season_dates(year)
    )

    rows = []

    current_start = start_date


    while current_start <= end_date:

        current_end = min(
            current_start
            + timedelta(days=29),
            end_date
        )

        start_string = (
            current_start.strftime(
                "%Y-%m-%d"
            )
        )

        end_string = (
            current_end.strftime(
                "%Y-%m-%d"
            )
        )


        print(
            f"{year}: "
            f"{start_string} through "
            f"{end_string}"
        )


        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            "?sportId=1"
            f"&startDate={start_string}"
            f"&endDate={end_string}"
            "&gameType=R"
            "&hydrate=linescore"
        )


        try:

            data = get_json(url)

        except Exception as error:

            print(
                f"Request failed: {error}"
            )

            print(
                "Retrying once..."
            )

            time.sleep(2)

            data = get_json(url)


        for date_data in data.get(
            "dates",
            []
        ):

            official_date = (
                date_data.get(
                    "date",
                    ""
                )
            )


            for game in date_data.get(
                "games",
                []
            ):

                game_pk = game.get(
                    "gamePk"
                )

                status = game.get(
                    "status",
                    {}
                )

                abstract_state = (
                    status.get(
                        "abstractGameState",
                        ""
                    )
                )


                # Only completed games.

                if abstract_state != "Final":
                    continue


                innings = (
                    game
                    .get("linescore", {})
                    .get("innings", [])
                )


                first_inning = None


                for inning in innings:

                    if inning.get(
                        "num"
                    ) == 1:

                        first_inning = inning
                        break


                if not first_inning:

                    fallback = (
                        get_first_inning_fallback(
                            game_pk
                        )
                    )

                    if fallback:

                        (
                            away_runs,
                            home_runs
                        ) = fallback

                    else:

                        continue

                else:

                    away_runs = (
                        first_inning
                        .get("away", {})
                        .get("runs")
                    )

                    home_runs = (
                        first_inning
                        .get("home", {})
                        .get("runs")
                    )


                if (
                    away_runs is None
                    or home_runs is None
                ):

                    continue


                away_runs = int(
                    away_runs
                )

                home_runs = int(
                    home_runs
                )


                teams = game.get(
                    "teams",
                    {}
                )


                away_team = (
                    teams
                    .get("away", {})
                    .get("team", {})
                )

                home_team = (
                    teams
                    .get("home", {})
                    .get("team", {})
                )


                venue = game.get(
                    "venue",
                    {}
                )


                top_scored = (
                    away_runs > 0
                )

                bottom_scored = (
                    home_runs > 0
                )

                nrfi = (
                    away_runs == 0
                    and home_runs == 0
                )

                yrfi = not nrfi


                rows.append({

                    "game_pk":
                        int(game_pk),

                    "date":
                        official_date,

                    "season":
                        year,

                    "away_team_id":
                        away_team.get(
                            "id"
                        ),

                    "away_team":
                        away_team.get(
                            "name",
                            ""
                        ),

                    "home_team_id":
                        home_team.get(
                            "id"
                        ),

                    "home_team":
                        home_team.get(
                            "name",
                            ""
                        ),

                    "venue_id":
                        venue.get(
                            "id"
                        ),

                    "venue_name":
                        venue.get(
                            "name",
                            ""
                        ),

                    "away_1st_runs":
                        away_runs,

                    "home_1st_runs":
                        home_runs,

                    "top_1st_scored":
                        int(
                            top_scored
                        ),

                    "bottom_1st_scored":
                        int(
                            bottom_scored
                        ),

                    "nrfi":
                        int(nrfi),

                    "yrfi":
                        int(yrfi),

                    "total_1st_runs":
                        (
                            away_runs
                            + home_runs
                        ),
                })


        current_start = (
            current_end
            + timedelta(days=1)
        )


        # Be polite to the API.

        time.sleep(0.15)


    return rows


# =========================================================
# BUILD DATABASE
# =========================================================

all_rows = []


print()
print(
    "BUILDING SHARPREPORT "
    "HISTORICAL NRFI DATABASE"
)

print(
    "=" * 60
)

print()


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    print()
    print(
        f"LOADING {year}"
    )

    print(
        "-" * 60
    )


    season_rows = (
        get_season_games(
            year
        )
    )


    print(
        f"{year} completed games: "
        f"{len(season_rows)}"
    )


    all_rows.extend(
        season_rows
    )


# =========================================================
# REMOVE DUPLICATES
# =========================================================

unique_games = {}


for row in all_rows:

    unique_games[
        row["game_pk"]
    ] = row


all_rows = list(
    unique_games.values()
)


all_rows.sort(
    key=lambda row: (
        row["date"],
        row["game_pk"],
    )
)


# =========================================================
# SAVE CSV
# =========================================================

csv_file = (
    "historical_first_innings.csv"
)


fieldnames = [

    "game_pk",
    "date",
    "season",

    "away_team_id",
    "away_team",

    "home_team_id",
    "home_team",

    "venue_id",
    "venue_name",

    "away_1st_runs",
    "home_1st_runs",

    "top_1st_scored",
    "bottom_1st_scored",

    "nrfi",
    "yrfi",

    "total_1st_runs",
]


with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


# =========================================================
# SUMMARY
# =========================================================

summary = []

summary.append(
    "SHARPREPORT HISTORICAL "
    "FIRST-INNING DATABASE"
)

summary.append(
    "=" * 60
)

summary.append("")

summary.append(
    f"Total games: "
    f"{len(all_rows)}"
)

summary.append("")


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    year_rows = [

        row

        for row in all_rows

        if row[
            "season"
        ] == year
    ]


    total = len(
        year_rows
    )


    nrfi_count = sum(
        row["nrfi"]
        for row in year_rows
    )


    yrfi_count = sum(
        row["yrfi"]
        for row in year_rows
    )


    top_scored = sum(
        row[
            "top_1st_scored"
        ]
        for row in year_rows
    )


    bottom_scored = sum(
        row[
            "bottom_1st_scored"
        ]
        for row in year_rows
    )


    if total:

        nrfi_rate = (
            nrfi_count
            / total
            * 100
        )

        yrfi_rate = (
            yrfi_count
            / total
            * 100
        )

        top_rate = (
            top_scored
            / total
            * 100
        )

        bottom_rate = (
            bottom_scored
            / total
            * 100
        )

    else:

        nrfi_rate = 0
        yrfi_rate = 0
        top_rate = 0
        bottom_rate = 0


    summary.append(
        f"{year}"
    )

    summary.append(
        f"Games: {total}"
    )

    summary.append(
        f"NRFI: "
        f"{nrfi_count} "
        f"({nrfi_rate:.1f}%)"
    )

    summary.append(
        f"YRFI: "
        f"{yrfi_count} "
        f"({yrfi_rate:.1f}%)"
    )

    summary.append(
        f"Top 1st scored: "
        f"{top_rate:.1f}%"
    )

    summary.append(
        f"Bottom 1st scored: "
        f"{bottom_rate:.1f}%"
    )

    summary.append(
        "-" * 60
    )


with open(
    "historical_outcomes_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary)
    )


print()
print(
    "HISTORICAL DATABASE COMPLETE"
)

print()

print(
    "Created:"
)

print(
    "historical_first_innings.csv"
)

print(
    "historical_outcomes_summary.txt"
)