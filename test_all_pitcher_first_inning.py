import json

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


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
        timeout=45
    ) as response:

        return json.load(response)


# =========================================================
# TODAY'S STARTING PITCHERS
# =========================================================

def get_today_pitchers():

    date_string = TODAY.strftime(
        "%Y-%m-%d"
    )

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_string}"
        "&hydrate=probablePitcher"
    )

    data = get_json(url)

    pitchers = []

    seen = set()

    for date_data in data.get(
        "dates",
        []
    ):

        for game in date_data.get(
            "games",
            []
        ):

            teams = game.get(
                "teams",
                {}
            )

            for side in [
                "away",
                "home"
            ]:

                pitcher = (
                    teams
                    .get(side, {})
                    .get(
                        "probablePitcher",
                        {}
                    )
                )

                player_id = pitcher.get(
                    "id"
                )

                player_name = pitcher.get(
                    "fullName"
                )

                if (
                    player_id
                    and player_name
                    and player_id not in seen
                ):

                    seen.add(player_id)

                    pitchers.append({
                        "id": int(player_id),
                        "name": player_name,
                    })

    return pitchers


# =========================================================
# LOAD SEASON FIRST-INNING SCORES
# =========================================================

def get_season_first_innings(
    year
):

    games = {}

    start_date = datetime(
        year,
        3,
        1
    ).date()

    end_date = TODAY

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
            "Loading games "
            f"{start_string} "
            f"through {end_string}..."
        )

        url = (
            "https://statsapi.mlb.com/"
            "api/v1/schedule"
            "?sportId=1"
            f"&startDate={start_string}"
            f"&endDate={end_string}"
            "&gameType=R"
            "&hydrate=linescore"
        )

        data = get_json(url)


        for date_data in data.get(
            "dates",
            []
        ):

            for game in date_data.get(
                "games",
                []
            ):

                game_pk = game.get(
                    "gamePk"
                )

                teams = game.get(
                    "teams",
                    {}
                )

                away_team_id = (
                    teams
                    .get("away", {})
                    .get("team", {})
                    .get("id")
                )

                home_team_id = (
                    teams
                    .get("home", {})
                    .get("team", {})
                    .get("id")
                )

                linescore = game.get(
                    "linescore",
                    {}
                )

                first_inning = None


                for inning in linescore.get(
                    "innings",
                    []
                ):

                    if inning.get(
                        "num"
                    ) == 1:

                        first_inning = inning
                        break


                if (
                    not game_pk
                    or not first_inning
                ):

                    continue


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


                games[int(game_pk)] = {

                    "away_team_id":
                        away_team_id,

                    "home_team_id":
                        home_team_id,

                    "away_runs":
                        int(away_runs),

                    "home_runs":
                        int(home_runs),
                }


        current_start = (
            current_end
            + timedelta(days=1)
        )


    return games


# =========================================================
# PITCHER START LOG
# =========================================================

def get_pitcher_starts(
    player_id,
    year
):

    url = (
        "https://statsapi.mlb.com/api/v1/"
        f"people/{player_id}/stats"
        "?stats=gameLog"
        "&group=pitching"
        f"&season={year}"
        "&gameType=R"
    )

    data = get_json(url)

    starts = []


    stats_groups = data.get(
        "stats",
        []
    )


    if not stats_groups:

        return starts


    for split in stats_groups[0].get(
        "splits",
        []
    ):

        stat = split.get(
            "stat",
            {}
        )

        games_started = stat.get(
            "gamesStarted",
            0
        )

        game_pk = (
            split
            .get("game", {})
            .get("gamePk")
        )

        team_id = (
            split
            .get("team", {})
            .get("id")
        )


        if (
            games_started == 1
            and game_pk
            and team_id
        ):

            starts.append({

                "game_pk":
                    int(game_pk),

                "team_id":
                    int(team_id),

                "date":
                    split.get(
                        "date",
                        ""
                    ),
            })


    return starts


# =========================================================
# CALCULATE HISTORY
# =========================================================

def calculate_history(
    starts,
    first_inning_games
):

    evaluated = []


    for start in starts:

        game = first_inning_games.get(
            start["game_pk"]
        )

        if not game:

            continue


        team_id = start["team_id"]


        if (
            team_id
            == game["away_team_id"]
        ):

            opponent_runs = (
                game["home_runs"]
            )

        elif (
            team_id
            == game["home_team_id"]
        ):

            opponent_runs = (
                game["away_runs"]
            )

        else:

            continue


        nrfi = (
            game["away_runs"] == 0
            and game["home_runs"] == 0
        )


        evaluated.append({

            "opponent_runs":
                opponent_runs,

            "nrfi":
                nrfi,
        })


    starts_evaluated = len(
        evaluated
    )

    scoreless_firsts = sum(
        1
        for game in evaluated
        if game["opponent_runs"] == 0
    )

    runs_allowed = sum(
        game["opponent_runs"]
        for game in evaluated
    )

    nrfi_games = sum(
        1
        for game in evaluated
        if game["nrfi"]
    )


    yrfi_games = (
        starts_evaluated
        - nrfi_games
    )


    if starts_evaluated:

        scoreless_percent = (
            scoreless_firsts
            / starts_evaluated
            * 100
        )

        runs_per_start = (
            runs_allowed
            / starts_evaluated
        )

        nrfi_percent = (
            nrfi_games
            / starts_evaluated
            * 100
        )

    else:

        scoreless_percent = None
        runs_per_start = None
        nrfi_percent = None


    return {

        "starts":
            starts_evaluated,

        "scoreless":
            scoreless_firsts,

        "scoreless_percent":
            scoreless_percent,

        "runs_allowed":
            runs_allowed,

        "runs_per_start":
            runs_per_start,

        "nrfi":
            nrfi_games,

        "yrfi":
            yrfi_games,

        "nrfi_percent":
            nrfi_percent,
    }


# =========================================================
# RUN TEST
# =========================================================

print("")
print(
    "Loading season first-inning data..."
)
print("")


first_inning_games = (
    get_season_first_innings(
        YEAR
    )
)


pitchers = get_today_pitchers()


output = []

output.append(
    "SHARPREPORT ALL-PITCHER "
    "FIRST-INNING TEST"
)

output.append(
    "=" * 70
)

output.append("")


for pitcher in pitchers:

    print(
        f"Calculating "
        f"{pitcher['name']}..."
    )


    starts = get_pitcher_starts(
        pitcher["id"],
        YEAR
    )


    history = calculate_history(
        starts,
        first_inning_games
    )


    output.append(
        f"Pitcher: "
        f"{pitcher['name']}"
    )

    output.append(
        f"Starts: "
        f"{history['starts']}"
    )

    output.append(
        "Scoreless 1st: "
        f"{history['scoreless']}"
    )


    if (
        history[
            "scoreless_percent"
        ]
        is not None
    ):

        output.append(
            "Scoreless 1st %: "
            f"{history['scoreless_percent']:.1f}%"
        )

        output.append(
            "1st-Inning Runs/Start: "
            f"{history['runs_per_start']:.2f}"
        )

        output.append(
            "NRFI Record: "
            f"{history['nrfi']}-"
            f"{history['yrfi']}"
        )

        output.append(
            "NRFI %: "
            f"{history['nrfi_percent']:.1f}%"
        )

    else:

        output.append(
            "Scoreless 1st %: MISSING"
        )

        output.append(
            "1st-Inning Runs/Start: MISSING"
        )

        output.append(
            "NRFI Record: MISSING"
        )

        output.append(
            "NRFI %: MISSING"
        )


    output.append(
        "-" * 70
    )


with open(
    "all_pitcher_first_inning_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )


print("")
print(
    "ALL-PITCHER TEST COMPLETE"
)

print("")

print(
    "Open "
    "all_pitcher_first_inning_output.txt"
)