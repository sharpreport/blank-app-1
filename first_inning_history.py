import json

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from functools import lru_cache


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
# SEASON FIRST-INNING RESULTS
# =========================================================

@lru_cache(maxsize=8)
def get_season_first_innings(
    year,
    end_date_string
):

    end_date = datetime.strptime(
        end_date_string,
        "%Y-%m-%d"
    ).date()

    start_date = datetime(
        year,
        3,
        1
    ).date()

    games = {}

    current_start = start_date


    while current_start <= end_date:

        current_end = min(
            current_start
            + timedelta(days=29),
            end_date
        )

        start_string = current_start.strftime(
            "%Y-%m-%d"
        )

        end_string = current_end.strftime(
            "%Y-%m-%d"
        )

        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
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

                first_inning = None

                for inning in (
                    game
                    .get("linescore", {})
                    .get("innings", [])
                ):

                    if inning.get("num") == 1:

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

@lru_cache(maxsize=256)
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
            stat.get("gamesStarted", 0) == 1
            and game_pk
            and team_id
        ):

            starts.append({

                "game_pk":
                    int(game_pk),

                "team_id":
                    int(team_id),
            })


    return starts


# =========================================================
# PITCHER FIRST-INNING HISTORY
# =========================================================

def get_pitcher_first_inning_history(
    player_id,
    year,
    first_inning_games
):

    starts = get_pitcher_starts(
        int(player_id),
        year
    )

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


    total_starts = len(
        evaluated
    )

    scoreless = sum(
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
        total_starts
        - nrfi_games
    )


    if total_starts:

        scoreless_percent = (
            scoreless
            / total_starts
            * 100
        )

        runs_per_start = (
            runs_allowed
            / total_starts
        )

        nrfi_percent = (
            nrfi_games
            / total_starts
            * 100
        )

    else:

        scoreless_percent = None
        runs_per_start = None
        nrfi_percent = None


    return {

        "starts":
            total_starts,

        "scoreless":
            scoreless,

        "scoreless_percent":
            scoreless_percent,

        "runs_per_start":
            runs_per_start,

        "nrfi":
            nrfi_games,

        "yrfi":
            yrfi_games,

        "nrfi_percent":
            nrfi_percent,
    }