import streamlit as st
import pandas as pd
import json
import io

from urllib.request import urlopen, Request
from datetime import datetime
from zoneinfo import ZoneInfo





# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="SharpReport NRFI Scanner",
    layout="wide",
)

st.title("⚾ SharpReport NRFI Scanner")

st.write(
    "Automatic MLB schedule, confirmed lineups, "
    "starting pitchers, and NRFI model inputs."
)

ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).date()
YEAR = TODAY.year


# =========================================================
# BASIC JSON / CSV REQUESTS
# =========================================================

def get_json(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(
        request,
        timeout=20
    ) as response:

        return json.load(response)


def get_csv(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        text = response.read().decode(
            "utf-8"
        )

    data = pd.read_csv(
        io.StringIO(text)
    )

    data.columns = (
        data.columns.str.strip()
    )

    return data


# =========================================================
# LINEUPS
# =========================================================

def get_team_top4(team_data):

    batting_order = team_data.get(
        "battingOrder",
        []
    )

    players = team_data.get(
        "players",
        {}
    )

    top4 = []

    for player_id in batting_order[:4]:

        player = players.get(
            f"ID{player_id}",
            {}
        )

        name = (
            player
            .get("person", {})
            .get("fullName")
        )

        if name:

            top4.append({
                "id": int(player_id),
                "name": name,
            })

    return top4


def get_lineups(game_pk):

    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/game/{game_pk}/boxscore"
    )

    try:

        data = get_json(url)

        teams = data.get(
            "teams",
            {}
        )

        away_top4 = get_team_top4(
            teams.get("away", {})
        )

        home_top4 = get_team_top4(
            teams.get("home", {})
        )

        return away_top4, home_top4

    except Exception:

        return [], []


def format_lineup(lineup):

    if not lineup:
        return "Not posted"

    return " | ".join(
        player["name"]
        for player in lineup
    )


# =========================================================
# BASEBALL SAVANT PITCHER DATA
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_savant_pitcher_data(year):

    expected_url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/expected_statistics"
        "?type=pitcher"
        f"&year={year}"
        "&position="
        "&team="
        "&filterType=pa"
        "&min=1"
        "&csv=true"
    )

    barrel_url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast"
        "?type=pitcher"
        f"&year={year}"
        "&position="
        "&team="
        "&min=1"
        "&csv=true"
    )

    expected = get_csv(
        expected_url
    )

    barrels = get_csv(
        barrel_url
    )

    xwoba = {}
    barrel_percent = {}

    for _, row in expected.iterrows():

        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "est_woba"
        )

        if (
            pd.notna(player_id)
            and pd.notna(value)
        ):

            xwoba[
                int(player_id)
            ] = float(value)


    for _, row in barrels.iterrows():

        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "brl_percent"
        )

        if (
            pd.notna(player_id)
            and pd.notna(value)
        ):

            barrel_percent[
                int(player_id)
            ] = float(value)


    return (
        xwoba,
        barrel_percent
    )


# =========================================================
# BASEBALL SAVANT HITTER DATA
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_savant_hitter_data(year):

    expected_url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/expected_statistics"
        "?type=batter"
        f"&year={year}"
        "&position="
        "&team="
        "&filterType=pa"
        "&min=1"
        "&csv=true"
    )

    barrel_url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast"
        "?type=batter"
        f"&year={year}"
        "&position="
        "&team="
        "&min=1"
        "&csv=true"
    )

    expected = get_csv(
        expected_url
    )

    barrels = get_csv(
        barrel_url
    )

    xwoba = {}
    barrel_percent = {}

    for _, row in expected.iterrows():

        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "est_woba"
        )

        if (
            pd.notna(player_id)
            and pd.notna(value)
        ):

            xwoba[
                int(player_id)
            ] = float(value)


    for _, row in barrels.iterrows():

        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "brl_percent"
        )

        if (
            pd.notna(player_id)
            and pd.notna(value)
        ):

            barrel_percent[
                int(player_id)
            ] = float(value)


    return (
        xwoba,
        barrel_percent
    )


# =========================================================
# MLB PITCHER K% / BB%
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_mlb_pitcher_rates(
    player_id,
    year
):

    if not player_id:
        return None, None, None


    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/people/{player_id}/stats"
        "?stats=season"
        "&group=pitching"
        f"&season={year}"
    )


    try:

        data = get_json(url)

        stat = (
            data["stats"][0]
            ["splits"][0]
            ["stat"]
        )

        strikeouts = stat.get(
            "strikeOuts",
            0
        )

        walks = stat.get(
            "baseOnBalls",
            0
        )

        batters_faced = stat.get(
            "battersFaced",
            0
        )


        if not batters_faced:
            return None, None, 0


        k_percent = (
            strikeouts
            / batters_faced
            * 100
        )

        bb_percent = (
            walks
            / batters_faced
            * 100
        )


        return (
            k_percent,
            bb_percent,
            batters_faced
        )


    except Exception:

        return None, None, None


# =========================================================
# MLB HITTER K% / BB%
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_mlb_hitter_rates(
    player_id,
    year
):

    if not player_id:
        return None, None, None


    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/people/{player_id}/stats"
        "?stats=season"
        "&group=hitting"
        f"&season={year}"
    )


    try:

        data = get_json(url)

        stat = (
            data["stats"][0]
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


        if not plate_appearances:
            return None, None, 0


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


        return (
            k_percent,
            bb_percent,
            plate_appearances
        )


    except Exception:

        return None, None, None


# =========================================================
# MLB SCHEDULE
# =========================================================

def get_mlb_schedule(date):

    date_string = date.strftime(
        "%Y-%m-%d"
    )

    url = (
        "https://statsapi.mlb.com/"
        "api/v1/schedule"
        "?sportId=1"
        f"&date={date_string}"
        "&hydrate=probablePitcher"
    )

    data = get_json(url)

    games = []

    dates = data.get(
        "dates",
        []
    )

    if not dates:
        return games


    for game in dates[0].get(
        "games",
        []
    ):

        try:

            game_pk = game.get(
                "gamePk"
            )


            away_team = (
                game
                .get("teams", {})
                .get("away", {})
                .get("team", {})
                .get("name", "Away")
            )

            home_team = (
                game
                .get("teams", {})
                .get("home", {})
                .get("team", {})
                .get("name", "Home")
            )


            away_probable = (
                game
                .get("teams", {})
                .get("away", {})
                .get("probablePitcher", {})
            )

            home_probable = (
                game
                .get("teams", {})
                .get("home", {})
                .get("probablePitcher", {})
            )


            away_pitcher = (
                away_probable.get(
                    "fullName",
                    "TBA"
                )
            )

            away_pitcher_id = (
                away_probable.get("id")
            )


            home_pitcher = (
                home_probable.get(
                    "fullName",
                    "TBA"
                )
            )

            home_pitcher_id = (
                home_probable.get("id")
            )


            (
                away_top4,
                home_top4
            ) = get_lineups(
                game_pk
            )


            if (
                len(away_top4) == 4
                and
                len(home_top4) == 4
            ):

                lineup_status = (
                    "✅ Confirmed"
                )

            else:

                lineup_status = (
                    "⏳ Waiting"
                )


            game_date = game.get(
                "gameDate"
            )


            if game_date:

                game_time = (
                    datetime
                    .fromisoformat(
                        game_date.replace(
                            "Z",
                            "+00:00"
                        )
                    )
                    .astimezone(ET)
                    .strftime(
                        "%I:%M %p ET"
                    )
                    .lstrip("0")
                )

            else:

                game_time = "TBA"


            venue = (
                game
                .get("venue", {})
                .get("name", "")
            )

            status = (
                game
                .get("status", {})
                .get(
                    "detailedState",
                    ""
                )
            )


            games.append({

                "Game":
                    f"{away_team} @ "
                    f"{home_team}",

                "Away Team":
                    away_team,

                "Away SP":
                    away_pitcher,

                "Away SP ID":
                    away_pitcher_id,

                "Away Top 4":
                    away_top4,

                "Home Team":
                    home_team,

                "Home SP":
                    home_pitcher,

                "Home SP ID":
                    home_pitcher_id,

                "Home Top 4":
                    home_top4,

                "Lineups":
                    lineup_status,

                "Start Time":
                    game_time,

                "Venue":
                    venue,

                "Status":
                    status,

                "Game ID":
                    game_pk,
            })


        except Exception as error:

            st.warning(
                "One game could not be "
                f"processed: {error}"
            )


    return games


# =========================================================
# FORMATTERS
# =========================================================

def format_xwoba(value):

    if value is None:
        return "—"

    return f"{value:.3f}"


def format_percent(value):

    if value is None:
        return "—"

    return f"{value:.1f}%"


def format_number(value):

    if value is None:
        return "—"

    return int(value)


def strict_average(values):

    valid = [
        value
        for value in values
        if value is not None
    ]

    if len(valid) != 4:
        return None

    return sum(valid) / 4


# =========================================================
# PITCHER TABLE
# =========================================================

def build_pitcher_table(
    games,
    xwoba_data,
    barrel_data
):

    rows = []


    for game in games:

        pitchers = [

            (
                game["Away Team"],
                game["Away SP"],
                game["Away SP ID"],
                game["Game"],
            ),

            (
                game["Home Team"],
                game["Home SP"],
                game["Home SP ID"],
                game["Game"],
            ),
        ]


        for (
            team,
            pitcher_name,
            pitcher_id,
            matchup
        ) in pitchers:


            if not pitcher_id:

                rows.append({

                    "Game": matchup,
                    "Team": team,
                    "Pitcher": pitcher_name,
                    "xwOBA Allowed": "—",
                    "K%": "—",
                    "BB%": "—",
                    "Barrel% Allowed": "—",
                    "Batters Faced": "—",
                })

                continue


            (
                k_percent,
                bb_percent,
                batters_faced
            ) = get_mlb_pitcher_rates(
                pitcher_id,
                YEAR
            )


            xwoba = xwoba_data.get(
                int(pitcher_id)
            )

            barrel = barrel_data.get(
                int(pitcher_id)
            )


            rows.append({

                "Game":
                    matchup,

                "Team":
                    team,

                "Pitcher":
                    pitcher_name,

                "xwOBA Allowed":
                    format_xwoba(
                        xwoba
                    ),

                "K%":
                    format_percent(
                        k_percent
                    ),

                "BB%":
                    format_percent(
                        bb_percent
                    ),

                "Barrel% Allowed":
                    format_percent(
                        barrel
                    ),

                "Batters Faced":
                    format_number(
                        batters_faced
                    ),
            })


    return rows


# =========================================================
# HITTER TABLES
# =========================================================

def build_hitter_tables(
    games,
    hitter_xwoba,
    hitter_barrels
):

    individual_rows = []
    team_rows = []


    for game in games:

        offenses = [

            {
                "team":
                    game["Away Team"],

                "opponent_sp":
                    game["Home SP"],

                "top4":
                    game["Away Top 4"],
            },

            {
                "team":
                    game["Home Team"],

                "opponent_sp":
                    game["Away SP"],

                "top4":
                    game["Home Top 4"],
            },
        ]


        for offense in offenses:

            team = offense["team"]

            opponent_sp = (
                offense["opponent_sp"]
            )

            top4 = offense["top4"]


            xwoba_values = []
            k_values = []
            bb_values = []
            barrel_values = []

            total_pa = 0


            for batting_spot, player in enumerate(
                top4,
                start=1
            ):

                player_id = player["id"]
                player_name = player["name"]


                (
                    k_percent,
                    bb_percent,
                    plate_appearances
                ) = get_mlb_hitter_rates(
                    player_id,
                    YEAR
                )


                xwoba = hitter_xwoba.get(
                    player_id
                )

                barrel = hitter_barrels.get(
                    player_id
                )


                xwoba_values.append(
                    xwoba
                )

                k_values.append(
                    k_percent
                )

                bb_values.append(
                    bb_percent
                )

                barrel_values.append(
                    barrel
                )


                if plate_appearances:
                    total_pa += plate_appearances


                individual_rows.append({

                    "Game":
                        game["Game"],

                    "Team":
                        team,

                    "Batting Spot":
                        batting_spot,

                    "Hitter":
                        player_name,

                    "xwOBA":
                        format_xwoba(
                            xwoba
                        ),

                    "K%":
                        format_percent(
                            k_percent
                        ),

                    "BB%":
                        format_percent(
                            bb_percent
                        ),

                    "Barrel%":
                        format_percent(
                            barrel
                        ),

                    "PA":
                        format_number(
                            plate_appearances
                        ),
                })


            team_xwoba = strict_average(
                xwoba_values
            )

            team_k = strict_average(
                k_values
            )

            team_bb = strict_average(
                bb_values
            )

            team_barrel = strict_average(
                barrel_values
            )


            team_rows.append({

                "Game":
                    game["Game"],

                "Offense":
                    team,

                "Opposing SP":
                    opponent_sp,

                "Top-4 xwOBA":
                    format_xwoba(
                        team_xwoba
                    ),

                "Top-4 K%":
                    format_percent(
                        team_k
                    ),

                "Top-4 BB%":
                    format_percent(
                        team_bb
                    ),

                "Top-4 Barrel%":
                    format_percent(
                        team_barrel
                    ),

                "Combined Top-4 PA":
                    total_pa,
            })


    return (
        team_rows,
        individual_rows
    )


# =========================================================
# APP
# =========================================================

st.write(
    f"**Date:** "
    f"{TODAY.strftime('%B %d, %Y')}"
)


if st.button(
    "RUN TODAY'S MLB SLATE",
    type="primary"
):

    try:

        with st.spinner(
            "Loading MLB slate and "
            "confirmed lineups..."
        ):

            games = get_mlb_schedule(
                TODAY
            )


        if not games:

            st.info(
                "No MLB games were found "
                "for today."
            )


        else:

            confirmed = sum(
                1
                for game in games
                if game["Lineups"]
                == "✅ Confirmed"
            )

            waiting = (
                len(games)
                - confirmed
            )


            # ---------------------------------------------
            # MLB SLATE
            # ---------------------------------------------

            st.subheader(
                "Today's MLB Slate"
            )

            st.write(
                f"**MLB Games:** "
                f"{len(games)}"
            )

            st.write(
                f"**Confirmed Lineups:** "
                f"{confirmed}"
            )

            st.write(
                f"**Waiting:** "
                f"{waiting}"
            )


            game_table = []

            for game in games:

                game_table.append({

                    "Game":
                        game["Game"],

                    "Away SP":
                        game["Away SP"],

                    "Away Top 4":
                        format_lineup(
                            game["Away Top 4"]
                        ),

                    "Home SP":
                        game["Home SP"],

                    "Home Top 4":
                        format_lineup(
                            game["Home Top 4"]
                        ),

                    "Lineups":
                        game["Lineups"],

                    "Start Time":
                        game["Start Time"],

                    "Venue":
                        game["Venue"],

                    "Status":
                        game["Status"],

                    "Game ID":
                        game["Game ID"],
                })


            st.dataframe(
                pd.DataFrame(
                    game_table
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # PITCHER DATA
            # ---------------------------------------------

            st.subheader(
                "Starting Pitcher Model Inputs"
            )

            st.caption(
                "Season-to-date pitcher "
                "statistics."
            )


            with st.spinner(
                "Loading pitcher metrics..."
            ):

                (
                    pitcher_xwoba,
                    pitcher_barrels
                ) = get_savant_pitcher_data(
                    YEAR
                )


                pitcher_rows = (
                    build_pitcher_table(
                        games,
                        pitcher_xwoba,
                        pitcher_barrels,
                    )
                )


            st.dataframe(
                pd.DataFrame(
                    pitcher_rows
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # HITTER DATA
            # ---------------------------------------------

            st.subheader(
                "Top-4 Offensive Model Inputs"
            )

            st.caption(
                "Equal-weight average of the "
                "first four confirmed hitters "
                "in each batting order."
            )


            with st.spinner(
                "Loading top-four hitter "
                "metrics..."
            ):

                (
                    hitter_xwoba,
                    hitter_barrels
                ) = get_savant_hitter_data(
                    YEAR
                )


                (
                    team_hitter_rows,
                    individual_hitter_rows
                ) = build_hitter_tables(
                    games,
                    hitter_xwoba,
                    hitter_barrels,
                )


            st.dataframe(
                pd.DataFrame(
                    team_hitter_rows
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # INDIVIDUAL HITTER DETAIL
            # ---------------------------------------------

            with st.expander(
                "View Individual Top-4 "
                "Hitter Metrics"
            ):

                st.dataframe(
                    pd.DataFrame(
                        individual_hitter_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


    except Exception as error:

        st.error(
            "The scanner could not "
            "complete today's slate."
        )

        st.exception(error)