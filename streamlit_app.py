import streamlit as st
import json
from urllib.request import urlopen, Request
from datetime import datetime
from zoneinfo import ZoneInfo


# -------------------------
# PAGE SETUP
# -------------------------

st.set_page_config(
    page_title="SharpReport NRFI Scanner",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ SharpReport NRFI Scanner")
st.subheader("MLB First-Inning Probability Model")

eastern = ZoneInfo("America/New_York")
today = datetime.now(eastern)

st.write(f"Date: {today.strftime('%B %d, %Y')}")

st.divider()


# -------------------------
# GENERAL API FUNCTION
# -------------------------

def get_json(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=20) as response:
        return json.load(response)


# -------------------------
# GET LINEUP FOR ONE TEAM
# -------------------------

def get_team_top4(team_data):

    batting_order = team_data.get("battingOrder", [])
    players = team_data.get("players", {})

    top4 = []

    for player_id in batting_order[:4]:

        key = f"ID{player_id}"

        player_info = players.get(key, {})

        name = (
            player_info
            .get("person", {})
            .get("fullName")
        )

        if name:
            top4.append(name)

    return top4


# -------------------------
# GET GAME LINEUPS
# -------------------------

def get_lineups(game_pk):

    url = (
        f"https://statsapi.mlb.com/api/v1/"
        f"game/{game_pk}/boxscore"
    )

    try:

        data = get_json(url)

        teams = data.get("teams", {})

        away_data = teams.get("away", {})
        home_data = teams.get("home", {})

        away_top4 = get_team_top4(away_data)
        home_top4 = get_team_top4(home_data)

        return away_top4, home_top4

    except Exception:

        # One lineup failure should NOT crash the whole slate
        return [], []


# -------------------------
# FORMAT LINEUP
# -------------------------

def format_lineup(players):

    if len(players) < 4:
        return "Not posted"

    return " | ".join(
        [
            f"1. {players[0]}",
            f"2. {players[1]}",
            f"3. {players[2]}",
            f"4. {players[3]}"
        ]
    )


# -------------------------
# GET MLB SCHEDULE
# -------------------------

def get_mlb_schedule():

    date_string = today.strftime("%Y-%m-%d")

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1"
        f"&date={date_string}"
        f"&hydrate=probablePitcher"
    )

    data = get_json(url)

    dates = data.get("dates", [])

    if not dates:
        return []

    games = []

    for game in dates[0].get("games", []):

        try:

            game_pk = game.get("gamePk")

            away_team_data = (
                game.get("teams", {})
                .get("away", {})
            )

            home_team_data = (
                game.get("teams", {})
                .get("home", {})
            )

            away_team = (
                away_team_data
                .get("team", {})
                .get("name", "Away")
            )

            home_team = (
                home_team_data
                .get("team", {})
                .get("name", "Home")
            )

            away_pitcher = (
                away_team_data
                .get("probablePitcher", {})
                .get("fullName", "TBD")
            )

            home_pitcher = (
                home_team_data
                .get("probablePitcher", {})
                .get("fullName", "TBD")
            )

            game_date = game.get("gameDate")

            if game_date:

                game_time_utc = datetime.fromisoformat(
                    game_date.replace("Z", "+00:00")
                )

                game_time_et = (
                    game_time_utc
                    .astimezone(eastern)
                    .strftime("%I:%M %p ET")
                )

            else:

                game_time_et = "TBD"


            venue = (
                game.get("venue", {})
                .get("name", "TBD")
            )

            status = (
                game.get("status", {})
                .get("detailedState", "Unknown")
            )


            # -------------------------
            # LINEUPS
            # -------------------------

            away_top4 = []
            home_top4 = []

            if game_pk:

                away_top4, home_top4 = (
                    get_lineups(game_pk)
                )


            if (
                len(away_top4) == 4
                and len(home_top4) == 4
            ):

                lineup_status = "✅ Confirmed"

            else:

                lineup_status = "⏳ Waiting"


            games.append(
                {
                    "Game":
                        f"{away_team} @ {home_team}",

                    "Away SP":
                        away_pitcher,

                    "Away Top 4":
                        format_lineup(away_top4),

                    "Home SP":
                        home_pitcher,

                    "Home Top 4":
                        format_lineup(home_top4),

                    "Lineups":
                        lineup_status,

                    "Start Time":
                        game_time_et,

                    "Venue":
                        venue,

                    "Status":
                        status,

                    "Game ID":
                        game_pk
                }
            )

        except Exception as game_error:

            # Even if one game has strange data,
            # continue loading the rest.
            games.append(
                {
                    "Game": "Unable to parse game",
                    "Away SP": "TBD",
                    "Away Top 4": "Not posted",
                    "Home SP": "TBD",
                    "Home Top 4": "Not posted",
                    "Lineups": "⚠️ Data issue",
                    "Start Time": "TBD",
                    "Venue": "TBD",
                    "Status": str(game_error),
                    "Game ID": ""
                }
            )

    return games


# -------------------------
# RUN BUTTON
# -------------------------

if st.button(
    "RUN TODAY'S MLB SLATE",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Loading MLB schedule and lineups..."
    ):

        try:

            games = get_mlb_schedule()

            if games:

                confirmed = sum(
                    1
                    for game in games
                    if game["Lineups"] == "✅ Confirmed"
                )

                waiting = sum(
                    1
                    for game in games
                    if game["Lineups"] == "⏳ Waiting"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "MLB Games",
                    len(games)
                )

                col2.metric(
                    "Confirmed Lineups",
                    confirmed
                )

                col3.metric(
                    "Waiting",
                    waiting
                )

                st.divider()

                st.dataframe(
                    games,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No MLB games were found today."
                )

        except Exception as error:

            st.error(
                "The MLB schedule could not be loaded."
            )

            st.write(
                "Copy the error below and send it to me:"
            )

            st.code(str(error))


st.divider()

st.caption(
    "SharpReport • Data-Driven • Transparent • Consistent"
)