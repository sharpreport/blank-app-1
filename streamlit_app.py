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

# Use Eastern Time
eastern = ZoneInfo("America/New_York")
today = datetime.now(eastern)

st.write(f"Date: {today.strftime('%B %d, %Y')}")

st.divider()


# -------------------------
# MLB SCHEDULE FUNCTION
# -------------------------

def get_mlb_schedule():

    date_string = today.strftime("%Y-%m-%d")

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_string}&hydrate=probablePitcher"
    )

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=15) as response:
        data = json.load(response)

    games = []

    if not data.get("dates"):
        return games

    for game in data["dates"][0]["games"]:

        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]

        away_pitcher = (
            game["teams"]["away"]
            .get("probablePitcher", {})
            .get("fullName", "TBD")
        )

        home_pitcher = (
            game["teams"]["home"]
            .get("probablePitcher", {})
            .get("fullName", "TBD")
        )

        game_time_utc = datetime.fromisoformat(
            game["gameDate"].replace("Z", "+00:00")
        )

        game_time_et = game_time_utc.astimezone(eastern)

        status = game["status"]["detailedState"]

        venue = game.get("venue", {}).get("name", "TBD")

        games.append(
            {
                "Game": f"{away_team} @ {home_team}",
                "Away SP": away_pitcher,
                "Home SP": home_pitcher,
                "Start Time": game_time_et.strftime("%I:%M %p ET"),
                "Venue": venue,
                "Status": status,
                "Game ID": game["gamePk"],
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

    with st.spinner("Loading today's MLB games..."):

        try:

            games = get_mlb_schedule()

            if games:

                st.success(
                    f"Found {len(games)} MLB games."
                )

                st.dataframe(
                    games,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No MLB games were found for today."
                )

        except Exception as error:

            st.error(
                "There was a problem retrieving the MLB schedule."
            )

            st.write("Error details:")

            st.code(str(error))


st.divider()

st.caption(
    "SharpReport • Data-Driven • Transparent • Consistent"
)