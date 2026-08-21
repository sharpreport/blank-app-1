import streamlit as st
import pandas as pd
import json
import io

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from first_inning_history import (
    get_season_first_innings,
    get_pitcher_first_inning_history,
)

from game_weather import (
    get_venue_info,
    get_game_weather,
    weather_usage,
)

from park_factors import (
    get_park_factors,
    classify_run_factor,
)

from model_inputs import (
    build_half_inning_model_table,
)

from nrfi_probability_model import (
    load_nrfi_model,
    build_half_features,
    predict_nrfi_yrfi,
)

from nrfi_market_odds import (
    get_mlb_events,
    get_first_inning_event_odds,
    find_odds_event,
    parse_first_inning_market,
    summarize_market,
    american_implied_probability,
)

from nrfi_data_logger import (
    save_slate_snapshot,
)




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
    "starting pitchers, and trained NRFI/YRFI probabilities."
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


            venue_data = game.get(
                "venue",
                {}
            )

            venue = venue_data.get(
                "name",
                ""
            )

            venue_id = venue_data.get(
                "id"
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

                "Venue ID":
                    venue_id,

                "Game Date":
                    game_date,

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
# TRAINED NRFI / YRFI PROBABILITY HELPERS
# =========================================================

def _live_pitcher_model_inputs(
    pitcher_id,
    pitcher_xwoba
):

    if not pitcher_id:
        return {
            "xwoba": None,
            "k_pct": None,
            "pa": 0,
        }

    (
        k_percent,
        _bb_percent,
        batters_faced
    ) = get_mlb_pitcher_rates(
        pitcher_id,
        YEAR
    )

    return {
        "xwoba":
            pitcher_xwoba.get(
                int(pitcher_id)
            ),
        "k_pct":
            k_percent,
        "pa":
            batters_faced or 0,
    }


def _live_top4_model_inputs(
    top4,
    hitter_xwoba,
    hitter_barrels
):

    # The trained final model is a confirmed Top-4 model.
    # Before a full lineup is available, use legitimate
    # missing-data handling from the trained inference engine.
    if len(top4) != 4:

        return {
            "xwoba": None,
            "k_pct": None,
            "bb_pct": None,
            "barrel_pct": None,
            "combined_pa": 0,
            "min_pa": 0,
            "complete_core": False,
            "hitters": [
                {
                    "player_id":
                        player.get("id"),
                    "name":
                        player.get("name"),
                }
                for player in top4
            ],
        }


    xwoba_values = []
    k_values = []
    bb_values = []
    barrel_values = []
    pa_values = []
    hitter_rows = []


    for player in top4:

        player_id = player["id"]

        (
            k_percent,
            bb_percent,
            plate_appearances
        ) = get_mlb_hitter_rates(
            player_id,
            YEAR
        )

        xwoba_values.append(
            hitter_xwoba.get(
                player_id
            )
        )

        k_values.append(
            k_percent
        )

        bb_values.append(
            bb_percent
        )

        barrel_values.append(
            hitter_barrels.get(
                player_id
            )
        )

        pa_values.append(
            plate_appearances
            if plate_appearances is not None
            else 0
        )

        hitter_rows.append({
            "player_id":
                player_id,

            "name":
                player.get(
                    "name"
                ),

            "xwoba":
                hitter_xwoba.get(
                    player_id
                ),

            "k_pct":
                k_percent,

            "bb_pct":
                bb_percent,

            "barrel_pct":
                hitter_barrels.get(
                    player_id
                ),

            "pa":
                (
                    plate_appearances
                    if plate_appearances is not None
                    else 0
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

    complete_core = all(
        value is not None
        for value in [
            team_xwoba,
            team_k,
            team_bb,
            team_barrel,
        ]
    )


    return {
        "xwoba":
            team_xwoba,
        "k_pct":
            team_k,
        "bb_pct":
            team_bb,
        "barrel_pct":
            team_barrel,
        "combined_pa":
            sum(pa_values),
        "min_pa":
            min(pa_values)
            if pa_values
            else 0,
        "complete_core":
            complete_core,

        "hitters":
            hitter_rows,
    }


def _value_available(value):

    if value is None:
        return False

    try:
        return bool(
            pd.notna(value)
        )

    except Exception:
        return False


def build_trained_probability_rows(
    games,
    pitcher_xwoba,
    hitter_xwoba,
    hitter_barrels,
    model_park_lookup,
    trained_model
):

    rows = []


    for game in games:

        away_pitcher = (
            _live_pitcher_model_inputs(
                game["Away SP ID"],
                pitcher_xwoba
            )
        )

        home_pitcher = (
            _live_pitcher_model_inputs(
                game["Home SP ID"],
                pitcher_xwoba
            )
        )

        away_offense = (
            _live_top4_model_inputs(
                game["Away Top 4"],
                hitter_xwoba,
                hitter_barrels
            )
        )

        home_offense = (
            _live_top4_model_inputs(
                game["Home Top 4"],
                hitter_xwoba,
                hitter_barrels
            )
        )


        venue_id = game.get(
            "Venue ID"
        )

        model_park = (
            model_park_lookup.get(
                int(venue_id)
            )
            if venue_id
            else None
        )

        model_run_factor = (
            model_park.get(
                "run_factor"
            )
            if model_park
            else None
        )


        # Top 1st:
        # Away offense vs Home starting pitcher.
        top_features = build_half_features(
            pitcher_xwoba=
                home_pitcher["xwoba"],
            pitcher_k_pct=
                home_pitcher["k_pct"],
            pitcher_pa=
                home_pitcher["pa"],
            offense_xwoba=
                away_offense["xwoba"],
            offense_k_pct=
                away_offense["k_pct"],
            offense_bb_pct=
                away_offense["bb_pct"],
            offense_barrel_pct=
                away_offense["barrel_pct"],
            offense_combined_pa=
                away_offense["combined_pa"],
            offense_min_pa=
                away_offense["min_pa"],
            offense_complete_core=
                away_offense["complete_core"],
            park_runs=
                model_run_factor,
        )


        # If today's away lineup is not posted yet, the lineup is
        # unknown — it is NOT a four-hitter group with zero PA.
        # Neutralize all offense sample/missingness features to the
        # exact training means inside the inference engine.
        if len(
            game["Away Top 4"]
        ) != 4:

            for feature_name in [
                "o_xwoba",
                "o_k_pct",
                "o_bb_pct",
                "o_barrel_pct",
                "o_log_combined_pa",
                "o_log_min_pa",
                "o_missing_core",
            ]:

                top_features[
                    feature_name
                ] = None


        # A TBA home starter is also unknown, not a true zero-PA
        # pitcher. Neutralize the pitcher sample controls.
        if not game[
            "Home SP ID"
        ]:

            top_features[
                "p_log_pa"
            ] = None

            top_features[
                "p_no_prior_pa"
            ] = None


        # Bottom 1st:
        # Home offense vs Away starting pitcher.
        bottom_features = build_half_features(
            pitcher_xwoba=
                away_pitcher["xwoba"],
            pitcher_k_pct=
                away_pitcher["k_pct"],
            pitcher_pa=
                away_pitcher["pa"],
            offense_xwoba=
                home_offense["xwoba"],
            offense_k_pct=
                home_offense["k_pct"],
            offense_bb_pct=
                home_offense["bb_pct"],
            offense_barrel_pct=
                home_offense["barrel_pct"],
            offense_combined_pa=
                home_offense["combined_pa"],
            offense_min_pa=
                home_offense["min_pa"],
            offense_complete_core=
                home_offense["complete_core"],
            park_runs=
                model_run_factor,
        )


        # Same neutral treatment for an unposted home lineup.
        if len(
            game["Home Top 4"]
        ) != 4:

            for feature_name in [
                "o_xwoba",
                "o_k_pct",
                "o_bb_pct",
                "o_barrel_pct",
                "o_log_combined_pa",
                "o_log_min_pa",
                "o_missing_core",
            ]:

                bottom_features[
                    feature_name
                ] = None


        # Same neutral treatment for a TBA away starter.
        if not game[
            "Away SP ID"
        ]:

            bottom_features[
                "p_log_pa"
            ] = None

            bottom_features[
                "p_no_prior_pa"
            ] = None


        result = predict_nrfi_yrfi(
            trained_model,
            top_features=
                top_features,
            bottom_features=
                bottom_features,
        )


        completeness_values = [
            away_pitcher["xwoba"],
            away_pitcher["k_pct"],
            home_pitcher["xwoba"],
            home_pitcher["k_pct"],

            away_offense["xwoba"],
            away_offense["k_pct"],
            away_offense["bb_pct"],
            away_offense["barrel_pct"],

            home_offense["xwoba"],
            home_offense["k_pct"],
            home_offense["bb_pct"],
            home_offense["barrel_pct"],

            model_run_factor,
        ]

        available = sum(
            1
            for value
            in completeness_values
            if _value_available(
                value
            )
        )

        completeness = (
            available
            / len(
                completeness_values
            )
            * 100
        )


        pitchers_known = (
            game["Away SP ID"]
            and
            game["Home SP ID"]
        )

        lineups_confirmed = (
            len(
                game["Away Top 4"]
            ) == 4
            and
            len(
                game["Home Top 4"]
            ) == 4
        )


        if (
            pitchers_known
            and
            lineups_confirmed
        ):

            status = "FINAL"

        elif pitchers_known:

            status = (
                "PROVISIONAL — WAITING LINEUPS"
            )

        else:

            status = (
                "LOW DATA — STARTER TBA"
            )


        rows.append({

            "Game":
                game["Game"],

            "Game ID":
                game.get(
                    "Game ID"
                ),

            "Game Date":
                game.get(
                    "Game Date"
                ),

            "Away Team":
                game.get(
                    "Away Team"
                ),

            "Home Team":
                game.get(
                    "Home Team"
                ),

            "Away SP":
                game.get(
                    "Away SP"
                ),

            "Away SP ID":
                game.get(
                    "Away SP ID"
                ),

            "Home SP":
                game.get(
                    "Home SP"
                ),

            "Home SP ID":
                game.get(
                    "Home SP ID"
                ),

            "Away Pitcher xwOBA":
                away_pitcher[
                    "xwoba"
                ],

            "Away Pitcher K%":
                away_pitcher[
                    "k_pct"
                ],

            "Away Pitcher PA":
                away_pitcher[
                    "pa"
                ],

            "Home Pitcher xwOBA":
                home_pitcher[
                    "xwoba"
                ],

            "Home Pitcher K%":
                home_pitcher[
                    "k_pct"
                ],

            "Home Pitcher PA":
                home_pitcher[
                    "pa"
                ],

            "Away Top4 xwOBA":
                away_offense[
                    "xwoba"
                ],

            "Away Top4 K%":
                away_offense[
                    "k_pct"
                ],

            "Away Top4 BB%":
                away_offense[
                    "bb_pct"
                ],

            "Away Top4 Barrel%":
                away_offense[
                    "barrel_pct"
                ],

            "Away Top4 Combined PA":
                away_offense[
                    "combined_pa"
                ],

            "Away Top4 Min PA":
                away_offense[
                    "min_pa"
                ],

            "Away Top4 Complete Core":
                away_offense[
                    "complete_core"
                ],

            "Away Top4 Hitters":
                away_offense[
                    "hitters"
                ],

            "Home Top4 xwOBA":
                home_offense[
                    "xwoba"
                ],

            "Home Top4 K%":
                home_offense[
                    "k_pct"
                ],

            "Home Top4 BB%":
                home_offense[
                    "bb_pct"
                ],

            "Home Top4 Barrel%":
                home_offense[
                    "barrel_pct"
                ],

            "Home Top4 Combined PA":
                home_offense[
                    "combined_pa"
                ],

            "Home Top4 Min PA":
                home_offense[
                    "min_pa"
                ],

            "Home Top4 Complete Core":
                home_offense[
                    "complete_core"
                ],

            "Home Top4 Hitters":
                home_offense[
                    "hitters"
                ],

            "Model Side":
                result["model_side"],

            "Model Probability":
                result["model_probability"]
                * 100,

            "NRFI Probability":
                result["nrfi_probability"]
                * 100,

            "YRFI Probability":
                result["yrfi_probability"]
                * 100,

            "Top 1st Score Probability":
                result[
                    "top_scoring_probability"
                ] * 100,

            "Bottom 1st Score Probability":
                result[
                    "bottom_scoring_probability"
                ] * 100,

            "Input Completeness":
                completeness,

            "Status":
                status,

            "Model Run Factor":
                model_run_factor,

            "Start Time":
                game["Start Time"],

            "Venue":
                game["Venue"],
        })


    rows.sort(
        key=lambda row:
            row["Model Probability"],
        reverse=True
    )


    for rank, row in enumerate(
        rows,
        start=1
    ):

        row["Rank"] = rank


    return rows


def format_probability(value):

    return f"{value:.1f}%"




# =========================================================
# MARKET ODDS / EDGE HELPERS
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_cached_mlb_odds_events(
    api_key
):

    return get_mlb_events(
        api_key
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_cached_first_inning_event_odds(
    api_key,
    event_id
):

    return get_first_inning_event_odds(
        api_key,
        event_id
    )


def format_american_odds(
    value
):

    if value is None:
        return "—"

    value = int(
        round(
            float(value)
        )
    )

    if value > 0:
        return f"+{value}"

    return str(
        value
    )


def attach_market_data(
    games,
    probability_rows,
    api_key
):

    probability_lookup = {
        row["Game"]:
            row
        for row in probability_rows
    }

    events, event_usage = (
        get_cached_mlb_odds_events(
            api_key
        )
    )

    output_rows = []

    last_usage = event_usage


    for game in games:

        probability_row = dict(
            probability_lookup[
                game["Game"]
            ]
        )

        probability_row.update({

            "Market No-Vig":
                None,

            "Market Raw Implied":
                None,

            "Edge":
                None,

            "Price Edge":
                None,

            "Market NRFI No-Vig":
                None,

            "Market YRFI No-Vig":
                None,

            "Best NRFI Price":
                None,

            "Best NRFI Book":
                None,

            "Best YRFI Price":
                None,

            "Best YRFI Book":
                None,

            "NRFI Break-Even":
                None,

            "YRFI Break-Even":
                None,

            "NRFI Market Edge":
                None,

            "YRFI Market Edge":
                None,

            "NRFI Price Edge":
                None,

            "YRFI Price Edge":
                None,

            "Market Bookmaker Rows":
                [],

            "Best Price":
                None,

            "Best Book":
                None,

            "Books":
                0,

            "Market Status":
                "MARKET UNAVAILABLE",
        })


        odds_event = find_odds_event(

            events,

            game["Away Team"],

            game["Home Team"],
        )


        if not odds_event:

            probability_row[
                "Market Status"
            ] = "ODDS EVENT NOT FOUND"

            output_rows.append(
                probability_row
            )

            continue


        try:

            event_odds, usage = (
                get_cached_first_inning_event_odds(

                    api_key,

                    odds_event["id"],
                )
            )

            last_usage = usage

            bookmaker_rows = (
                parse_first_inning_market(
                    event_odds
                )
            )

            market_summary = (
                summarize_market(
                    bookmaker_rows
                )
            )


            if not market_summary:

                probability_row[
                    "Market Status"
                ] = "1ST-INNING MARKET NOT POSTED"

                output_rows.append(
                    probability_row
                )

                continue


            if (
                probability_row[
                    "Model Side"
                ] == "NRFI"
            ):

                market_no_vig = (
                    market_summary[
                        "consensus_nrfi_no_vig"
                    ]
                )

                best_price = (
                    market_summary[
                        "best_nrfi_price"
                    ]
                )

                best_book = (
                    market_summary[
                        "best_nrfi_book"
                    ]
                )

                model_probability = (
                    probability_row[
                        "NRFI Probability"
                    ]
                    / 100.0
                )

            else:

                market_no_vig = (
                    market_summary[
                        "consensus_yrfi_no_vig"
                    ]
                )

                best_price = (
                    market_summary[
                        "best_yrfi_price"
                    ]
                )

                best_book = (
                    market_summary[
                        "best_yrfi_book"
                    ]
                )

                model_probability = (
                    probability_row[
                        "YRFI Probability"
                    ]
                    / 100.0
                )


            raw_implied = (
                american_implied_probability(
                    best_price
                )
            )

            edge = (
                model_probability
                - market_no_vig
            )


            price_edge = (
                model_probability
                - raw_implied
                if raw_implied is not None
                else None
            )


            nrfi_best_price = (
                market_summary[
                    "best_nrfi_price"
                ]
            )

            yrfi_best_price = (
                market_summary[
                    "best_yrfi_price"
                ]
            )

            nrfi_break_even = (
                american_implied_probability(
                    nrfi_best_price
                )
            )

            yrfi_break_even = (
                american_implied_probability(
                    yrfi_best_price
                )
            )

            nrfi_model_probability = (
                probability_row[
                    "NRFI Probability"
                ]
                / 100.0
            )

            yrfi_model_probability = (
                probability_row[
                    "YRFI Probability"
                ]
                / 100.0
            )

            nrfi_market_no_vig = (
                market_summary[
                    "consensus_nrfi_no_vig"
                ]
            )

            yrfi_market_no_vig = (
                market_summary[
                    "consensus_yrfi_no_vig"
                ]
            )


            probability_row.update({

                "Market No-Vig":
                    market_no_vig
                    * 100.0,

                "Market Raw Implied":
                    raw_implied
                    * 100.0
                    if raw_implied is not None
                    else None,

                "Edge":
                    edge
                    * 100.0,

                "Price Edge":
                    price_edge
                    * 100.0
                    if price_edge is not None
                    else None,

                "Market NRFI No-Vig":
                    nrfi_market_no_vig
                    * 100.0,

                "Market YRFI No-Vig":
                    yrfi_market_no_vig
                    * 100.0,

                "Best NRFI Price":
                    nrfi_best_price,

                "Best NRFI Book":
                    market_summary[
                        "best_nrfi_book"
                    ],

                "Best YRFI Price":
                    yrfi_best_price,

                "Best YRFI Book":
                    market_summary[
                        "best_yrfi_book"
                    ],

                "NRFI Break-Even":
                    (
                        nrfi_break_even
                        * 100.0
                        if nrfi_break_even is not None
                        else None
                    ),

                "YRFI Break-Even":
                    (
                        yrfi_break_even
                        * 100.0
                        if yrfi_break_even is not None
                        else None
                    ),

                "NRFI Market Edge":
                    (
                        nrfi_model_probability
                        - nrfi_market_no_vig
                    )
                    * 100.0,

                "YRFI Market Edge":
                    (
                        yrfi_model_probability
                        - yrfi_market_no_vig
                    )
                    * 100.0,

                "NRFI Price Edge":
                    (
                        (
                            nrfi_model_probability
                            - nrfi_break_even
                        )
                        * 100.0
                        if nrfi_break_even is not None
                        else None
                    ),

                "YRFI Price Edge":
                    (
                        (
                            yrfi_model_probability
                            - yrfi_break_even
                        )
                        * 100.0
                        if yrfi_break_even is not None
                        else None
                    ),

                "Market Bookmaker Rows":
                    bookmaker_rows,

                "Best Price":
                    best_price,

                "Best Book":
                    best_book,

                "Books":
                    market_summary[
                        "book_count"
                    ],

                "Market Status":
                    "LIVE",
            })


        except Exception as error:

            probability_row[
                "Market Status"
            ] = (
                f"ERROR: {error}"
            )


        output_rows.append(
            probability_row
        )


    # Preserve model-probability ranking.
    output_rows.sort(
        key=lambda row:
            row["Model Probability"],
        reverse=True
    )


    for rank, row in enumerate(
        output_rows,
        start=1
    ):

        row["Rank"] = rank


    return (
        output_rows,
        last_usage
    )


def format_edge(
    value
):

    if value is None:
        return "—"

    return f"{value:+.1f}%"



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
            # GAME WEATHER
            # ---------------------------------------------

            st.subheader(
                "Game Weather & Roof Conditions"
            )

            st.caption(
                "Game-time weather at the ballpark. "
                "Indoor parks ignore outdoor weather. "
                "Retractable-roof games require roof verification."
            )

            weather_rows = []

            with st.spinner(
                "Loading game-time weather..."
            ):

                for game in games:

                    try:

                        venue_info = get_venue_info(
                            game["Venue ID"]
                        )

                        if not venue_info:
                            raise Exception(
                                "Venue data unavailable"
                            )


                        weather = get_game_weather(
                            venue_info["latitude"],
                            venue_info["longitude"],
                            venue_info["timezone"],
                            game["Game Date"],
                        )

                        if not weather:
                            raise Exception(
                                "Weather unavailable"
                            )


                        temperature = (
                            f"{weather['temperature']:.1f}°F"
                            if weather["temperature"] is not None
                            else "—"
                        )

                        humidity = (
                            f"{weather['humidity']:.0f}%"
                            if weather["humidity"] is not None
                            else "—"
                        )

                        dew_point = (
                            f"{weather['dew_point']:.1f}°F"
                            if weather["dew_point"] is not None
                            else "—"
                        )

                        pressure = (
                            f"{weather['pressure']:.1f} hPa"
                            if weather["pressure"] is not None
                            else "—"
                        )

                        wind = (
                            f"{weather['wind_speed']:.1f} mph"
                            if weather["wind_speed"] is not None
                            else "—"
                        )

                        if (
                            weather["wind_direction_degrees"]
                            is not None
                        ):

                            wind_direction = (
                                f"{weather['wind_direction']} "
                                f"({weather['wind_direction_degrees']:.0f}°)"
                            )

                        else:

                            wind_direction = "—"


                        gusts = (
                            f"{weather['wind_gusts']:.1f} mph"
                            if weather["wind_gusts"] is not None
                            else "—"
                        )

                        precip = (
                            f"{weather['precipitation_probability']:.0f}%"
                            if weather[
                                "precipitation_probability"
                            ] is not None
                            else "—"
                        )


                        weather_rows.append({

                            "Game":
                                game["Game"],

                            "Venue":
                                venue_info["name"],

                            "Roof":
                                venue_info["roof_type"],

                            "Weather Use":
                                weather_usage(
                                    venue_info["roof_type"]
                                ),

                            "Temp":
                                temperature,

                            "Humidity":
                                humidity,

                            "Dew Point":
                                dew_point,

                            "Pressure":
                                pressure,

                            "Wind":
                                wind,

                            "Wind Direction":
                                wind_direction,

                            "Wind Gusts":
                                gusts,

                            "Precip":
                                precip,

                            "Local Game Time":
                                weather["game_local"],
                        })


                    except Exception as error:

                        weather_rows.append({

                            "Game":
                                game["Game"],

                            "Venue":
                                game["Venue"],

                            "Roof":
                                "—",

                            "Weather Use":
                                f"ERROR: {error}",

                            "Temp":
                                "—",

                            "Humidity":
                                "—",

                            "Dew Point":
                                "—",

                            "Pressure":
                                "—",

                            "Wind":
                                "—",

                            "Wind Direction":
                                "—",

                            "Wind Gusts":
                                "—",

                            "Precip":
                                "—",

                            "Local Game Time":
                                game["Start Time"],
                        })


            st.dataframe(
                pd.DataFrame(
                    weather_rows
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # BALLPARK FACTORS
            # ---------------------------------------------

            st.subheader(
                "Ballpark Run Environment"
            )

            st.caption(
                "Baseball Savant 3-year rolling park factors. "
                "100 = MLB average."
            )


            with st.spinner(
                "Loading ballpark factors..."
            ):

                park_lookup = (
                    get_park_factors(
                        YEAR
                    )
                )

                park_rows = []


                for game in games:

                    venue_id = game.get(
                        "Venue ID"
                    )

                    park = (
                        park_lookup.get(
                            int(venue_id)
                        )
                        if venue_id
                        else None
                    )


                    if park:

                        park_rows.append({

                            "Game":
                                game["Game"],

                            "Venue":
                                park[
                                    "venue_name"
                                ],

                            "Run Factor":
                                park[
                                    "run_factor"
                                ],

                            "Run Environment":
                                classify_run_factor(
                                    park[
                                        "run_factor"
                                    ]
                                ),

                            "wOBA Factor":
                                park[
                                    "woba_factor"
                                ],

                            "HR Factor":
                                park[
                                    "hr_factor"
                                ],

                            "Sample":
                                park[
                                    "year_range"
                                ],
                        })

                    else:

                        park_rows.append({

                            "Game":
                                game["Game"],

                            "Venue":
                                game["Venue"],

                            "Run Factor":
                                "—",

                            "Run Environment":
                                "MISSING",

                            "wOBA Factor":
                                "—",

                            "HR Factor":
                                "—",

                            "Sample":
                                "—",
                        })


            st.dataframe(
                pd.DataFrame(
                    park_rows
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
            # PITCHER FIRST-INNING HISTORY
            # ---------------------------------------------

            st.subheader(
                "Pitcher YTD First-Inning History"
            )

            st.caption(
                "Regular-season results through "
                "yesterday. Current-day games are "
                "excluded to prevent data leakage."
            )


            with st.spinner(
                "Loading YTD first-inning history..."
            ):

                history_end_date = (
                    TODAY
                    - timedelta(days=1)
                )


                season_first_innings = (
                    get_season_first_innings(
                        YEAR,
                        history_end_date.isoformat()
                    )
                )


                first_inning_rows = []


                for game in games:

                    pitchers = [

                        (
                            game["Away Team"],
                            game["Away SP"],
                            game["Away SP ID"],
                        ),

                        (
                            game["Home Team"],
                            game["Home SP"],
                            game["Home SP ID"],
                        ),
                    ]


                    for (
                        team,
                        pitcher_name,
                        pitcher_id
                    ) in pitchers:


                        if not pitcher_id:

                            first_inning_rows.append({

                                "Game":
                                    game["Game"],

                                "Team":
                                    team,

                                "Pitcher":
                                    pitcher_name,

                                "Starts":
                                    "—",

                                "Scoreless 1st":
                                    "—",

                                "Scoreless 1st %":
                                    "—",

                                "1st-Inning Runs/Start":
                                    "—",

                                "NRFI Record":
                                    "—",

                                "NRFI %":
                                    "—",
                            })

                            continue


                        history = (
                            get_pitcher_first_inning_history(
                                pitcher_id,
                                YEAR,
                                season_first_innings,
                            )
                        )


                        starts = history[
                            "starts"
                        ]


                        if starts:

                            scoreless_rate = (
                                f"{history['scoreless_percent']:.1f}%"
                            )

                            runs_per_start = (
                                f"{history['runs_per_start']:.2f}"
                            )

                            nrfi_record = (
                                f"{history['nrfi']}-"
                                f"{history['yrfi']}"
                            )

                            nrfi_rate = (
                                f"{history['nrfi_percent']:.1f}%"
                            )

                        else:

                            scoreless_rate = "—"
                            runs_per_start = "—"
                            nrfi_record = "—"
                            nrfi_rate = "—"


                        first_inning_rows.append({

                            "Game":
                                game["Game"],

                            "Team":
                                team,

                            "Pitcher":
                                pitcher_name,

                            "Starts":
                                starts,

                            "Scoreless 1st":
                                history[
                                    "scoreless"
                                ],

                            "Scoreless 1st %":
                                scoreless_rate,

                            "1st-Inning Runs/Start":
                                runs_per_start,

                            "NRFI Record":
                                nrfi_record,

                            "NRFI %":
                                nrfi_rate,
                        })


            st.dataframe(
                pd.DataFrame(
                    first_inning_rows
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


            # ---------------------------------------------
            # COMBINED HALF-INNING MODEL INPUTS
            # ---------------------------------------------

            st.subheader(
                "Combined Half-Inning Model Inputs"
            )

            st.caption(
                "Diagnostic context table. "
                "The production v1 probability model uses "
                "the validated pitcher, Top-4 offense, "
                "sample-size, and prior-season Run Factor inputs."
            )


            combined_model_rows = (
                build_half_inning_model_table(

                    games,
                    pitcher_rows,
                    first_inning_rows,
                    team_hitter_rows,
                    weather_rows,
                    park_rows,
                )
            )


            combined_df = pd.DataFrame(
                combined_model_rows
            )


            st.dataframe(
                combined_df,
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # TRAINED v1 NRFI / YRFI PROBABILITIES
            # ---------------------------------------------

            st.subheader(
                "Trained NRFI / YRFI Probabilities"
            )

            st.caption(
                "SharpReport NRFI/YRFI Model v1. "
                "Top and Bottom 1st are modeled separately. "
                "The production model uses the prior completed "
                "season's 3-year Run Factor to match historical training."
            )


            with st.spinner(
                "Running trained NRFI/YRFI probability model..."
            ):

                trained_model = (
                    load_nrfi_model()
                )

                # Match the historical training rule:
                # 2026 games use the 2025 three-year rolling
                # park-factor table, not the in-progress 2026 table.
                model_park_lookup = (
                    get_park_factors(
                        YEAR - 1
                    )
                )

                probability_rows = (
                    build_trained_probability_rows(
                        games,
                        pitcher_xwoba,
                        hitter_xwoba,
                        hitter_barrels,
                        model_park_lookup,
                        trained_model,
                    )
                )


            st.write(
                f"**Model:** "
                f"{trained_model.get('model_name', 'SharpReport NRFI/YRFI Model v1')}"
            )

            st.write(
                f"**Training Through:** "
                f"{trained_model.get('training_end_date', '—')}"
            )

            st.caption(
                "FINAL = both starting pitchers and both Top-4 "
                "lineups are known. Missing season metrics are "
                "handled by the same trained missing-data logic "
                "used in the historical model."
            )


            # ---------------------------------------------
            # LIVE MARKET PRICES / EDGE
            # ---------------------------------------------

            try:

                odds_api_key = (
                    st.secrets[
                        "ODDS_API_KEY"
                    ]
                )

            except Exception:

                odds_api_key = None


            odds_usage = None


            if odds_api_key:

                with st.spinner(
                    "Loading live first-inning prices "
                    "and calculating no-vig market edge..."
                ):

                    (
                        market_rows,
                        odds_usage
                    ) = attach_market_data(
                        games,
                        probability_rows,
                        odds_api_key,
                    )


                st.caption(
                    "Market = first-inning total 0.5. "
                    "Under 0.5 = NRFI; Over 0.5 = YRFI. "
                    "Consensus no-vig probability is the average "
                    "of each sportsbook's two-way no-vig market. "
                    "Odds requests are cached for 30 minutes to preserve API credits."
                )


                requests_remaining = (
                    odds_usage.get(
                        "requests_remaining"
                    )
                    if odds_usage
                    else None
                )

                if requests_remaining is not None:

                    st.write(
                        f"**Odds API Credits Remaining:** "
                        f"{requests_remaining}"
                    )

            else:

                market_rows = [
                    dict(row)
                    for row in probability_rows
                ]

                for row in market_rows:

                    row.update({

                        "Market No-Vig":
                            None,

                        "Market Raw Implied":
                            None,

                        "Edge":
                            None,

                        "Price Edge":
                            None,

                        "Market NRFI No-Vig":
                            None,

                        "Market YRFI No-Vig":
                            None,

                        "Best NRFI Price":
                            None,

                        "Best NRFI Book":
                            None,

                        "Best YRFI Price":
                            None,

                        "Best YRFI Book":
                            None,

                        "NRFI Break-Even":
                            None,

                        "YRFI Break-Even":
                            None,

                        "NRFI Market Edge":
                            None,

                        "YRFI Market Edge":
                            None,

                        "NRFI Price Edge":
                            None,

                        "YRFI Price Edge":
                            None,

                        "Market Bookmaker Rows":
                            [],

                        "Best Price":
                            None,

                        "Best Book":
                            None,

                        "Books":
                            0,

                        "Market Status":
                            "ODDS API KEY NOT FOUND",
                    })


                st.warning(
                    "ODDS_API_KEY was not found in "
                    "Streamlit secrets. Model probabilities "
                    "are available, but market edge is disabled."
                )


            # ---------------------------------------------
            # PERSISTENT PREGAME DATA SNAPSHOT
            # ---------------------------------------------

            try:

                github_data_token = (
                    st.secrets[
                        "GITHUB_DATA_TOKEN"
                    ]
                )

                github_data_repo = (
                    st.secrets[
                        "GITHUB_DATA_REPO"
                    ]
                )

            except Exception:

                github_data_token = None
                github_data_repo = None


            if (
                github_data_token
                and
                github_data_repo
            ):

                try:

                    snapshot_time = (
                        datetime.now(
                            ET
                        )
                    )

                    model_metadata = {
                        "model_name":
                            trained_model.get(
                                "model_name",
                                "SharpReport NRFI/YRFI Model v1",
                            ),

                        "training_start_date":
                            trained_model.get(
                                "training_start_date"
                            ),

                        "training_end_date":
                            trained_model.get(
                                "training_end_date"
                            ),

                        "training_games":
                            trained_model.get(
                                "training_games"
                            ),
                    }


                    with st.spinner(
                        "Saving this pregame model + market "
                        "snapshot to the private data repository..."
                    ):

                        snapshot_result = (
                            save_slate_snapshot(
                                token=
                                    github_data_token,

                                repo=
                                    github_data_repo,

                                rows=
                                    market_rows,

                                snapshot_time=
                                    snapshot_time,

                                model_metadata=
                                    model_metadata,

                                odds_usage=
                                    odds_usage,
                            )
                        )


                    if snapshot_result.get(
                        "ok"
                    ):

                        st.success(
                            "Pregame data snapshot saved "
                            "to the private SharpReport data repository."
                        )

                        st.caption(
                            "Saved snapshot: "
                            f"{snapshot_result.get('path')}"
                        )

                except Exception as error:

                    st.warning(
                        "The scanner completed, but the private "
                        "data snapshot could not be saved: "
                        f"{error}"
                    )

            else:

                st.info(
                    "Persistent data logging is not active because "
                    "GITHUB_DATA_TOKEN or GITHUB_DATA_REPO "
                    "was not found in Streamlit secrets."
                )


            # ---------------------------------------------
            # TOP MODEL PROBABILITIES
            # ---------------------------------------------

            st.subheader(
                "Top Model Probabilities"
            )

            st.caption(
                "Top four games ranked only by the trained model's "
                "stronger NRFI/YRFI side probability. "
                "This ranking ignores sportsbook price."
            )


            top_model_display = []

            for row in market_rows[:4]:

                top_model_display.append({

                    "Rank":
                        row["Rank"],

                    "Game":
                        row["Game"],

                    "Model Side":
                        row["Model Side"],

                    "Model Probability":
                        format_probability(
                            row["Model Probability"]
                        ),

                    "NRFI":
                        format_probability(
                            row["NRFI Probability"]
                        ),

                    "YRFI":
                        format_probability(
                            row["YRFI Probability"]
                        ),

                    "Input Completeness":
                        f"{row['Input Completeness']:.0f}%",

                    "Status":
                        row["Status"],
                })


            st.dataframe(
                pd.DataFrame(
                    top_model_display
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # BEST FINAL EDGES
            # ---------------------------------------------

            st.subheader(
                "Best Final Edges"
            )

            st.caption(
                "Only games with FINAL model status appear here. "
                "Price Edge compares the model probability directly "
                "with the break-even probability of the best currently "
                "available price. Market Edge compares the model with "
                "the consensus no-vig market."
            )


            final_edge_rows = [
                row
                for row in market_rows
                if (
                    row["Status"] == "FINAL"
                    and
                    row["Price Edge"] is not None
                    and
                    row["Price Edge"] > 0
                )
            ]

            final_edge_rows.sort(
                key=lambda row:
                    row["Price Edge"],
                reverse=True
            )


            final_edge_display = []

            for rank, row in enumerate(
                final_edge_rows[:4],
                start=1
            ):

                final_edge_display.append({

                    "Edge Rank":
                        rank,

                    "Game":
                        row["Game"],

                    "Model Side":
                        row["Model Side"],

                    "Model":
                        format_probability(
                            row["Model Probability"]
                        ),

                    "Market No-Vig":
                        format_probability(
                            row["Market No-Vig"]
                        ),

                    "Market Edge":
                        format_edge(
                            row["Edge"]
                        ),

                    "Best Price":
                        format_american_odds(
                            row["Best Price"]
                        ),

                    "Break-Even":
                        (
                            format_probability(
                                row["Market Raw Implied"]
                            )
                            if row["Market Raw Implied"] is not None
                            else "—"
                        ),

                    "Price Edge":
                        format_edge(
                            row["Price Edge"]
                        ),

                    "Sportsbook":
                        row["Best Book"]
                        or "—",

                    "Books":
                        row["Books"],
                })


            if final_edge_display:

                st.dataframe(
                    pd.DataFrame(
                        final_edge_display
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No positive FINAL price edges are available yet. "
                    "Re-run after confirmed lineups and starters are posted."
                )


            # ---------------------------------------------
            # PROVISIONAL MARKET WATCH
            # ---------------------------------------------

            st.subheader(
                "Provisional Market Watch"
            )

            st.caption(
                "These games have live market prices but are still "
                "waiting on confirmed Top-4 lineups. Their probabilities "
                "and edges can change when the real lineup inputs arrive. "
                "They are not treated as FINAL model edges."
            )


            provisional_rows = [
                row
                for row in market_rows
                if (
                    row["Status"]
                    == "PROVISIONAL — WAITING LINEUPS"
                    and
                    row["Price Edge"] is not None
                    and
                    row["Price Edge"] > 0
                )
            ]

            provisional_rows.sort(
                key=lambda row:
                    row["Price Edge"],
                reverse=True
            )


            provisional_display = []

            for rank, row in enumerate(
                provisional_rows[:6],
                start=1
            ):

                provisional_display.append({

                    "Watch Rank":
                        rank,

                    "Game":
                        row["Game"],

                    "Model Side":
                        row["Model Side"],

                    "Model":
                        format_probability(
                            row["Model Probability"]
                        ),

                    "Market No-Vig":
                        format_probability(
                            row["Market No-Vig"]
                        ),

                    "Market Edge":
                        format_edge(
                            row["Edge"]
                        ),

                    "Best Price":
                        format_american_odds(
                            row["Best Price"]
                        ),

                    "Break-Even":
                        (
                            format_probability(
                                row["Market Raw Implied"]
                            )
                            if row["Market Raw Implied"] is not None
                            else "—"
                        ),

                    "Price Edge":
                        format_edge(
                            row["Price Edge"]
                        ),

                    "Sportsbook":
                        row["Best Book"]
                        or "—",

                    "Input Completeness":
                        f"{row['Input Completeness']:.0f}%",

                    "Status":
                        row["Status"],
                })


            if provisional_display:

                st.dataframe(
                    pd.DataFrame(
                        provisional_display
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No positive provisional price edges "
                    "are currently available."
                )


            # ---------------------------------------------
            # ALL GAMES RANKED
            # ---------------------------------------------

            st.subheader(
                "All Games — Model + Market"
            )

            st.caption(
                "Market Edge = model probability minus consensus "
                "no-vig market probability. Price Edge = model probability "
                "minus the break-even probability of the best available price. "
                "Before confirmed lineups are posted, unknown lineup inputs "
                "are neutralized to the model's training means."
            )


            all_game_display = []

            for row in market_rows:

                all_game_display.append({

                    "Rank":
                        row["Rank"],

                    "Game":
                        row["Game"],

                    "Model Side":
                        row["Model Side"],

                    "Model":
                        format_probability(
                            row["Model Probability"]
                        ),

                    "Market No-Vig":
                        (
                            format_probability(
                                row["Market No-Vig"]
                            )
                            if row["Market No-Vig"] is not None
                            else "—"
                        ),

                    "Market Edge":
                        format_edge(
                            row["Edge"]
                        ),

                    "Best Price":
                        format_american_odds(
                            row["Best Price"]
                        ),

                    "Break-Even":
                        (
                            format_probability(
                                row["Market Raw Implied"]
                            )
                            if row["Market Raw Implied"] is not None
                            else "—"
                        ),

                    "Price Edge":
                        format_edge(
                            row["Price Edge"]
                        ),

                    "Sportsbook":
                        row["Best Book"]
                        or "—",

                    "NRFI":
                        format_probability(
                            row["NRFI Probability"]
                        ),

                    "YRFI":
                        format_probability(
                            row["YRFI Probability"]
                        ),

                    "Top 1st Scores":
                        format_probability(
                            row[
                                "Top 1st Score Probability"
                            ]
                        ),

                    "Bottom 1st Scores":
                        format_probability(
                            row[
                                "Bottom 1st Score Probability"
                            ]
                        ),

                    "Run Factor":
                        (
                            f"{row['Model Run Factor']:.0f}"
                            if row["Model Run Factor"] is not None
                            else "100 (neutral)"
                        ),

                    "Input Completeness":
                        f"{row['Input Completeness']:.0f}%",

                    "Status":
                        row["Status"],

                    "Market":
                        row["Market Status"],
                })


            st.dataframe(
                pd.DataFrame(
                    all_game_display
                ),
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # HALF-INNING PROBABILITY DETAIL
            # ---------------------------------------------

            with st.expander(
                "View Half-Inning Probability Detail"
            ):

                half_probability_rows = []

                for row in market_rows:

                    half_probability_rows.append({

                        "Game":
                            row["Game"],

                        "Half":
                            "Top 1st",

                        "Scores":
                            format_probability(
                                row[
                                    "Top 1st Score Probability"
                                ]
                            ),

                        "Scoreless":
                            format_probability(
                                100
                                - row[
                                    "Top 1st Score Probability"
                                ]
                            ),

                        "Status":
                            row["Status"],
                    })

                    half_probability_rows.append({

                        "Game":
                            row["Game"],

                        "Half":
                            "Bottom 1st",

                        "Scores":
                            format_probability(
                                row[
                                    "Bottom 1st Score Probability"
                                ]
                            ),

                        "Scoreless":
                            format_probability(
                                100
                                - row[
                                    "Bottom 1st Score Probability"
                                ]
                            ),

                        "Status":
                            row["Status"],
                    })


                st.dataframe(
                    pd.DataFrame(
                        half_probability_rows
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