import streamlit as st
import pandas as pd
import json
import io
import base64
import html

from pathlib import Path

from urllib.request import urlopen, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pitcher_first_inning_windows import (
    get_pitcher_first_inning_windows,
    empty_pitcher_first_inning_windows,
    flatten_pitcher_first_inning_windows,
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

from nrfi_result_grader import (
    grade_recent_results,
)

from nrfi_edge_dashboard import (
    render_edge_performance_dashboard,
)

from nrfi_pitcher_history_research import (
    render_pitcher_history_research,
)

from nrfi_clv_dashboard import (
    render_clv_dashboard,
)

from nrfi_model_governance import (
    update_model_governance,
    render_model_governance_dashboard,
)

from nrfi_model_alerts import (
    update_model_attention_alerts,
    render_model_attention_banner,
    render_model_attention_dashboard,
)




# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="SharpReport NRFI / YRFI Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_DIR = Path(
    __file__
).resolve().parent

ASSET_DIR = (
    APP_DIR
    / "assets"
)


def _asset_data_uri(
    filename,
):
    path = (
        ASSET_DIR
        / filename
    )

    if not path.exists():
        return None

    suffix = (
        path.suffix
        .lower()
        .lstrip(".")
    )

    mime = (
        "image/jpeg"
        if suffix in {
            "jpg",
            "jpeg",
        }
        else "image/png"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode(
        "ascii"
    )

    return (
        f"data:{mime};base64,"
        f"{encoded}"
    )


SHARPREPORT_LOGO_URI = (
    _asset_data_uri(
        "sharpreport_logo.jpeg"
    )
)

SHARPREPORT_BACKGROUND_URI = (
    _asset_data_uri(
        "sharpreport_background.jpeg"
    )
)


background_css = (
    f'url("{SHARPREPORT_BACKGROUND_URI}")'
    if SHARPREPORT_BACKGROUND_URI
    else "none"
)


st.markdown(
    f"""
    <style>
    :root {{
        --sr-black: #050505;
        --sr-panel: #101010;
        --sr-panel-2: #171717;
        --sr-gold: #F6C431;
        --sr-gold-bright: #FFD95A;
        --sr-gold-deep: #B98408;
        --sr-white: #F7F7F7;
        --sr-muted: #B7B7B7;
        --sr-border: rgba(246, 196, 49, 0.32);
    }}

    .stApp {{
        background:
            radial-gradient(
                circle at 78% 0%,
                rgba(246,196,49,0.10),
                transparent 28rem
            ),
            linear-gradient(
                180deg,
                #080808 0%,
                #050505 44%,
                #050505 100%
            );
        color: var(--sr-white);
    }}

    [data-testid="stHeader"] {{
        background: rgba(5,5,5,0.90);
        border-bottom: 1px solid rgba(246,196,49,0.16);
    }}

    [data-testid="stToolbar"] {{
        right: 0.5rem;
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 4rem;
    }}

    .sr-brand-hero {{
        min-height: 235px;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 1.35rem 1.55rem;
        margin-bottom: 1.15rem;
        border: 1px solid var(--sr-border);
        border-radius: 18px;
        overflow: hidden;
        background-image:
            linear-gradient(
                90deg,
                rgba(0,0,0,0.96) 0%,
                rgba(0,0,0,0.82) 37%,
                rgba(0,0,0,0.36) 72%,
                rgba(0,0,0,0.48) 100%
            ),
            {background_css};
        background-size: cover;
        background-position: center;
        box-shadow:
            0 14px 40px rgba(0,0,0,0.42),
            inset 0 0 45px rgba(246,196,49,0.035);
    }}

    .sr-brand-logo {{
        width: 145px;
        height: 145px;
        object-fit: cover;
        border-radius: 50%;
        border: 2px solid rgba(255,217,90,0.72);
        box-shadow:
            0 0 0 5px rgba(246,196,49,0.08),
            0 0 30px rgba(246,196,49,0.20);
        flex: 0 0 auto;
    }}

    .sr-brand-copy {{
        max-width: 610px;
    }}

    .sr-brand-eyebrow {{
        color: var(--sr-gold);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .sr-brand-title {{
        color: #FFFFFF;
        font-size: 2.25rem;
        line-height: 1.03;
        font-weight: 900;
        letter-spacing: -0.035em;
        text-shadow: 0 2px 15px rgba(0,0,0,0.75);
    }}

    .sr-brand-title span {{
        color: var(--sr-gold-bright);
    }}

    .sr-brand-subtitle {{
        margin-top: 0.65rem;
        color: #D2D2D2;
        font-size: 0.98rem;
        line-height: 1.45;
        max-width: 560px;
    }}

    h1, h2, h3 {{
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }}

    h2 {{
        border-bottom: 1px solid rgba(246,196,49,0.18);
        padding-bottom: 0.32rem;
    }}

    p, li, label {{
        color: #D6D6D6;
    }}

    a {{
        color: var(--sr-gold-bright) !important;
    }}

    div[data-testid="stButton"] > button {{
        background:
            linear-gradient(
                180deg,
                #FFDA57 0%,
                #E8B51E 100%
            );
        color: #090909;
        border: 1px solid #FFDD62;
        border-radius: 10px;
        font-weight: 900;
        letter-spacing: 0.015em;
        box-shadow:
            0 5px 18px rgba(246,196,49,0.16);
        min-height: 3rem;
    }}

    div[data-testid="stButton"] > button:hover {{
        color: #000000;
        border-color: #FFF0A5;
        box-shadow:
            0 7px 24px rgba(246,196,49,0.26);
        transform: translateY(-1px);
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(246,196,49,0.19);
        border-radius: 11px;
        overflow: hidden;
        background: rgba(10,10,10,0.76);
    }}

    .sr-responsive-wrap {{
        width: 100%;
        overflow: hidden;
        margin: 0.45rem 0 0.8rem 0;
        border: 1px solid rgba(246,196,49,0.22);
        border-radius: 12px;
        background: rgba(8,8,8,0.88);
    }}

    .sr-responsive-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
        color: #F4F4F4;
        font-size: 0.84rem;
    }}

    .sr-responsive-table th {{
        padding: 0.68rem 0.62rem;
        text-align: left;
        color: #D8D8D8;
        background: #0B0B0B;
        border-bottom: 1px solid rgba(246,196,49,0.28);
        font-weight: 800;
        white-space: nowrap;
    }}

    .sr-responsive-table td {{
        padding: 0.70rem 0.62rem;
        vertical-align: top;
        border-bottom: 1px solid rgba(255,255,255,0.075);
        line-height: 1.30;
    }}

    .sr-responsive-table tr:last-child td {{
        border-bottom: 0;
    }}

    .sr-responsive-table .sr-game-cell {{
        font-weight: 800;
        min-width: 220px;
    }}

    .sr-responsive-table .sr-qualified {{
        color: #FFD95A;
        font-weight: 900;
    }}

    .sr-responsive-table .sr-watch {{
        color: #D6D6D6;
        font-weight: 800;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid rgba(246,196,49,0.20);
        border-radius: 11px;
        background: rgba(13,13,13,0.78);
    }}

    div[data-testid="stAlert"] {{
        border-radius: 11px;
    }}

    [data-testid="stCaptionContainer"] {{
        color: #AFAFAF;
    }}

    hr {{
        border-color: rgba(246,196,49,0.16) !important;
    }}

    @media (max-width: 800px) {{
        .block-container {{
            padding-left: 0.72rem;
            padding-right: 0.72rem;
            padding-top: 0.65rem;
        }}

        .sr-brand-hero {{
            min-height: 205px;
            padding: 1rem;
            align-items: flex-end;
            background-position: 63% center;
        }}

        .sr-brand-logo {{
            width: 86px;
            height: 86px;
        }}

        .sr-brand-eyebrow {{
            font-size: 0.64rem;
            letter-spacing: 0.12em;
        }}

        .sr-brand-title {{
            font-size: 1.52rem;
        }}

        .sr-brand-subtitle {{
            font-size: 0.83rem;
            line-height: 1.35;
        }}

        div[data-testid="stButton"] > button {{
            width: 100%;
        }}

        .sr-play-card {{
            padding: 11px 12px !important;
        }}

        .sr-play-game {{
            font-size: 0.98rem !important;
        }}

        .sr-play-line {{
            font-size: 0.80rem !important;
        }}

        .sr-responsive-wrap {{
            border: 0;
            background: transparent;
            overflow: visible;
        }}

        .sr-responsive-table,
        .sr-responsive-table tbody,
        .sr-responsive-table tr,
        .sr-responsive-table td {{
            display: block;
            width: 100%;
        }}

        .sr-responsive-table thead {{
            display: none;
        }}

        .sr-responsive-table tr {{
            box-sizing: border-box;
            margin: 0 0 0.78rem 0;
            padding: 0.30rem 0.80rem;
            border: 1px solid rgba(246,196,49,0.30);
            border-radius: 12px;
            background:
                linear-gradient(
                    180deg,
                    rgba(24,24,24,0.98) 0%,
                    rgba(10,10,10,0.98) 100%
                );
            box-shadow: 0 5px 16px rgba(0,0,0,0.20);
        }}

        .sr-responsive-table td {{
            box-sizing: border-box;
            display: grid;
            grid-template-columns: minmax(104px, 38%) minmax(0, 62%);
            gap: 0.60rem;
            padding: 0.52rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }}

        .sr-responsive-table td:last-child {{
            border-bottom: 0;
        }}

        .sr-responsive-table td::before {{
            content: attr(data-label);
            color: #AFAFAF;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }}

        .sr-responsive-table .sr-game-cell {{
            min-width: 0;
            font-size: 0.91rem;
        }}
    }}

    @media (max-width: 520px) {{
        .sr-brand-hero {{
            display: block;
            min-height: 240px;
            background-position: 69% center;
        }}

        .sr-brand-logo {{
            width: 74px;
            height: 74px;
            margin-bottom: 0.65rem;
        }}

        .sr-brand-title {{
            font-size: 1.38rem;
        }}

        .sr-brand-subtitle {{
            max-width: 92%;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


if SHARPREPORT_LOGO_URI:

    logo_html = (
        f'<img class="sr-brand-logo" '
        f'src="{SHARPREPORT_LOGO_URI}" '
        f'alt="SharpReport logo">'
    )

else:

    logo_html = ""


st.markdown(
    f"""
    <div class="sr-brand-hero">
        {logo_html}
        <div class="sr-brand-copy">
            <div class="sr-brand-eyebrow">
                Analytics · Performance · Results
            </div>
            <div class="sr-brand-title">
                MLB <span>NRFI / YRFI</span> Scanner
            </div>
            <div class="sr-brand-subtitle">
                Trained probability model, confirmed-lineup inputs,
                live first-inning market pricing, executable edge,
                automated tracking, and forward model validation.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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


def render_responsive_results(
    rows,
    *,
    game_column="Game",
    qualification_column=None,
):
    """
    Render result rows as a normal table on desktop and as
    stacked full-width cards on mobile. Every field stays visible
    without horizontal scrolling.
    """

    if not rows:
        return

    columns = list(
        rows[0].keys()
    )

    header_html = "".join(
        f"<th>{html.escape(str(column))}</th>"
        for column in columns
    )

    body_rows = []

    for row in rows:

        cells = []

        for column in columns:

            value = row.get(
                column,
                "—",
            )

            if (
                value is None
                or value == ""
            ):
                value = "—"

            class_names = []

            if column == game_column:
                class_names.append(
                    "sr-game-cell"
                )

            if (
                qualification_column
                and
                column == qualification_column
            ):

                value_text = str(
                    value
                ).upper()

                if "QUALIFIED" in value_text:
                    class_names.append(
                        "sr-qualified"
                    )

                elif "WATCH" in value_text:
                    class_names.append(
                        "sr-watch"
                    )

            class_attr = (
                f' class="{" ".join(class_names)}"'
                if class_names
                else ""
            )

            cells.append(
                f'<td data-label="{html.escape(str(column))}"'
                f'{class_attr}>'
                f'{html.escape(str(value))}'
                f'</td>'
            )

        body_rows.append(
            "<tr>"
            + "".join(
                cells
            )
            + "</tr>"
        )

    st.markdown(
        """
        <div class="sr-responsive-wrap">
            <table class="sr-responsive-table">
                <thead>
                    <tr>
        """
        + header_html
        + """
                    </tr>
                </thead>
                <tbody>
        """
        + "".join(
            body_rows
        )
        + """
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


# Historical out-of-sample qualification line for Model v1.
# Probabilities in this app are stored in percentage-point form.
NRFI_PLAY_THRESHOLD = 57.0




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


                first_inning_rows = []

                # Context/research only. These rolling first-inning
                # windows are logged for future testing but are NOT
                # included in SharpReport Model v1 probabilities.
                pitcher_first_inning_windows = {}


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


                        try:

                            rolling_history = (
                                get_pitcher_first_inning_windows(
                                    pitcher_id,
                                    YEAR,
                                    history_end_date.isoformat(),
                                )
                            )

                        except Exception as error:

                            rolling_history = (
                                empty_pitcher_first_inning_windows(
                                    pitcher_id=
                                        pitcher_id,
                                    season=
                                        YEAR,
                                    end_date=
                                        history_end_date.isoformat(),
                                    error=
                                        error,
                                )
                            )


                        pitcher_first_inning_windows[
                            int(pitcher_id)
                        ] = rolling_history

                        history = rolling_history[
                            "season_window"
                        ]

                        starts = history[
                            "starts"
                        ]


                        if starts:

                            scoreless_rate = (
                                f"{history['scoreless_opponent_first_pct']:.1f}%"
                            )

                            runs_per_start = (
                                f"{history['first_inning_runs_allowed_per_start']:.2f}"
                            )

                            nrfi_record = (
                                f"{history['game_nrfi']}-"
                                f"{history['game_yrfi']}"
                            )

                            nrfi_rate = (
                                f"{history['game_nrfi_pct']:.1f}%"
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
                                    "scoreless_opponent_first"
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
                "Hitter Metrics",
                expanded=True,
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
            # ROLLING PITCHER FIRST-INNING RESEARCH CONTEXT
            # ---------------------------------------------

            for row in market_rows:

                away_pitcher_id = row.get(
                    "Away SP ID"
                )

                home_pitcher_id = row.get(
                    "Home SP ID"
                )

                away_history = (
                    pitcher_first_inning_windows.get(
                        int(away_pitcher_id)
                    )
                    if away_pitcher_id
                    else None
                )

                home_history = (
                    pitcher_first_inning_windows.get(
                        int(home_pitcher_id)
                    )
                    if home_pitcher_id
                    else None
                )

                row.update(
                    flatten_pitcher_first_inning_windows(
                        "Away Pitcher",
                        away_history,
                    )
                )

                row.update(
                    flatten_pitcher_first_inning_windows(
                        "Home Pitcher",
                        home_history,
                    )
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
            # AUTOMATIC COMPLETED-GAME GRADING
            # ---------------------------------------------

            if (
                github_data_token
                and
                github_data_repo
            ):

                try:

                    with st.spinner(
                        "Checking recent completed games "
                        "and grading saved FINAL pregame edges..."
                    ):

                        grading_summary = (
                            grade_recent_results(
                                token=
                                    github_data_token,

                                repo=
                                    github_data_repo,

                                reference_date=
                                    TODAY,

                                days_back=
                                    7,
                            )
                        )


                    st.caption(
                        "Automatic result grading: "
                        f"{grading_summary['graded_games']} "
                        "completed games currently have a "
                        "matched FINAL pregame model + market snapshot; "
                        f"{grading_summary['result_files_updated']} "
                        "daily result files were updated on this run."
                    )

                except Exception as error:

                    st.warning(
                        "Pregame logging completed, but automatic "
                        "result grading could not finish: "
                        f"{error}"
                    )


            # ---------------------------------------------
            # MODEL GOVERNANCE / SAFE RETRAINING STATUS
            # ---------------------------------------------

            if (
                github_data_token
                and
                github_data_repo
            ):

                try:

                    update_model_governance(
                        token=
                            github_data_token,

                        repo=
                            github_data_repo,
                    )

                except Exception as error:

                    st.warning(
                        "The scanner completed, but model governance "
                        "could not be refreshed: "
                        f"{error}"
                    )


            # ---------------------------------------------
            # MODEL ATTENTION ALERTS
            # ---------------------------------------------

            if (
                github_data_token
                and
                github_data_repo
            ):

                try:

                    update_model_attention_alerts(
                        token=
                            github_data_token,

                        repo=
                            github_data_repo,
                    )

                    render_model_attention_banner(
                        token=
                            github_data_token,

                        repo=
                            github_data_repo,
                    )

                except Exception as error:

                    st.warning(
                        "The scanner completed, but the model-attention "
                        "alert layer could not be refreshed: "
                        f"{error}"
                    )


            # ---------------------------------------------
            # DAILY DECISION BOARD
            # ---------------------------------------------

            st.markdown(
                """
                <style>
                .sr-decision-title {
                    color: #FFD95A;
                    font-size: 1.75rem;
                    font-weight: 900;
                    letter-spacing: -0.025em;
                    margin-top: 0.35rem;
                    margin-bottom: 0.15rem;
                    text-shadow: 0 0 18px rgba(246,196,49,0.12);
                }

                .sr-decision-subtitle {
                    color: #BEBEBE;
                    margin-bottom: 1rem;
                }

                .sr-summary-card {
                    border: 1px solid rgba(246,196,49,0.28);
                    border-radius: 12px;
                    padding: 12px 14px;
                    min-height: 82px;
                    background:
                        linear-gradient(
                            145deg,
                            rgba(24,24,24,0.94),
                            rgba(10,10,10,0.92)
                        );
                    box-shadow:
                        inset 0 1px 0 rgba(255,255,255,0.03),
                        0 5px 16px rgba(0,0,0,0.22);
                }

                .sr-summary-label {
                    font-size: 0.76rem;
                    color: #D5B44D;
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                    margin-bottom: 4px;
                    font-weight: 800;
                }

                .sr-summary-value {
                    color: #FFFFFF;
                    font-size: 1.45rem;
                    font-weight: 900;
                    line-height: 1.1;
                }

                .sr-play-card {
                    background:
                        linear-gradient(
                            135deg,
                            #FFE978 0%,
                            #F6C431 52%,
                            #DDAA16 100%
                        );
                    color: #090909;
                    border: 2px solid #FFF0A0;
                    border-radius: 14px;
                    padding: 14px 16px;
                    margin: 8px 0 12px 0;
                    box-shadow:
                        0 8px 24px rgba(246,196,49,0.18),
                        inset 0 1px 0 rgba(255,255,255,0.55);
                }

                .sr-play-rank {
                    font-size: 0.76rem;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                    text-transform: uppercase;
                }

                .sr-play-game {
                    font-size: 1.08rem;
                    font-weight: 800;
                    margin-top: 2px;
                    margin-bottom: 7px;
                }

                .sr-play-line {
                    font-size: 0.92rem;
                    line-height: 1.45;
                }

                .sr-tier {
                    display: inline-block;
                    border: 1px solid rgba(0,0,0,0.30);
                    border-radius: 999px;
                    padding: 2px 8px;
                    margin-left: 6px;
                    font-size: 0.72rem;
                    font-weight: 800;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )


            final_games = [
                row
                for row in market_rows
                if row["Status"] == "FINAL"
            ]

            # Primary decision-board ranking:
            # FINAL games sorted ONLY by Model v1 NRFI probability.
            # Sportsbook price and market edge do not affect this order.
            final_nrfi_rows = [
                row
                for row in final_games
                if row.get(
                    "NRFI Probability"
                ) is not None
            ]

            final_nrfi_rows.sort(
                key=lambda row:
                    row.get(
                        "NRFI Probability",
                        -1.0,
                    ),
                reverse=True,
            )

            qualified_final_nrfi_rows = [
                row
                for row in final_nrfi_rows
                if row.get(
                    "NRFI Probability",
                    -1.0,
                ) >= NRFI_PLAY_THRESHOLD
            ]

            positive_final_rows = [
                row
                for row in final_games
                if (
                    row["Price Edge"] is not None
                    and
                    row["Price Edge"] > 0
                )
            ]

            positive_final_rows.sort(
                key=lambda row:
                    row["Price Edge"],
                reverse=True,
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
                reverse=True,
            )

            low_data_rows = [
                row
                for row in market_rows
                if str(
                    row.get(
                        "Status",
                        ""
                    )
                ).startswith(
                    "LOW DATA"
                )
            ]

            best_final_edge = (
                positive_final_rows[
                    0
                ][
                    "Price Edge"
                ]
                if positive_final_rows
                else None
            )


            st.markdown(
                '<div class="sr-decision-title">'
                'SharpReport Daily Decision Board'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="sr-decision-subtitle">'
                'Highest-probability FINAL NRFI games first. Market value '
                'remains a separate secondary view, and sections open by '
                'default so the full slate can be reviewed by scrolling.'
                '</div>',
                unsafe_allow_html=True,
            )


            summary_columns = st.columns(
                4
            )

            with summary_columns[0]:

                st.markdown(
                    f"""
                    <div class="sr-summary-card">
                        <div class="sr-summary-label">FINAL Games</div>
                        <div class="sr-summary-value">{len(final_games)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with summary_columns[1]:

                highest_final_nrfi = (
                    final_nrfi_rows[
                        0
                    ].get(
                        "NRFI Probability"
                    )
                    if final_nrfi_rows
                    else None
                )

                highest_final_nrfi_text = (
                    format_probability(
                        highest_final_nrfi
                    )
                    if highest_final_nrfi is not None
                    else "—"
                )

                st.markdown(
                    f"""
                    <div class="sr-summary-card">
                        <div class="sr-summary-label">Highest FINAL NRFI</div>
                        <div class="sr-summary-value">{highest_final_nrfi_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with summary_columns[2]:

                st.markdown(
                    f"""
                    <div class="sr-summary-card">
                        <div class="sr-summary-label">Qualified NRFI Plays</div>
                        <div class="sr-summary-value">{len(qualified_final_nrfi_rows)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with summary_columns[3]:

                st.markdown(
                    f"""
                    <div class="sr-summary-card">
                        <div class="sr-summary-label">Positive FINAL Edges</div>
                        <div class="sr-summary-value">{len(positive_final_rows)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ---------------------------------------------
            # HIGHEST-PROBABILITY FINAL NRFI PLAYS
            # ---------------------------------------------

            st.markdown(
                '<div style="font-size:1.55rem;font-weight:900;'
                'color:#FFD95A;margin-top:0.6rem;margin-bottom:0.15rem;">'
                'Highest Probability FINAL NRFI Board'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "FINAL games ranked strictly by Model v1 NRFI probability. "
                "A game is a Qualified NRFI Play only at 57.0% or higher. "
                "Sportsbook price, break-even probability, and market edge "
                "do NOT determine the probability ranking. On mobile, result "
                "tables automatically stack so every field is visible without "
                "sideways scrolling."
            )

            if final_nrfi_rows:

                for rank, row in enumerate(
                    final_nrfi_rows[:4],
                    start=1,
                ):

                    qualification = (
                        "QUALIFIED NRFI PLAY"
                        if row["NRFI Probability"]
                        >= NRFI_PLAY_THRESHOLD
                        else "WATCH — BELOW 57% QUALIFICATION"
                    )

                    st.markdown(
                        f"""
                        <div class="sr-play-card">
                            <div class="sr-play-rank">
                                NRFI MODEL RANK #{rank} · {qualification}
                            </div>
                            <div class="sr-play-game">
                                {row["Game"]} — NRFI
                            </div>
                            <div class="sr-play-line">
                                <b>Model NRFI Probability:</b> {
                                    format_probability(
                                        row["NRFI Probability"]
                                    )
                                }
                                &nbsp;&nbsp;|&nbsp;&nbsp;
                                <b>Inputs:</b> {
                                    f'{row["Input Completeness"]:.0f}%'
                                }
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                final_nrfi_display = []

                for rank, row in enumerate(
                    final_nrfi_rows[:4],
                    start=1,
                ):

                    qualification = (
                        "QUALIFIED PLAY"
                        if row["NRFI Probability"]
                        >= NRFI_PLAY_THRESHOLD
                        else "WATCH"
                    )

                    final_nrfi_display.append({
                        "Rank":
                            rank,

                        "Game":
                            row["Game"],

                        "Model NRFI":
                            format_probability(
                                row["NRFI Probability"]
                            ),

                        "Qualification":
                            qualification,

                        "Input Completeness":
                            f"{row['Input Completeness']:.0f}%",

                        "Status":
                            row["Status"],
                    })

                render_responsive_results(
                    final_nrfi_display,
                    qualification_column=
                        "Qualification",
                )

                if qualified_final_nrfi_rows:

                    st.success(
                        f"{len(qualified_final_nrfi_rows)} FINAL game(s) "
                        "currently meet the official 57.0% Model v1 "
                        "NRFI qualification threshold."
                    )

                else:

                    st.info(
                        "No FINAL game currently reaches the official "
                        "57.0% Model v1 NRFI qualification threshold. "
                        "The ranked games above are WATCHES, not plays."
                    )

            else:

                st.info(
                    "No FINAL games are available yet. Once both starting "
                    "pitchers and both confirmed Top-4 lineups are available, "
                    "the highest Model v1 NRFI probabilities will appear here."
                )


            # ---------------------------------------------
            # SECONDARY: BEST MARKET VALUE — FINAL
            # ---------------------------------------------

            st.subheader(
                "Best Market Value — FINAL"
            )

            st.caption(
                "Separate secondary view. FINAL games are ranked here by "
                "positive executable Price Edge. This section does not alter "
                "the NRFI probability ranking above."
            )

            if positive_final_rows:

                market_value_display = []

                for rank, row in enumerate(
                    positive_final_rows[:4],
                    start=1,
                ):

                    market_value_display.append({
                        "Value Rank":
                            rank,

                        "Game":
                            row["Game"],

                        "Side":
                            row["Model Side"],

                        "Model":
                            format_probability(
                                row["Model Probability"]
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

                        "Market Edge":
                            format_edge(
                                row["Edge"]
                            ),

                        "Sportsbook":
                            row["Best Book"]
                            or "—",
                    })

                render_responsive_results(
                    market_value_display
                )

            else:

                st.info(
                    "No positive FINAL Price Edges are available right now. "
                    "A game can still rank highly for NRFI probability even "
                    "when the current sportsbook price offers no positive edge."
                )


            # ---------------------------------------------
            # PROVISIONAL WATCH
            # ---------------------------------------------

            with st.expander(
                f"Provisional Market Watch ({len(provisional_rows)})",
                expanded=True,
            ):

                st.caption(
                    "Live market opportunities that are still waiting on "
                    "confirmed Top-4 lineups. These are watch-list items, "
                    "not FINAL plays."
                )

                provisional_display = []

                for rank, row in enumerate(
                    provisional_rows[:8],
                    start=1,
                ):

                    provisional_display.append({
                        "Watch Rank":
                            rank,

                        "Game":
                            row["Game"],

                        "Side":
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
                    })

                if provisional_display:

                    render_responsive_results(
                        provisional_display
                    )

                else:

                    st.info(
                        "No positive provisional Price Edges "
                        "are currently available."
                    )


            # ---------------------------------------------
            # ALL GAMES — DAILY OPERATING TABLE
            # ---------------------------------------------

            st.subheader(
                "All Games — Model + Market"
            )

            st.caption(
                "The full slate remains visible for transparency. "
                "Price Edge is model probability minus the break-even "
                "probability of the best available price."
            )

            all_game_display = []

            for row in market_rows:

                all_game_display.append({
                    "Rank":
                        row["Rank"],

                    "Game":
                        row["Game"],

                    "Side":
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

                    "Inputs":
                        f"{row['Input Completeness']:.0f}%",

                    "Status":
                        row["Status"],
                })


            render_responsive_results(
                all_game_display
            )


            # ---------------------------------------------
            # MODEL-ONLY RANKING
            # ---------------------------------------------

            with st.expander(
                "Highest Model Probability — Either Side",
                expanded=True,
            ):

                st.caption(
                    "The four strongest model-side probabilities before "
                    "considering sportsbook price. A high probability is "
                    "not automatically a betting edge."
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

                render_responsive_results(
                    top_model_display
                )


            # ---------------------------------------------
            # PERFORMANCE TRACKING — COLLAPSED
            # ---------------------------------------------

            with st.expander(
                "Performance Tracking — ROI, Edge Bands & CLV",
                expanded=True,
            ):

                if (
                    github_data_token
                    and
                    github_data_repo
                ):

                    try:

                        render_edge_performance_dashboard(
                            token=
                                github_data_token,

                            repo=
                                github_data_repo,
                        )

                    except Exception as error:

                        st.warning(
                            "The live scanner is working, but the "
                            "edge performance dashboard could not load: "
                            f"{error}"
                        )

                    try:

                        render_clv_dashboard(
                            token=
                                github_data_token,

                            repo=
                                github_data_repo,
                        )

                    except Exception as error:

                        st.warning(
                            "The live scanner is working, but the "
                            "near-close / CLV dashboard could not load: "
                            f"{error}"
                        )


            # ---------------------------------------------
            # MODEL RESEARCH & GOVERNANCE — COLLAPSED
            # ---------------------------------------------

            with st.expander(
                "Model Research & Governance",
                expanded=True,
            ):

                if (
                    github_data_token
                    and
                    github_data_repo
                ):

                    try:

                        render_model_attention_dashboard(
                            token=
                                github_data_token,

                            repo=
                                github_data_repo,
                        )

                    except Exception as error:

                        st.warning(
                            "The live scanner is working, but the model "
                            "attention dashboard could not load: "
                            f"{error}"
                        )

                    try:

                        render_model_governance_dashboard(
                            token=
                                github_data_token,

                            repo=
                                github_data_repo,
                        )

                    except Exception as error:

                        st.warning(
                            "The live scanner is working, but the model "
                            "governance dashboard could not load: "
                            f"{error}"
                        )

                    try:

                        render_pitcher_history_research(
                            token=
                                github_data_token,

                            repo=
                                github_data_repo,
                        )

                    except Exception as error:

                        st.warning(
                            "The live scanner is working, but the rolling "
                            "pitcher-history research monitor could not load: "
                            f"{error}"
                        )


            # ---------------------------------------------
            # HALF-INNING PROBABILITY DETAIL
            # ---------------------------------------------

            with st.expander(
                "View Half-Inning Probability Detail",
                expanded=True,
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