import io
import json
import math
import os
import tomllib

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

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
    read_json_file,
    save_slate_snapshot,
    upsert_json_file,
)

from nrfi_result_grader import (
    grade_recent_results,
)

from nrfi_model_governance import (
    update_model_governance,
)

from nrfi_model_alerts import (
    update_model_attention_alerts,
)

from park_factors import (
    get_park_factors,
)

from pitcher_first_inning_windows import (
    get_pitcher_first_inning_windows,
    empty_pitcher_first_inning_windows,
    flatten_pitcher_first_inning_windows,
)


ET = ZoneInfo("America/New_York")

SECRETS_FILE = Path(
    ".streamlit/secrets.toml"
)

CAPTURE_MAX_MINUTES = int(
    os.getenv(
        "NRFI_CAPTURE_MAX_MINUTES",
        "60",
    )
)

CAPTURE_MIN_MINUTES = int(
    os.getenv(
        "NRFI_CAPTURE_MIN_MINUTES",
        "5",
    )
)


CLOSE_MAX_MINUTES = int(
    os.getenv(
        "NRFI_CLOSE_MAX_MINUTES",
        "25",
    )
)

CLOSE_MIN_MINUTES = int(
    os.getenv(
        "NRFI_CLOSE_MIN_MINUTES",
        "0",
    )
)

CLV_MIN_ENTRY_PRICE_EDGE = float(
    os.getenv(
        "NRFI_CLV_MIN_PRICE_EDGE",
        "2.0",
    )
)


# =========================================================
# CONFIG
# =========================================================

def load_secret(
    name,
):
    env_value = os.getenv(
        name
    )

    if env_value:
        return str(
            env_value
        ).strip()

    if SECRETS_FILE.exists():
        with SECRETS_FILE.open(
            "rb"
        ) as handle:
            data = tomllib.load(
                handle
            )

        value = data.get(
            name
        )

        if value:
            return str(
                value
            ).strip()

    return None


def load_config():
    config = {
        "odds_api_key":
            load_secret(
                "ODDS_API_KEY"
            ),

        "github_data_token":
            load_secret(
                "GITHUB_DATA_TOKEN"
            ),

        "github_data_repo":
            load_secret(
                "GITHUB_DATA_REPO"
            ),
    }

    missing = [
        name
        for name, value in config.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required configuration: "
            + ", ".join(
                missing
            )
        )

    return config


# =========================================================
# BASIC REQUESTS
# =========================================================

def get_json(
    url,
):
    request = Request(
        url,
        headers={
            "User-Agent":
                "SharpReport-NRFI-Scheduled-Collector/1.0"
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        return json.load(
            response
        )


def get_csv(
    url,
):
    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
    )

    with urlopen(
        request,
        timeout=45,
    ) as response:
        text = (
            response.read()
            .decode(
                "utf-8"
            )
        )

    data = pd.read_csv(
        io.StringIO(
            text
        )
    )

    data.columns = (
        data.columns
        .str.strip()
    )

    return data


# =========================================================
# MLB SCHEDULE / LINEUPS
# =========================================================

def get_schedule(
    game_date,
):
    date_text = (
        game_date.strftime(
            "%Y-%m-%d"
        )
    )

    url = (
        "https://statsapi.mlb.com/"
        "api/v1/schedule"
        "?sportId=1"
        f"&date={date_text}"
        "&hydrate=probablePitcher"
    )

    payload = get_json(
        url
    )

    games = []

    for date_block in payload.get(
        "dates",
        []
    ):
        for game in date_block.get(
            "games",
            []
        ):
            away_data = (
                game
                .get("teams", {})
                .get("away", {})
            )

            home_data = (
                game
                .get("teams", {})
                .get("home", {})
            )

            away_team = (
                away_data
                .get("team", {})
                .get(
                    "name",
                    "Away",
                )
            )

            home_team = (
                home_data
                .get("team", {})
                .get(
                    "name",
                    "Home",
                )
            )

            away_probable = (
                away_data.get(
                    "probablePitcher",
                    {},
                )
            )

            home_probable = (
                home_data.get(
                    "probablePitcher",
                    {},
                )
            )

            venue_data = game.get(
                "venue",
                {},
            )

            games.append({
                "Game":
                    f"{away_team} @ {home_team}",

                "Game ID":
                    game.get(
                        "gamePk"
                    ),

                "Game Date":
                    game.get(
                        "gameDate"
                    ),

                "Away Team":
                    away_team,

                "Home Team":
                    home_team,

                "Away SP":
                    away_probable.get(
                        "fullName",
                        "TBA",
                    ),

                "Away SP ID":
                    away_probable.get(
                        "id"
                    ),

                "Home SP":
                    home_probable.get(
                        "fullName",
                        "TBA",
                    ),

                "Home SP ID":
                    home_probable.get(
                        "id"
                    ),

                "Venue":
                    venue_data.get(
                        "name",
                        "",
                    ),

                "Venue ID":
                    venue_data.get(
                        "id"
                    ),

                "Status":
                    (
                        game
                        .get("status", {})
                        .get(
                            "detailedState",
                            "",
                        )
                    ),
            })

    return games


def parse_game_time(
    game_date,
):
    if not game_date:
        return None

    try:
        return (
            datetime.fromisoformat(
                str(
                    game_date
                ).replace(
                    "Z",
                    "+00:00",
                )
            )
            .astimezone(
                ET
            )
        )
    except Exception:
        return None


def minutes_until_game(
    game,
    now_et,
):
    game_time = parse_game_time(
        game.get(
            "Game Date"
        )
    )

    if game_time is None:
        return None

    return (
        game_time
        - now_et
    ).total_seconds() / 60.0


def get_team_top4(
    team_data,
):
    batting_order = team_data.get(
        "battingOrder",
        [],
    )

    players = team_data.get(
        "players",
        {},
    )

    top4 = []

    for player_id in batting_order[
        :4
    ]:
        player = players.get(
            f"ID{player_id}",
            {},
        )

        name = (
            player
            .get("person", {})
            .get(
                "fullName"
            )
        )

        if name:
            top4.append({
                "id":
                    int(
                        player_id
                    ),

                "name":
                    name,
            })

    return top4


def get_lineups(
    game_pk,
):
    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/game/{game_pk}/boxscore"
    )

    try:
        data = get_json(
            url
        )

        teams = data.get(
            "teams",
            {},
        )

        return (
            get_team_top4(
                teams.get(
                    "away",
                    {},
                )
            ),
            get_team_top4(
                teams.get(
                    "home",
                    {},
                )
            ),
        )

    except Exception:
        return [], []


def find_final_capture_candidates(
    games,
    now_et,
    captured_game_ids,
):
    candidates = []

    for game in games:
        minutes = minutes_until_game(
            game,
            now_et,
        )

        if minutes is None:
            continue

        if (
            minutes
            < CAPTURE_MIN_MINUTES
            or
            minutes
            > CAPTURE_MAX_MINUTES
        ):
            continue

        game_id = game.get(
            "Game ID"
        )

        if str(
            game_id
        ) in captured_game_ids:
            continue

        if (
            not game.get(
                "Away SP ID"
            )
            or
            not game.get(
                "Home SP ID"
            )
        ):
            continue

        away_top4, home_top4 = (
            get_lineups(
                game_id
            )
        )

        if (
            len(
                away_top4
            ) != 4
            or
            len(
                home_top4
            ) != 4
        ):
            continue

        game = dict(
            game
        )

        game[
            "Away Top 4"
        ] = away_top4

        game[
            "Home Top 4"
        ] = home_top4

        game[
            "Lineups"
        ] = "✅ Confirmed"

        game[
            "Minutes Before First Pitch"
        ] = minutes

        game_time = parse_game_time(
            game.get(
                "Game Date"
            )
        )

        game[
            "Start Time"
        ] = (
            game_time
            .strftime(
                "%I:%M %p ET"
            )
            .lstrip(
                "0"
            )
            if game_time is not None
            else "TBA"
        )

        candidates.append(
            game
        )

    return candidates


def find_near_close_candidates(
    games,
    now_et,
    initial_capture_state,
    near_close_capture_state,
):
    candidates = []

    for game in games:
        game_id = game.get(
            "Game ID"
        )

        game_key = str(
            game_id
        )

        entry = initial_capture_state.get(
            game_key
        )

        if not entry:
            continue

        if game_key in near_close_capture_state:
            continue

        try:
            entry_price_edge = float(
                entry.get(
                    "price_edge"
                )
            )
        except Exception:
            continue

        # Spend a second Odds API credit only on actual edge candidates.
        if (
            entry_price_edge
            < CLV_MIN_ENTRY_PRICE_EDGE
        ):
            continue

        minutes = minutes_until_game(
            game,
            now_et,
        )

        if minutes is None:
            continue

        if (
            minutes
            < CLOSE_MIN_MINUTES
            or
            minutes
            > CLOSE_MAX_MINUTES
        ):
            continue

        if (
            not game.get(
                "Away SP ID"
            )
            or
            not game.get(
                "Home SP ID"
            )
        ):
            continue

        away_top4, home_top4 = (
            get_lineups(
                game_id
            )
        )

        if (
            len(
                away_top4
            ) != 4
            or
            len(
                home_top4
            ) != 4
        ):
            continue

        game = dict(
            game
        )

        game[
            "Away Top 4"
        ] = away_top4

        game[
            "Home Top 4"
        ] = home_top4

        game[
            "Lineups"
        ] = "✅ Confirmed"

        game[
            "Minutes Before First Pitch"
        ] = minutes

        game_time = parse_game_time(
            game.get(
                "Game Date"
            )
        )

        game[
            "Start Time"
        ] = (
            game_time
            .strftime(
                "%I:%M %p ET"
            )
            .lstrip(
                "0"
            )
            if game_time is not None
            else "TBA"
        )

        candidates.append(
            game
        )

    return candidates


# =========================================================
# LIVE SEASON METRICS
# =========================================================

def get_savant_pitcher_xwoba(
    year,
):
    url = (
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

    data = get_csv(
        url
    )

    lookup = {}

    for _, row in data.iterrows():
        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "est_woba"
        )

        if (
            pd.notna(
                player_id
            )
            and
            pd.notna(
                value
            )
        ):
            lookup[
                int(
                    player_id
                )
            ] = float(
                value
            )

    return lookup


def get_savant_hitter_data(
    year,
):
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
            pd.notna(
                player_id
            )
            and
            pd.notna(
                value
            )
        ):
            xwoba[
                int(
                    player_id
                )
            ] = float(
                value
            )

    for _, row in barrels.iterrows():
        player_id = row.get(
            "player_id"
        )

        value = row.get(
            "brl_percent"
        )

        if (
            pd.notna(
                player_id
            )
            and
            pd.notna(
                value
            )
        ):
            barrel_percent[
                int(
                    player_id
                )
            ] = float(
                value
            )

    return (
        xwoba,
        barrel_percent,
    )


@lru_cache(
    maxsize=None
)
def get_mlb_pitcher_rates(
    player_id,
    year,
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
        data = get_json(
            url
        )

        stat = (
            data[
                "stats"
            ][0][
                "splits"
            ][0][
                "stat"
            ]
        )

        strikeouts = stat.get(
            "strikeOuts",
            0,
        )

        walks = stat.get(
            "baseOnBalls",
            0,
        )

        batters_faced = stat.get(
            "battersFaced",
            0,
        )

        if not batters_faced:
            return None, None, 0

        return (
            strikeouts
            / batters_faced
            * 100.0,

            walks
            / batters_faced
            * 100.0,

            batters_faced,
        )

    except Exception:
        return None, None, None


@lru_cache(
    maxsize=None
)
def get_mlb_hitter_rates(
    player_id,
    year,
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
        data = get_json(
            url
        )

        stat = (
            data[
                "stats"
            ][0][
                "splits"
            ][0][
                "stat"
            ]
        )

        plate_appearances = stat.get(
            "plateAppearances",
            0,
        )

        strikeouts = stat.get(
            "strikeOuts",
            0,
        )

        walks = stat.get(
            "baseOnBalls",
            0,
        )

        if not plate_appearances:
            return None, None, 0

        return (
            strikeouts
            / plate_appearances
            * 100.0,

            walks
            / plate_appearances
            * 100.0,

            plate_appearances,
        )

    except Exception:
        return None, None, None


def strict_average(
    values,
):
    if (
        not values
        or
        any(
            value is None
            for value in values
        )
    ):
        return None

    return (
        sum(
            float(
                value
            )
            for value in values
        )
        / len(
            values
        )
    )


def value_available(
    value,
):
    if value is None:
        return False

    try:
        return bool(
            pd.notna(
                value
            )
        )
    except Exception:
        return False


def pitcher_inputs(
    pitcher_id,
    pitcher_xwoba,
    year,
):
    k_percent, _bb_percent, pa = (
        get_mlb_pitcher_rates(
            int(
                pitcher_id
            ),
            int(
                year
            ),
        )
    )

    return {
        "xwoba":
            pitcher_xwoba.get(
                int(
                    pitcher_id
                )
            ),

        "k_pct":
            k_percent,

        "pa":
            pa or 0,
    }


def top4_inputs(
    top4,
    hitter_xwoba,
    hitter_barrels,
    year,
):
    xwoba_values = []
    k_values = []
    bb_values = []
    barrel_values = []
    pa_values = []
    hitter_rows = []

    for player in top4:
        player_id = int(
            player[
                "id"
            ]
        )

        (
            k_percent,
            bb_percent,
            plate_appearances,
        ) = get_mlb_hitter_rates(
            player_id,
            int(
                year
            ),
        )

        player_xwoba = (
            hitter_xwoba.get(
                player_id
            )
        )

        player_barrel = (
            hitter_barrels.get(
                player_id
            )
        )

        xwoba_values.append(
            player_xwoba
        )

        k_values.append(
            k_percent
        )

        bb_values.append(
            bb_percent
        )

        barrel_values.append(
            player_barrel
        )

        pa_values.append(
            plate_appearances
            or 0
        )

        hitter_rows.append({
            "player_id":
                player_id,

            "name":
                player.get(
                    "name"
                ),

            "xwoba":
                player_xwoba,

            "k_pct":
                k_percent,

            "bb_pct":
                bb_percent,

            "barrel_pct":
                player_barrel,

            "pa":
                plate_appearances
                or 0,
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
            sum(
                pa_values
            ),

        "min_pa":
            min(
                pa_values
            )
            if pa_values
            else 0,

        "complete_core":
            complete_core,

        "hitters":
            hitter_rows,
    }


# =========================================================
# MODEL PROBABILITIES
# =========================================================

def build_probability_rows(
    games,
    pitcher_xwoba,
    hitter_xwoba,
    hitter_barrels,
    park_lookup,
    trained_model,
    year,
):
    rows = []

    for game in games:
        away_pitcher = pitcher_inputs(
            game[
                "Away SP ID"
            ],
            pitcher_xwoba,
            year,
        )

        home_pitcher = pitcher_inputs(
            game[
                "Home SP ID"
            ],
            pitcher_xwoba,
            year,
        )

        away_offense = top4_inputs(
            game[
                "Away Top 4"
            ],
            hitter_xwoba,
            hitter_barrels,
            year,
        )

        home_offense = top4_inputs(
            game[
                "Home Top 4"
            ],
            hitter_xwoba,
            hitter_barrels,
            year,
        )

        venue_id = game.get(
            "Venue ID"
        )

        park = (
            park_lookup.get(
                int(
                    venue_id
                )
            )
            if venue_id
            else None
        )

        run_factor = (
            park.get(
                "run_factor"
            )
            if park
            else None
        )

        top_features = build_half_features(
            pitcher_xwoba=
                home_pitcher[
                    "xwoba"
                ],

            pitcher_k_pct=
                home_pitcher[
                    "k_pct"
                ],

            pitcher_pa=
                home_pitcher[
                    "pa"
                ],

            offense_xwoba=
                away_offense[
                    "xwoba"
                ],

            offense_k_pct=
                away_offense[
                    "k_pct"
                ],

            offense_bb_pct=
                away_offense[
                    "bb_pct"
                ],

            offense_barrel_pct=
                away_offense[
                    "barrel_pct"
                ],

            offense_combined_pa=
                away_offense[
                    "combined_pa"
                ],

            offense_min_pa=
                away_offense[
                    "min_pa"
                ],

            offense_complete_core=
                away_offense[
                    "complete_core"
                ],

            park_runs=
                run_factor,
        )

        bottom_features = build_half_features(
            pitcher_xwoba=
                away_pitcher[
                    "xwoba"
                ],

            pitcher_k_pct=
                away_pitcher[
                    "k_pct"
                ],

            pitcher_pa=
                away_pitcher[
                    "pa"
                ],

            offense_xwoba=
                home_offense[
                    "xwoba"
                ],

            offense_k_pct=
                home_offense[
                    "k_pct"
                ],

            offense_bb_pct=
                home_offense[
                    "bb_pct"
                ],

            offense_barrel_pct=
                home_offense[
                    "barrel_pct"
                ],

            offense_combined_pa=
                home_offense[
                    "combined_pa"
                ],

            offense_min_pa=
                home_offense[
                    "min_pa"
                ],

            offense_complete_core=
                home_offense[
                    "complete_core"
                ],

            park_runs=
                run_factor,
        )

        result = predict_nrfi_yrfi(
            trained_model,
            top_features=
                top_features,
            bottom_features=
                bottom_features,
        )

        completeness_values = [
            away_pitcher[
                "xwoba"
            ],
            away_pitcher[
                "k_pct"
            ],
            home_pitcher[
                "xwoba"
            ],
            home_pitcher[
                "k_pct"
            ],
            away_offense[
                "xwoba"
            ],
            away_offense[
                "k_pct"
            ],
            away_offense[
                "bb_pct"
            ],
            away_offense[
                "barrel_pct"
            ],
            home_offense[
                "xwoba"
            ],
            home_offense[
                "k_pct"
            ],
            home_offense[
                "bb_pct"
            ],
            home_offense[
                "barrel_pct"
            ],
            run_factor,
        ]

        available = sum(
            1
            for value in completeness_values
            if value_available(
                value
            )
        )

        completeness = (
            available
            / len(
                completeness_values
            )
            * 100.0
        )

        rows.append({
            "Game":
                game[
                    "Game"
                ],

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
                result[
                    "model_side"
                ],

            "Model Probability":
                result[
                    "model_probability"
                ]
                * 100.0,

            "NRFI Probability":
                result[
                    "nrfi_probability"
                ]
                * 100.0,

            "YRFI Probability":
                result[
                    "yrfi_probability"
                ]
                * 100.0,

            "Top 1st Score Probability":
                result[
                    "top_scoring_probability"
                ]
                * 100.0,

            "Bottom 1st Score Probability":
                result[
                    "bottom_scoring_probability"
                ]
                * 100.0,

            "Input Completeness":
                completeness,

            "Status":
                "FINAL",

            "Model Run Factor":
                run_factor,

            "Start Time":
                game.get(
                    "Start Time"
                ),

            "Venue":
                game.get(
                    "Venue"
                ),

            "Scheduled Capture Minutes Before First Pitch":
                round(
                    game.get(
                        "Minutes Before First Pitch",
                        0.0,
                    ),
                    2,
                ),
        })

    rows.sort(
        key=lambda row:
            row[
                "Model Probability"
            ],
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "Rank"
        ] = rank

    return rows


# =========================================================
# MARKET
# =========================================================

def attach_market(
    games,
    probability_rows,
    api_key,
):
    probability_lookup = {
        row[
            "Game"
        ]:
            row
        for row in probability_rows
    }

    events, event_usage = (
        get_mlb_events(
            api_key
        )
    )

    output_rows = []
    last_usage = event_usage

    for game in games:
        row = dict(
            probability_lookup[
                game[
                    "Game"
                ]
            ]
        )

        odds_event = find_odds_event(
            events,
            game[
                "Away Team"
            ],
            game[
                "Home Team"
            ],
        )

        if not odds_event:
            print(
                f"Market event not found: "
                f"{game['Game']}"
            )
            continue

        try:
            event_odds, usage = (
                get_first_inning_event_odds(
                    api_key,
                    odds_event[
                        "id"
                    ],
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
                print(
                    f"1st-inning market not posted: "
                    f"{game['Game']}"
                )
                continue

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

            nrfi_best_book = (
                market_summary[
                    "best_nrfi_book"
                ]
            )

            yrfi_best_book = (
                market_summary[
                    "best_yrfi_book"
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
                row[
                    "NRFI Probability"
                ]
                / 100.0
            )

            yrfi_model_probability = (
                row[
                    "YRFI Probability"
                ]
                / 100.0
            )

            if row[
                "Model Side"
            ] == "NRFI":
                market_no_vig = (
                    nrfi_market_no_vig
                )

                best_price = (
                    nrfi_best_price
                )

                best_book = (
                    nrfi_best_book
                )

                raw_implied = (
                    nrfi_break_even
                )

                model_probability = (
                    nrfi_model_probability
                )

            else:
                market_no_vig = (
                    yrfi_market_no_vig
                )

                best_price = (
                    yrfi_best_price
                )

                best_book = (
                    yrfi_best_book
                )

                raw_implied = (
                    yrfi_break_even
                )

                model_probability = (
                    yrfi_model_probability
                )

            row.update({
                "Market No-Vig":
                    market_no_vig
                    * 100.0,

                "Market Raw Implied":
                    raw_implied
                    * 100.0
                    if raw_implied is not None
                    else None,

                "Edge":
                    (
                        model_probability
                        - market_no_vig
                    )
                    * 100.0,

                "Price Edge":
                    (
                        (
                            model_probability
                            - raw_implied
                        )
                        * 100.0
                        if raw_implied is not None
                        else None
                    ),

                "Market NRFI No-Vig":
                    nrfi_market_no_vig
                    * 100.0,

                "Market YRFI No-Vig":
                    yrfi_market_no_vig
                    * 100.0,

                "Best NRFI Price":
                    nrfi_best_price,

                "Best NRFI Book":
                    nrfi_best_book,

                "Best YRFI Price":
                    yrfi_best_price,

                "Best YRFI Book":
                    yrfi_best_book,

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

            output_rows.append(
                row
            )

        except Exception as error:
            print(
                f"Market error for "
                f"{game['Game']}: {error}"
            )

    return output_rows, last_usage


# =========================================================
# AUTOMATION STATE
# =========================================================

def capture_state_path(
    game_date,
):
    return (
        f"automation/captured/"
        f"{game_date.isoformat()}.json"
    )


def load_capture_state(
    token,
    repo,
    game_date,
):
    path = capture_state_path(
        game_date
    )

    loaded = read_json_file(
        token=token,
        repo=repo,
        path=path,
    )

    if not loaded:
        return {
            "game_date":
                game_date.isoformat(),

            "captured":
                {},
        }

    payload = loaded.get(
        "data",
        {},
    )

    if not isinstance(
        payload.get(
            "captured"
        ),
        dict,
    ):
        payload[
            "captured"
        ] = {}

    return payload


def save_capture_state(
    token,
    repo,
    game_date,
    state,
):
    path = capture_state_path(
        game_date
    )

    return upsert_json_file(
        token=token,
        repo=repo,
        path=path,
        payload=state,
        commit_message=(
            "Update scheduled NRFI capture state "
            f"{game_date.isoformat()}"
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():
    config = load_config()

    now_et = datetime.now(
        ET
    )

    today = now_et.date()
    year = today.year

    print(
        "SharpReport scheduled collector"
    )

    print(
        f"Current ET: "
        f"{now_et.isoformat()}"
    )

    state = load_capture_state(
        token=
            config[
                "github_data_token"
            ],

        repo=
            config[
                "github_data_repo"
            ],

        game_date=
            today,
    )

    captured = state.get(
        "captured",
        {},
    )

    near_close_captured = state.get(
        "near_close_captured",
        {},
    )

    # A game must have been captured on a PRIOR automation run
    # before it can receive a separate near-close pull.
    captured_before_run = dict(
        captured
    )

    games = get_schedule(
        today
    )

    print(
        f"MLB games on slate: "
        f"{len(games)}"
    )

    initial_candidates = (
        find_final_capture_candidates(
            games=
                games,

            now_et=
                now_et,

            captured_game_ids=
                set(
                    captured.keys()
                ),
        )
    )

    print(
        f"FINAL uncaptured games within "
        f"{CAPTURE_MIN_MINUTES}-"
        f"{CAPTURE_MAX_MINUTES} minutes: "
        f"{len(initial_candidates)}"
    )

    near_close_candidates = (
        find_near_close_candidates(
            games=
                games,

            now_et=
                now_et,

            initial_capture_state=
                captured_before_run,

            near_close_capture_state=
                near_close_captured,
        )
    )

    print(
        f"Near-close candidates within "
        f"{CLOSE_MIN_MINUTES}-"
        f"{CLOSE_MAX_MINUTES} minutes "
        f"with entry Price Edge >= "
        f"{CLV_MIN_ENTRY_PRICE_EDGE:.1f}%: "
        f"{len(near_close_candidates)}"
    )


    all_model_candidates = []

    seen_game_ids = set()

    for game in (
        initial_candidates
        +
        near_close_candidates
    ):
        game_key = str(
            game.get(
                "Game ID"
            )
        )

        if game_key in seen_game_ids:
            continue

        seen_game_ids.add(
            game_key
        )

        all_model_candidates.append(
            game
        )


    last_usage = None

    if all_model_candidates:

        print(
            "Loading current pitcher/hitter "
            "metrics and model inputs..."
        )

        pitcher_xwoba = (
            get_savant_pitcher_xwoba(
                year
            )
        )

        (
            hitter_xwoba,
            hitter_barrels,
        ) = get_savant_hitter_data(
            year
        )

        park_lookup = (
            get_park_factors(
                year - 1
            )
        )

        trained_model = (
            load_nrfi_model()
        )

        probability_rows = (
            build_probability_rows(
                games=
                    all_model_candidates,

                pitcher_xwoba=
                    pitcher_xwoba,

                hitter_xwoba=
                    hitter_xwoba,

                hitter_barrels=
                    hitter_barrels,

                park_lookup=
                    park_lookup,

                trained_model=
                    trained_model,

                year=
                    year,
            )
        )


        # Rolling pitcher first-inning history remains
        # research-only and does NOT alter Model v1.
        history_end_date = (
            today
            - timedelta(
                days=1
            )
        )

        pitcher_history_lookup = {}

        unique_pitcher_ids = sorted({
            int(pitcher_id)
            for game in all_model_candidates
            for pitcher_id in [
                game.get(
                    "Away SP ID"
                ),
                game.get(
                    "Home SP ID"
                ),
            ]
            if pitcher_id
        })

        for pitcher_id in unique_pitcher_ids:

            try:

                pitcher_history_lookup[
                    pitcher_id
                ] = get_pitcher_first_inning_windows(
                    pitcher_id,
                    year,
                    history_end_date.isoformat(),
                )

            except Exception as error:

                pitcher_history_lookup[
                    pitcher_id
                ] = empty_pitcher_first_inning_windows(
                    pitcher_id=
                        pitcher_id,

                    season=
                        year,

                    end_date=
                        history_end_date.isoformat(),

                    error=
                        error,
                )


        for row in probability_rows:

            away_pitcher_id = row.get(
                "Away SP ID"
            )

            home_pitcher_id = row.get(
                "Home SP ID"
            )

            away_history = (
                pitcher_history_lookup.get(
                    int(
                        away_pitcher_id
                    )
                )
                if away_pitcher_id
                else None
            )

            home_history = (
                pitcher_history_lookup.get(
                    int(
                        home_pitcher_id
                    )
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


        print(
            "Rolling first-inning history logged for "
            f"{len(unique_pitcher_ids)} pitcher(s) "
            "(Season/L30/L20/L10; research only)."
        )


        probability_lookup = {
            row[
                "Game"
            ]:
                row
            for row in probability_rows
        }


        # =================================================
        # INITIAL FINAL CAPTURE
        # =================================================

        if initial_candidates:

            print(
                "Fetching initial FINAL first-inning market "
                "for eligible games..."
            )

            initial_probability_rows = [
                probability_lookup[
                    game[
                        "Game"
                    ]
                ]
                for game in initial_candidates
            ]

            (
                initial_market_rows,
                initial_usage,
            ) = attach_market(
                games=
                    initial_candidates,

                probability_rows=
                    initial_probability_rows,

                api_key=
                    config[
                        "odds_api_key"
                    ],
            )

            if initial_market_rows:

                initial_metadata = {
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

                    "collection_mode":
                        "scheduled_final_pregame",

                    "capture_window_minutes":
                        {
                            "minimum":
                                CAPTURE_MIN_MINUTES,

                            "maximum":
                                CAPTURE_MAX_MINUTES,
                        },
                }

                initial_snapshot = (
                    save_slate_snapshot(
                        token=
                            config[
                                "github_data_token"
                            ],

                        repo=
                            config[
                                "github_data_repo"
                            ],

                        rows=
                            initial_market_rows,

                        snapshot_time=
                            now_et,

                        model_metadata=
                            initial_metadata,

                        odds_usage=
                            initial_usage,
                    )
                )

                for row in initial_market_rows:
                    game_key = str(
                        row[
                            "Game ID"
                        ]
                    )

                    captured[
                        game_key
                    ] = {
                        "game":
                            row[
                                "Game"
                            ],

                        "snapshot_path":
                            initial_snapshot.get(
                                "path"
                            ),

                        "snapshot_time_et":
                            now_et.isoformat(),

                        "minutes_before_first_pitch":
                            row.get(
                                "Scheduled Capture Minutes Before First Pitch"
                            ),

                        "model_side":
                            row.get(
                                "Model Side"
                            ),

                        "model_probability":
                            row.get(
                                "Model Probability"
                            ),

                        "market_no_vig":
                            row.get(
                                "Market No-Vig"
                            ),

                        "break_even":
                            row.get(
                                "Market Raw Implied"
                            ),

                        "price_edge":
                            row.get(
                                "Price Edge"
                            ),

                        "best_price":
                            row.get(
                                "Best Price"
                            ),

                        "best_book":
                            row.get(
                                "Best Book"
                            ),
                    }

                last_usage = initial_usage

                print(
                    f"Saved initial FINAL snapshot: "
                    f"{initial_snapshot.get('path')}"
                )

                print(
                    f"Initial games captured this run: "
                    f"{len(initial_market_rows)}"
                )

            else:

                print(
                    "No initial FINAL candidates had a live "
                    "first-inning market."
                )


        # =================================================
        # ONE NEAR-CLOSE CAPTURE FOR QUALIFIED EDGE GAMES
        # =================================================

        if near_close_candidates:

            print(
                "Fetching one near-close first-inning market "
                "for qualified edge candidates..."
            )

            close_probability_rows = [
                probability_lookup[
                    game[
                        "Game"
                    ]
                ]
                for game in near_close_candidates
            ]

            (
                close_market_rows,
                close_usage,
            ) = attach_market(
                games=
                    near_close_candidates,

                probability_rows=
                    close_probability_rows,

                api_key=
                    config[
                        "odds_api_key"
                    ],
            )

            if close_market_rows:

                close_metadata = {
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

                    "collection_mode":
                        "scheduled_near_close",

                    "entry_price_edge_minimum":
                        CLV_MIN_ENTRY_PRICE_EDGE,

                    "capture_window_minutes":
                        {
                            "minimum":
                                CLOSE_MIN_MINUTES,

                            "maximum":
                                CLOSE_MAX_MINUTES,
                        },
                }

                close_snapshot = (
                    save_slate_snapshot(
                        token=
                            config[
                                "github_data_token"
                            ],

                        repo=
                            config[
                                "github_data_repo"
                            ],

                        rows=
                            close_market_rows,

                        snapshot_time=
                            now_et,

                        model_metadata=
                            close_metadata,

                        odds_usage=
                            close_usage,
                    )
                )

                for row in close_market_rows:
                    game_key = str(
                        row[
                            "Game ID"
                        ]
                    )

                    entry = captured_before_run.get(
                        game_key,
                        {},
                    )

                    entry_side = entry.get(
                        "model_side"
                    )

                    if entry_side == "NRFI":
                        close_price = row.get(
                            "Best NRFI Price"
                        )

                        close_book = row.get(
                            "Best NRFI Book"
                        )

                        close_no_vig = row.get(
                            "Market NRFI No-Vig"
                        )

                        close_break_even = row.get(
                            "NRFI Break-Even"
                        )

                    elif entry_side == "YRFI":
                        close_price = row.get(
                            "Best YRFI Price"
                        )

                        close_book = row.get(
                            "Best YRFI Book"
                        )

                        close_no_vig = row.get(
                            "Market YRFI No-Vig"
                        )

                        close_break_even = row.get(
                            "YRFI Break-Even"
                        )

                    else:
                        close_price = None
                        close_book = None
                        close_no_vig = None
                        close_break_even = None

                    near_close_captured[
                        game_key
                    ] = {
                        "game":
                            row[
                                "Game"
                            ],

                        "snapshot_path":
                            close_snapshot.get(
                                "path"
                            ),

                        "snapshot_time_et":
                            now_et.isoformat(),

                        "minutes_before_first_pitch":
                            row.get(
                                "Scheduled Capture Minutes Before First Pitch"
                            ),

                        "entry_model_side":
                            entry_side,

                        "entry_price_edge":
                            entry.get(
                                "price_edge"
                            ),

                        "close_price":
                            close_price,

                        "close_book":
                            close_book,

                        "close_no_vig":
                            close_no_vig,

                        "close_break_even":
                            close_break_even,
                    }

                last_usage = close_usage

                print(
                    f"Saved near-close snapshot: "
                    f"{close_snapshot.get('path')}"
                )

                print(
                    f"Near-close games captured this run: "
                    f"{len(close_market_rows)}"
                )

            else:

                print(
                    "No qualified near-close candidates had a "
                    "live first-inning market."
                )


        state[
            "captured"
        ] = captured

        state[
            "near_close_captured"
        ] = near_close_captured

        state[
            "updated_at_et"
        ] = now_et.isoformat()

        save_capture_state(
            token=
                config[
                    "github_data_token"
                ],

            repo=
                config[
                    "github_data_repo"
                ],

            game_date=
                today,

            state=
                state,
        )


        if last_usage:
            remaining = (
                last_usage.get(
                    "requests_remaining"
                )
            )

            if remaining is not None:
                print(
                    "Odds API credits remaining: "
                    f"{remaining}"
                )

    else:

        print(
            "No initial or near-close odds request needed on this run."
        )


    grading = grade_recent_results(
        token=
            config[
                "github_data_token"
            ],

        repo=
            config[
                "github_data_repo"
            ],

        reference_date=
            today,

        days_back=
            7,
    )

    print(
        "Automatic grading check complete: "
        f"{grading.get('graded_games', 0)} "
        "matched FINAL games in the recent window."
    )

    try:

        governance = update_model_governance(
            token=
                config[
                    "github_data_token"
                ],

            repo=
                config[
                    "github_data_repo"
                ],
        )

        print(
            "Model governance: "
            f"{governance.get('eligible_forward_games', 0)}/"
            f"{governance.get('minimum_games_for_challenger_test', 300)} "
            "eligible forward games; "
            f"status={governance.get('recommendation', 'COLLECTING')}."
        )

    except Exception as error:

        print(
            "Model governance update warning: "
            f"{error}"
        )


    try:

        alerts = update_model_attention_alerts(
            token=
                config[
                    "github_data_token"
                ],

            repo=
                config[
                    "github_data_repo"
                ],
        )

        print(
            "Model attention alerts: "
            f"status={alerts.get('overall_status', 'COLLECTING')}; "
            f"unreviewed={alerts.get('unreviewed_alert_count', 0)}; "
            f"research_watches="
            f"{len(alerts.get('research_watches', []))}."
        )

        if alerts.get(
            "unreviewed_alert_count",
            0,
        ):

            print(
                "ATTENTION: open the SharpReport scanner "
                "Model Research & Governance section for review."
            )

    except Exception as error:

        print(
            "Model attention alert update warning: "
            f"{error}"
        )


    print(
        "Scheduled collection run complete."
    )


if __name__ == "__main__":
    main()
