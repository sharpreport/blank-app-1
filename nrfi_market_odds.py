
import json
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


SPORT_KEY = "baseball_mlb"
MARKET_KEY = "totals_1st_1_innings"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"

SECRETS_FILE = Path(
    ".streamlit/secrets.toml"
)


# =========================================================
# API KEY
# =========================================================

def load_odds_api_key():
    """
    Load the key without ever printing it.

    Priority:
      1) ODDS_API_KEY environment variable
      2) .streamlit/secrets.toml
    """

    env_key = os.getenv(
        "ODDS_API_KEY"
    )

    if env_key:
        return env_key.strip()

    if SECRETS_FILE.exists():
        with SECRETS_FILE.open(
            "rb"
        ) as handle:
            data = tomllib.load(
                handle
            )

        key = data.get(
            "ODDS_API_KEY"
        )

        if key:
            return str(
                key
            ).strip()

    raise RuntimeError(
        "ODDS_API_KEY was not found. "
        "Add it to .streamlit/secrets.toml "
        "or the ODDS_API_KEY environment variable."
    )


# =========================================================
# REQUESTS
# =========================================================

def _api_get(
    url,
):
    request = Request(
        url,
        headers={
            "User-Agent":
                "SharpReport-NRFI-Scanner/1.0"
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:

            payload = json.load(
                response
            )

            usage = {
                "requests_remaining":
                    response.headers.get(
                        "x-requests-remaining"
                    ),

                "requests_used":
                    response.headers.get(
                        "x-requests-used"
                    ),

                "requests_last":
                    response.headers.get(
                        "x-requests-last"
                    ),
            }

            return payload, usage

    except HTTPError as error:
        body = (
            error.read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            f"The Odds API returned HTTP "
            f"{error.code}: {body}"
        ) from error


def get_mlb_events(
    api_key,
):
    params = urlencode({
        "apiKey":
            api_key,
    })

    url = (
        "https://api.the-odds-api.com/"
        f"v4/sports/{SPORT_KEY}/events"
        f"?{params}"
    )

    return _api_get(
        url
    )


def get_first_inning_event_odds(
    api_key,
    event_id,
    regions=DEFAULT_REGIONS,
):
    params = urlencode({
        "apiKey":
            api_key,

        "regions":
            regions,

        "markets":
            MARKET_KEY,

        "oddsFormat":
            DEFAULT_ODDS_FORMAT,

        "dateFormat":
            "iso",
    })

    url = (
        "https://api.the-odds-api.com/"
        f"v4/sports/{SPORT_KEY}/"
        f"events/{event_id}/odds"
        f"?{params}"
    )

    return _api_get(
        url
    )


# =========================================================
# TEAM MATCHING
# =========================================================

TEAM_ALIASES = {
    "oakland athletics":
        "athletics",

    "sacramento athletics":
        "athletics",

    "the athletics":
        "athletics",

    "la angels":
        "los angeles angels",

    "ny yankees":
        "new york yankees",

    "ny mets":
        "new york mets",
}


def normalize_team_name(
    name,
):
    text = (
        str(name)
        .strip()
        .lower()
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return TEAM_ALIASES.get(
        text,
        text,
    )


def find_odds_event(
    events,
    away_team,
    home_team,
):
    away_norm = normalize_team_name(
        away_team
    )

    home_norm = normalize_team_name(
        home_team
    )

    for event in events:
        event_away = normalize_team_name(
            event.get(
                "away_team",
                ""
            )
        )

        event_home = normalize_team_name(
            event.get(
                "home_team",
                ""
            )
        )

        if (
            event_away == away_norm
            and
            event_home == home_norm
        ):
            return event

    return None


# =========================================================
# MARKET MATH
# =========================================================

def american_implied_probability(
    price,
):
    price = float(
        price
    )

    if price == 0:
        return None

    if price < 0:
        return (
            abs(price)
            / (
                abs(price)
                + 100.0
            )
        )

    return (
        100.0
        / (
            price
            + 100.0
        )
    )


def no_vig_two_way(
    nrfi_price,
    yrfi_price,
):
    nrfi_raw = (
        american_implied_probability(
            nrfi_price
        )
    )

    yrfi_raw = (
        american_implied_probability(
            yrfi_price
        )
    )

    total = (
        nrfi_raw
        + yrfi_raw
    )

    if total <= 0:
        return None

    return {
        "nrfi_raw_implied":
            nrfi_raw,

        "yrfi_raw_implied":
            yrfi_raw,

        "nrfi_no_vig":
            nrfi_raw
            / total,

        "yrfi_no_vig":
            yrfi_raw
            / total,

        "hold":
            total
            - 1.0,
    }


def _point_is_half(
    outcome,
):
    point = outcome.get(
        "point"
    )

    # NRFI/YRFI is specifically the 0.5-run first-inning total.
    #
    # totals_1st_1_innings can contain a featured first-inning
    # total other than 0.5 (for example 1.5). If a bookmaker
    # omits the point, we cannot safely infer that it means 0.5.
    # Reject unknown points rather than risk treating an Over 1.5
    # price as a YRFI price.
    if point is None:
        return False

    try:
        return abs(
            float(point)
            - 0.5
        ) < 1e-9

    except Exception:
        return False


def parse_first_inning_market(
    event_odds,
):
    """
    Under 0.5 first-inning runs = NRFI
    Over 0.5 first-inning runs = YRFI
    """

    rows = []

    for bookmaker in event_odds.get(
        "bookmakers",
        []
    ):

        book_title = bookmaker.get(
            "title",
            bookmaker.get(
                "key",
                "Unknown"
            )
        )

        book_key = bookmaker.get(
            "key"
        )

        last_update = bookmaker.get(
            "last_update"
        )

        market = None

        for candidate in bookmaker.get(
            "markets",
            []
        ):

            if candidate.get(
                "key"
            ) == MARKET_KEY:

                market = candidate
                break

        if not market:
            continue

        nrfi_price = None
        yrfi_price = None

        for outcome in market.get(
            "outcomes",
            []
        ):

            if not _point_is_half(
                outcome
            ):
                continue

            name = str(
                outcome.get(
                    "name",
                    ""
                )
            ).strip().lower()

            price = outcome.get(
                "price"
            )

            if price is None:
                continue

            if name == "under":
                nrfi_price = float(
                    price
                )

            elif name == "over":
                yrfi_price = float(
                    price
                )

        if (
            nrfi_price is None
            or yrfi_price is None
        ):
            continue

        math_result = no_vig_two_way(
            nrfi_price,
            yrfi_price,
        )

        rows.append({
            "bookmaker":
                book_title,

            "bookmaker_key":
                book_key,

            "nrfi_price":
                nrfi_price,

            "yrfi_price":
                yrfi_price,

            "nrfi_raw_implied":
                math_result[
                    "nrfi_raw_implied"
                ],

            "yrfi_raw_implied":
                math_result[
                    "yrfi_raw_implied"
                ],

            "nrfi_no_vig":
                math_result[
                    "nrfi_no_vig"
                ],

            "yrfi_no_vig":
                math_result[
                    "yrfi_no_vig"
                ],

            "hold":
                math_result[
                    "hold"
                ],

            "last_update":
                last_update,
        })

    return rows


def summarize_market(
    bookmaker_rows,
):
    if not bookmaker_rows:
        return None

    nrfi_no_vig_values = [
        row[
            "nrfi_no_vig"
        ]
        for row in bookmaker_rows
    ]

    yrfi_no_vig_values = [
        row[
            "yrfi_no_vig"
        ]
        for row in bookmaker_rows
    ]

    best_nrfi = max(
        bookmaker_rows,
        key=lambda row:
            row[
                "nrfi_price"
            ],
    )

    best_yrfi = max(
        bookmaker_rows,
        key=lambda row:
            row[
                "yrfi_price"
            ],
    )

    return {
        "book_count":
            len(
                bookmaker_rows
            ),

        "consensus_nrfi_no_vig":
            sum(
                nrfi_no_vig_values
            )
            / len(
                nrfi_no_vig_values
            ),

        "consensus_yrfi_no_vig":
            sum(
                yrfi_no_vig_values
            )
            / len(
                yrfi_no_vig_values
            ),

        "best_nrfi_price":
            best_nrfi[
                "nrfi_price"
            ],

        "best_nrfi_book":
            best_nrfi[
                "bookmaker"
            ],

        "best_yrfi_price":
            best_yrfi[
                "yrfi_price"
            ],

        "best_yrfi_book":
            best_yrfi[
                "bookmaker"
            ],
    }


def model_market_edge(
    nrfi_model_probability,
    yrfi_model_probability,
    market_summary,
):
    if market_summary is None:
        return None

    nrfi_edge = (
        float(
            nrfi_model_probability
        )
        - market_summary[
            "consensus_nrfi_no_vig"
        ]
    )

    yrfi_edge = (
        float(
            yrfi_model_probability
        )
        - market_summary[
            "consensus_yrfi_no_vig"
        ]
    )

    if nrfi_edge >= yrfi_edge:
        best_side = "NRFI"
        best_edge = nrfi_edge
    else:
        best_side = "YRFI"
        best_edge = yrfi_edge

    return {
        "nrfi_edge":
            nrfi_edge,

        "yrfi_edge":
            yrfi_edge,

        "best_edge_side":
            best_side,

        "best_edge":
            best_edge,
    }


# =========================================================
# SAFE LOCAL TEST
# =========================================================

def _format_american(
    value,
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


def run_test():
    api_key = load_odds_api_key()

    events, event_usage = (
        get_mlb_events(
            api_key
        )
    )

    print()
    print(
        "SHARPREPORT NRFI ODDS API TEST"
    )
    print(
        "=" * 64
    )

    print(
        f"MLB events found: "
        f"{len(events)}"
    )

    print(
        "The API key was loaded successfully "
        "without printing it."
    )

    print()

    # Limit the validation test to the first 3 events so the
    # user does not burn unnecessary usage credits.
    events_to_test = events[:3]

    market_found = False

    for event in events_to_test:
        event_id = event.get(
            "id"
        )

        away = event.get(
            "away_team"
        )

        home = event.get(
            "home_team"
        )

        print(
            f"Checking: "
            f"{away} @ {home}"
        )

        event_odds, usage = (
            get_first_inning_event_odds(
                api_key,
                event_id,
            )
        )

        rows = (
            parse_first_inning_market(
                event_odds
            )
        )

        summary = summarize_market(
            rows
        )

        if not summary:
            print(
                "  No paired NRFI/YRFI "
                "0.5 market found."
            )

        else:
            market_found = True

            print(
                f"  Books with paired market: "
                f"{summary['book_count']}"
            )

            print(
                f"  Consensus no-vig NRFI: "
                f"{summary['consensus_nrfi_no_vig'] * 100:.1f}%"
            )

            print(
                f"  Consensus no-vig YRFI: "
                f"{summary['consensus_yrfi_no_vig'] * 100:.1f}%"
            )

            print(
                f"  Best NRFI: "
                f"{_format_american(summary['best_nrfi_price'])} "
                f"({summary['best_nrfi_book']})"
            )

            print(
                f"  Best YRFI: "
                f"{_format_american(summary['best_yrfi_price'])} "
                f"({summary['best_yrfi_book']})"
            )

        remaining = usage.get(
            "requests_remaining"
        )

        last_cost = usage.get(
            "requests_last"
        )

        if remaining is not None:
            print(
                f"  API credits remaining: "
                f"{remaining}"
            )

        if last_cost is not None:
            print(
                f"  This event request cost: "
                f"{last_cost}"
            )

        print()

    if market_found:
        print(
            "PASS: first-inning NRFI/YRFI "
            "market data is available."
        )
    else:
        print(
            "TEST COMPLETE: API connection works, "
            "but none of the first 3 MLB events had a "
            "paired first-inning 0.5 market at this moment."
        )


if __name__ == "__main__":
    run_test()
