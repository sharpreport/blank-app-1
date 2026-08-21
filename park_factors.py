import json
import re

from urllib.request import urlopen, Request
from functools import lru_cache


def get_text(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return response.read().decode(
            "utf-8"
        )


@lru_cache(maxsize=8)
def get_park_factors(year):

    url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast-park-factors"
        "?type=year"
        f"&year={year}"
        "&batSide="
        "&stat=index_wOBA"
        "&condition=All"
        "&rolling=3"
        "&parks=mlb"
    )

    html = get_text(url)

    match = re.search(
        r"\bdata\s*=\s*(\[.*?\]);",
        html,
        re.DOTALL
    )

    if not match:

        raise Exception(
            "Could not locate Baseball Savant "
            "park-factor data."
        )

    raw_data = json.loads(
        match.group(1)
    )

    parks = {}

    for park in raw_data:

        venue_id = park.get(
            "venue_id"
        )

        if not venue_id:
            continue

        parks[int(venue_id)] = {

            "venue_name":
                park.get(
                    "venue_name",
                    ""
                ),

            "run_factor":
                int(
                    park.get(
                        "index_runs",
                        100
                    )
                ),

            "woba_factor":
                int(
                    park.get(
                        "index_woba",
                        100
                    )
                ),

            "hr_factor":
                int(
                    park.get(
                        "index_hr",
                        100
                    )
                ),

            "year_range":
                park.get(
                    "year_range",
                    ""
                ),
        }

    return parks


def classify_run_factor(
    run_factor
):

    if run_factor >= 110:
        return "Very Hitter-Friendly"

    if run_factor >= 103:
        return "Hitter-Friendly"

    if run_factor <= 90:
        return "Very Pitcher-Friendly"

    if run_factor <= 97:
        return "Pitcher-Friendly"

    return "Neutral"