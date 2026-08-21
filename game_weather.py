import json

from urllib.request import urlopen, Request
from urllib.parse import urlencode
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import lru_cache


def get_json(url):

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return json.load(response)


# =========================================================
# VENUE INFORMATION
# =========================================================

@lru_cache(maxsize=64)
def get_venue_info(venue_id):

    if not venue_id:
        return None

    url = (
        "https://statsapi.mlb.com/api/v1/"
        f"venues/{venue_id}"
        "?hydrate=location,fieldInfo,timezone"
    )

    data = get_json(url)

    venues = data.get(
        "venues",
        []
    )

    if not venues:
        return None


    venue = venues[0]

    location = venue.get(
        "location",
        {}
    )

    timezone = venue.get(
        "timeZone",
        {}
    )

    field_info = venue.get(
        "fieldInfo",
        {}
    )


    latitude = location.get(
        "latitude"
    )

    longitude = location.get(
        "longitude"
    )


    if (
        latitude is None
        or longitude is None
    ):

        default_coordinates = (
            location.get(
                "defaultCoordinates",
                {}
            )
        )

        latitude = (
            default_coordinates.get(
                "latitude"
            )
        )

        longitude = (
            default_coordinates.get(
                "longitude"
            )
        )


    if (
        latitude is None
        or longitude is None
    ):

        return None


    return {

        "name":
            venue.get(
                "name",
                ""
            ),

        "latitude":
            float(latitude),

        "longitude":
            float(longitude),

        "timezone":
            timezone.get(
                "id",
                "America/New_York"
            ),

        "roof_type":
            field_info.get(
                "roofType",
                "Unknown"
            ),
    }


# =========================================================
# WIND DIRECTION
# =========================================================

def compass_direction(degrees):

    if degrees is None:
        return "—"

    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    index = round(
        degrees / 45
    ) % 8

    return directions[index]


# =========================================================
# WEATHER USE RULE
# =========================================================

def weather_usage(roof_type):

    roof = (
        str(roof_type)
        .strip()
        .lower()
    )

    if roof == "indoor":

        return "IGNORE OUTDOOR WEATHER"

    if roof == "retractable":

        return "VERIFY ROOF OPEN/CLOSED"

    if roof == "open":

        return "USE WEATHER"

    return "VERIFY ROOF"


# =========================================================
# GAME-TIME WEATHER
# =========================================================

def get_game_weather(
    latitude,
    longitude,
    timezone_name,
    game_date
):

    if not game_date:
        return None


    game_utc = datetime.fromisoformat(
        game_date.replace(
            "Z",
            "+00:00"
        )
    )


    try:

        local_timezone = ZoneInfo(
            timezone_name
        )

    except Exception:

        local_timezone = ZoneInfo(
            "America/New_York"
        )


    game_local = game_utc.astimezone(
        local_timezone
    )


    local_date = game_local.strftime(
        "%Y-%m-%d"
    )


    hourly_variables = ",".join([

        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation_probability",
        "weather_code",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ])


    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "hourly":
            hourly_variables,

        "temperature_unit":
            "fahrenheit",

        "wind_speed_unit":
            "mph",

        "precipitation_unit":
            "inch",

        "timezone":
            timezone_name,

        "start_date":
            local_date,

        "end_date":
            local_date,
    }


    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urlencode(params)
    )


    data = get_json(url)

    hourly = data.get(
        "hourly",
        {}
    )

    times = hourly.get(
        "time",
        []
    )


    if not times:
        return None


    target_time = (
        game_local.replace(
            tzinfo=None
        )
    )


    closest_index = min(

        range(len(times)),

        key=lambda i: abs(

            datetime.fromisoformat(
                times[i]
            )

            - target_time

        ),
    )


    def value(field):

        values = hourly.get(
            field,
            []
        )

        if closest_index < len(values):

            return values[
                closest_index
            ]

        return None


    wind_degrees = value(
        "wind_direction_10m"
    )


    return {

        "game_local":
            game_local.strftime(
                "%I:%M %p"
            ).lstrip("0"),

        "weather_hour":
            times[
                closest_index
            ],

        "temperature":
            value(
                "temperature_2m"
            ),

        "humidity":
            value(
                "relative_humidity_2m"
            ),

        "dew_point":
            value(
                "dew_point_2m"
            ),

        "pressure":
            value(
                "surface_pressure"
            ),

        "precipitation_probability":
            value(
                "precipitation_probability"
            ),

        "weather_code":
            value(
                "weather_code"
            ),

        "wind_speed":
            value(
                "wind_speed_10m"
            ),

        "wind_direction_degrees":
            wind_degrees,

        "wind_direction":
            compass_direction(
                wind_degrees
            ),

        "wind_gusts":
            value(
                "wind_gusts_10m"
            ),
    }