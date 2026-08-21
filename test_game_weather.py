import json

from urllib.request import urlopen, Request
from urllib.parse import urlencode
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================================================
# TIME
# =========================================================

ET = ZoneInfo("America/New_York")

TODAY = datetime.now(ET).date()


# =========================================================
# BASIC REQUEST
# =========================================================

def get_json(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return json.load(response)


# =========================================================
# TODAY'S MLB GAMES
# =========================================================

def get_today_games():

    date_string = TODAY.strftime(
        "%Y-%m-%d"
    )

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_string}"
    )

    data = get_json(url)

    games = []


    for date_data in data.get(
        "dates",
        []
    ):

        for game in date_data.get(
            "games",
            []
        ):

            away = (
                game
                .get("teams", {})
                .get("away", {})
                .get("team", {})
                .get("name", "Away")
            )

            home = (
                game
                .get("teams", {})
                .get("home", {})
                .get("team", {})
                .get("name", "Home")
            )

            venue = game.get(
                "venue",
                {}
            )


            games.append({

                "game":
                    f"{away} @ {home}",

                "game_date":
                    game.get(
                        "gameDate"
                    ),

                "venue_id":
                    venue.get(
                        "id"
                    ),

                "venue_name":
                    venue.get(
                        "name",
                        ""
                    ),
            })


    return games


# =========================================================
# VENUE INFORMATION
# =========================================================

def get_venue_info(
    venue_id
):

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


    # MLB may return coordinates directly
    # OR inside defaultCoordinates.

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

def compass_direction(
    degrees
):

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
# OPEN-METEO WEATHER
# =========================================================

def get_game_weather(
    latitude,
    longitude,
    timezone_name,
    game_date
):

    game_utc = datetime.fromisoformat(
        game_date.replace(
            "Z",
            "+00:00"
        )
    )

    game_local = game_utc.astimezone(
        ZoneInfo(timezone_name)
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
        game_local
        .replace(
            tzinfo=None
        )
    )


    closest_index = min(

        range(
            len(times)
        ),

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
                "%Y-%m-%d %I:%M %p"
            ),

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


# =========================================================
# ROOF WEATHER RULE
# =========================================================

def weather_usage(
    roof_type
):

    roof = (
        roof_type
        .strip()
        .lower()
    )


    if roof == "indoor":

        return (
            "IGNORE OUTDOOR WEATHER"
        )


    if roof == "retractable":

        return (
            "CHECK ROOF OPEN/CLOSED"
        )


    if roof == "open":

        return (
            "USE WEATHER"
        )


    return (
        "VERIFY ROOF"
    )


# =========================================================
# RUN TEST
# =========================================================

games = get_today_games()

output = []

output.append(
    "SHARPREPORT GAME WEATHER TEST"
)

output.append(
    "=" * 70
)

output.append("")


for game in games:

    print(
        f"Loading weather: "
        f"{game['game']}"
    )


    try:

        venue = get_venue_info(
            game["venue_id"]
        )


        if not venue:

            raise Exception(
                "Venue information missing"
            )


        weather = get_game_weather(

            venue["latitude"],
            venue["longitude"],
            venue["timezone"],
            game["game_date"],
        )


        if not weather:

            raise Exception(
                "Weather information missing"
            )


        output.append(
            f"Game: {game['game']}"
        )

        output.append(
            f"Venue: {venue['name']}"
        )

        output.append(
            f"Roof Type: "
            f"{venue['roof_type']}"
        )

        output.append(
            f"Weather Rule: "
            f"{weather_usage(venue['roof_type'])}"
        )

        output.append(
            f"Coordinates: "
            f"{venue['latitude']}, "
            f"{venue['longitude']}"
        )

        output.append(
            f"Venue Timezone: "
            f"{venue['timezone']}"
        )

        output.append(
            f"Game Time: "
            f"{weather['game_local']}"
        )

        output.append(
            f"Weather Hour Used: "
            f"{weather['weather_hour']}"
        )

        output.append("")

        output.append(
            f"Temperature: "
            f"{weather['temperature']} F"
        )

        output.append(
            f"Humidity: "
            f"{weather['humidity']}%"
        )

        output.append(
            f"Dew Point: "
            f"{weather['dew_point']} F"
        )

        output.append(
            f"Surface Pressure: "
            f"{weather['pressure']} hPa"
        )

        output.append(
            f"Wind Speed: "
            f"{weather['wind_speed']} mph"
        )

        output.append(
            f"Wind Direction: "
            f"{weather['wind_direction']} "
            f"({weather['wind_direction_degrees']}°)"
        )

        output.append(
            f"Wind Gusts: "
            f"{weather['wind_gusts']} mph"
        )

        output.append(
            f"Precipitation Probability: "
            f"{weather['precipitation_probability']}%"
        )

        output.append(
            f"Weather Code: "
            f"{weather['weather_code']}"
        )

        output.append(
            "-" * 70
        )


    except Exception as error:

        output.append(
            f"Game: {game['game']}"
        )

        output.append(
            f"ERROR: {error}"
        )

        output.append(
            "-" * 70
        )


with open(
    "game_weather_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )


print("")
print(
    "WEATHER TEST COMPLETE"
)

print("")

print(
    "Open game_weather_output.txt"
)