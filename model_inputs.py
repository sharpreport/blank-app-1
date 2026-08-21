def to_number(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if text in {
        "",
        "—",
        "MISSING",
        "None",
    }:
        return None

    replacements = [
        "%",
        "°F",
        "mph",
        "hPa",
    ]

    for item in replacements:
        text = text.replace(
            item,
            ""
        )

    text = text.strip()

    try:
        return float(text)

    except Exception:
        return None


def build_half_inning_model_table(
    games,
    pitcher_rows,
    first_inning_rows,
    team_hitter_rows,
    weather_rows,
    park_rows,
):

    pitcher_lookup = {

        (
            row["Game"],
            row["Team"]
        ): row

        for row in pitcher_rows
    }


    history_lookup = {

        (
            row["Game"],
            row["Team"]
        ): row

        for row in first_inning_rows
    }


    offense_lookup = {

        (
            row["Game"],
            row["Offense"]
        ): row

        for row in team_hitter_rows
    }


    weather_lookup = {

        row["Game"]: row

        for row in weather_rows
    }


    park_lookup = {

        row["Game"]: row

        for row in park_rows
    }


    model_rows = []


    for game in games:

        matchup = game[
            "Game"
        ]


        halves = [

            {
                "half":
                    "Top 1st",

                "offense":
                    game["Away Team"],

                "pitcher_team":
                    game["Home Team"],

                "pitcher":
                    game["Home SP"],
            },

            {
                "half":
                    "Bottom 1st",

                "offense":
                    game["Home Team"],

                "pitcher_team":
                    game["Away Team"],

                "pitcher":
                    game["Away SP"],
            },
        ]


        for half in halves:

            offense = offense_lookup.get(
                (
                    matchup,
                    half["offense"]
                ),
                {}
            )


            pitcher = pitcher_lookup.get(
                (
                    matchup,
                    half["pitcher_team"]
                ),
                {}
            )


            history = history_lookup.get(
                (
                    matchup,
                    half["pitcher_team"]
                ),
                {}
            )


            weather = weather_lookup.get(
                matchup,
                {}
            )


            park = park_lookup.get(
                matchup,
                {}
            )


            top4_xwoba = to_number(
                offense.get(
                    "Top-4 xwOBA"
                )
            )

            top4_k = to_number(
                offense.get(
                    "Top-4 K%"
                )
            )

            top4_bb = to_number(
                offense.get(
                    "Top-4 BB%"
                )
            )

            top4_barrel = to_number(
                offense.get(
                    "Top-4 Barrel%"
                )
            )


            sp_xwoba = to_number(
                pitcher.get(
                    "xwOBA Allowed"
                )
            )

            sp_k = to_number(
                pitcher.get(
                    "K%"
                )
            )

            sp_bb = to_number(
                pitcher.get(
                    "BB%"
                )
            )

            sp_barrel = to_number(
                pitcher.get(
                    "Barrel% Allowed"
                )
            )


            scoreless_first = to_number(
                history.get(
                    "Scoreless 1st %"
                )
            )

            first_runs_start = to_number(
                history.get(
                    "1st-Inning Runs/Start"
                )
            )

            nrfi_percent = to_number(
                history.get(
                    "NRFI %"
                )
            )


            run_factor = to_number(
                park.get(
                    "Run Factor"
                )
            )

            woba_factor = to_number(
                park.get(
                    "wOBA Factor"
                )
            )

            hr_factor = to_number(
                park.get(
                    "HR Factor"
                )
            )


            temperature = to_number(
                weather.get(
                    "Temp"
                )
            )

            humidity = to_number(
                weather.get(
                    "Humidity"
                )
            )

            dew_point = to_number(
                weather.get(
                    "Dew Point"
                )
            )

            pressure = to_number(
                weather.get(
                    "Pressure"
                )
            )

            wind_speed = to_number(
                weather.get(
                    "Wind"
                )
            )

            wind_gusts = to_number(
                weather.get(
                    "Wind Gusts"
                )
            )

            precipitation = to_number(
                weather.get(
                    "Precip"
                )
            )


            required_inputs = {

                "Top-4 xwOBA":
                    top4_xwoba,

                "Top-4 K%":
                    top4_k,

                "Top-4 BB%":
                    top4_bb,

                "Top-4 Barrel%":
                    top4_barrel,

                "SP xwOBA":
                    sp_xwoba,

                "SP K%":
                    sp_k,

                "SP BB%":
                    sp_bb,

                "SP Barrel%":
                    sp_barrel,

                "SP Scoreless 1st %":
                    scoreless_first,

                "SP 1st Runs/Start":
                    first_runs_start,

                "Run Factor":
                    run_factor,
            }


            missing = [

                name

                for name, value
                in required_inputs.items()

                if value is None
            ]


            lineup_status = game.get(
                "Lineups",
                ""
            )


            weather_rule = weather.get(
                "Weather Use",
                "—"
            )


            if (
                lineup_status
                != "✅ Confirmed"
            ):

                model_status = (
                    "WAITING LINEUP"
                )

            elif missing:

                model_status = (
                    "MISSING INPUTS"
                )

            elif (
                "VERIFY ROOF"
                in str(weather_rule)
            ):

                model_status = (
                    "ROOF VERIFY"
                )

            else:

                model_status = (
                    "READY"
                )


            model_rows.append({

                "Game":
                    matchup,

                "Half":
                    half["half"],

                "Offense":
                    half["offense"],

                "Opposing SP":
                    half["pitcher"],

                "Lineup Status":
                    lineup_status,

                "Model Status":
                    model_status,

                "Missing Inputs":
                    (
                        ", ".join(missing)
                        if missing
                        else ""
                    ),

                # OFFENSE

                "Top-4 xwOBA":
                    top4_xwoba,

                "Top-4 K%":
                    top4_k,

                "Top-4 BB%":
                    top4_bb,

                "Top-4 Barrel%":
                    top4_barrel,

                "Top-4 Combined PA":
                    to_number(
                        offense.get(
                            "Combined Top-4 PA"
                        )
                    ),

                # STARTING PITCHER

                "SP xwOBA Allowed":
                    sp_xwoba,

                "SP K%":
                    sp_k,

                "SP BB%":
                    sp_bb,

                "SP Barrel% Allowed":
                    sp_barrel,

                "SP Batters Faced":
                    to_number(
                        pitcher.get(
                            "Batters Faced"
                        )
                    ),

                # FIRST-INNING HISTORY

                "SP Starts":
                    to_number(
                        history.get(
                            "Starts"
                        )
                    ),

                "SP Scoreless 1st %":
                    scoreless_first,

                "SP 1st-Inning Runs/Start":
                    first_runs_start,

                "SP NRFI Record":
                    history.get(
                        "NRFI Record",
                        "—"
                    ),

                "SP NRFI % Context":
                    nrfi_percent,

                # PARK

                "Park Run Factor":
                    run_factor,

                "Park wOBA Factor":
                    woba_factor,

                "Park HR Factor":
                    hr_factor,

                # WEATHER

                "Roof":
                    weather.get(
                        "Roof",
                        "—"
                    ),

                "Weather Use":
                    weather_rule,

                "Temperature F":
                    temperature,

                "Humidity %":
                    humidity,

                "Dew Point F":
                    dew_point,

                "Pressure hPa":
                    pressure,

                "Wind mph":
                    wind_speed,

                "Wind Direction":
                    weather.get(
                        "Wind Direction",
                        "—"
                    ),

                "Wind Gusts mph":
                    wind_gusts,

                "Precip %":
                    precipitation,
            })


    return model_rows