# =========================================================
# SHARPREPORT NRFI / YRFI RISK ENGINE
#
# CURRENT PURPOSE:
# - Score every half-inning
# - Score every game
# - Rank strongest NRFI / YRFI signals
#
# IMPORTANT:
# These are provisional model indexes.
# They are NOT calibrated probabilities yet.
# =========================================================


def clamp(value, minimum=0.0, maximum=100.0):

    return max(
        minimum,
        min(maximum, value)
    )


def scaled_delta(
    value,
    baseline,
    scale,
    higher_is_risk=True
):

    if value is None:
        value = baseline

    delta = (
        value - baseline
    ) / scale

    if not higher_is_risk:
        delta = -delta

    return max(
        -2.5,
        min(2.5, delta)
    )


# =========================================================
# OFFENSE
# =========================================================

def offense_risk(row):

    # League-average placeholders are used
    # when confirmed lineup data is missing.

    xwoba = scaled_delta(
        row.get("Top-4 xwOBA"),
        0.330,
        0.035,
        True,
    )

    k_rate = scaled_delta(
        row.get("Top-4 K%"),
        22.0,
        4.0,
        False,
    )

    bb_rate = scaled_delta(
        row.get("Top-4 BB%"),
        9.0,
        2.5,
        True,
    )

    barrel = scaled_delta(
        row.get("Top-4 Barrel%"),
        8.5,
        3.0,
        True,
    )


    combined = (
        xwoba * 0.45
        + k_rate * 0.20
        + bb_rate * 0.15
        + barrel * 0.20
    )


    return clamp(
        50
        + combined * 15
    )


# =========================================================
# STARTING PITCHER
# =========================================================

def pitcher_risk(row):

    xwoba = scaled_delta(
        row.get("SP xwOBA Allowed"),
        0.320,
        0.030,
        True,
    )

    k_rate = scaled_delta(
        row.get("SP K%"),
        22.5,
        4.0,
        False,
    )

    bb_rate = scaled_delta(
        row.get("SP BB%"),
        8.0,
        2.5,
        True,
    )

    barrel = scaled_delta(
        row.get("SP Barrel% Allowed"),
        7.5,
        3.0,
        True,
    )


    combined = (
        xwoba * 0.35
        + k_rate * 0.30
        + bb_rate * 0.18
        + barrel * 0.17
    )


    return clamp(
        50
        + combined * 15
    )


# =========================================================
# FIRST-INNING HISTORY
# =========================================================

def history_risk(row):

    scoreless_percent = row.get(
        "SP Scoreless 1st %"
    )

    starts = row.get(
        "SP Starts"
    )


    # Neutral placeholder if starter
    # history is unavailable.

    if (
        scoreless_percent is None
        or starts is None
        or starts == 0
    ):

        return 50.0


    raw_scoring_rate = (
        100.0
        - scoreless_percent
    )


    # Provisional league prior:
    # 25.5% scoring in one half-inning.

    prior_scoring_rate = 25.5


    # Shrink small samples toward league average.

    sample_weight = (
        starts
        / (starts + 15.0)
    )


    adjusted_rate = (
        prior_scoring_rate
        + sample_weight
        * (
            raw_scoring_rate
            - prior_scoring_rate
        )
    )


    return clamp(
        50
        + (
            adjusted_rate
            - prior_scoring_rate
        )
        * 1.5
    )


# =========================================================
# PARK
# =========================================================

def park_risk(row):

    run_factor = row.get(
        "Park Run Factor"
    )


    if run_factor is None:
        run_factor = 100


    return clamp(
        50
        + (
            run_factor - 100
        )
        * 2.0
    )


# =========================================================
# DATA COMPLETENESS
# =========================================================

def data_quality(row):

    score = 0


    # OFFENSE — 30 points

    offense_fields = [
        "Top-4 xwOBA",
        "Top-4 K%",
        "Top-4 BB%",
        "Top-4 Barrel%",
    ]

    offense_present = sum(
        row.get(field) is not None
        for field in offense_fields
    )

    score += (
        offense_present
        / 4
        * 30
    )


    # PITCHER — 35 points

    pitcher_fields = [
        "SP xwOBA Allowed",
        "SP K%",
        "SP BB%",
        "SP Barrel% Allowed",
    ]

    pitcher_present = sum(
        row.get(field) is not None
        for field in pitcher_fields
    )

    score += (
        pitcher_present
        / 4
        * 35
    )


    # FIRST-INNING HISTORY — 20 points

    if (
        row.get(
            "SP Scoreless 1st %"
        )
        is not None
    ):

        score += 20


    # PARK — 10 points

    if (
        row.get(
            "Park Run Factor"
        )
        is not None
    ):

        score += 10


    # CONFIRMED LINEUP — 5 points

    if (
        row.get(
            "Lineup Status"
        )
        == "✅ Confirmed"
    ):

        score += 5


    return round(
        clamp(score),
        1
    )


# =========================================================
# HALF-INNING MODEL
# =========================================================

def build_half_inning_scores(
    combined_rows
):

    output = []


    for row in combined_rows:

        offense = offense_risk(
            row
        )

        pitcher = pitcher_risk(
            row
        )

        history = history_risk(
            row
        )

        park = park_risk(
            row
        )


        # Temporary weights.
        # Historical training will determine
        # final coefficients.

        overall = (
            offense * 0.32
            + pitcher * 0.38
            + history * 0.20
            + park * 0.10
        )


        overall = clamp(
            overall
        )


        quality = data_quality(
            row
        )


        if quality >= 95:

            status = "FINAL INPUTS"

        elif quality >= 60:

            status = "PROVISIONAL"

        else:

            status = "LOW DATA"


        output.append({

            "Game":
                row["Game"],

            "Half":
                row["Half"],

            "Offense":
                row["Offense"],

            "Opposing SP":
                row["Opposing SP"],

            "Input Status":
                status,

            "Data Quality":
                quality,

            "Offense Risk":
                round(
                    offense,
                    1
                ),

            "Pitcher Risk":
                round(
                    pitcher,
                    1
                ),

            "History Risk":
                round(
                    history,
                    1
                ),

            "Park Risk":
                round(
                    park,
                    1
                ),

            "Half-Inning Risk":
                round(
                    overall,
                    1
                ),
        })


    return output


# =========================================================
# GAME-LEVEL MODEL
# =========================================================

def build_game_scores(
    half_rows
):

    games = {}


    for row in half_rows:

        game = row[
            "Game"
        ]

        games.setdefault(
            game,
            []
        )

        games[game].append(
            row
        )


    output = []


    for game, halves in games.items():

        if len(halves) < 2:
            continue


        top = next(
            (
                row
                for row in halves
                if row["Half"]
                == "Top 1st"
            ),
            halves[0]
        )


        bottom = next(
            (
                row
                for row in halves
                if row["Half"]
                == "Bottom 1st"
            ),
            halves[-1]
        )


        top_risk = top[
            "Half-Inning Risk"
        ]

        bottom_risk = bottom[
            "Half-Inning Risk"
        ]


        # Current raw game scoring-risk index.
        # 50 = neutral reference.

        game_risk = (
            top_risk
            + bottom_risk
        ) / 2


        if game_risk >= 50:

            model_side = "YRFI"

        else:

            model_side = "NRFI"


        distance = abs(
            game_risk - 50
        )


        quality = (
            top["Data Quality"]
            + bottom["Data Quality"]
        ) / 2


        # Used only to rank games.
        # NOT a win probability.

        ranking_score = (
            distance
            * (
                quality / 100
            )
        )


        if (
            top["Input Status"]
            == "FINAL INPUTS"
            and
            bottom["Input Status"]
            == "FINAL INPUTS"
        ):

            game_status = (
                "FINAL INPUTS"
            )

        else:

            game_status = (
                "PROVISIONAL"
            )


        output.append({

            "Game":
                game,

            "Model Side":
                model_side,

            "Game Risk Index":
                round(
                    game_risk,
                    1
                ),

            "Distance From Neutral":
                round(
                    distance,
                    1
                ),

            "Data Quality":
                round(
                    quality,
                    1
                ),

            "Ranking Score":
                round(
                    ranking_score,
                    2
                ),

            "Status":
                game_status,

            "Top 1st Risk":
                round(
                    top_risk,
                    1
                ),

            "Bottom 1st Risk":
                round(
                    bottom_risk,
                    1
                ),

            "Away vs Home SP":
                top[
                    "Opposing SP"
                ],

            "Home vs Away SP":
                bottom[
                    "Opposing SP"
                ],
        })


    output.sort(
        key=lambda row: row[
            "Ranking Score"
        ],
        reverse=True
    )


    for rank, row in enumerate(
        output,
        start=1
    ):

        row["Rank"] = rank


    return output