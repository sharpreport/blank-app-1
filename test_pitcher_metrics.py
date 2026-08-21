from pybaseball import (
    pitching_stats,
    statcast_pitcher_exitvelo_barrels
)

YEAR = 2026

output = []

output.append("SHARPREPORT PITCHER METRICS TEST")
output.append("=" * 50)
output.append("")


# ---------------------------------
# TEST FANGRAPHS PITCHING DATA
# ---------------------------------

try:

    output.append("LOADING FANGRAPHS PITCHING DATA...")

    fg = pitching_stats(
        YEAR,
        qual=1
    )

    output.append(
        f"Pitchers returned: {len(fg)}"
    )

    output.append("")
    output.append("FANGRAPHS COLUMN NAMES:")
    output.append("")

    for column in fg.columns:
        output.append(str(column))

    output.append("")
    output.append(
        "POSSIBLE K / BB COLUMNS:"
    )
    output.append("")

    for column in fg.columns:

        name = str(column).lower()

        if (
            "k%" in name
            or "bb%" in name
            or "strike" in name
            or "walk" in name
        ):
            output.append(str(column))

    output.append("")
    output.append("FIRST 5 PITCHERS:")
    output.append("")
    output.append(
        fg.head(5).to_string()
    )

except Exception as error:

    output.append("")
    output.append(
        "FANGRAPHS ERROR:"
    )
    output.append(str(error))


output.append("")
output.append("=" * 50)
output.append("")


# ---------------------------------
# TEST BASEBALL SAVANT BARREL DATA
# ---------------------------------

try:

    output.append(
        "LOADING BASEBALL SAVANT BARREL DATA..."
    )

    barrels = (
        statcast_pitcher_exitvelo_barrels(
            YEAR,
            minBBE=1
        )
    )

    output.append(
        f"Pitchers returned: {len(barrels)}"
    )

    output.append("")
    output.append("BARREL DATA COLUMN NAMES:")
    output.append("")

    for column in barrels.columns:
        output.append(str(column))

    output.append("")
    output.append(
        "POSSIBLE BARREL COLUMNS:"
    )
    output.append("")

    for column in barrels.columns:

        name = str(column).lower()

        if (
            "barrel" in name
            or "batted" in name
            or "exit" in name
            or "hard" in name
        ):
            output.append(str(column))

    output.append("")
    output.append("FIRST 5 PITCHERS:")
    output.append("")
    output.append(
        barrels.head(5).to_string()
    )

except Exception as error:

    output.append("")
    output.append(
        "BARREL DATA ERROR:"
    )
    output.append(str(error))


# ---------------------------------
# WRITE RESULTS TO FILE
# ---------------------------------

with open(
    "pitcher_metrics_output.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(output))


print("")
print("TEST COMPLETE")
print("")
print(
    "Open pitcher_metrics_output.txt "
    "from the file list."
)