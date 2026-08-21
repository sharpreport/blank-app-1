
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7D2
#
# Validate reconstructed full-season pitcher metrics
# against official/public season totals.
#
# Compares:
# - Reconstructed xwOBA allowed vs Baseball Savant
# - Reconstructed Barrel% vs Baseball Savant
# - Reconstructed K% vs MLB Stats API
# - Reconstructed BB% vs MLB Stats API
#
# Uses 2022-2025 completed seasons only.
# =========================================================


YEARS = [2022, 2023, 2024, 2025]

SUMMARY_FILE = Path(
    "historical_pitcher_metric_validation_summary.txt"
)

DETAIL_FILE = Path(
    "historical_pitcher_metric_validation_details.csv"
)


# =========================================================
# SETTINGS / TOLERANCES
# =========================================================

MIN_PA = 100

XWOBA_TOLERANCE = 0.020
K_TOLERANCE_PP = 1.50
BB_TOLERANCE_PP = 1.50
BARREL_TOLERANCE_PP = 1.50


# =========================================================
# REQUEST HELPERS
# =========================================================

def get_text(url):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        return (
            response
            .read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )


def get_json(url):

    return json.loads(
        get_text(url)
    )


def get_csv(url):

    text = get_text(url)

    return pd.read_csv(
        io.StringIO(text)
    )


# =========================================================
# RECONSTRUCT SEASON TOTALS FROM OUR DAILY FILES
# =========================================================

def reconstructed_season(year):

    path = Path(
        f"statcast_daily_pitcher_{year}.csv"
    )

    if not path.exists():

        raise SystemExit(
            f"ERROR: {path} was not found."
        )


    daily = pd.read_csv(
        path
    )


    numeric = [
        "pitcher",
        "pa",
        "strikeouts",
        "walks",
        "xwoba_num",
        "xwoba_denom",
        "bbe",
        "barrels",
    ]


    for column in numeric:

        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce",
        ).fillna(
            0
        )


    grouped = (
        daily
        .groupby(
            "pitcher",
            as_index=False,
        )[
            [
                "pa",
                "strikeouts",
                "walks",
                "xwoba_num",
                "xwoba_denom",
                "bbe",
                "barrels",
            ]
        ]
        .sum()
    )


    grouped[
        "pitcher"
    ] = grouped[
        "pitcher"
    ].astype(int)


    grouped[
        "our_xwoba"
    ] = (
        grouped[
            "xwoba_num"
        ]
        /
        grouped[
            "xwoba_denom"
        ].replace(
            0,
            pd.NA,
        )
    )


    grouped[
        "our_k_pct"
    ] = (
        grouped[
            "strikeouts"
        ]
        /
        grouped[
            "pa"
        ].replace(
            0,
            pd.NA,
        )
        * 100
    )


    grouped[
        "our_bb_pct"
    ] = (
        grouped[
            "walks"
        ]
        /
        grouped[
            "pa"
        ].replace(
            0,
            pd.NA,
        )
        * 100
    )


    grouped[
        "our_barrel_pct"
    ] = (
        grouped[
            "barrels"
        ]
        /
        grouped[
            "bbe"
        ].replace(
            0,
            pd.NA,
        )
        * 100
    )


    return grouped


# =========================================================
# BASEBALL SAVANT EXPECTED STATS
# =========================================================

def savant_expected(year):

    url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/expected_statistics"
        f"?type=pitcher&year={year}"
        "&position=&team="
        "&filterType=pa&min=1"
        "&csv=true"
    )


    df = get_csv(url)


    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    id_candidates = [
        "player_id",
        "playerid",
        "id",
    ]


    id_column = None

    for candidate in id_candidates:

        if candidate in df.columns:

            id_column = candidate
            break


    if id_column is None:

        raise RuntimeError(
            f"Could not find player ID "
            f"in Savant expected stats for {year}."
        )


    if "est_woba" not in df.columns:

        raise RuntimeError(
            f"Could not find est_woba "
            f"in Savant expected stats for {year}."
        )


    result = pd.DataFrame({

        "pitcher":
            pd.to_numeric(
                df[
                    id_column
                ],
                errors="coerce",
            ),

        "savant_xwoba":
            pd.to_numeric(
                df[
                    "est_woba"
                ],
                errors="coerce",
            ),
    })


    result = result.dropna(
        subset=[
            "pitcher"
        ]
    )


    result[
        "pitcher"
    ] = result[
        "pitcher"
    ].astype(int)


    return result


# =========================================================
# BASEBALL SAVANT BARREL%
# =========================================================

def savant_barrels(year):

    url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast"
        f"?type=pitcher&year={year}"
        "&position=&team=&min=1"
        "&csv=true"
    )


    df = get_csv(url)


    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    id_candidates = [
        "player_id",
        "playerid",
        "id",
    ]


    id_column = None

    for candidate in id_candidates:

        if candidate in df.columns:

            id_column = candidate
            break


    if id_column is None:

        raise RuntimeError(
            f"Could not find player ID "
            f"in Savant barrel stats for {year}."
        )


    if "brl_percent" not in df.columns:

        raise RuntimeError(
            f"Could not find brl_percent "
            f"in Savant barrel stats for {year}."
        )


    result = pd.DataFrame({

        "pitcher":
            pd.to_numeric(
                df[
                    id_column
                ],
                errors="coerce",
            ),

        "savant_barrel_pct":
            pd.to_numeric(
                df[
                    "brl_percent"
                ],
                errors="coerce",
            ),
    })


    result = result.dropna(
        subset=[
            "pitcher"
        ]
    )


    result[
        "pitcher"
    ] = result[
        "pitcher"
    ].astype(int)


    return result


# =========================================================
# MLB STATS API K% / BB%
# =========================================================

def mlb_pitching_rates(
    pitcher_id,
    year,
):

    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/people/{pitcher_id}/stats"
        "?stats=season"
        "&group=pitching"
        f"&season={year}"
    )


    data = get_json(url)


    stats = (
        data
        .get(
            "stats",
            [{}]
        )[0]
        .get(
            "splits",
            []
        )
    )


    if not stats:

        return (
            None,
            None,
            None,
        )


    stat = stats[
        0
    ].get(
        "stat",
        {}
    )


    batters_faced = stat.get(
        "battersFaced"
    )

    strikeouts = stat.get(
        "strikeOuts"
    )

    walks = stat.get(
        "baseOnBalls"
    )


    try:

        bf = float(
            batters_faced
        )

        so = float(
            strikeouts
        )

        bb = float(
            walks
        )

    except Exception:

        return (
            None,
            None,
            None,
        )


    if bf <= 0:

        return (
            None,
            None,
            None,
        )


    return (
        bf,
        so / bf * 100,
        bb / bf * 100,
    )


# =========================================================
# VALIDATION
# =========================================================

detail_rows = []


for year in YEARS:

    print()
    print(
        f"VALIDATING {year}"
    )
    print(
        "-" * 60
    )


    ours = reconstructed_season(
        year
    )


    expected = savant_expected(
        year
    )


    barrels = savant_barrels(
        year
    )


    merged = (
        ours
        .merge(
            expected,
            on="pitcher",
            how="left",
        )
        .merge(
            barrels,
            on="pitcher",
            how="left",
        )
    )


    merged = merged[
        merged[
            "pa"
        ] >= MIN_PA
    ].copy()


    for index, row in merged.iterrows():

        pitcher_id = int(
            row[
                "pitcher"
            ]
        )


        (
            mlb_bf,
            mlb_k_pct,
            mlb_bb_pct,
        ) = mlb_pitching_rates(
            pitcher_id,
            year,
        )


        our_xwoba = row[
            "our_xwoba"
        ]

        savant_xwoba = row[
            "savant_xwoba"
        ]


        our_barrel = row[
            "our_barrel_pct"
        ]

        savant_barrel = row[
            "savant_barrel_pct"
        ]


        xwoba_diff = (
            abs(
                our_xwoba
                - savant_xwoba
            )
            if (
                pd.notna(
                    our_xwoba
                )
                and pd.notna(
                    savant_xwoba
                )
            )
            else None
        )


        k_diff = (
            abs(
                row[
                    "our_k_pct"
                ]
                - mlb_k_pct
            )
            if mlb_k_pct
            is not None
            else None
        )


        bb_diff = (
            abs(
                row[
                    "our_bb_pct"
                ]
                - mlb_bb_pct
            )
            if mlb_bb_pct
            is not None
            else None
        )


        barrel_diff = (
            abs(
                our_barrel
                - savant_barrel
            )
            if (
                pd.notna(
                    our_barrel
                )
                and pd.notna(
                    savant_barrel
                )
            )
            else None
        )


        detail_rows.append({

            "year":
                year,

            "pitcher_id":
                pitcher_id,

            "our_pa":
                round(
                    float(
                        row[
                            "pa"
                        ]
                    ),
                    0,
                ),

            "mlb_batters_faced":
                mlb_bf,

            "our_xwoba":
                round(
                    float(
                        our_xwoba
                    ),
                    4,
                )
                if pd.notna(
                    our_xwoba
                )
                else None,

            "savant_xwoba":
                round(
                    float(
                        savant_xwoba
                    ),
                    4,
                )
                if pd.notna(
                    savant_xwoba
                )
                else None,

            "xwoba_abs_diff":
                round(
                    float(
                        xwoba_diff
                    ),
                    4,
                )
                if xwoba_diff
                is not None
                else None,

            "our_k_pct":
                round(
                    float(
                        row[
                            "our_k_pct"
                        ]
                    ),
                    2,
                ),

            "mlb_k_pct":
                round(
                    float(
                        mlb_k_pct
                    ),
                    2,
                )
                if mlb_k_pct
                is not None
                else None,

            "k_abs_diff_pp":
                round(
                    float(
                        k_diff
                    ),
                    2,
                )
                if k_diff
                is not None
                else None,

            "our_bb_pct":
                round(
                    float(
                        row[
                            "our_bb_pct"
                        ]
                    ),
                    2,
                ),

            "mlb_bb_pct":
                round(
                    float(
                        mlb_bb_pct
                    ),
                    2,
                )
                if mlb_bb_pct
                is not None
                else None,

            "bb_abs_diff_pp":
                round(
                    float(
                        bb_diff
                    ),
                    2,
                )
                if bb_diff
                is not None
                else None,

            "our_barrel_pct":
                round(
                    float(
                        our_barrel
                    ),
                    2,
                )
                if pd.notna(
                    our_barrel
                )
                else None,

            "savant_barrel_pct":
                round(
                    float(
                        savant_barrel
                    ),
                    2,
                )
                if pd.notna(
                    savant_barrel
                )
                else None,

            "barrel_abs_diff_pp":
                round(
                    float(
                        barrel_diff
                    ),
                    2,
                )
                if barrel_diff
                is not None
                else None,
        })


details = pd.DataFrame(
    detail_rows
)


details.to_csv(
    DETAIL_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

lines = [
    "SHARPREPORT HISTORICAL PITCHER METRIC VALIDATION",
    "=" * 72,
    "",
    f"Completed seasons checked: "
    f"{', '.join(str(y) for y in YEARS)}",
    f"Minimum reconstructed PA: {MIN_PA}",
    "",
]


def metric_summary(
    label,
    diff_column,
    tolerance,
):

    valid = details[
        diff_column
    ].dropna()


    if valid.empty:

        lines.extend([
            label,
            "No comparable rows.",
            "-" * 72,
        ])

        return


    within = int(
        (
            valid
            <= tolerance
        ).sum()
    )


    lines.extend([
        label,
        f"Comparable pitcher-seasons: {len(valid)}",
        (
            f"Mean absolute difference: "
            f"{valid.mean():.4f}"
        ),
        (
            f"Median absolute difference: "
            f"{valid.median():.4f}"
        ),
        (
            f"90th percentile difference: "
            f"{valid.quantile(0.90):.4f}"
        ),
        (
            f"Within tolerance: "
            f"{within} of {len(valid)} "
            f"({within / len(valid) * 100:.2f}%)"
        ),
        f"Tolerance: {tolerance}",
        "-" * 72,
    ])


metric_summary(
    "xwOBA validation",
    "xwoba_abs_diff",
    XWOBA_TOLERANCE,
)

metric_summary(
    "K% validation",
    "k_abs_diff_pp",
    K_TOLERANCE_PP,
)

metric_summary(
    "BB% validation",
    "bb_abs_diff_pp",
    BB_TOLERANCE_PP,
)

metric_summary(
    "Barrel% validation",
    "barrel_abs_diff_pp",
    BARREL_TOLERANCE_PP,
)


lines.extend([
    "",
    "INTERPRETATION:",
    (
        "Small differences are expected because detailed Statcast "
        "plate-appearance reconstruction and leaderboard definitions "
        "can differ slightly in denominator handling."
    ),
    (
        "If large systematic differences appear, do not train the "
        "prediction model until the reconstruction is corrected."
    ),
])


SUMMARY_FILE.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print()
print(
    "PITCHER METRIC VALIDATION COMPLETE"
)
print()

print(
    f"Created: {SUMMARY_FILE}"
)

print(
    f"Created: {DETAIL_FILE}"
)
