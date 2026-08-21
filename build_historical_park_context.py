
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7G1
#
# Build leakage-safe historical park context.
#
# For each game season, use the PRIOR completed season's
# 3-year rolling Baseball Savant park factors:
#
#   2022 games -> 2021 park factors
#   2023 games -> 2022 park factors
#   2024 games -> 2023 park factors
#   2025 games -> 2024 park factors
#   2026 games -> 2025 park factors
#
# This avoids using future/current-season results when
# evaluating an earlier game.
#
# Inputs:
#   historical_first_innings.csv
#
# Outputs:
#   historical_park_context.csv
#   historical_park_context_summary.txt
# =========================================================


INPUT_FILE = Path("historical_first_innings.csv")
OUTPUT_FILE = Path("historical_park_context.csv")
SUMMARY_FILE = Path("historical_park_context_summary.txt")

ROLLING = 3


def get_text(url):
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8",
            errors="replace",
        )


def normalize_name(value):
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def find_column(
    df,
    candidates,
    label,
    required=True,
):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    if required:
        raise SystemExit(
            f"ERROR: Could not find {label}. "
            f"Available columns:\n"
            + ", ".join(
                map(str, df.columns)
            )
        )

    return None


def parse_embedded_json(text):
    patterns = [
        r"\bdata\s*=\s*(\[\{.*?\}\])\s*;",
        r"\bdata\s*:\s*(\[\{.*?\}\])\s*[,}]",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.S,
        )

        if match:
            return json.loads(
                match.group(1)
            )

    raise RuntimeError(
        "Could not locate embedded park-factor "
        "data on Baseball Savant page."
    )


def get_park_factor_table(source_year):
    url = (
        "https://baseballsavant.mlb.com/"
        "leaderboard/statcast-park-factors"
        f"?type=year"
        f"&year={source_year}"
        "&batSide="
        "&stat=index_wOBA"
        "&condition=All"
        f"&rolling={ROLLING}"
        "&parks=mlb"
    )

    text = get_text(url)
    records = parse_embedded_json(text)

    rows = []

    for record in records:
        venue_id = (
            record.get("venue_id")
            or record.get("venueId")
            or record.get("stadium_id")
            or record.get("stadiumId")
        )

        venue_name = (
            record.get("name")
            or record.get("venue_name")
            or record.get("venueName")
            or record.get("park_name")
            or record.get("parkName")
            or ""
        )

        def number(*keys):
            for key in keys:
                if key in record:
                    value = pd.to_numeric(
                        record.get(key),
                        errors="coerce",
                    )

                    if pd.notna(value):
                        return float(value)

            return None

        if venue_id is not None:
            try:
                venue_id = int(
                    float(venue_id)
                )
            except Exception:
                venue_id = None

        rows.append({
            "source_year":
                int(source_year),
            "venue_id":
                venue_id,
            "venue_name":
                venue_name,
            "venue_name_norm":
                normalize_name(
                    venue_name
                ),
            "park_runs":
                number(
                    "index_runs",
                    "index_R",
                    "runs",
                ),
            "park_woba":
                number(
                    "index_woba",
                    "index_wOBA",
                    "woba",
                ),
            "park_hr":
                number(
                    "index_hr",
                    "index_HR",
                    "hr",
                ),
            "park_hits":
                number(
                    "index_hits",
                    "hits",
                ),
            "park_so":
                number(
                    "index_so",
                    "index_SO",
                    "so",
                ),
            "park_bb":
                number(
                    "index_bb",
                    "index_BB",
                    "bb",
                ),
        })

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError(
            f"No park factors returned for {source_year}."
        )

    return table


if not INPUT_FILE.exists():
    raise SystemExit(
        "ERROR: historical_first_innings.csv "
        "was not found."
    )


games = pd.read_csv(
    INPUT_FILE
)

game_pk_col = find_column(
    games,
    [
        "game_pk",
        "gamepk",
        "game_id",
    ],
    "game ID",
)

season_col = find_column(
    games,
    [
        "season",
        "year",
    ],
    "season",
)

date_col = find_column(
    games,
    [
        "date",
        "game_date",
    ],
    "game date",
)

venue_id_col = find_column(
    games,
    [
        "venue_id",
        "venueid",
        "stadium_id",
    ],
    "venue ID",
    required=False,
)

venue_name_col = find_column(
    games,
    [
        "venue_name",
        "venue",
        "stadium",
        "park",
    ],
    "venue name",
    required=False,
)

if (
    venue_id_col is None
    and venue_name_col is None
):
    raise SystemExit(
        "ERROR: Historical outcomes contain "
        "neither a venue ID nor a venue name."
    )


games[
    "season_model"
] = pd.to_numeric(
    games[
        season_col
    ],
    errors="raise",
).astype(int)

games[
    "date_model"
] = pd.to_datetime(
    games[
        date_col
    ],
    errors="raise",
)

if venue_id_col is not None:
    games[
        "venue_id_model"
    ] = pd.to_numeric(
        games[
            venue_id_col
        ],
        errors="coerce",
    )
else:
    games[
        "venue_id_model"
    ] = pd.NA

if venue_name_col is not None:
    games[
        "venue_name_model"
    ] = games[
        venue_name_col
    ].fillna(
        ""
    ).astype(str)
else:
    games[
        "venue_name_model"
    ] = ""

games[
    "venue_name_norm"
] = games[
    "venue_name_model"
].map(
    normalize_name
)


seasons = sorted(
    games[
        "season_model"
    ].unique()
)

source_years = sorted(
    {
        int(season) - 1
        for season in seasons
    }
)


park_tables = {}

for source_year in source_years:
    print(
        f"Downloading {ROLLING}-year "
        f"park factors ending {source_year}..."
    )

    park_tables[
        source_year
    ] = get_park_factor_table(
        source_year
    )


output_rows = []


for _, game in games.iterrows():
    season = int(
        game[
            "season_model"
        ]
    )

    source_year = (
        season - 1
    )

    table = park_tables[
        source_year
    ]

    matched = None
    match_method = "missing"

    venue_id = game[
        "venue_id_model"
    ]

    if pd.notna(
        venue_id
    ):
        venue_id_int = int(
            float(venue_id)
        )

        by_id = table[
            table[
                "venue_id"
            ] == venue_id_int
        ]

        if not by_id.empty:
            matched = by_id.iloc[0]
            match_method = "venue_id"

    if matched is None:
        venue_norm = game[
            "venue_name_norm"
        ]

        if venue_norm:
            by_name = table[
                table[
                    "venue_name_norm"
                ] == venue_norm
            ]

            if not by_name.empty:
                matched = (
                    by_name.iloc[0]
                )
                match_method = (
                    "venue_name"
                )

    row = {
        "game_pk":
            int(
                game[
                    game_pk_col
                ]
            ),
        "date":
            game[
                "date_model"
            ].strftime(
                "%Y-%m-%d"
            ),
        "season":
            season,
        "source_year":
            source_year,
        "venue_id":
            (
                int(
                    float(venue_id)
                )
                if pd.notna(
                    venue_id
                )
                else None
            ),
        "venue_name":
            game[
                "venue_name_model"
            ],
        "match_method":
            match_method,
    }

    if matched is None:
        row.update({
            "park_runs":
                None,
            "park_woba":
                None,
            "park_hr":
                None,
            "park_hits":
                None,
            "park_so":
                None,
            "park_bb":
                None,
            "matched_park_name":
                "",
        })
    else:
        row.update({
            "park_runs":
                matched[
                    "park_runs"
                ],
            "park_woba":
                matched[
                    "park_woba"
                ],
            "park_hr":
                matched[
                    "park_hr"
                ],
            "park_hits":
                matched[
                    "park_hits"
                ],
            "park_so":
                matched[
                    "park_so"
                ],
            "park_bb":
                matched[
                    "park_bb"
                ],
            "matched_park_name":
                matched[
                    "venue_name"
                ],
        })

    output_rows.append(
        row
    )


output = pd.DataFrame(
    output_rows
)

output = output.sort_values(
    [
        "date",
        "game_pk",
    ]
).reset_index(
    drop=True
)

if len(output) != len(games):
    raise RuntimeError(
        "ERROR: Historical park context "
        "row count does not match outcomes."
    )

output.to_csv(
    OUTPUT_FILE,
    index=False,
)


total = len(output)

matched_any = int(
    output[
        "park_runs"
    ].notna().sum()
)

matched_id = int(
    (
        output[
            "match_method"
        ] == "venue_id"
    ).sum()
)

matched_name = int(
    (
        output[
            "match_method"
        ] == "venue_name"
    ).sum()
)

missing = (
    total - matched_any
)


lines = [
    "SHARPREPORT HISTORICAL PARK CONTEXT SUMMARY",
    "=" * 72,
    "",
    f"Historical games: {total}",
    (
        f"Games with prior-season park factor: "
        f"{matched_any} "
        f"({matched_any / total * 100:.2f}%)"
    ),
    (
        f"Matched by venue ID: "
        f"{matched_id}"
    ),
    (
        f"Matched by venue name: "
        f"{matched_name}"
    ),
    (
        f"Missing park factor: "
        f"{missing}"
    ),
    "",
    "NO-LOOKAHEAD METHOD:",
    (
        "Each game uses the prior completed season's "
        "3-year rolling Baseball Savant park factor."
    ),
    (
        "Example: 2026 games use the 2025 park-factor table, "
        "not the completed 2026 season."
    ),
    "",
    "PRIMARY MODEL CANDIDATE:",
    "Run Factor (100 = neutral)",
    "",
    "SECONDARY CONTEXT ONLY FOR NOW:",
    "wOBA factor",
    "HR factor",
    "Hits factor",
    "SO factor",
    "BB factor",
]

for season in seasons:
    year_rows = output[
        output[
            "season"
        ] == season
    ]

    year_total = len(
        year_rows
    )

    year_match = int(
        year_rows[
            "park_runs"
        ].notna().sum()
    )

    lines.extend([
        "",
        str(season),
        (
            f"Games: "
            f"{year_total}"
        ),
        (
            f"Park factor available: "
            f"{year_match} "
            f"({year_match / year_total * 100:.2f}%)"
        ),
        (
            f"Source park-factor year: "
            f"{season - 1}"
        ),
        "-" * 72,
    ])


if missing > 0:
    lines.extend([
        "",
        "MISSING VENUES",
        "-" * 72,
    ])

    missing_rows = (
        output[
            output[
                "park_runs"
            ].isna()
        ][
            [
                "season",
                "venue_id",
                "venue_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "venue_name",
            ]
        )
    )

    for _, row in (
        missing_rows.iterrows()
    ):
        lines.append(
            f"{int(row['season'])} | "
            f"venue_id={row['venue_id']} | "
            f"{row['venue_name']}"
        )


SUMMARY_FILE.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print()
print(
    "HISTORICAL PARK CONTEXT BUILD COMPLETE"
)
print()
print(
    f"Created: {OUTPUT_FILE}"
)
print(
    f"Created: {SUMMARY_FILE}"
)
