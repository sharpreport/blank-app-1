
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


# =========================================================
# SHARPREPORT STAGE 7E1
#
# Build historical ACTUAL starting Top-4 batting orders
# from MLB completed-game boxscores.
#
# Input:
#   historical_actual_starters.csv
#
# Outputs:
#   historical_top4_lineups.csv
#   historical_top4_lineups_summary.txt
#   historical_top4_lineups_progress.csv
#
# Resumable. Saves after every 100 games.
# =========================================================


INPUT_FILE = Path("historical_actual_starters.csv")
OUTPUT_FILE = Path("historical_top4_lineups.csv")
PROGRESS_FILE = Path("historical_top4_lineups_progress.csv")
SUMMARY_FILE = Path("historical_top4_lineups_summary.txt")

SAVE_EVERY = 100
SLEEP_SECONDS = 0.08


def get_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            with urlopen(
                request,
                timeout=45,
            ) as response:
                return json.loads(
                    response.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                )
        except Exception as error:
            last_error = error
            print(
                f"Attempt {attempt} failed: {error}"
            )
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Request failed after 3 attempts: {last_error}"
    )


def player_name(player):
    person = player.get("person", {})
    return (
        person.get("fullName")
        or person.get("name")
        or ""
    )


def player_id(player):
    person = player.get("person", {})
    value = person.get("id")

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def extract_starting_top4(team_box):
    players = team_box.get("players", {}) or {}

    # Primary method:
    # MLB completed boxscores generally preserve battingOrder
    # as 100, 200, 300 ... for starters, with later substitutions
    # receiving values such as 101, 201, etc.
    #
    # For each of the first four batting slots, choose the lowest
    # battingOrder value in that hundred-band. This preserves the
    # original starter rather than a later substitute.
    by_slot = {}

    for pdata in players.values():
        raw_order = pdata.get("battingOrder")

        if raw_order in (None, ""):
            continue

        try:
            order = int(raw_order)
        except Exception:
            continue

        slot = order // 100
        suffix = order % 100

        if slot not in (1, 2, 3, 4):
            continue

        pid = player_id(pdata)

        if pid is None:
            continue

        candidate = {
            "order": order,
            "suffix": suffix,
            "id": pid,
            "name": player_name(pdata),
        }

        if (
            slot not in by_slot
            or candidate["suffix"] < by_slot[slot]["suffix"]
        ):
            by_slot[slot] = candidate

    if all(slot in by_slot for slot in (1, 2, 3, 4)):
        return [
            by_slot[slot]
            for slot in (1, 2, 3, 4)
        ], "player_battingOrder"

    # Fallback:
    # Use the team's ordered battingOrder list if player-level
    # batting-order fields are incomplete.
    ordered_ids = team_box.get("battingOrder", []) or []

    fallback = []

    for raw_id in ordered_ids:
        try:
            pid = int(raw_id)
        except Exception:
            continue

        pdata = players.get(
            f"ID{pid}",
            {}
        )

        fallback.append({
            "order": None,
            "suffix": None,
            "id": pid,
            "name": player_name(pdata),
        })

        if len(fallback) == 4:
            break

    if len(fallback) == 4:
        return fallback, "team_battingOrder_fallback"

    return [], "missing"


def build_row(game):
    game_pk = int(game["game_pk"])

    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/game/{game_pk}/boxscore"
    )

    box = get_json(url)

    teams = box.get("teams", {}) or {}

    away_box = teams.get("away", {}) or {}
    home_box = teams.get("home", {}) or {}

    away_top4, away_method = extract_starting_top4(
        away_box
    )
    home_top4, home_method = extract_starting_top4(
        home_box
    )

    row = {
        "game_pk": game_pk,
        "date": game["date"],
        "season": int(game["season"]),
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "away_method": away_method,
        "home_method": home_method,
        "away_top4_complete": int(len(away_top4) == 4),
        "home_top4_complete": int(len(home_top4) == 4),
    }

    for side, top4 in [
        ("away", away_top4),
        ("home", home_top4),
    ]:
        for slot in range(1, 5):
            if len(top4) >= slot:
                player = top4[slot - 1]
                row[f"{side}_b{slot}_id"] = player["id"]
                row[f"{side}_b{slot}_name"] = player["name"]
            else:
                row[f"{side}_b{slot}_id"] = None
                row[f"{side}_b{slot}_name"] = ""

    return row


if not INPUT_FILE.exists():
    raise SystemExit(
        "ERROR: historical_actual_starters.csv was not found."
    )


games = pd.read_csv(INPUT_FILE)

required = [
    "game_pk",
    "date",
    "season",
    "away_team",
    "home_team",
]

for column in required:
    if column not in games.columns:
        raise SystemExit(
            f"ERROR: Missing required column: {column}"
        )


games["game_pk"] = pd.to_numeric(
    games["game_pk"],
    errors="raise",
).astype(int)

games["season"] = pd.to_numeric(
    games["season"],
    errors="raise",
).astype(int)


existing_rows = []

if PROGRESS_FILE.exists():
    progress = pd.read_csv(PROGRESS_FILE)

    if not progress.empty:
        existing_rows = progress.to_dict(
            orient="records"
        )

completed_ids = {
    int(row["game_pk"])
    for row in existing_rows
}


remaining = games[
    ~games["game_pk"].isin(completed_ids)
].copy()


print()
print("SHARPREPORT HISTORICAL TOP-4 LINEUP BUILD")
print("=" * 70)
print(f"Total historical games: {len(games)}")
print(f"Already completed: {len(completed_ids)}")
print(f"Remaining: {len(remaining)}")
print()


rows = list(existing_rows)


for count, (_, game) in enumerate(
    remaining.iterrows(),
    start=1,
):
    try:
        row = build_row(game)
    except Exception as error:
        print(
            f"ERROR game {int(game['game_pk'])}: {error}"
        )

        row = {
            "game_pk": int(game["game_pk"]),
            "date": game["date"],
            "season": int(game["season"]),
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "away_method": "request_error",
            "home_method": "request_error",
            "away_top4_complete": 0,
            "home_top4_complete": 0,
        }

        for side in ["away", "home"]:
            for slot in range(1, 5):
                row[f"{side}_b{slot}_id"] = None
                row[f"{side}_b{slot}_name"] = ""

    rows.append(row)

    total_done = len(rows)

    if (
        count % SAVE_EVERY == 0
        or count == len(remaining)
    ):
        progress_df = pd.DataFrame(rows)

        progress_df = progress_df.drop_duplicates(
            subset=["game_pk"],
            keep="last",
        )

        progress_df = progress_df.sort_values(
            ["date", "game_pk"]
        ).reset_index(drop=True)

        progress_df.to_csv(
            PROGRESS_FILE,
            index=False,
        )

        print(
            f"Saved progress: "
            f"{len(progress_df)} / {len(games)} games"
        )

    time.sleep(SLEEP_SECONDS)


final = pd.DataFrame(rows)

final = final.drop_duplicates(
    subset=["game_pk"],
    keep="last",
)

final = final.sort_values(
    ["date", "game_pk"]
).reset_index(drop=True)

final.to_csv(
    OUTPUT_FILE,
    index=False,
)


total_games = len(final)

away_complete = int(
    final["away_top4_complete"]
    .fillna(0)
    .astype(int)
    .sum()
)

home_complete = int(
    final["home_top4_complete"]
    .fillna(0)
    .astype(int)
    .sum()
)

both_complete = int(
    (
        final["away_top4_complete"]
        .fillna(0)
        .astype(int)
        .eq(1)
        &
        final["home_top4_complete"]
        .fillna(0)
        .astype(int)
        .eq(1)
    ).sum()
)

primary_away = int(
    (
        final["away_method"]
        == "player_battingOrder"
    ).sum()
)

primary_home = int(
    (
        final["home_method"]
        == "player_battingOrder"
    ).sum()
)

fallback_away = int(
    (
        final["away_method"]
        == "team_battingOrder_fallback"
    ).sum()
)

fallback_home = int(
    (
        final["home_method"]
        == "team_battingOrder_fallback"
    ).sum()
)


lines = [
    "SHARPREPORT HISTORICAL TOP-4 LINEUP SUMMARY",
    "=" * 70,
    "",
    f"Historical games: {total_games}",
    "",
    (
        f"Away Top-4 complete: "
        f"{away_complete} "
        f"({away_complete / total_games * 100:.2f}%)"
    ),
    (
        f"Home Top-4 complete: "
        f"{home_complete} "
        f"({home_complete / total_games * 100:.2f}%)"
    ),
    (
        f"Both teams Top-4 complete: "
        f"{both_complete} "
        f"({both_complete / total_games * 100:.2f}%)"
    ),
    "",
    "EXTRACTION METHODS",
    "-" * 70,
    (
        f"Away primary starter-order method: "
        f"{primary_away}"
    ),
    (
        f"Away fallback method: "
        f"{fallback_away}"
    ),
    (
        f"Home primary starter-order method: "
        f"{primary_home}"
    ),
    (
        f"Home fallback method: "
        f"{fallback_home}"
    ),
    "",
    "NOTE:",
    (
        "Primary extraction prefers the earliest player in each "
        "batting-order slot so completed-game substitutions do not "
        "replace the original starting hitter."
    ),
]


SUMMARY_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print("HISTORICAL TOP-4 LINEUP BUILD COMPLETE")
print()
print(f"Created: {OUTPUT_FILE}")
print(f"Created: {SUMMARY_FILE}")
print(f"Progress file retained: {PROGRESS_FILE}")
