import json

from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode

from nrfi_data_logger import (
    list_repo_directory,
    read_json_file,
    upsert_json_file,
)


MLB_API_ROOT = "https://statsapi.mlb.com/api"


def _get_json(
    url,
):
    request = Request(
        url,
        headers={
            "User-Agent":
                "SharpReport-NRFI-Result-Grader"
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:

        return json.load(
            response
        )


def _parse_datetime(
    value,
):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )
    except Exception:
        return None


def _american_profit_on_one_unit_risk(
    price,
    won,
):
    if price is None:
        return None

    try:
        price = float(
            price
        )
    except Exception:
        return None

    if not won:
        return -1.0

    if price > 0:
        return (
            price
            / 100.0
        )

    if price < 0:
        return (
            100.0
            / abs(
                price
            )
        )

    return None


def _first_inning_runs_from_innings(
    innings,
):
    for inning in innings or []:
        try:
            inning_number = int(
                inning.get(
                    "num",
                    0,
                )
            )
        except Exception:
            inning_number = 0

        if inning_number != 1:
            continue

        away = (
            inning
            .get("away", {})
            .get("runs")
        )

        home = (
            inning
            .get("home", {})
            .get("runs")
        )

        if away is None or home is None:
            return None

        return (
            int(away),
            int(home),
        )

    return None


def _fetch_first_inning_runs(
    game,
):
    linescore = game.get(
        "linescore",
        {}
    )

    result = (
        _first_inning_runs_from_innings(
            linescore.get(
                "innings",
                []
            )
        )
    )

    if result is not None:
        return result


    game_pk = game.get(
        "gamePk"
    )

    if not game_pk:
        return None


    feed = _get_json(
        f"{MLB_API_ROOT}/v1.1/"
        f"game/{game_pk}/feed/live"
    )

    innings = (
        feed
        .get("liveData", {})
        .get("linescore", {})
        .get(
            "innings",
            []
        )
    )

    return (
        _first_inning_runs_from_innings(
            innings
        )
    )


def fetch_completed_games(
    game_date,
):
    date_text = (
        game_date.isoformat()
        if isinstance(
            game_date,
            date
        )
        else str(
            game_date
        )
    )

    query = urlencode({
        "sportId":
            1,

        "date":
            date_text,

        "hydrate":
            "linescore",
    })

    payload = _get_json(
        f"{MLB_API_ROOT}/v1/schedule?"
        f"{query}"
    )

    games = []

    for date_block in payload.get(
        "dates",
        []
    ):
        for game in date_block.get(
            "games",
            []
        ):
            status = game.get(
                "status",
                {}
            )

            abstract_state = status.get(
                "abstractGameState",
                ""
            )

            coded_state = status.get(
                "codedGameState",
                ""
            )

            detailed_state = status.get(
                "detailedState",
                ""
            )


            is_final = (
                abstract_state == "Final"
                or
                coded_state == "F"
                or
                detailed_state.startswith(
                    "Final"
                )
            )

            if not is_final:
                continue


            first_runs = (
                _fetch_first_inning_runs(
                    game
                )
            )

            if first_runs is None:
                continue


            away_first_runs, home_first_runs = (
                first_runs
            )

            nrfi = (
                away_first_runs == 0
                and
                home_first_runs == 0
            )

            away_team = (
                game
                .get("teams", {})
                .get("away", {})
                .get("team", {})
                .get(
                    "name",
                    ""
                )
            )

            home_team = (
                game
                .get("teams", {})
                .get("home", {})
                .get("team", {})
                .get(
                    "name",
                    ""
                )
            )

            away_final_runs = (
                game
                .get("teams", {})
                .get("away", {})
                .get(
                    "score"
                )
            )

            home_final_runs = (
                game
                .get("teams", {})
                .get("home", {})
                .get(
                    "score"
                )
            )

            games.append({
                "game_id":
                    game.get(
                        "gamePk"
                    ),

                "game":
                    f"{away_team} @ {home_team}",

                "away_team":
                    away_team,

                "home_team":
                    home_team,

                "game_start_utc":
                    game.get(
                        "gameDate"
                    ),

                "status":
                    detailed_state,

                "away_first_inning_runs":
                    away_first_runs,

                "home_first_inning_runs":
                    home_first_runs,

                "total_first_inning_runs":
                    (
                        away_first_runs
                        + home_first_runs
                    ),

                "actual_side":
                    (
                        "NRFI"
                        if nrfi
                        else "YRFI"
                    ),

                "nrfi":
                    nrfi,

                "yrfi":
                    not nrfi,

                "away_final_runs":
                    away_final_runs,

                "home_final_runs":
                    home_final_runs,
            })

    return games


def _load_snapshots_for_date(
    token,
    repo,
    game_date,
):
    date_text = (
        game_date.isoformat()
        if isinstance(
            game_date,
            date
        )
        else str(
            game_date
        )
    )

    folder = (
        f"snapshots/{date_text}"
    )

    items = list_repo_directory(
        token=token,
        repo=repo,
        path=folder,
    )

    snapshots = []

    for item in items:
        if (
            item.get("type") != "file"
            or
            not str(
                item.get(
                    "name",
                    ""
                )
            ).endswith(
                ".json"
            )
        ):
            continue

        loaded = read_json_file(
            token=token,
            repo=repo,
            path=item["path"],
        )

        if not loaded:
            continue

        snapshot = loaded.get(
            "data",
            {}
        )

        snapshot["_path"] = (
            item["path"]
        )

        snapshot["_snapshot_time"] = (
            _parse_datetime(
                snapshot.get(
                    "snapshot_time_utc"
                )
            )
        )

        snapshots.append(
            snapshot
        )


    snapshots.sort(
        key=lambda snapshot:
            (
                snapshot.get(
                    "_snapshot_time"
                )
                or
                datetime.min.replace(
                    tzinfo=timezone.utc
                )
            )
    )

    return snapshots


def _matching_game_row(
    snapshot,
    game_id,
):
    for row in snapshot.get(
        "games",
        []
    ):
        try:
            row_game_id = int(
                row.get(
                    "Game ID"
                )
            )
        except Exception:
            continue

        if row_game_id == int(
            game_id
        ):
            return row

    return None


def _latest_final_pregame_snapshot(
    snapshots,
    completed_game,
):
    game_start = _parse_datetime(
        completed_game.get(
            "game_start_utc"
        )
    )

    if game_start is None:
        return None


    eligible = []

    for snapshot in snapshots:
        snapshot_time = snapshot.get(
            "_snapshot_time"
        )

        if (
            snapshot_time is None
            or
            snapshot_time >= game_start
        ):
            continue


        row = _matching_game_row(
            snapshot,
            completed_game[
                "game_id"
            ],
        )

        if row is None:
            continue


        if row.get(
            "Status"
        ) != "FINAL":
            continue


        if row.get(
            "Market Status"
        ) != "LIVE":
            continue


        if row.get(
            "Best Price"
        ) is None:
            continue


        eligible.append({
            "snapshot":
                snapshot,

            "row":
                row,

            "snapshot_time":
                snapshot_time,
        })


    if not eligible:
        return None

    return max(
        eligible,
        key=lambda item:
            item["snapshot_time"],
    )


def _grade_side(
    actual_side,
    side,
    price,
):
    if side not in {
        "NRFI",
        "YRFI",
    }:
        return {
            "won":
                None,

            "profit_units_1u_risk":
                None,
        }

    won = (
        actual_side == side
    )

    return {
        "won":
            won,

        "profit_units_1u_risk":
            _american_profit_on_one_unit_risk(
                price=price,
                won=won,
            ),
    }


def _build_game_grade(
    completed_game,
    selected,
):
    base = {
        **completed_game,

        "grading_status":
            "NO_ELIGIBLE_FINAL_PREGAME_SNAPSHOT",

        "selected_snapshot_path":
            None,

        "selected_snapshot_time_utc":
            None,

        "minutes_before_first_pitch":
            None,

        "model_side":
            None,

        "model_probability":
            None,

        "market_no_vig":
            None,

        "market_edge":
            None,

        "best_price":
            None,

        "best_book":
            None,

        "break_even":
            None,

        "price_edge":
            None,

        "model_side_won":
            None,

        "model_side_profit_units_1u_risk":
            None,

        "nrfi_probability":
            None,

        "yrfi_probability":
            None,

        "best_nrfi_price":
            None,

        "best_nrfi_book":
            None,

        "nrfi_break_even":
            None,

        "nrfi_market_edge":
            None,

        "nrfi_price_edge":
            None,

        "nrfi_won":
            None,

        "nrfi_profit_units_1u_risk":
            None,

        "best_yrfi_price":
            None,

        "best_yrfi_book":
            None,

        "yrfi_break_even":
            None,

        "yrfi_market_edge":
            None,

        "yrfi_price_edge":
            None,

        "yrfi_won":
            None,

        "yrfi_profit_units_1u_risk":
            None,
    }


    if selected is None:
        return base


    snapshot = selected[
        "snapshot"
    ]

    row = selected[
        "row"
    ]

    snapshot_time = selected[
        "snapshot_time"
    ]

    game_start = _parse_datetime(
        completed_game[
            "game_start_utc"
        ]
    )

    minutes_before = (
        (
            game_start
            - snapshot_time
        ).total_seconds()
        / 60.0
    )


    model_side = row.get(
        "Model Side"
    )

    model_grade = _grade_side(
        actual_side=
            completed_game[
                "actual_side"
            ],

        side=
            model_side,

        price=
            row.get(
                "Best Price"
            ),
    )

    nrfi_grade = _grade_side(
        actual_side=
            completed_game[
                "actual_side"
            ],

        side=
            "NRFI",

        price=
            row.get(
                "Best NRFI Price"
            ),
    )

    yrfi_grade = _grade_side(
        actual_side=
            completed_game[
                "actual_side"
            ],

        side=
            "YRFI",

        price=
            row.get(
                "Best YRFI Price"
            ),
    )


    base.update({
        "grading_status":
            "GRADED_FINAL_PREGAME",

        "selected_snapshot_path":
            snapshot.get(
                "_path"
            ),

        "selected_snapshot_time_utc":
            snapshot.get(
                "snapshot_time_utc"
            ),

        "minutes_before_first_pitch":
            round(
                minutes_before,
                2,
            ),

        "model_side":
            model_side,

        "model_probability":
            row.get(
                "Model Probability"
            ),

        "market_no_vig":
            row.get(
                "Market No-Vig"
            ),

        "market_edge":
            row.get(
                "Edge"
            ),

        "best_price":
            row.get(
                "Best Price"
            ),

        "best_book":
            row.get(
                "Best Book"
            ),

        "break_even":
            row.get(
                "Market Raw Implied"
            ),

        "price_edge":
            row.get(
                "Price Edge"
            ),

        "model_side_won":
            model_grade[
                "won"
            ],

        "model_side_profit_units_1u_risk":
            model_grade[
                "profit_units_1u_risk"
            ],

        "nrfi_probability":
            row.get(
                "NRFI Probability"
            ),

        "yrfi_probability":
            row.get(
                "YRFI Probability"
            ),

        "best_nrfi_price":
            row.get(
                "Best NRFI Price"
            ),

        "best_nrfi_book":
            row.get(
                "Best NRFI Book"
            ),

        "nrfi_break_even":
            row.get(
                "NRFI Break-Even"
            ),

        "nrfi_market_edge":
            row.get(
                "NRFI Market Edge"
            ),

        "nrfi_price_edge":
            row.get(
                "NRFI Price Edge"
            ),

        "nrfi_won":
            nrfi_grade[
                "won"
            ],

        "nrfi_profit_units_1u_risk":
            nrfi_grade[
                "profit_units_1u_risk"
            ],

        "best_yrfi_price":
            row.get(
                "Best YRFI Price"
            ),

        "best_yrfi_book":
            row.get(
                "Best YRFI Book"
            ),

        "yrfi_break_even":
            row.get(
                "YRFI Break-Even"
            ),

        "yrfi_market_edge":
            row.get(
                "YRFI Market Edge"
            ),

        "yrfi_price_edge":
            row.get(
                "YRFI Price Edge"
            ),

        "yrfi_won":
            yrfi_grade[
                "won"
            ],

        "yrfi_profit_units_1u_risk":
            yrfi_grade[
                "profit_units_1u_risk"
            ],
    })

    # Preserve research-only rolling pitcher first-inning
    # fields from the exact selected FINAL pregame snapshot.
    # These values are not used to change Model v1.
    for key, value in row.items():
        if (
            str(key).startswith(
                "Away Pitcher FI "
            )
            or
            str(key).startswith(
                "Home Pitcher FI "
            )
        ):
            base[key] = value


    return base


def grade_date(
    token,
    repo,
    game_date,
):
    date_text = (
        game_date.isoformat()
        if isinstance(
            game_date,
            date
        )
        else str(
            game_date
        )
    )

    snapshots = _load_snapshots_for_date(
        token=token,
        repo=repo,
        game_date=date_text,
    )

    if not snapshots:
        return {
            "date":
                date_text,

            "snapshot_count":
                0,

            "completed_games":
                0,

            "graded_games":
                0,

            "result_file_changed":
                False,

            "result_path":
                None,

            "grades":
                [],
        }


    completed_games = (
        fetch_completed_games(
            date_text
        )
    )

    if not completed_games:
        return {
            "date":
                date_text,

            "snapshot_count":
                len(
                    snapshots
                ),

            "completed_games":
                0,

            "graded_games":
                0,

            "result_file_changed":
                False,

            "result_path":
                None,

            "grades":
                [],
        }


    grades = []

    for completed_game in completed_games:
        selected = (
            _latest_final_pregame_snapshot(
                snapshots=snapshots,
                completed_game=completed_game,
            )
        )

        grades.append(
            _build_game_grade(
                completed_game=
                    completed_game,

                selected=
                    selected,
            )
        )


    graded_count = sum(
        1
        for row in grades
        if row.get(
            "grading_status"
        ) == "GRADED_FINAL_PREGAME"
    )


    result_payload = {
        "schema_version":
            "1.0",

        "result_type":
            "graded_slate",

        "game_date":
            date_text,

        "snapshot_count":
            len(
                snapshots
            ),

        "completed_games":
            len(
                completed_games
            ),

        "graded_final_pregame_games":
            graded_count,

        "grading_rule":
            (
                "Latest FINAL snapshot strictly before first pitch "
                "with LIVE first-inning market and a best available price."
            ),

        "profit_convention":
            (
                "Profit/loss is measured in units with 1.00 unit risked "
                "per graded side."
            ),

        "games":
            grades,
    }


    result_path = (
        f"results/{date_text}.json"
    )

    save_result = upsert_json_file(
        token=token,
        repo=repo,
        path=result_path,
        payload=result_payload,
        commit_message=(
            f"Grade NRFI results "
            f"{date_text}"
        ),
    )


    return {
        "date":
            date_text,

        "snapshot_count":
            len(
                snapshots
            ),

        "completed_games":
            len(
                completed_games
            ),

        "graded_games":
            graded_count,

        "result_file_changed":
            bool(
                save_result.get(
                    "changed"
                )
            ),

        "result_path":
            result_path,

        "grades":
            grades,
    }


def _sync_edge_ledger(
    token,
    repo,
    summaries,
):
    ledger_path = (
        "analytics/final_pregame_edge_ledger.json"
    )

    existing = read_json_file(
        token=token,
        repo=repo,
        path=ledger_path,
    )

    existing_payload = (
        existing.get(
            "data",
            {}
        )
        if existing
        else {}
    )

    existing_games = (
        existing_payload.get(
            "games",
            []
        )
        or []
    )

    by_game_id = {}

    for row in existing_games:
        game_id = row.get(
            "game_id"
        )

        if game_id is None:
            continue

        by_game_id[
            str(game_id)
        ] = row


    changed = False

    for summary in summaries:
        game_date = summary.get(
            "date"
        )

        for row in summary.get(
            "grades",
            []
        ):
            if (
                row.get(
                    "grading_status"
                )
                !=
                "GRADED_FINAL_PREGAME"
            ):
                continue

            game_id = row.get(
                "game_id"
            )

            if game_id is None:
                continue

            ledger_row = {
                **row,

                "game_date":
                    game_date,
            }

            key = str(
                game_id
            )

            if (
                by_game_id.get(
                    key
                )
                != ledger_row
            ):
                by_game_id[
                    key
                ] = ledger_row

                changed = True


    if (
        not changed
        and
        existing is not None
    ):
        return {
            "changed":
                False,

            "path":
                ledger_path,

            "games":
                len(
                    by_game_id
                ),
        }


    if not by_game_id:
        return {
            "changed":
                False,

            "path":
                None,

            "games":
                0,
        }


    games = sorted(
        by_game_id.values(),
        key=lambda row: (
            str(
                row.get(
                    "game_date",
                    ""
                )
            ),
            str(
                row.get(
                    "game_start_utc",
                    ""
                )
            ),
            str(
                row.get(
                    "game_id",
                    ""
                )
            ),
        ),
    )


    payload = {
        "schema_version":
            "1.0",

        "ledger_type":
            "final_pregame_edge_ledger",

        "updated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "grading_rule":
            (
                "Latest FINAL snapshot strictly before first pitch "
                "with LIVE first-inning market and a best available price."
            ),

        "profit_convention":
            (
                "One unit risked per model-side observation. "
                "Winning profit follows the saved American price; "
                "a loss is -1.00 unit."
            ),

        "games":
            games,
    }


    save_result = upsert_json_file(
        token=token,
        repo=repo,
        path=ledger_path,
        payload=payload,
        commit_message=(
            "Update FINAL pregame edge performance ledger"
        ),
    )


    return {
        "changed":
            bool(
                save_result.get(
                    "changed"
                )
            ),

        "path":
            ledger_path,

        "games":
            len(
                games
            ),
    }


def grade_recent_results(
    token,
    repo,
    reference_date,
    days_back=7,
):
    if not isinstance(
        reference_date,
        date
    ):
        reference_date = date.fromisoformat(
            str(
                reference_date
            )
        )


    summaries = []

    for offset in range(
        days_back
    ):
        target_date = (
            reference_date
            - timedelta(
                days=offset
            )
        )

        summaries.append(
            grade_date(
                token=token,
                repo=repo,
                game_date=target_date,
            )
        )


    ledger_summary = (
        _sync_edge_ledger(
            token=token,
            repo=repo,
            summaries=summaries,
        )
    )


    return {
        "dates_checked":
            len(
                summaries
            ),

        "dates_with_snapshots":
            sum(
                1
                for row in summaries
                if row[
                    "snapshot_count"
                ] > 0
            ),

        "completed_games_seen":
            sum(
                row[
                    "completed_games"
                ]
                for row in summaries
            ),

        "graded_games":
            sum(
                row[
                    "graded_games"
                ]
                for row in summaries
            ),

        "result_files_updated":
            sum(
                1
                for row in summaries
                if row[
                    "result_file_changed"
                ]
            ),

        "edge_ledger_games":
            ledger_summary[
                "games"
            ],

        "edge_ledger_updated":
            ledger_summary[
                "changed"
            ],

        "edge_ledger_path":
            ledger_summary[
                "path"
            ],

        "dates":
            summaries,
    }
