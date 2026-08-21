import json

from datetime import date, datetime, timedelta
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MLB_API_ROOT = "https://statsapi.mlb.com/api"


def _get_json(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                "SharpReport-Pitcher-First-Inning-Windows/1.0"
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        return json.load(
            response
        )


def _date_text(value):
    if isinstance(
        value,
        datetime,
    ):
        return value.date().isoformat()

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    return str(
        value
    )[:10]


def _chunk_dates(
    start_date,
    end_date,
    chunk_days=31,
):
    cursor = start_date

    while cursor <= end_date:
        chunk_end = min(
            cursor
            + timedelta(
                days=chunk_days - 1
            ),
            end_date,
        )

        yield cursor, chunk_end

        cursor = (
            chunk_end
            + timedelta(
                days=1
            )
        )


def _first_inning_from_game(game):
    innings = (
        game
        .get("linescore", {})
        .get("innings", [])
    )

    for inning in innings:
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

        away_runs = (
            inning
            .get("away", {})
            .get("runs")
        )

        home_runs = (
            inning
            .get("home", {})
            .get("runs")
        )

        if (
            away_runs is None
            or
            home_runs is None
        ):
            return None

        return {
            "away_team_id":
                (
                    game
                    .get("teams", {})
                    .get("away", {})
                    .get("team", {})
                    .get("id")
                ),

            "home_team_id":
                (
                    game
                    .get("teams", {})
                    .get("home", {})
                    .get("team", {})
                    .get("id")
                ),

            "away_first_runs":
                int(
                    away_runs
                ),

            "home_first_runs":
                int(
                    home_runs
                ),

            "total_first_runs":
                int(
                    away_runs
                )
                + int(
                    home_runs
                ),
        }

    return None


@lru_cache(
    maxsize=16
)
def get_season_first_inning_lookup(
    season,
    end_date,
):
    season = int(
        season
    )

    end_date = date.fromisoformat(
        _date_text(
            end_date
        )
    )

    # March 1 safely includes special/international regular-season
    # openers while avoiding unnecessary offseason schedule data.
    start_date = date(
        season,
        3,
        1,
    )

    if end_date < start_date:
        return {}

    lookup = {}

    for chunk_start, chunk_end in _chunk_dates(
        start_date,
        end_date,
    ):
        query = urlencode({
            "sportId":
                1,

            "startDate":
                chunk_start.isoformat(),

            "endDate":
                chunk_end.isoformat(),

            "gameType":
                "R",

            "hydrate":
                "linescore",
        })

        payload = _get_json(
            f"{MLB_API_ROOT}/v1/schedule?{query}"
        )

        for date_block in payload.get(
            "dates",
            [],
        ):
            for game in date_block.get(
                "games",
                [],
            ):
                game_pk = game.get(
                    "gamePk"
                )

                if not game_pk:
                    continue

                first_inning = (
                    _first_inning_from_game(
                        game
                    )
                )

                if first_inning is None:
                    continue

                lookup[
                    int(
                        game_pk
                    )
                ] = first_inning

    return lookup


def _pitcher_game_log(
    pitcher_id,
    season,
):
    query = urlencode({
        "stats":
            "gameLog",

        "group":
            "pitching",

        "season":
            int(
                season
            ),

        "gameType":
            "R",
    })

    payload = _get_json(
        f"{MLB_API_ROOT}/v1/people/"
        f"{int(pitcher_id)}/stats?{query}"
    )

    splits = []

    for stat_group in payload.get(
        "stats",
        [],
    ):
        splits.extend(
            stat_group.get(
                "splits",
                [],
            )
        )

    return splits


def _is_start(split):
    stat = split.get(
        "stat",
        {},
    )

    try:
        return int(
            stat.get(
                "gamesStarted",
                0,
            )
        ) == 1
    except Exception:
        return False


def _opponent_first_runs(
    split,
    first_inning,
):
    team_id = (
        split
        .get("team", {})
        .get("id")
    )

    away_team_id = first_inning.get(
        "away_team_id"
    )

    home_team_id = first_inning.get(
        "home_team_id"
    )

    if (
        team_id is not None
        and
        away_team_id is not None
        and
        int(team_id) == int(away_team_id)
    ):
        return first_inning.get(
            "home_first_runs"
        )

    if (
        team_id is not None
        and
        home_team_id is not None
        and
        int(team_id) == int(home_team_id)
    ):
        return first_inning.get(
            "away_first_runs"
        )

    # Some Stats API game-log payloads expose isHome even when
    # the team object is incomplete. Use it as a safe fallback.
    is_home = split.get(
        "isHome"
    )

    if is_home is True:
        return first_inning.get(
            "away_first_runs"
        )

    if is_home is False:
        return first_inning.get(
            "home_first_runs"
        )

    opponent_id = (
        split
        .get("opponent", {})
        .get("id")
    )

    if (
        opponent_id is not None
        and
        away_team_id is not None
        and
        int(opponent_id) == int(away_team_id)
    ):
        return first_inning.get(
            "away_first_runs"
        )

    if (
        opponent_id is not None
        and
        home_team_id is not None
        and
        int(opponent_id) == int(home_team_id)
    ):
        return first_inning.get(
            "home_first_runs"
        )

    return None


def _summarize_start_rows(
    rows,
    requested_starts=None,
):
    selected = (
        rows[:requested_starts]
        if requested_starts is not None
        else rows
    )

    starts = len(
        selected
    )

    if starts == 0:
        return {
            "requested_starts":
                requested_starts,

            "starts":
                0,

            "scoreless_opponent_first":
                0,

            "scoreless_opponent_first_pct":
                None,

            "first_inning_runs_allowed":
                0,

            "first_inning_runs_allowed_per_start":
                None,

            "game_nrfi":
                0,

            "game_yrfi":
                0,

            "game_nrfi_pct":
                None,
        }

    scoreless = sum(
        1
        for row in selected
        if row[
            "opponent_first_runs"
        ] == 0
    )

    runs_allowed = sum(
        row[
            "opponent_first_runs"
        ]
        for row in selected
    )

    nrfi = sum(
        1
        for row in selected
        if row[
            "game_nrfi"
        ]
    )

    return {
        "requested_starts":
            requested_starts,

        "starts":
            starts,

        "scoreless_opponent_first":
            scoreless,

        "scoreless_opponent_first_pct":
            scoreless
            / starts
            * 100.0,

        "first_inning_runs_allowed":
            runs_allowed,

        "first_inning_runs_allowed_per_start":
            runs_allowed
            / starts,

        "game_nrfi":
            nrfi,

        "game_yrfi":
            starts
            - nrfi,

        "game_nrfi_pct":
            nrfi
            / starts
            * 100.0,
    }


def empty_pitcher_first_inning_windows(
    pitcher_id=None,
    season=None,
    end_date=None,
    error=None,
):
    return {
        "pitcher_id":
            pitcher_id,

        "season":
            season,

        "through_date":
            _date_text(
                end_date
            )
            if end_date is not None
            else None,

        "data_status":
            "UNAVAILABLE"
            if error
            else "NO_STARTS",

        "error":
            str(
                error
            )
            if error
            else None,

        "season_window":
            _summarize_start_rows(
                []
            ),

        "last_30":
            _summarize_start_rows(
                [],
                30,
            ),

        "last_20":
            _summarize_start_rows(
                [],
                20,
            ),

        "last_10":
            _summarize_start_rows(
                [],
                10,
            ),
    }


def get_pitcher_first_inning_windows(
    pitcher_id,
    season,
    end_date,
):
    pitcher_id = int(
        pitcher_id
    )

    season = int(
        season
    )

    end_date_text = _date_text(
        end_date
    )

    end_date_value = date.fromisoformat(
        end_date_text
    )

    season_lookup = (
        get_season_first_inning_lookup(
            season,
            end_date_text,
        )
    )

    splits = _pitcher_game_log(
        pitcher_id,
        season,
    )

    start_rows = []

    for split in splits:
        if not _is_start(
            split
        ):
            continue

        split_date_text = _date_text(
            split.get(
                "date",
                ""
            )
        )

        try:
            split_date = date.fromisoformat(
                split_date_text
            )
        except Exception:
            continue

        if split_date > end_date_value:
            continue

        game_pk = (
            split
            .get("game", {})
            .get("gamePk")
        )

        if game_pk is None:
            continue

        first_inning = season_lookup.get(
            int(
                game_pk
            )
        )

        if first_inning is None:
            continue

        opponent_runs = _opponent_first_runs(
            split,
            first_inning,
        )

        if opponent_runs is None:
            continue

        total_first_runs = first_inning.get(
            "total_first_runs"
        )

        start_rows.append({
            "date":
                split_date_text,

            "game_id":
                int(
                    game_pk
                ),

            "opponent_first_runs":
                int(
                    opponent_runs
                ),

            "game_nrfi":
                bool(
                    total_first_runs == 0
                ),
        })

    # Most recent start first so [:10], [:20], [:30]
    # are true rolling windows.
    start_rows.sort(
        key=lambda row:
            row[
                "date"
            ],
        reverse=True,
    )

    data_status = (
        "OK"
        if start_rows
        else "NO_STARTS"
    )

    return {
        "pitcher_id":
            pitcher_id,

        "season":
            season,

        "through_date":
            end_date_text,

        "data_status":
            data_status,

        "error":
            None,

        "season_window":
            _summarize_start_rows(
                start_rows
            ),

        "last_30":
            _summarize_start_rows(
                start_rows,
                30,
            ),

        "last_20":
            _summarize_start_rows(
                start_rows,
                20,
            ),

        "last_10":
            _summarize_start_rows(
                start_rows,
                10,
            ),
    }


def flatten_pitcher_first_inning_windows(
    prefix,
    history,
):
    if not history:
        history = empty_pitcher_first_inning_windows()

    output = {
        f"{prefix} FI History Through":
            history.get(
                "through_date"
            ),

        f"{prefix} FI History Status":
            history.get(
                "data_status"
            ),

        f"{prefix} FI Used As Model Weight":
            False,
    }

    windows = [
        (
            "Season",
            "season_window",
        ),
        (
            "L30",
            "last_30",
        ),
        (
            "L20",
            "last_20",
        ),
        (
            "L10",
            "last_10",
        ),
    ]

    for label, key in windows:
        window = history.get(
            key,
            {},
        )

        output.update({
            f"{prefix} FI {label} Starts":
                window.get(
                    "starts"
                ),

            f"{prefix} FI {label} Scoreless Opp 1st":
                window.get(
                    "scoreless_opponent_first"
                ),

            f"{prefix} FI {label} Scoreless Opp 1st %":
                window.get(
                    "scoreless_opponent_first_pct"
                ),

            f"{prefix} FI {label} Runs/Start":
                window.get(
                    "first_inning_runs_allowed_per_start"
                ),

            f"{prefix} FI {label} Game NRFI %":
                window.get(
                    "game_nrfi_pct"
                ),
        })

    return output
