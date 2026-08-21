import base64
import json
import math
import re
import uuid

from datetime import date, datetime, timezone
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


GITHUB_API_ROOT = "https://api.github.com"
SCHEMA_VERSION = "1.0"


def _json_safe(value):
    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    # Handles numpy/pandas scalar types without importing them.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item_value)
            for key, item_value in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item_value)
            for item_value in value
        ]

    return str(value)


def build_slate_snapshot(
    rows,
    snapshot_time,
    model_metadata=None,
    odds_usage=None,
):
    if snapshot_time.tzinfo is None:
        raise ValueError(
            "snapshot_time must be timezone-aware."
        )

    game_date = None

    for row in rows:
        candidate = row.get("Game Date")

        if not candidate:
            continue

        candidate_text = str(
            candidate
        ).strip()

        # MLB schedule data may provide a full ISO timestamp
        # such as 2026-08-21T20:10:00Z. The repository folder
        # should be the baseball calendar date only.
        match = re.match(
            r"^(\d{4}-\d{2}-\d{2})",
            candidate_text,
        )

        if match:
            game_date = match.group(1)
            break

    if game_date is None:
        game_date = snapshot_time.date().isoformat()

    utc_time = snapshot_time.astimezone(
        timezone.utc
    )

    return _json_safe({
        "schema_version":
            SCHEMA_VERSION,

        "snapshot_type":
            "pregame_slate",

        "snapshot_time_et":
            snapshot_time.isoformat(),

        "snapshot_time_utc":
            utc_time.isoformat(),

        "game_date":
            game_date,

        "model":
            model_metadata or {},

        "odds_api":
            odds_usage or {},

        "game_count":
            len(rows),

        "games":
            rows,
    })


def _github_contents_url(
    repo,
    path,
):
    repo = str(repo).strip()

    if "/" not in repo:
        raise ValueError(
            "GITHUB_DATA_REPO must look like owner/repository."
        )

    encoded_repo = quote(
        repo,
        safe="/"
    )

    encoded_path = quote(
        path,
        safe="/"
    )

    return (
        f"{GITHUB_API_ROOT}/repos/"
        f"{encoded_repo}/contents/{encoded_path}"
    )


def write_json_file(
    token,
    repo,
    path,
    payload,
    commit_message,
):
    token = str(token).strip()
    repo = str(repo).strip()

    if not token:
        raise ValueError(
            "GitHub data token is empty."
        )

    if not repo:
        raise ValueError(
            "GitHub data repository is empty."
        )

    json_text = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )

    encoded_content = base64.b64encode(
        json_text.encode("utf-8")
    ).decode("ascii")

    body = json.dumps({
        "message":
            commit_message,

        "content":
            encoded_content,
    }).encode("utf-8")

    request = Request(
        _github_contents_url(
            repo,
            path,
        ),
        data=body,
        method="PUT",
        headers={
            "Accept":
                "application/vnd.github+json",

            "Authorization":
                f"Bearer {token}",

            "X-GitHub-Api-Version":
                "2022-11-28",

            "User-Agent":
                "SharpReport-NRFI-Data-Logger",

            "Content-Type":
                "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:

            response_payload = json.load(
                response
            )

            return {
                "ok":
                    True,

                "status":
                    response.status,

                "path":
                    path,

                "commit_sha":
                    (
                        response_payload
                        .get("commit", {})
                        .get("sha")
                    ),
            }

    except HTTPError as error:
        try:
            error_payload = json.loads(
                error.read().decode(
                    "utf-8"
                )
            )

            message = error_payload.get(
                "message",
                str(error),
            )

        except Exception:
            message = str(error)

        raise RuntimeError(
            f"GitHub data write failed "
            f"(HTTP {error.code}): {message}"
        ) from error


def save_slate_snapshot(
    token,
    repo,
    rows,
    snapshot_time,
    model_metadata=None,
    odds_usage=None,
):
    snapshot = build_slate_snapshot(
        rows=rows,
        snapshot_time=snapshot_time,
        model_metadata=model_metadata,
        odds_usage=odds_usage,
    )

    day = snapshot[
        "game_date"
    ]

    timestamp = snapshot_time.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    unique_suffix = uuid.uuid4().hex[:8]

    path = (
        f"snapshots/{day}/"
        f"{timestamp}_{unique_suffix}.json"
    )

    return write_json_file(
        token=token,
        repo=repo,
        path=path,
        payload=snapshot,
        commit_message=(
            f"Log NRFI slate snapshot "
            f"{snapshot_time.isoformat()}"
        ),
    )
