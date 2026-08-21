import math

import pandas as pd
import streamlit as st

from nrfi_data_logger import (
    read_json_file,
    upsert_json_file,
)

from nrfi_pitcher_history_research import (
    WINDOWS,
    MIN_TESTABLE_GAMES,
    _eligible_rows,
    _evaluate_candidate,
)


ALERT_PATH = (
    "analytics/model_attention_alerts.json"
)

GOVERNANCE_PATH = (
    "analytics/model_governance.json"
)

LEDGER_PATH = (
    "analytics/final_pregame_edge_ledger.json"
)

HISTORY_FEATURE_SETS = [
    "Scoreless %",
    "Runs/Start",
    "Scoreless % + Runs/Start",
]

# A positive rolling-history result becomes visible as a research WATCH
# after the existing 120-game minimum.
HISTORY_WATCH_MIN_GAMES = MIN_TESTABLE_GAMES
HISTORY_WATCH_MIN_BRIER_GAIN = 0.00010

# A rolling-history feature does NOT become an actionable "review Model v2"
# alert until it has a larger forward sample.
HISTORY_REVIEW_MIN_GAMES = 250
HISTORY_REVIEW_MIN_BRIER_GAIN = 0.00010

MODEL_REVIEW_PROMPT = (
    "Review the SharpReport model attention alert and evaluate "
    "whether a Model v2 challenger should be built. Do not change "
    "the production model until the out-of-sample evidence is reviewed."
)


def _number(
    value,
):
    try:
        number = float(
            value
        )
    except Exception:
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def _load_payload(
    token,
    repo,
    path,
):
    loaded = read_json_file(
        token=token,
        repo=repo,
        path=path,
    )

    if not loaded:
        return {}

    payload = loaded.get(
        "data",
        {}
    )

    if not isinstance(
        payload,
        dict,
    ):
        return {}

    return payload


def _best_history_candidate(
    games,
    window,
):
    rows = _eligible_rows(
        games,
        window,
    )

    best = None

    for feature_set in HISTORY_FEATURE_SETS:

        result = _evaluate_candidate(
            rows,
            feature_set,
        )

        gain = _number(
            result.get(
                "candidate_gain"
            )
        )

        candidate = {
            "window":
                window,

            "candidate":
                feature_set,

            "eligible_games":
                int(
                    result.get(
                        "games",
                        0,
                    )
                    or 0
                ),

            "train_games":
                result.get(
                    "train"
                ),

            "test_games":
                result.get(
                    "test"
                ),

            "incremental_brier_gain":
                gain,

            "research_status":
                result.get(
                    "status",
                    "COLLECTING",
                ),
        }

        if gain is None:
            continue

        if (
            best is None
            or
            gain
            >
            (
                best.get(
                    "incremental_brier_gain"
                )
                or float(
                    "-inf"
                )
            )
        ):
            best = candidate

    if best is None:
        return {
            "window":
                window,

            "candidate":
                None,

            "eligible_games":
                len(
                    rows
                ),

            "train_games":
                None,

            "test_games":
                None,

            "incremental_brier_gain":
                None,

            "research_status":
                "COLLECTING",
        }

    return best


def _governance_detection(
    governance,
):
    recommendation = str(
        governance.get(
            "recommendation",
            "COLLECTING",
        )
    ).strip().upper()

    if recommendation != "REVIEW_FOR_PROMOTION":
        return None

    evaluation = governance.get(
        "walk_forward",
        {}
    ) or {}

    return {
        "id":
            "governance-review-for-promotion",

        "category":
            "MODEL_GOVERNANCE",

        "severity":
            "REVIEW_REQUIRED",

        "title":
            "Shadow challenger passed the promotion-review gates",

        "message":
            (
                "Model Governance is reporting REVIEW_FOR_PROMOTION. "
                "This means the shadow challenger passed the configured "
                "out-of-sample Brier, log-loss, calibration, and "
                "chronological-fold review gates."
            ),

        "action":
            (
                "Ask ChatGPT to review the challenger against production "
                "Model v1 and decide whether a new model version should "
                "be built. Production v1 remains unchanged."
            ),

        "condition_current":
            True,

        "metrics":
            {
                "eligible_forward_games":
                    governance.get(
                        "eligible_forward_games"
                    ),

                "evaluated_through":
                    governance.get(
                        "evaluated_through"
                    ),

                "brier_gain":
                    evaluation.get(
                        "brier_gain"
                    ),

                "logloss_gain":
                    evaluation.get(
                        "logloss_gain"
                    ),

                "ece_change":
                    evaluation.get(
                        "ece_change"
                    ),

                "fold_wins":
                    evaluation.get(
                        "fold_wins"
                    ),
            },
    }


def _history_detections_and_watches(
    games,
):
    alerts = []
    watches = []

    for window in WINDOWS:

        best = _best_history_candidate(
            games,
            window,
        )

        games_count = int(
            best.get(
                "eligible_games",
                0,
            )
            or 0
        )

        gain = _number(
            best.get(
                "incremental_brier_gain"
            )
        )

        if (
            gain is not None
            and
            games_count
            >=
            HISTORY_WATCH_MIN_GAMES
            and
            gain
            >=
            HISTORY_WATCH_MIN_BRIER_GAIN
        ):

            watch = {
                **best,

                "watch_status":
                    (
                        "REVIEW"
                        if games_count
                        >=
                        HISTORY_REVIEW_MIN_GAMES
                        else "WATCH"
                    ),
            }

            watches.append(
                watch
            )

        if (
            gain is None
            or
            games_count
            <
            HISTORY_REVIEW_MIN_GAMES
            or
            gain
            <
            HISTORY_REVIEW_MIN_BRIER_GAIN
        ):
            continue

        candidate_name = str(
            best.get(
                "candidate",
                "Unknown",
            )
        )

        safe_candidate = (
            candidate_name
            .lower()
            .replace(
                " ",
                "-"
            )
            .replace(
                "%",
                "pct"
            )
            .replace(
                "/",
                "-"
            )
            .replace(
                "+",
                "plus"
            )
        )

        alert_id = (
            "pitcher-history-"
            f"{window.lower()}-"
            f"{safe_candidate}"
        )

        alerts.append({
            "id":
                alert_id,

            "category":
                "PITCHER_HISTORY_RESEARCH",

            "severity":
                "RESEARCH_REVIEW",

            "title":
                (
                    f"{window} pitcher-history signal reached "
                    "the Model v2 review threshold"
                ),

            "message":
                (
                    f"The best {window} rolling pitcher-history "
                    f"challenger is {candidate_name} with "
                    f"{games_count} eligible forward FINAL games "
                    f"and an incremental Brier gain of "
                    f"{gain:+.6f}."
                ),

            "action":
                (
                    "Ask ChatGPT to build and test a dedicated "
                    f"Model v1 + {window} pitcher-history challenger. "
                    "Do not add this feature to production solely from "
                    "this research alert."
                ),

            "condition_current":
                True,

            "metrics":
                best,
        })

    watches.sort(
        key=lambda row:
            (
                _number(
                    row.get(
                        "incremental_brier_gain"
                    )
                )
                or float(
                    "-inf"
                )
            ),
        reverse=True,
    )

    return alerts, watches


def _merge_latched_alerts(
    existing_alerts,
    current_detections,
):
    existing_by_id = {
        str(
            alert.get(
                "id"
            )
        ):
            alert
        for alert in existing_alerts
        if isinstance(
            alert,
            dict,
        )
        and alert.get(
            "id"
        )
    }

    current_by_id = {
        str(
            alert.get(
                "id"
            )
        ):
            alert
        for alert in current_detections
        if alert.get(
            "id"
        )
    }

    merged = []

    for alert_id, detection in current_by_id.items():

        previous = existing_by_id.get(
            alert_id
        )

        acknowledged = False
        first_detected_through = None

        if previous:

            was_current = bool(
                previous.get(
                    "condition_current",
                    False,
                )
            )

            was_acknowledged = bool(
                previous.get(
                    "acknowledged",
                    False,
                )
            )

            # If the condition had cleared after acknowledgement and
            # later returns, reopen it as a fresh attention item.
            if (
                was_acknowledged
                and
                not was_current
            ):
                acknowledged = False

            else:
                acknowledged = (
                    was_acknowledged
                )

            first_detected_through = (
                previous.get(
                    "first_detected_through"
                )
            )

        metrics = detection.get(
            "metrics",
            {}
        ) or {}

        if not first_detected_through:
            first_detected_through = (
                metrics.get(
                    "evaluated_through"
                )
                or
                metrics.get(
                    "eligible_games"
                )
            )

        merged.append({
            **detection,

            "acknowledged":
                acknowledged,

            "first_detected_through":
                first_detected_through,
        })


    # Keep any unreviewed alert latched even if its triggering condition
    # is no longer present. This prevents an alert from disappearing
    # before the user sees/reviews it.
    for alert_id, previous in existing_by_id.items():

        if alert_id in current_by_id:
            continue

        if bool(
            previous.get(
                "acknowledged",
                False,
            )
        ):
            continue

        merged.append({
            **previous,

            "condition_current":
                False,
        })


    merged.sort(
        key=lambda alert:
            (
                0
                if alert.get(
                    "severity"
                )
                ==
                "REVIEW_REQUIRED"
                else 1,

                str(
                    alert.get(
                        "title",
                        "",
                    )
                ),
            )
    )

    return merged


def update_model_attention_alerts(
    token,
    repo,
):
    governance = _load_payload(
        token,
        repo,
        GOVERNANCE_PATH,
    )

    ledger = _load_payload(
        token,
        repo,
        LEDGER_PATH,
    )

    games = ledger.get(
        "games",
        []
    )

    if not isinstance(
        games,
        list,
    ):
        games = []

    existing = _load_payload(
        token,
        repo,
        ALERT_PATH,
    )

    existing_alerts = existing.get(
        "alerts",
        []
    )

    if not isinstance(
        existing_alerts,
        list,
    ):
        existing_alerts = []

    current_detections = []

    governance_alert = (
        _governance_detection(
            governance
        )
    )

    if governance_alert:
        current_detections.append(
            governance_alert
        )

    (
        history_alerts,
        history_watches,
    ) = _history_detections_and_watches(
        games
    )

    current_detections.extend(
        history_alerts
    )

    merged_alerts = _merge_latched_alerts(
        existing_alerts=
            existing_alerts,

        current_detections=
            current_detections,
    )

    unreviewed = [
        alert
        for alert in merged_alerts
        if not bool(
            alert.get(
                "acknowledged",
                False,
            )
        )
    ]

    current_unreviewed = [
        alert
        for alert in unreviewed
        if bool(
            alert.get(
                "condition_current",
                False,
            )
        )
    ]

    governance_status = str(
        governance.get(
            "recommendation",
            "COLLECTING",
        )
    )

    if unreviewed:
        overall_status = (
            "ATTENTION_REQUIRED"
        )

    elif history_watches:
        overall_status = (
            "RESEARCH_WATCH"
        )

    elif governance_status == "KEEP_V1":
        overall_status = (
            "KEEP_V1"
        )

    else:
        overall_status = (
            "COLLECTING"
        )

    payload = {
        "schema_version":
            "1.0",

        "production_model_change":
            "NEVER_AUTOMATIC",

        "overall_status":
            overall_status,

        "governance_status":
            governance_status,

        "eligible_forward_games":
            governance.get(
                "eligible_forward_games",
                0,
            ),

        "minimum_games_for_full_challenger":
            governance.get(
                "minimum_games_for_challenger_test",
                300,
            ),

        "unreviewed_alert_count":
            len(
                unreviewed
            ),

        "current_unreviewed_alert_count":
            len(
                current_unreviewed
            ),

        "alerts":
            merged_alerts,

        "research_watches":
            history_watches,

        "thresholds":
            {
                "pitcher_history_watch_min_games":
                    HISTORY_WATCH_MIN_GAMES,

                "pitcher_history_watch_min_brier_gain":
                    HISTORY_WATCH_MIN_BRIER_GAIN,

                "pitcher_history_review_min_games":
                    HISTORY_REVIEW_MIN_GAMES,

                "pitcher_history_review_min_brier_gain":
                    HISTORY_REVIEW_MIN_BRIER_GAIN,
            },

        "what_to_tell_chatgpt":
            MODEL_REVIEW_PROMPT,
    }

    write_result = upsert_json_file(
        token=token,
        repo=repo,
        path=ALERT_PATH,
        payload=payload,
        commit_message=(
            "Update NRFI model attention alerts"
        ),
    )

    return {
        **payload,

        "alert_path":
            ALERT_PATH,

        "alerts_updated":
            bool(
                write_result.get(
                    "changed"
                )
            ),
    }


def acknowledge_model_attention_alerts(
    token,
    repo,
):
    current = _load_payload(
        token,
        repo,
        ALERT_PATH,
    )

    alerts = current.get(
        "alerts",
        []
    )

    if not isinstance(
        alerts,
        list,
    ):
        alerts = []

    changed = False

    for alert in alerts:

        if not isinstance(
            alert,
            dict,
        ):
            continue

        if not bool(
            alert.get(
                "acknowledged",
                False,
            )
        ):
            alert[
                "acknowledged"
            ] = True

            changed = True

    if not changed:
        return {
            **current,

            "alerts_updated":
                False,
        }

    current[
        "alerts"
    ] = alerts

    current[
        "unreviewed_alert_count"
    ] = 0

    current[
        "current_unreviewed_alert_count"
    ] = 0

    if current.get(
        "research_watches"
    ):
        current[
            "overall_status"
        ] = "RESEARCH_WATCH"

    elif current.get(
        "governance_status"
    ) == "KEEP_V1":
        current[
            "overall_status"
        ] = "KEEP_V1"

    else:
        current[
            "overall_status"
        ] = "COLLECTING"

    write_result = upsert_json_file(
        token=token,
        repo=repo,
        path=ALERT_PATH,
        payload=current,
        commit_message=(
            "Acknowledge NRFI model attention alerts"
        ),
    )

    return {
        **current,

        "alerts_updated":
            bool(
                write_result.get(
                    "changed"
                )
            ),
    }


def _read_alert_data(
    token,
    repo,
):
    return _load_payload(
        token,
        repo,
        ALERT_PATH,
    )


def render_model_attention_banner(
    token,
    repo,
):
    data = _read_alert_data(
        token,
        repo,
    )

    if not data:
        return

    alerts = data.get(
        "alerts",
        []
    )

    if not isinstance(
        alerts,
        list,
    ):
        alerts = []

    unreviewed = [
        alert
        for alert in alerts
        if not bool(
            alert.get(
                "acknowledged",
                False,
            )
        )
    ]

    if not unreviewed:
        return

    review_required = any(
        alert.get(
            "severity"
        )
        ==
        "REVIEW_REQUIRED"
        for alert in unreviewed
    )

    if review_required:

        st.error(
            "⚠ SHARPREPORT MODEL ATTENTION REQUIRED — "
            "A challenger has reached a formal review condition. "
            "Production Model v1 has NOT changed."
        )

    else:

        st.warning(
            "⚠ SHARPREPORT MODEL RESEARCH ATTENTION — "
            "A rolling pitcher-history candidate has reached the "
            "configured Model v2 review threshold. "
            "Production Model v1 has NOT changed."
        )


    for alert in unreviewed:

        title = str(
            alert.get(
                "title",
                "Model attention item",
            )
        )

        message = str(
            alert.get(
                "message",
                "",
            )
        )

        action = str(
            alert.get(
                "action",
                "",
            )
        )

        condition_note = ""

        if not bool(
            alert.get(
                "condition_current",
                False,
            )
        ):
            condition_note = (
                " The triggering condition is no longer current, "
                "but this alert remains latched until reviewed."
            )

        st.markdown(
            f"**{title}**"
        )

        st.caption(
            message
            +
            condition_note
        )

        if action:
            st.caption(
                "Next action: "
                + action
            )


    st.caption(
        "What to tell ChatGPT: "
        + str(
            data.get(
                "what_to_tell_chatgpt",
                MODEL_REVIEW_PROMPT,
            )
        )
    )


def render_model_attention_dashboard(
    token,
    repo,
):
    st.subheader(
        "Model Attention Alerts"
    )

    st.caption(
        "This is the plain-English alert layer for Model v1. "
        "It never changes production automatically. Alerts stay "
        "latched until they are explicitly marked reviewed."
    )

    data = _read_alert_data(
        token,
        repo,
    )

    if not data:
        st.info(
            "The alert monitor has not been initialized yet. "
            "Run the scanner or allow the scheduled collector to run."
        )

        return

    status_rows = [{
        "Overall Status":
            data.get(
                "overall_status",
                "COLLECTING",
            ),

        "Governance":
            data.get(
                "governance_status",
                "COLLECTING",
            ),

        "Forward FINAL Games":
            data.get(
                "eligible_forward_games",
                0,
            ),

        "Full Challenger Minimum":
            data.get(
                "minimum_games_for_full_challenger",
                300,
            ),

        "Unreviewed Alerts":
            data.get(
                "unreviewed_alert_count",
                0,
            ),
    }]

    st.dataframe(
        pd.DataFrame(
            status_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    alerts = data.get(
        "alerts",
        []
    )

    if not isinstance(
        alerts,
        list,
    ):
        alerts = []

    unreviewed = [
        alert
        for alert in alerts
        if not bool(
            alert.get(
                "acknowledged",
                False,
            )
        )
    ]

    if unreviewed:

        alert_rows = []

        for alert in unreviewed:
            alert_rows.append({
                "Severity":
                    alert.get(
                        "severity"
                    ),

                "Alert":
                    alert.get(
                        "title"
                    ),

                "Current":
                    (
                        "YES"
                        if alert.get(
                            "condition_current"
                        )
                        else "LATCHED"
                    ),

                "What To Do":
                    alert.get(
                        "action"
                    ),
            })

        st.markdown(
            "**Unreviewed Attention Items**"
        )

        st.dataframe(
            pd.DataFrame(
                alert_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "After you have reviewed these alerts with ChatGPT, "
            "you can mark them reviewed below. This does not alter "
            "Model v1 or promote any challenger."
        )

        if st.button(
            "Mark current model alert(s) reviewed",
            key=
                "acknowledge_nrfi_model_alerts",
        ):

            try:

                acknowledge_model_attention_alerts(
                    token=token,
                    repo=repo,
                )

                st.success(
                    "Model attention alert(s) marked reviewed. "
                    "Production Model v1 was not changed."
                )

            except Exception as error:

                st.warning(
                    "The alert could not be marked reviewed: "
                    f"{error}"
                )

    else:

        st.success(
            "No unreviewed model-attention alert is active."
        )


    watches = data.get(
        "research_watches",
        []
    )

    if not isinstance(
        watches,
        list,
    ):
        watches = []

    st.markdown(
        "**Rolling Pitcher-History Research Watch**"
    )

    if watches:

        rows = []

        for row in watches:
            gain = _number(
                row.get(
                    "incremental_brier_gain"
                )
            )

            rows.append({
                "Window":
                    row.get(
                        "window"
                    ),

                "Best Candidate":
                    row.get(
                        "candidate"
                    ),

                "Eligible Games":
                    row.get(
                        "eligible_games"
                    ),

                "Incremental Brier Gain":
                    (
                        f"{gain:+.6f}"
                        if gain is not None
                        else "—"
                    ),

                "Status":
                    row.get(
                        "watch_status"
                    ),
            })

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No L10/L20/L30/Season pitcher-history candidate "
            "has reached the research-watch threshold."
        )


    thresholds = data.get(
        "thresholds",
        {}
    )

    st.caption(
        "Research WATCH: "
        f"{thresholds.get('pitcher_history_watch_min_games', 120)}+ "
        "eligible FINAL games and Incremental Brier Gain >= "
        f"{thresholds.get('pitcher_history_watch_min_brier_gain', 0.00010):.5f}. "
        "Actionable pitcher-history REVIEW: "
        f"{thresholds.get('pitcher_history_review_min_games', 250)}+ "
        "eligible FINAL games with the same minimum gain. "
        "Formal full-model promotion review still comes from the "
        "separate 300-game Model Governance system."
    )

    st.info(
        "If the status becomes ATTENTION_REQUIRED, tell ChatGPT: "
        + str(
            data.get(
                "what_to_tell_chatgpt",
                MODEL_REVIEW_PROMPT,
            )
        )
    )
