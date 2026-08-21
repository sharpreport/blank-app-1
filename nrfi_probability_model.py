
import json
import math
from pathlib import Path


# =========================================================
# SHARPREPORT LIVE NRFI / YRFI PROBABILITY ENGINE
#
# Loads the trained:
#   sharpreport_nrfi_model_v1.json
#
# Produces:
#   P(Top 1st scores)
#   P(Bottom 1st scores)
#   P(NRFI)
#   P(YRFI)
#
# IMPORTANT:
# - K%, BB%, Barrel% are expected in percentage-point form:
#     22.5 means 22.5%, NOT 0.225.
# - xwOBA is expected as a decimal:
#     .320 means .320.
# - Run Factor uses 100 as neutral.
# - Missing model values are imputed with the exact training
#   mean stored in the trained JSON.
# =========================================================


DEFAULT_MODEL_PATH = Path(
    "sharpreport_nrfi_model_v1.json"
)


def _sigmoid(value):
    value = max(
        min(float(value), 30.0),
        -30.0,
    )

    return (
        1.0
        / (
            1.0
            + math.exp(-value)
        )
    )


def _clean_number(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def load_nrfi_model(
    model_path=DEFAULT_MODEL_PATH,
):
    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: "
            f"{model_path}"
        )

    with model_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        model = json.load(
            handle
        )

    required_top = (
        model
        .get("models", {})
        .get("Top")
    )

    required_bottom = (
        model
        .get("models", {})
        .get("Bottom")
    )

    if (
        required_top is None
        or required_bottom is None
    ):
        raise ValueError(
            "Model JSON must contain "
            "Top and Bottom models."
        )

    return model


def build_half_features(
    *,
    pitcher_xwoba=None,
    pitcher_k_pct=None,
    pitcher_pa=None,
    offense_xwoba=None,
    offense_k_pct=None,
    offense_bb_pct=None,
    offense_barrel_pct=None,
    offense_combined_pa=None,
    offense_min_pa=None,
    offense_complete_core=None,
    park_runs=None,
):
    """
    Convert LIVE raw inputs into the exact feature names used
    during historical model training.

    Percent metrics use percentage points:
      22.5 = 22.5%
    """

    p_pa = _clean_number(
        pitcher_pa
    )

    if p_pa is None:
        p_pa = 0.0

    p_pa = max(
        p_pa,
        0.0,
    )

    combined_pa = _clean_number(
        offense_combined_pa
    )

    if combined_pa is None:
        combined_pa = 0.0

    combined_pa = max(
        combined_pa,
        0.0,
    )

    min_pa = _clean_number(
        offense_min_pa
    )

    if min_pa is None:
        min_pa = 0.0

    min_pa = max(
        min_pa,
        0.0,
    )

    complete_core = (
        bool(
            offense_complete_core
        )
        if offense_complete_core is not None
        else False
    )

    park_value = _clean_number(
        park_runs
    )

    if park_value is None:
        park_value = 100.0
        park_missing = 1.0
    else:
        park_missing = 0.0

    return {
        "p_xwoba":
            _clean_number(
                pitcher_xwoba
            ),

        "p_k_pct":
            _clean_number(
                pitcher_k_pct
            ),

        "o_xwoba":
            _clean_number(
                offense_xwoba
            ),

        "o_k_pct":
            _clean_number(
                offense_k_pct
            ),

        "o_bb_pct":
            _clean_number(
                offense_bb_pct
            ),

        "o_barrel_pct":
            _clean_number(
                offense_barrel_pct
            ),

        "p_log_pa":
            math.log1p(
                p_pa
            ),

        "p_no_prior_pa":
            1.0
            if p_pa <= 0
            else 0.0,

        "o_log_combined_pa":
            math.log1p(
                combined_pa
            ),

        "o_log_min_pa":
            math.log1p(
                min_pa
            ),

        "o_missing_core":
            0.0
            if complete_core
            else 1.0,

        "park_runs":
            park_value,

        "park_missing":
            park_missing,
    }


def predict_half_scoring_probability(
    model,
    half,
    features,
):
    """
    half must be:
      "Top"
      "Bottom"
    """

    if half not in (
        "Top",
        "Bottom",
    ):
        raise ValueError(
            "half must be "
            "'Top' or 'Bottom'"
        )

    half_model = (
        model[
            "models"
        ][half]
    )

    intercept = float(
        half_model[
            "intercept"
        ]
    )

    coefficients = (
        half_model[
            "coefficients"
        ]
    )

    means = (
        half_model[
            "means"
        ]
    )

    stds = (
        half_model[
            "stds"
        ]
    )

    linear_score = (
        intercept
    )

    standardized = {}

    for feature_name in model[
        "features"
    ]:
        raw_value = _clean_number(
            features.get(
                feature_name
            )
        )

        mean_value = float(
            means[
                feature_name
            ]
        )

        std_value = float(
            stds[
                feature_name
            ]
        )

        if raw_value is None:
            raw_value = (
                mean_value
            )

        if (
            not math.isfinite(
                std_value
            )
            or abs(
                std_value
            ) < 1e-12
        ):
            std_value = 1.0

        z_value = (
            raw_value
            - mean_value
        ) / std_value

        standardized[
            feature_name
        ] = z_value

        linear_score += (
            float(
                coefficients[
                    feature_name
                ]
            )
            * z_value
        )

    probability = _sigmoid(
        linear_score
    )

    return {
        "half":
            half,

        "scoring_probability":
            probability,

        "scoreless_probability":
            1.0 - probability,

        "linear_score":
            linear_score,

        "standardized_features":
            standardized,
    }


def predict_nrfi_yrfi(
    model,
    *,
    top_features,
    bottom_features,
):
    top_result = (
        predict_half_scoring_probability(
            model,
            "Top",
            top_features,
        )
    )

    bottom_result = (
        predict_half_scoring_probability(
            model,
            "Bottom",
            bottom_features,
        )
    )

    p_top_score = (
        top_result[
            "scoring_probability"
        ]
    )

    p_bottom_score = (
        bottom_result[
            "scoring_probability"
        ]
    )

    nrfi_probability = (
        (1.0 - p_top_score)
        * (1.0 - p_bottom_score)
    )

    yrfi_probability = (
        1.0
        - nrfi_probability
    )

    model_side = (
        "NRFI"
        if nrfi_probability
        >= yrfi_probability
        else "YRFI"
    )

    model_probability = max(
        nrfi_probability,
        yrfi_probability,
    )

    return {
        "top_scoring_probability":
            p_top_score,

        "bottom_scoring_probability":
            p_bottom_score,

        "nrfi_probability":
            nrfi_probability,

        "yrfi_probability":
            yrfi_probability,

        "model_side":
            model_side,

        "model_probability":
            model_probability,

        "top_detail":
            top_result,

        "bottom_detail":
            bottom_result,
    }


def american_odds_implied_probability(
    american_odds,
):
    odds = _clean_number(
        american_odds
    )

    if odds is None or odds == 0:
        return None

    if odds < 0:
        return (
            abs(odds)
            / (
                abs(odds)
                + 100.0
            )
        )

    return (
        100.0
        / (
            odds
            + 100.0
        )
    )


def market_edge(
    model_probability,
    american_odds,
):
    """
    Raw market edge versus the quoted side's implied
    probability.

    Positive = model probability is greater than the
    market-implied probability.

    This is NOT a no-vig calculation. A future market module
    can calculate no-vig NRFI/YRFI probabilities when both
    sides' prices are available.
    """

    model_p = _clean_number(
        model_probability
    )

    implied = (
        american_odds_implied_probability(
            american_odds
        )
    )

    if (
        model_p is None
        or implied is None
    ):
        return None

    return {
        "model_probability":
            model_p,

        "market_implied_probability":
            implied,

        "edge_probability":
            model_p - implied,

        "edge_percentage_points":
            (
                model_p
                - implied
            ) * 100.0,
    }


def _mean_case_features(
    half_model,
):
    """
    Pure smoke-test row:
    every feature equals its training mean.

    Therefore all z-scores should be zero and predicted
    probability should equal sigmoid(intercept).
    """

    return {
        feature:
            float(
                value
            )
        for feature, value in (
            half_model[
                "means"
            ].items()
        )
    }


def run_self_test(
    model_path=DEFAULT_MODEL_PATH,
):
    model = load_nrfi_model(
        model_path
    )

    top_model = (
        model[
            "models"
        ]["Top"]
    )

    bottom_model = (
        model[
            "models"
        ]["Bottom"]
    )

    top_features = (
        _mean_case_features(
            top_model
        )
    )

    bottom_features = (
        _mean_case_features(
            bottom_model
        )
    )

    top = (
        predict_half_scoring_probability(
            model,
            "Top",
            top_features,
        )
    )

    bottom = (
        predict_half_scoring_probability(
            model,
            "Bottom",
            bottom_features,
        )
    )

    nrfi = (
        (
            1.0
            - top[
                "scoring_probability"
            ]
        )
        *
        (
            1.0
            - bottom[
                "scoring_probability"
            ]
        )
    )

    print()
    print(
        "SHARPREPORT NRFI MODEL SELF-TEST"
    )
    print(
        "=" * 56
    )
    print(
        f"Model: "
        f"{model.get('model_name')}"
    )
    print(
        f"Training games: "
        f"{model.get('training_games')}"
    )
    print(
        f"Training through: "
        f"{model.get('training_end_date')}"
    )
    print()

    print(
        "Mean-feature sanity check:"
    )

    print(
        f"Top 1st scores: "
        f"{top['scoring_probability'] * 100:.2f}%"
    )

    print(
        f"Bottom 1st scores: "
        f"{bottom['scoring_probability'] * 100:.2f}%"
    )

    print(
        f"Game NRFI: "
        f"{nrfi * 100:.2f}%"
    )

    print(
        f"Game YRFI: "
        f"{(1.0 - nrfi) * 100:.2f}%"
    )
    print()

    print(
        "Expected mean-feature check:"
    )
    print(
        "Top probability should equal "
        "sigmoid(Top intercept)."
    )
    print(
        "Bottom probability should equal "
        "sigmoid(Bottom intercept)."
    )

    expected_top = _sigmoid(
        top_model[
            "intercept"
        ]
    )

    expected_bottom = _sigmoid(
        bottom_model[
            "intercept"
        ]
    )

    top_ok = abs(
        expected_top
        - top[
            "scoring_probability"
        ]
    ) < 1e-12

    bottom_ok = abs(
        expected_bottom
        - bottom[
            "scoring_probability"
        ]
    ) < 1e-12

    if top_ok and bottom_ok:
        print()
        print(
            "PASS: model JSON and inference "
            "math are aligned."
        )
    else:
        raise RuntimeError(
            "SELF-TEST FAILED: "
            "inference math is not aligned "
            "with the trained JSON."
        )


if __name__ == "__main__":
    run_self_test()
