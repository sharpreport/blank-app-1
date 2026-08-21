
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


YEARS = [2022, 2023, 2024, 2025]
MIN_PA = 100

TOP4_FILE = Path("historical_top4_skill_context.csv")
SUMMARY_FILE = Path("historical_top4_metric_validation_summary.txt")
DETAIL_FILE = Path("historical_top4_metric_validation_details.csv")

XWOBA_TOLERANCE = 0.020
K_TOLERANCE_PP = 1.50
BB_TOLERANCE_PP = 1.50
BARREL_TOLERANCE_PP = 1.50

MAX_WORKERS = 4


def get_text(url, timeout=60):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Request failed after 3 attempts: {last_error}")


def get_json(url):
    return json.loads(get_text(url))


def get_csv(url):
    return pd.read_csv(io.StringIO(get_text(url)))


coverage_lines = []

if TOP4_FILE.exists():
    top4 = pd.read_csv(TOP4_FILE)
    total_sides = len(top4) * 2

    prior_all4 = 0
    core_all4 = 0
    prior_but_missing_core = 0
    missing_due_zero_bbe = 0

    for side in ["away", "home"]:
        prior_cols = [f"{side}_b{i}_pa_before_game" for i in range(1, 5)]
        bbe_cols = [f"{side}_b{i}_bbe_before_game" for i in range(1, 5)]

        metric_cols = []
        for i in range(1, 5):
            metric_cols.extend([
                f"{side}_b{i}_xwoba",
                f"{side}_b{i}_k_pct",
                f"{side}_b{i}_bb_pct",
                f"{side}_b{i}_barrel_pct",
            ])

        prior_matrix = top4[prior_cols].apply(pd.to_numeric, errors="coerce")
        bbe_matrix = top4[bbe_cols].apply(pd.to_numeric, errors="coerce")
        metric_matrix = top4[metric_cols].apply(pd.to_numeric, errors="coerce")

        all_prior_mask = (prior_matrix > 0).all(axis=1)
        all_core_mask = metric_matrix.notna().all(axis=1)

        missing_core_mask = all_prior_mask & ~all_core_mask

        zero_bbe_mask = (
            missing_core_mask
            & (((bbe_matrix <= 0) | bbe_matrix.isna()).any(axis=1))
        )

        prior_all4 += int(all_prior_mask.sum())
        core_all4 += int(all_core_mask.sum())
        prior_but_missing_core += int(missing_core_mask.sum())
        missing_due_zero_bbe += int(zero_bbe_mask.sum())

    coverage_lines = [
        "TOP-4 COVERAGE AUDIT",
        "-" * 72,
        f"Team Top-4 sides: {total_sides}",
        f"All 4 hitters with prior PA: {prior_all4}",
        f"All 4 individual core metrics available: {core_all4}",
        f"All 4 with prior PA but incomplete core metrics: {prior_but_missing_core}",
        f"Those explained by at least one hitter having zero prior BBE: {missing_due_zero_bbe}",
        "",
    ]
else:
    coverage_lines = [
        "TOP-4 COVERAGE AUDIT",
        "-" * 72,
        "historical_top4_skill_context.csv was not found.",
        "",
    ]


def reconstructed_season(year):
    path = Path(f"statcast_daily_batter_{year}.csv")
    if not path.exists():
        raise SystemExit(f"ERROR: {path} was not found.")

    daily = pd.read_csv(path)

    numeric = [
        "batter",
        "pa",
        "strikeouts",
        "walks",
        "xwoba_num",
        "xwoba_denom",
        "bbe",
        "barrels",
    ]

    for column in numeric:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0)

    grouped = (
        daily
        .groupby("batter", as_index=False)[
            ["pa", "strikeouts", "walks", "xwoba_num", "xwoba_denom", "bbe", "barrels"]
        ]
        .sum()
    )

    grouped["batter"] = grouped["batter"].astype(int)
    grouped["our_xwoba"] = grouped["xwoba_num"] / grouped["xwoba_denom"].replace(0, pd.NA)
    grouped["our_k_pct"] = grouped["strikeouts"] / grouped["pa"].replace(0, pd.NA) * 100
    grouped["our_bb_pct"] = grouped["walks"] / grouped["pa"].replace(0, pd.NA) * 100
    grouped["our_barrel_pct"] = grouped["barrels"] / grouped["bbe"].replace(0, pd.NA) * 100

    return grouped


def find_id_column(df):
    for candidate in ["player_id", "playerid", "id"]:
        if candidate in df.columns:
            return candidate
    return None


def savant_expected(year):
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=batter&year={year}&position=&team=&filterType=pa&min=1&csv=true"
    )
    df = get_csv(url)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = find_id_column(df)
    if id_col is None:
        raise RuntimeError(f"Could not find player ID in Savant expected stats for {year}.")
    if "est_woba" not in df.columns:
        raise RuntimeError(f"Could not find est_woba in Savant expected stats for {year}.")

    result = pd.DataFrame({
        "batter": pd.to_numeric(df[id_col], errors="coerce"),
        "savant_xwoba": pd.to_numeric(df["est_woba"], errors="coerce"),
    }).dropna(subset=["batter"])

    result["batter"] = result["batter"].astype(int)
    return result


def savant_barrels(year):
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&year={year}&position=&team=&min=1&csv=true"
    )
    df = get_csv(url)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = find_id_column(df)
    if id_col is None:
        raise RuntimeError(f"Could not find player ID in Savant barrel stats for {year}.")
    if "brl_percent" not in df.columns:
        raise RuntimeError(f"Could not find brl_percent in Savant barrel stats for {year}.")

    result = pd.DataFrame({
        "batter": pd.to_numeric(df[id_col], errors="coerce"),
        "savant_barrel_pct": pd.to_numeric(df["brl_percent"], errors="coerce"),
    }).dropna(subset=["batter"])

    result["batter"] = result["batter"].astype(int)
    return result


def mlb_hitting_rates(args):
    batter_id, year = args
    url = (
        "https://statsapi.mlb.com/"
        f"api/v1/people/{batter_id}/stats"
        "?stats=season"
        "&group=hitting"
        f"&season={year}"
    )

    try:
        data = get_json(url)
        stats = data.get("stats", [{}])[0].get("splits", [])
        if not stats:
            return batter_id, year, None, None, None

        stat = stats[0].get("stat", {})
        pa = float(stat.get("plateAppearances"))
        so = float(stat.get("strikeOuts"))
        bb = float(stat.get("baseOnBalls"))

        if pa <= 0:
            return batter_id, year, None, None, None

        return batter_id, year, pa, so / pa * 100, bb / pa * 100
    except Exception:
        return batter_id, year, None, None, None


detail_rows = []

for year in YEARS:
    print()
    print(f"VALIDATING BATTERS {year}")
    print("-" * 60)

    ours = reconstructed_season(year)
    expected = savant_expected(year)
    barrels = savant_barrels(year)

    merged = (
        ours
        .merge(expected, on="batter", how="left")
        .merge(barrels, on="batter", how="left")
    )

    merged = merged[merged["pa"] >= MIN_PA].copy()

    ids = [(int(pid), year) for pid in merged["batter"]]
    mlb_rates = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(mlb_hitting_rates, item): item
            for item in ids
        }

        completed = 0
        for future in as_completed(future_map):
            batter_id, result_year, mlb_pa, mlb_k_pct, mlb_bb_pct = future.result()
            mlb_rates[batter_id] = (mlb_pa, mlb_k_pct, mlb_bb_pct)

            completed += 1
            if completed % 100 == 0:
                print(f"  MLB rate checks: {completed}/{len(ids)}")

    for _, row in merged.iterrows():
        batter_id = int(row["batter"])
        mlb_pa, mlb_k_pct, mlb_bb_pct = mlb_rates.get(
            batter_id,
            (None, None, None),
        )

        our_xwoba = row["our_xwoba"]
        savant_xwoba = row["savant_xwoba"]
        our_barrel = row["our_barrel_pct"]
        savant_barrel = row["savant_barrel_pct"]

        xwoba_diff = (
            abs(our_xwoba - savant_xwoba)
            if pd.notna(our_xwoba) and pd.notna(savant_xwoba)
            else None
        )

        k_diff = (
            abs(row["our_k_pct"] - mlb_k_pct)
            if mlb_k_pct is not None
            else None
        )

        bb_diff = (
            abs(row["our_bb_pct"] - mlb_bb_pct)
            if mlb_bb_pct is not None
            else None
        )

        barrel_diff = (
            abs(our_barrel - savant_barrel)
            if pd.notna(our_barrel) and pd.notna(savant_barrel)
            else None
        )

        detail_rows.append({
            "year": year,
            "batter_id": batter_id,
            "our_pa": row["pa"],
            "mlb_pa": mlb_pa,
            "our_xwoba": round(float(our_xwoba), 4) if pd.notna(our_xwoba) else None,
            "savant_xwoba": round(float(savant_xwoba), 4) if pd.notna(savant_xwoba) else None,
            "xwoba_abs_diff": round(float(xwoba_diff), 4) if xwoba_diff is not None else None,
            "our_k_pct": round(float(row["our_k_pct"]), 2),
            "mlb_k_pct": round(float(mlb_k_pct), 2) if mlb_k_pct is not None else None,
            "k_abs_diff_pp": round(float(k_diff), 2) if k_diff is not None else None,
            "our_bb_pct": round(float(row["our_bb_pct"]), 2),
            "mlb_bb_pct": round(float(mlb_bb_pct), 2) if mlb_bb_pct is not None else None,
            "bb_abs_diff_pp": round(float(bb_diff), 2) if bb_diff is not None else None,
            "our_barrel_pct": round(float(our_barrel), 2) if pd.notna(our_barrel) else None,
            "savant_barrel_pct": round(float(savant_barrel), 2) if pd.notna(savant_barrel) else None,
            "barrel_abs_diff_pp": round(float(barrel_diff), 2) if barrel_diff is not None else None,
        })


details = pd.DataFrame(detail_rows)
details.to_csv(DETAIL_FILE, index=False)


lines = [
    "SHARPREPORT HISTORICAL TOP-4 HITTER METRIC VALIDATION",
    "=" * 72,
    "",
]

lines.extend(coverage_lines)

lines.extend([
    f"Completed seasons checked: {', '.join(str(y) for y in YEARS)}",
    f"Minimum reconstructed PA per hitter: {MIN_PA}",
    "",
])


def metric_summary(label, diff_column, tolerance):
    valid = details[diff_column].dropna()

    if valid.empty:
        lines.extend([
            label,
            "No comparable rows.",
            "-" * 72,
        ])
        return

    within = int((valid <= tolerance).sum())

    lines.extend([
        label,
        f"Comparable hitter-seasons: {len(valid)}",
        f"Mean absolute difference: {valid.mean():.4f}",
        f"Median absolute difference: {valid.median():.4f}",
        f"90th percentile difference: {valid.quantile(0.90):.4f}",
        f"Within tolerance: {within} of {len(valid)} ({within / len(valid) * 100:.2f}%)",
        f"Tolerance: {tolerance}",
        "-" * 72,
    ])


metric_summary("xwOBA validation", "xwoba_abs_diff", XWOBA_TOLERANCE)
metric_summary("K% validation", "k_abs_diff_pp", K_TOLERANCE_PP)
metric_summary("BB% validation", "bb_abs_diff_pp", BB_TOLERANCE_PP)
metric_summary("Barrel% validation", "barrel_abs_diff_pp", BARREL_TOLERANCE_PP)

lines.extend([
    "",
    "INTERPRETATION:",
    (
        "Small differences are expected because detailed Statcast reconstruction "
        "and leaderboard / MLB denominator definitions can differ slightly."
    ),
    (
        "If large systematic differences appear, do not run the offensive "
        "predictive model until corrected."
    ),
])

SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

print()
print("TOP-4 HITTER METRIC VALIDATION COMPLETE")
print()
print(f"Created: {SUMMARY_FILE}")
print(f"Created: {DETAIL_FILE}")
