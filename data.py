"""Synthetic 12-week ramp dataset and schema validation.

One row per node per week. Everything downstream (capacity, utilization,
shortage, scenarios) is derived from these raw columns, so an uploaded CSV
with the same schema can replace the synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEKS = 12
HOURS_PER_SHIFT = 8
MAX_SHIFTS_PER_WEEK = 21  # 3 shifts x 7 days

DOWNTIME_COLS = {
    "dt_planned": "Planned maintenance",
    "dt_unplanned": "Unplanned equipment",
    "dt_material": "Material shortage",
    "dt_quality": "Quality hold",
}

REQUIRED_COLUMNS = [
    "week", "node", "site", "base_shifts", "rate_uph", "ramp_factor",
    "yield_pct", "demand", *DOWNTIME_COLS.keys(),
]


@dataclass(frozen=True)
class NodeSpec:
    node: str
    site: str
    nameplate: int            # units/week at full ramp, 100% yield, zero downtime
    base_shifts: int          # scheduled 8h shifts per week
    ramp_start: float         # fraction of nameplate rate achievable in week 1
    ramp_weeks: int           # weeks to reach full rate
    yield_start: float
    yield_exit: float
    unplanned_dt_mean: float  # mean unplanned downtime hours/week at full maturity
    demand_share: float
    pm_weeks: tuple = ()
    incidents: dict = field(default_factory=dict)  # {(week, col): extra hours}


NODES: list[NodeSpec] = [
    NodeSpec("AUS-L1", "Austin, US", 4000, 10, 0.70, 6, 0.93, 0.97, 4.0, 0.28,
             pm_weeks=(4, 8, 12)),
    NodeSpec("PEN-L2", "Penang, MY", 4500, 10, 0.55, 8, 0.90, 0.96, 5.0, 0.29,
             pm_weeks=(3, 7, 11), incidents={(9, "dt_material"): 12.0}),
    NodeSpec("GDL-L1", "Guadalajara, MX", 3200, 10, 0.40, 10, 0.85, 0.94, 6.0, 0.23,
             pm_weeks=(5, 10), incidents={(6, "dt_unplanned"): 22.0, (7, "dt_unplanned"): 10.0}),
    NodeSpec("SZH-L3", "Suzhou, CN", 3000, 15, 0.80, 4, 0.95, 0.98, 3.0, 0.20,
             pm_weeks=(2, 6, 10)),
]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ramp_curve(start: float, weeks_to_full: int, n_weeks: int = WEEKS) -> np.ndarray:
    w = np.arange(n_weeks)
    progress = w / max(weeks_to_full - 1, 1)
    return start + (1 - start) * _smoothstep(progress)


def generate_data(seed: int = 7, demand_volatility: float = 0.04,
                  downtime_severity: float = 1.0) -> pd.DataFrame:
    """Build the synthetic node-week dataset.

    demand_volatility: std-dev of multiplicative weekly demand noise.
    downtime_severity: multiplier on unplanned, material and quality downtime.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(1, WEEKS + 1)

    # Network launch demand: logistic climb from ~6.8k to ~12.0k units/week.
    total_demand = 6400 + 5800 / (1 + np.exp(-(weeks - 5.5) / 1.8))

    rows = []
    for spec in NODES:
        ramp = ramp_curve(spec.ramp_start, spec.ramp_weeks)
        yld = spec.yield_start + (spec.yield_exit - spec.yield_start) * _smoothstep(
            (weeks - 1) / (WEEKS - 1))
        rate = spec.nameplate / (spec.base_shifts * HOURS_PER_SHIFT)
        sched = spec.base_shifts * HOURS_PER_SHIFT
        noise = rng.normal(1.0, demand_volatility, WEEKS)

        for i, w in enumerate(weeks):
            maturity_drag = 1.35 - 0.35 * ramp[i]  # immature lines break more
            dt = {
                "dt_planned": 4.0 if w in spec.pm_weeks else 0.0,
                "dt_unplanned": rng.gamma(2.0, spec.unplanned_dt_mean / 2.0)
                * maturity_drag * downtime_severity,
                "dt_material": (rng.uniform(2, 8) if rng.random() < 0.25 else 0.0)
                * downtime_severity,
                "dt_quality": (rng.uniform(2, 10) if rng.random() < 0.30 - 0.015 * w else 0.0)
                * downtime_severity,
            }
            for (iw, col), hrs in spec.incidents.items():
                if iw == w:
                    dt[col] += hrs * downtime_severity
            # Never let downtime exceed 60% of scheduled time.
            total_dt = sum(dt.values())
            if total_dt > 0.6 * sched:
                scale = 0.6 * sched / total_dt
                dt = {k: v * scale for k, v in dt.items()}

            rows.append({
                "week": int(w),
                "node": spec.node,
                "site": spec.site,
                "base_shifts": spec.base_shifts,
                "rate_uph": round(rate, 3),
                "ramp_factor": round(float(ramp[i]), 4),
                "yield_pct": round(float(yld[i]), 4),
                "demand": int(round(total_demand[i] * spec.demand_share * noise[i])),
                **{k: round(float(v), 2) for k, v in dt.items()},
            })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; an empty list means the frame is usable."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    numeric = [c for c in REQUIRED_COLUMNS if c not in ("node", "site")]
    for c in numeric:
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"Column '{c}' must be numeric.")
    if problems:
        return problems
    if df.duplicated(["week", "node"]).any():
        problems.append("Each node may appear only once per week.")
    if (df[numeric] < 0).any().any():
        problems.append("Numeric columns cannot be negative.")
    if not df["ramp_factor"].between(0, 1.2).all():
        problems.append("ramp_factor should be a fraction of nameplate (0 to 1.2).")
    if not df["yield_pct"].between(0, 1).all():
        problems.append("yield_pct should be a fraction between 0 and 1.")
    if (df["base_shifts"] > MAX_SHIFTS_PER_WEEK).any():
        problems.append(f"base_shifts cannot exceed {MAX_SHIFTS_PER_WEEK} per week.")
    dt = df[list(DOWNTIME_COLS)].sum(axis=1)
    if (dt > df["base_shifts"] * HOURS_PER_SHIFT).any():
        problems.append("Downtime hours exceed scheduled hours for at least one node-week.")
    counts = df.groupby("node")["week"].nunique()
    if counts.nunique() > 1:
        problems.append("Every node needs the same set of weeks.")
    return problems
