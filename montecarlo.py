"""Monte Carlo layer: rerun every scenario over many random draws.

Two sampling modes:
  regenerate  synthetic data only. Each run calls generate_data with a new seed,
              so demand noise, random downtime and the timing of quality and
              material events all vary. Ramp and yield curves and the built-in
              incidents stay fixed.
  perturb     any dataset, including uploads. Demand gets multiplicative normal
              noise; unplanned, material and quality downtime get mean-preserving
              lognormal noise. Planned maintenance, ramp and yield stay fixed.

Each run evaluates baseline, uplift-only and uplift-with-levers on the same
draw (common random numbers), so differences between scenarios are not
blurred by sampling noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DOWNTIME_COLS, HOURS_PER_SHIFT, generate_data
from .model import Prepared, ScenarioConfig, ShiftConfig, TransferConfig, UpliftConfig, run_core
from .readiness import METRIC_INPUTS, Thresholds, core_metrics, status_array

SCENARIOS = ("Baseline", "Uplift, no levers", "Uplift with levers")
DT = list(DOWNTIME_COLS)
RANDOM_DT = ["dt_unplanned", "dt_material", "dt_quality"]


@dataclass(frozen=True)
class MCSettings:
    n_runs: int = 200
    base_seed: int = 1000
    mode: str = "regenerate"       # or "perturb"
    demand_volatility: float = 0.04
    downtime_severity: float = 1.0
    downtime_sigma: float = 0.5    # perturb mode: lognormal sigma on random downtime


@dataclass
class MCResult:
    settings: MCSettings
    nodes: list[str]
    weeks: np.ndarray
    network: pd.DataFrame       # run, scenario, demand, fulfilled, shortfall, fill_rate, cost, utilization
    node_runs: pd.DataFrame     # run, scenario, node, core metrics (status assigned at display time)
    weekly: dict                # scenario -> {"unfilled": (R, W), "fulfilled": (R, W)}
    drivers: pd.DataFrame       # run-level inputs for sensitivity


# --------------------------------------------------------------------------- arrays
@dataclass
class _Base:
    nodes: list[str]
    weeks: np.ndarray
    base_shifts: np.ndarray
    rate: np.ndarray
    ramp: np.ndarray
    yld: np.ndarray
    demand: np.ndarray
    dt: dict


def _base_arrays(raw: pd.DataFrame) -> _Base:
    node = raw["node"].astype(str).to_numpy()
    week = raw["week"].to_numpy()
    order = np.lexsort((week, node))
    nodes = list(dict.fromkeys(node[order]))
    weeks = np.unique(week)
    shape = (len(nodes), len(weeks))

    def arr(c):
        return raw[c].to_numpy(float)[order].reshape(shape)

    return _Base(nodes, weeks, arr("base_shifts"), arr("rate_uph"), arr("ramp_factor"),
                 arr("yield_pct"), arr("demand"), {c: arr(c) for c in DT})


def _prepared(b: _Base, demand: np.ndarray, dt: dict) -> Prepared:
    """Same arithmetic as model.derive_capacity, on arrays."""
    downtime = sum(dt.values())
    scheduled = b.base_shifts * HOURS_PER_SHIFT
    available = np.clip(scheduled - downtime, 0, None)
    good_uph = b.rate * b.ramp * b.yld
    return Prepared(b.nodes, b.weeks, b.base_shifts, demand, good_uph,
                    available * good_uph, downtime, scheduled, b.yld)


def _perturb(b: _Base, rng: np.random.Generator, s: MCSettings) -> tuple[np.ndarray, dict]:
    demand = np.clip(b.demand * rng.normal(1.0, s.demand_volatility, b.demand.shape), 0, None)
    sig = s.downtime_sigma
    dt = {c: b.dt[c].copy() for c in DT}
    for c in RANDOM_DT:
        dt[c] = dt[c] * rng.lognormal(-sig ** 2 / 2, sig, dt[c].shape) * s.downtime_severity
    sched = b.base_shifts * HOURS_PER_SHIFT
    total = sum(dt.values())
    scale = np.where(total > 0.6 * sched, 0.6 * sched / np.where(total > 0, total, 1), 1.0)
    return demand, {c: v * scale for c, v in dt.items()}


def scenario_set(cfg: ScenarioConfig) -> dict[str, ScenarioConfig]:
    off_s = ShiftConfig(**{**cfg.shifts.__dict__, "enabled": False})
    off_t = TransferConfig(**{**cfg.transfer.__dict__, "enabled": False})
    return {
        SCENARIOS[0]: ScenarioConfig(UpliftConfig(), off_s, off_t, cfg.carry_backlog),
        SCENARIOS[1]: ScenarioConfig(cfg.uplift, off_s, off_t, cfg.carry_backlog),
        SCENARIOS[2]: cfg,
    }


# --------------------------------------------------------------------------- run
def run_monte_carlo(raw: pd.DataFrame, cfg: ScenarioConfig, th: Thresholds,
                    s: MCSettings, progress=None) -> MCResult:
    scen = scenario_set(cfg)
    base = _base_arrays(raw)
    keep = METRIC_INPUTS + ("shift_cost", "transfer_cost", "lost_units")
    stacks = {name: {k: [] for k in keep} for name in scen}
    drivers = []

    for i in range(s.n_runs):
        seed = s.base_seed + i
        if s.mode == "regenerate":
            b = _base_arrays(generate_data(seed, s.demand_volatility, s.downtime_severity))
            demand, dt = b.demand, b.dt
        else:
            b = base
            demand, dt = _perturb(base, np.random.default_rng(seed), s)
        p = _prepared(b, demand, dt)
        drivers.append({"run": i, "seed": seed, "Network demand": demand.sum(),
                        **{f"{n} downtime": sum(v[j].sum() for v in dt.values())
                           for j, n in enumerate(b.nodes)}})
        for name, c in scen.items():
            out, _ = run_core(p, c)
            out.update(downtime_hours=p.downtime, scheduled_hours=p.scheduled,
                       yield_pct=p.yield_pct)
            for k in keep:
                stacks[name][k].append(out[k])
        if progress and (i % 10 == 0 or i == s.n_runs - 1):
            progress((i + 1) / s.n_runs)

    net_rows, node_frames, weekly = [], [], {}
    runs = np.arange(s.n_runs)
    for name in scen:
        a = {k: np.stack(v) for k, v in stacks[name].items()}       # (R, N, W)
        m = core_metrics(a, th)                                      # (R, N)
        nf = pd.DataFrame({k: v.ravel() for k, v in m.items()})
        nf.insert(0, "node", np.tile(base.nodes, s.n_runs))
        nf.insert(0, "scenario", name)
        nf.insert(0, "run", np.repeat(runs, len(base.nodes)))
        node_frames.append(nf)

        dem = a["demand"].sum((1, 2))
        ful = a["fulfilled"].sum((1, 2))
        cap = a["own_capacity"].sum((1, 2))
        net_rows.append(pd.DataFrame({
            "run": runs, "scenario": name, "demand": dem, "fulfilled": ful,
            "shortfall": dem - ful, "fill_rate": 1 - (dem - ful) / dem,
            "cost": (a["shift_cost"] + a["transfer_cost"]).sum((1, 2)),
            "utilization": (a["own_output"] + a["sent_capacity"]).sum((1, 2)) / cap,
        }))
        unfilled = a["backlog_out"] + a["lost_units"]
        weekly[name] = {"unfilled": unfilled.sum(1), "fulfilled": a["fulfilled"].sum(1)}

    return MCResult(s, base.nodes, base.weeks, pd.concat(net_rows, ignore_index=True),
                    pd.concat(node_frames, ignore_index=True), weekly, pd.DataFrame(drivers))


# --------------------------------------------------------------------------- summaries
def pct(x, q):
    return float(np.percentile(x, q))


def status_probabilities(r: MCResult, th: Thresholds) -> pd.DataFrame:
    """Share of runs in each status, per scenario and node, with a standard error."""
    nr = r.node_runs.copy()
    nr["status"] = status_array(nr, th)
    out = (nr.groupby(["scenario", "node"])["status"].value_counts(normalize=True)
           .unstack(fill_value=0.0).reindex(columns=["Red", "Amber", "Green"], fill_value=0.0)
           .reset_index())
    n = r.settings.n_runs
    out["red_se"] = np.sqrt(out["Red"] * (1 - out["Red"]) / n)
    out["not_green"] = out["Red"] + out["Amber"]
    return out


def network_summary(r: MCResult, target_fill: float) -> pd.DataFrame:
    g = r.network.groupby("scenario", sort=False)
    return pd.DataFrame({
        "fill_p10": g["fill_rate"].quantile(0.10), "fill_p50": g["fill_rate"].median(),
        "fill_p90": g["fill_rate"].quantile(0.90),
        "short_p10": g["shortfall"].quantile(0.10), "short_p50": g["shortfall"].median(),
        "short_p90": g["shortfall"].quantile(0.90),
        "cost_p50": g["cost"].median(),
        "p_meets_target": g["fill_rate"].apply(lambda x: (x >= target_fill).mean()),
    }).reindex(list(SCENARIOS)).reset_index()


def weekly_bands(r: MCResult, scenario: str, col: str = "unfilled") -> pd.DataFrame:
    x = r.weekly[scenario][col]
    return pd.DataFrame({"week": r.weeks, "p10": np.percentile(x, 10, 0),
                         "p50": np.percentile(x, 50, 0), "p90": np.percentile(x, 90, 0)})


def driver_correlations(r: MCResult, scenario: str = SCENARIOS[2]) -> pd.DataFrame:
    """Spearman rank correlation between run inputs and network shortfall."""
    y = r.network.loc[r.network["scenario"] == scenario, "shortfall"].reset_index(drop=True)
    x = r.drivers.drop(columns=["run", "seed"])
    if y.nunique() < 2:
        return pd.DataFrame(columns=["driver", "rho"])
    ranks = x.rank()
    yr = y.rank()
    rho = ranks.apply(lambda c: np.corrcoef(c, yr)[0, 1] if c.nunique() > 1 else 0.0)
    return (pd.DataFrame({"driver": rho.index, "rho": rho.to_numpy()})
            .assign(abs_rho=lambda d: d["rho"].abs())
            .sort_values("abs_rho", ascending=True).drop(columns="abs_rho"))


def percentile_of(value: float, sample: np.ndarray) -> float:
    return float((np.asarray(sample) < value).mean() * 100)
