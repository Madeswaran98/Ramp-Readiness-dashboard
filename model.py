"""Week-by-week ramp simulation.

Capacity chain (per node-week):
    scheduled_hours   = base_shifts x 8
    available_hours   = scheduled_hours - downtime_hours
    nominal_capacity  = scheduled_hours x rate_uph x ramp_factor          (no losses)
    effective_capacity= available_hours x rate_uph x ramp_factor x yield  (good units)

Order of operations inside each week:
    1. demand (after uplift) + backlog carried in = required load
    2. own capacity = effective capacity + recovery-shift capacity
    3. cross-node transfers move spare capacity to nodes that are short
    4. fulfilled = own output + units received; remainder becomes backlog (or lost)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DOWNTIME_COLS, HOURS_PER_SHIFT, MAX_SHIFTS_PER_WEEK


# --------------------------------------------------------------------------- config
@dataclass
class UpliftConfig:
    pct: float = 0.0              # 0.15 = +15% demand
    start_week: int = 1
    phase_in_weeks: int = 1       # 1 = step change
    nodes: list[str] | None = None  # None = every node


@dataclass
class ShiftConfig:
    enabled: bool = False
    extra_shifts: dict[str, int] = field(default_factory=dict)  # per week
    start_week: int = 1           # decision week
    lead_time_weeks: int = 1      # hiring/training before the shift produces
    efficiency: float = 0.85      # output of a recovery shift vs a regular shift
    cost_per_shift: float = 9000.0


@dataclass
class TransferConfig:
    enabled: bool = False
    start_week: int = 1
    lead_time_weeks: int = 2      # qualification of donor for receiver's demand
    max_share_of_donor: float = 0.25  # cap on donor capacity that can be lent
    efficiency: float = 0.92      # received units per unit of donor capacity
    cost_per_unit: float = 14.0   # freight + duties + handling per unit shipped
    compatible: dict[tuple[str, str], bool] = field(default_factory=dict)  # (donor, receiver)


@dataclass
class ScenarioConfig:
    uplift: UpliftConfig = field(default_factory=UpliftConfig)
    shifts: ShiftConfig = field(default_factory=ShiftConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    carry_backlog: bool = True


# --------------------------------------------------------------------------- derive
def derive_capacity(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values(["node", "week"]).reset_index(drop=True)
    df["downtime_hours"] = df[list(DOWNTIME_COLS)].sum(axis=1)
    df["scheduled_hours"] = df["base_shifts"] * HOURS_PER_SHIFT
    df["available_hours"] = (df["scheduled_hours"] - df["downtime_hours"]).clip(lower=0)
    df["nameplate_capacity"] = df["scheduled_hours"] * df["rate_uph"]
    df["nominal_capacity"] = df["nameplate_capacity"] * df["ramp_factor"]
    df["good_units_per_hour"] = df["rate_uph"] * df["ramp_factor"] * df["yield_pct"]
    df["effective_capacity"] = df["available_hours"] * df["good_units_per_hour"]
    df["downtime_pct"] = df["downtime_hours"] / df["scheduled_hours"]
    return df


def uplift_multiplier(weeks: pd.Series, cfg: UpliftConfig) -> np.ndarray:
    w = weeks.to_numpy()
    phase = np.clip((w - cfg.start_week + 1) / max(cfg.phase_in_weeks, 1), 0, 1)
    phase = np.where(w < cfg.start_week, 0.0, phase)
    return 1.0 + cfg.pct * phase


def shift_headroom(base_shifts: int) -> int:
    return max(MAX_SHIFTS_PER_WEEK - int(base_shifts), 0)


# --------------------------------------------------------------------------- core
@dataclass
class Prepared:
    """Node-major (nodes x weeks) arrays for one dataset."""
    nodes: list[str]
    weeks: np.ndarray
    base_shifts: np.ndarray
    demand: np.ndarray
    good_uph: np.ndarray
    effective: np.ndarray
    downtime: np.ndarray
    scheduled: np.ndarray
    yield_pct: np.ndarray


def prepare(df: pd.DataFrame) -> Prepared:
    """Arrays from a frame already passed through derive_capacity (sorted node, week)."""
    nodes = list(df["node"].unique())
    weeks = np.sort(df["week"].unique())
    shape = (len(nodes), len(weeks))

    def arr(c):
        return df[c].to_numpy(float).reshape(shape)

    return Prepared(nodes, weeks, arr("base_shifts"), arr("demand"),
                    arr("good_units_per_hour"), arr("effective_capacity"),
                    arr("downtime_hours"), arr("scheduled_hours"), arr("yield_pct"))


def run_core(p: Prepared, cfg: ScenarioConfig) -> tuple[dict, list]:
    """Scenario arithmetic on arrays. Returns (arrays keyed by column, transfer flows)."""
    n_n, n_w = p.demand.shape
    w = p.weeks[None, :]

    # 1. Demand uplift
    u = cfg.uplift
    phase = np.clip((w - u.start_week + 1) / max(u.phase_in_weeks, 1), 0, 1)
    mult = 1.0 + u.pct * np.where(w < u.start_week, 0.0, phase)
    if u.nodes is not None:
        mult = np.where(np.isin(p.nodes, u.nodes)[:, None], mult, 1.0)
    demand = p.demand * mult

    # 2. Recovery shifts
    s = cfg.shifts
    extra = np.zeros((n_n, n_w))
    if s.enabled:
        want = np.array([s.extra_shifts.get(n, 0) for n in p.nodes], float)[:, None]
        room = np.clip(MAX_SHIFTS_PER_WEEK - p.base_shifts, 0, None)
        extra = np.where(w >= s.start_week + s.lead_time_weeks, np.minimum(want, room), 0.0)
    shift_cap = extra * HOURS_PER_SHIFT * s.efficiency * p.good_uph
    own_cap = p.effective + shift_cap

    # 3/4. Week loop for backlog and transfers (plain floats keep it fast)
    t = cfg.transfer
    names = ("backlog_in", "required_load", "own_output", "sent_capacity",
             "received_units", "fulfilled", "backlog_out", "lost_units")
    out = {c: np.zeros((n_n, n_w)) for c in names}
    allowed = [[d != r and t.compatible.get((p.nodes[d], p.nodes[r]), False)
                for r in range(n_n)] for d in range(n_n)]
    live_from = t.start_week + t.lead_time_weeks
    transfers_on = t.enabled and t.efficiency > 0
    flows = []
    backlog = [0.0] * n_n

    for wi in range(n_w):
        week = int(p.weeks[wi])
        cap = own_cap[:, wi].tolist()
        dem = demand[:, wi].tolist()
        req = [dem[i] + backlog[i] for i in range(n_n)]
        need = [max(req[i] - cap[i], 0.0) for i in range(n_n)]
        lendable = [min(max(cap[i] - req[i], 0.0), t.max_share_of_donor * cap[i])
                    for i in range(n_n)]
        sent = [0.0] * n_n
        received = [0.0] * n_n

        if transfers_on and week >= live_from and any(need) and any(lendable):
            for r in sorted(range(n_n), key=lambda i: -need[i]):
                for d in sorted(range(n_n), key=lambda i: -lendable[i]):
                    if need[r] <= 1e-9:
                        break
                    if lendable[d] <= 1e-9 or not allowed[d][r]:
                        continue
                    give = min(lendable[d], need[r] / t.efficiency)
                    lendable[d] -= give
                    sent[d] += give
                    received[r] += give * t.efficiency
                    need[r] -= give * t.efficiency
                    flows.append((week, p.nodes[d], p.nodes[r], give,
                                  give * t.efficiency, give * t.cost_per_unit))

        for i in range(n_n):
            own = min(req[i], cap[i])
            fulfilled = own + received[i]
            open_units = max(req[i] - fulfilled, 0.0)
            out["backlog_in"][i, wi] = backlog[i]
            out["required_load"][i, wi] = req[i]
            out["own_output"][i, wi] = own
            out["sent_capacity"][i, wi] = sent[i]
            out["received_units"][i, wi] = received[i]
            out["fulfilled"][i, wi] = fulfilled
            if cfg.carry_backlog:
                out["backlog_out"][i, wi] = open_units
                backlog[i] = open_units
            else:
                out["lost_units"][i, wi] = open_units

    out.update(demand=demand, extra_shifts=extra, shift_capacity=shift_cap,
               own_capacity=own_cap, shift_cost=extra * s.cost_per_shift,
               transfer_cost=out["sent_capacity"] * t.cost_per_unit)
    return out, flows


# --------------------------------------------------------------------------- simulate
def simulate(raw: pd.DataFrame, cfg: ScenarioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the scenario. Returns (node-week results, transfer flows)."""
    df = derive_capacity(raw)
    df["baseline_demand"] = df["demand"].astype(float)
    out, flows = run_core(prepare(df), cfg)
    for c, v in out.items():
        df[c] = v.ravel()
    df["extra_shifts"] = df["extra_shifts"].astype(int)
    df["shift_hours"] = df["extra_shifts"] * HOURS_PER_SHIFT
    df["unfilled"] = df["backlog_out"] + df["lost_units"]
    df["capacity_gap"] = df["demand"] - df["own_capacity"]          # + short, - slack
    df["load_ratio"] = np.where(df["own_capacity"] > 0,
                                df["required_load"] / df["own_capacity"], np.inf)
    df["utilization"] = np.where(df["own_capacity"] > 0,
                                 (df["own_output"] + df["sent_capacity"]) / df["own_capacity"], 0)
    df["oee_proxy"] = (df["available_hours"] / df["scheduled_hours"]) \
        * df["ramp_factor"] * df["yield_pct"] * df["utilization"]
    flows_df = pd.DataFrame(flows, columns=["week", "donor", "receiver",
                                            "donor_capacity_used", "units_received", "cost"])
    return df, flows_df


# --------------------------------------------------------------------------- summaries
def network_weekly(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("week").agg(
        demand=("demand", "sum"), baseline_demand=("baseline_demand", "sum"),
        nominal_capacity=("nominal_capacity", "sum"),
        effective_capacity=("effective_capacity", "sum"),
        own_capacity=("own_capacity", "sum"), fulfilled=("fulfilled", "sum"),
        unfilled=("unfilled", "sum"), downtime_hours=("downtime_hours", "sum"),
        scheduled_hours=("scheduled_hours", "sum"),
    ).reset_index()
    g["utilization"] = (g["fulfilled"] / g["own_capacity"]).clip(upper=1.0)
    return g


def total_shortfall(res: pd.DataFrame) -> float:
    """Demand not fulfilled by the end of the horizon (backlog left or sales lost)."""
    return float(res["demand"].sum() - res["fulfilled"].sum())


def kpis(res: pd.DataFrame) -> dict:
    demand = res["demand"].sum()
    short = total_shortfall(res)
    return {
        "demand": demand,
        "fulfilled": res["fulfilled"].sum(),
        "shortfall": short,
        "fill_rate": 1 - short / demand if demand else 1.0,
        "utilization": (res["own_output"].sum() + res["sent_capacity"].sum())
        / res["own_capacity"].sum(),
        "downtime_pct": res["downtime_hours"].sum() / res["scheduled_hours"].sum(),
        "peak_backlog": res.groupby("week")["unfilled"].sum().max(),
        "mitigation_cost": res["shift_cost"].sum() + res["transfer_cost"].sum(),
    }


def shortage_bridge(raw: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """Decompose shortfall: baseline -> +uplift -> -recovery shifts -> -transfers.

    Steps are applied in this fixed order, so each lever's credit depends on the
    levers ahead of it (transfers are credited only with what shifts did not fix).
    """
    off_s, off_t = ShiftConfig(**{**cfg.shifts.__dict__, "enabled": False}), \
        TransferConfig(**{**cfg.transfer.__dict__, "enabled": False})
    base = ScenarioConfig(UpliftConfig(), off_s, off_t, cfg.carry_backlog)
    up = ScenarioConfig(cfg.uplift, off_s, off_t, cfg.carry_backlog)
    up_s = ScenarioConfig(cfg.uplift, cfg.shifts, off_t, cfg.carry_backlog)
    vals = [total_shortfall(simulate(raw, c)[0]) for c in (base, up, up_s, cfg)]
    return pd.DataFrame({
        "step": ["Baseline shortfall", "Demand uplift", "Recovery shifts",
                 "Cross-node transfers", "Residual shortfall"],
        "value": [vals[0], vals[1] - vals[0], vals[2] - vals[1], vals[3] - vals[2], vals[3]],
        "measure": ["absolute", "relative", "relative", "relative", "total"],
    })
