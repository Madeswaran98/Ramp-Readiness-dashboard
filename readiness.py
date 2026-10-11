"""Node readiness status (Green / Amber / Red) and mitigation recommendations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DOWNTIME_COLS, HOURS_PER_SHIFT
from .model import ScenarioConfig, shift_headroom


@dataclass
class Thresholds:
    red_fill_rate: float = 0.95        # below -> Red
    amber_fill_rate: float = 0.99      # below -> Amber
    exit_coverage_red: float = 0.98    # exit-week capacity / demand below -> Red
    exit_coverage_amber: float = 1.05  # thin buffer -> Amber
    high_util: float = 0.90            # utilization considered "running hot"
    hot_weeks_amber: int = 4           # weeks at/above high_util -> Amber
    downtime_amber: float = 0.12       # downtime share of scheduled hours
    min_exit_yield: float = 0.95
    exit_window: int = 2               # last N weeks define the exit rate


STATUS_ORDER = {"Red": 0, "Amber": 1, "Green": 2}


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


METRIC_INPUTS = ("demand", "fulfilled", "own_capacity", "received_units", "sent_capacity",
                 "own_output", "downtime_hours", "scheduled_hours", "yield_pct", "backlog_out")


def core_metrics(a: dict, th: Thresholds) -> dict:
    """Readiness metrics from arrays shaped (..., nodes, weeks).

    Shared by the single-run view and the Monte Carlo layer, so both rate nodes
    on identical definitions. Returns arrays shaped (..., nodes).
    """
    ew = slice(-th.exit_window, None)
    dem = a["demand"].sum(-1)
    short = dem - a["fulfilled"].sum(-1)
    net_cap = a["own_capacity"] + a["received_units"] - a["sent_capacity"]
    exit_cap = net_cap[..., ew].mean(-1)
    exit_dem = a["demand"][..., ew].mean(-1)
    cap = a["own_capacity"]
    util = np.divide(a["own_output"] + a["sent_capacity"], cap,
                     out=np.zeros_like(cap), where=cap > 0)
    return {
        "fill_rate": np.where(dem > 0, 1 - short / np.where(dem > 0, dem, 1), 1.0),
        "shortfall": short,
        "ending_backlog": a["backlog_out"][..., -1],
        "exit_coverage": np.where(exit_dem > 0, exit_cap / np.where(exit_dem > 0, exit_dem, 1), 1.0),
        "exit_gap_per_week": np.clip(exit_dem - exit_cap, 0, None),
        "exit_spare_per_week": np.clip(exit_cap - exit_dem, 0, None),
        "hot_weeks": (util >= th.high_util).sum(-1),
        "downtime_pct": a["downtime_hours"].sum(-1) / a["scheduled_hours"].sum(-1),
        "exit_yield": a["yield_pct"][..., ew].mean(-1),
    }


def status_array(m: dict | pd.DataFrame, th: Thresholds) -> np.ndarray:
    """Vectorised Green/Amber/Red. Must agree with classify() (see tests)."""
    fr, ec = np.asarray(m["fill_rate"]), np.asarray(m["exit_coverage"])
    red = (fr < th.red_fill_rate) | (ec < th.exit_coverage_red)
    amber = ((fr < th.amber_fill_rate) | (ec < th.exit_coverage_amber)
             | (np.asarray(m["hot_weeks"]) >= th.hot_weeks_amber)
             | (np.asarray(m["downtime_pct"]) > th.downtime_amber)
             | (np.asarray(m["exit_yield"]) < th.min_exit_yield))
    return np.select([red, amber], ["Red", "Amber"], "Green")


def node_metrics(res: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    res = res.sort_values(["node", "week"])
    nodes = list(res["node"].unique())
    shape = (len(nodes), res["week"].nunique())

    def arr(c):
        return res[c].to_numpy(float).reshape(shape)

    m = core_metrics({c: arr(c) for c in METRIC_INPUTS}, th)
    ew = slice(-th.exit_window, None)
    out = pd.DataFrame({"node": nodes,
                        "site": res.groupby("node", sort=False)["site"].first().to_numpy(),
                        **m})
    out["hot_weeks"] = out["hot_weeks"].astype(int)
    out["peak_load_ratio"] = arr("load_ratio").max(-1)
    out["exit_ramp"] = arr("ramp_factor")[:, ew].mean(-1)
    out["extra_shifts_exit"] = arr("extra_shifts")[:, ew].max(-1).astype(int)
    out["transfer_in"] = arr("received_units").sum(-1)
    out["transfer_out"] = arr("sent_capacity").sum(-1)
    return out


def classify(m: pd.Series, th: Thresholds) -> tuple[str, list[str]]:
    red, amber = [], []
    if m.fill_rate < th.red_fill_rate:
        red.append(f"Fill rate {m.fill_rate:.1%} is below {th.red_fill_rate:.0%}")
    elif m.fill_rate < th.amber_fill_rate:
        amber.append(f"Fill rate {m.fill_rate:.1%} is below {th.amber_fill_rate:.0%}")
    if m.exit_coverage < th.exit_coverage_red:
        red.append(f"Exit capacity covers {m.exit_coverage:.0%} of exit demand")
    elif m.exit_coverage < th.exit_coverage_amber:
        amber.append(f"Exit buffer is thin ({m.exit_coverage:.0%} coverage)")
    if m.hot_weeks >= th.hot_weeks_amber:
        amber.append(f"{m.hot_weeks} weeks at or above {th.high_util:.0%} utilization")
    if m.downtime_pct > th.downtime_amber:
        amber.append(f"Downtime is {m.downtime_pct:.1%} of scheduled hours")
    if m.exit_yield < th.min_exit_yield:
        amber.append(f"Exit yield {m.exit_yield:.1%} is below {th.min_exit_yield:.0%}")
    if red:
        return "Red", red + amber
    if amber:
        return "Amber", amber
    return "Green", ["Meets all readiness thresholds"]


def readiness_table(res: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    m = node_metrics(res, th)
    out = m.apply(lambda r: classify(r, th), axis=1, result_type="expand")
    m["status"], m["drivers"] = out[0], out[1]
    m["rank"] = m["status"].map(STATUS_ORDER)
    return m.sort_values(["rank", "fill_rate"]).drop(columns="rank").reset_index(drop=True)


def _units_per_shift(g: pd.DataFrame, efficiency: float, exit_window: int) -> float:
    ex = g.tail(exit_window)
    return float((HOURS_PER_SHIFT * efficiency * ex["good_units_per_hour"]).mean())


def suggest_shifts(res: pd.DataFrame, cfg: ScenarioConfig, th: Thresholds) -> dict[str, int]:
    """Extra shifts/week needed to close each node's exit-rate gap on its own capacity."""
    out = {}
    for node, g in res.groupby("node"):
        g = g.sort_values("week")
        ex = g.tail(th.exit_window)
        # Target the Amber buffer so a Green rating is reachable on own capacity.
        target = ex["demand"].mean() * th.exit_coverage_amber
        gap = max(target - ex["effective_capacity"].mean(), 0.0)
        per_shift = _units_per_shift(g, cfg.shifts.efficiency, th.exit_window)
        need = math.ceil(gap / per_shift) if per_shift > 0 and gap > 0 else 0
        out[node] = int(min(need, shift_headroom(g["base_shifts"].iloc[0])))
    return out


def recommendations(res: pd.DataFrame, flows: pd.DataFrame, cfg: ScenarioConfig,
                    th: Thresholds) -> pd.DataFrame:
    table = readiness_table(res, th)
    recs = []
    donors = table[table["exit_spare_per_week"] > 0].set_index("node")

    for _, m in table.iterrows():
        node = m["node"]
        g = res[res["node"] == node].sort_values("week")
        per_shift = _units_per_shift(g, cfg.shifts.efficiency, th.exit_window)
        headroom = shift_headroom(g["base_shifts"].iloc[0]) - m["extra_shifts_exit"]
        gap = m["exit_gap_per_week"]
        prio = {"Red": "High", "Amber": "Medium", "Green": "Low"}[m["status"]]

        def add(lever, action, impact, cost, p=prio):
            recs.append({"node": node, "status": m["status"], "priority": p, "lever": lever,
                         "action": action, "impact_units_per_week": impact,
                         "cost_per_week": cost})

        if gap > 0:
            n = math.ceil(gap / per_shift) if per_shift else 0
            more = " more" if m["extra_shifts_exit"] else ""
            if 0 < n <= headroom:
                add("Recovery shifts",
                    f"Add {n}{more} recovery {'shift' if n == 1 else 'shifts'} a week, staffed "
                    f"{_n(cfg.shifts.lead_time_weeks, 'week')} ahead. Closes the "
                    f"{gap:,.0f} units/week exit gap.",
                    n * per_shift, n * cfg.shifts.cost_per_shift)
            elif headroom > 0:
                add("Recovery shifts",
                    f"Use all {_n(headroom, 'remaining shift slot')}; still {gap - headroom * per_shift:,.0f} "
                    f"units/week short, so pair with transfers or capital.",
                    headroom * per_shift, headroom * cfg.shifts.cost_per_shift)
            else:
                add("Capital / new tool", "Line is at 21 shifts/week. Only added equipment, "
                    "transfers or rate improvements raise output.", None, None)

            t = cfg.transfer
            others = [d for d in donors.index if d != node]
            qualified = [d for d in others if t.compatible.get((d, node), False)]
            unqualified = [d for d in others if d not in qualified]

            def _avail(ds):
                return min(sum(donors.at[d, "exit_spare_per_week"] for d in ds)
                           * t.efficiency, gap)

            if qualified and _avail(qualified) > 1:
                move = _avail(qualified)
                if t.enabled:
                    text = (f"{', '.join(qualified)} still hold spare exit capacity. The "
                            f"{t.max_share_of_donor:.0%} lending cap may be holding it back; "
                            f"raising it could free about {move:,.0f} units/week for {node}.")
                else:
                    text = (f"Turn on transfers: {', '.join(qualified)} are qualified and could "
                            f"supply about {move:,.0f} units/week after transfer losses.")
                add("Cross-node transfer", text, move, move / t.efficiency * t.cost_per_unit)
            elif unqualified and _avail(unqualified) > 1:
                move = _avail(unqualified)
                add("Cross-node transfer",
                    f"Start qualifying {', '.join(unqualified)} to build for {node}; about "
                    f"{move:,.0f} units/week of spare exit capacity after transfer losses.",
                    move, move / t.efficiency * t.cost_per_unit)
            elif m["transfer_in"] > 0:
                add("Cross-node transfer",
                    f"Already receiving {m['transfer_in']:,.0f} units over the horizon; "
                    "no donor has spare exit capacity left.", None, None, "Low")
            else:
                add("Cross-node transfer",
                    "No node has spare exit capacity to lend. Capacity has to come from "
                    "this node or from new equipment.", None, None, "Low")

        if gap <= 0 and m["fill_rate"] < th.amber_fill_rate:
            short = g[g["unfilled"] > 0.5]
            live = [w for w, on in (
                (cfg.shifts.start_week + cfg.shifts.lead_time_weeks,
                 cfg.shifts.enabled and any(cfg.shifts.extra_shifts.values())),
                (cfg.transfer.start_week + cfg.transfer.lead_time_weeks, cfg.transfer.enabled))
                if on]
            first_live = min(live) if live else None
            peak_row = g.loc[g["unfilled"].idxmax()]
            when = (f" Levers only take effect from week {first_live}." if first_live else "")
            first = f" from week {int(short['week'].min())}" if len(short) else ""
            add("Earlier action",
                f"Exit rate is covered, but the node runs short{first} and unfilled demand "
                f"peaks at {peak_row['unfilled']:,.0f} units in week {int(peak_row['week'])}."
                f"{when} Bring the decision forward, shorten lead times or build ahead.",
                None, None)

        if m["downtime_pct"] > th.downtime_amber or m["status"] != "Green":
            cats = g[list(DOWNTIME_COLS)].sum()
            top = cats.idxmax()
            hrs = cats[top] / len(g)
            if hrs > 1:
                gain = 0.5 * hrs * g.tail(th.exit_window)["good_units_per_hour"].mean()
                add("Downtime reduction",
                    f"Halve {DOWNTIME_COLS[top].lower()} downtime (avg {hrs:.1f} h/week, the "
                    f"largest loss category) through spares, PM tuning or supplier expedite.",
                    gain, None, "Medium" if prio == "High" else prio)

        if m["exit_yield"] < th.min_exit_yield - 0.005:
            ex = g.tail(th.exit_window)
            per_pt = (ex["available_hours"] * ex["rate_uph"] * ex["ramp_factor"]).mean() * 0.01
            target_pts = (th.min_exit_yield - m["exit_yield"]) * 100
            add("Yield improvement",
                f"Lift yield {target_pts:.1f} pts to {th.min_exit_yield:.0%}; "
                f"each point is worth about {per_pt:,.0f} units/week.",
                per_pt * target_pts, None)

        if m["ending_backlog"] > 0.5 * g["demand"].tail(1).iloc[0]:
            add("Demand shaping",
                f"{m['ending_backlog']:,.0f} units remain open at week {g['week'].max()}. "
                "Agree customer allocation and re-commit dates.", None, None)

        if m["status"] == "Green" and m["exit_spare_per_week"] > 0.10 * g["demand"].tail(1).iloc[0]:
            add("Act as donor",
                f"Holds ~{m['exit_spare_per_week']:,.0f} units/week spare at exit; offer it "
                "to short nodes once qualified.", m["exit_spare_per_week"], None, "Low")

    cols = ["node", "status", "priority", "lever", "action", "impact_units_per_week",
            "cost_per_week"]
    df = pd.DataFrame(recs, columns=cols)
    df["_p"] = df["priority"].map({"High": 0, "Medium": 1, "Low": 2})
    df["_s"] = df["status"].map(STATUS_ORDER)
    return df.sort_values(["_s", "_p", "node"]).drop(columns=["_p", "_s"]).reset_index(drop=True)
