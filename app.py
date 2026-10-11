"""Manufacturing ramp readiness dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ramp import charts as ch
from ramp.data import (DOWNTIME_COLS, HOURS_PER_SHIFT, MAX_SHIFTS_PER_WEEK, REQUIRED_COLUMNS,
                       generate_data, validate)
from ramp.model import (ScenarioConfig, ShiftConfig, TransferConfig, UpliftConfig, kpis,
                        network_weekly, shift_headroom, shortage_bridge, simulate)
from ramp.montecarlo import (SCENARIOS, MCSettings, driver_correlations, network_summary,
                             percentile_of, run_monte_carlo, status_probabilities, weekly_bands)
from ramp.readiness import Thresholds, readiness_table, recommendations, suggest_shifts

st.set_page_config(page_title="Ramp readiness", page_icon="🏭", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  h1 {{ font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0; }}
  .lede {{ color: {ch.MUTED}; font-size: 1.02rem; margin: .2rem 0 1.4rem; max-width: 72ch; }}
  .tile {{ background: #fff; border-left: 6px solid var(--c); border-radius: 4px;
          padding: .8rem 1rem .7rem; height: 100%; }}
  .tile .top {{ display: flex; justify-content: space-between; align-items: baseline;
               gap: .5rem; }}
  .tile .node {{ font-size: 1.25rem; font-weight: 700; color: {ch.INK}; white-space: nowrap; }}
  .tile .status {{ font-weight: 700; color: var(--c); white-space: nowrap; }}
  .tile .site {{ color: {ch.MUTED}; font-size: .85rem; margin-bottom: .5rem; }}
  .tile .site em {{ font-style: normal; color: {ch.INK}; }}
  .tile .figs {{ display: grid; grid-template-columns: 1fr 1fr; gap: .1rem .8rem; font-size: .9rem; }}
  .tile .figs b {{ font-size: 1.05rem; color: {ch.INK}; }}
  .act {{ display: grid; grid-template-columns: 11rem 1fr 13rem; gap: .25rem 1rem;
          padding: .55rem 0; border-top: 1px solid {ch.GRID}; font-size: .93rem; }}
  .act:first-of-type {{ border-top: 0; }}
  .act .lever {{ font-weight: 700; }}
  .act .lever small {{ display: block; font-weight: 400; color: {ch.MUTED}; }}
  .act .est {{ text-align: right; color: {ch.INK}; font-variant-numeric: tabular-nums; }}
  .act .est small {{ display: block; color: {ch.MUTED}; }}
  .nodehead {{ font-size: 1.1rem; font-weight: 700; }}
  .nodehead span {{ color: var(--c); margin-left: .5rem; }}
  .drivers {{ color: {ch.MUTED}; font-size: .88rem; margin: .1rem 0 .4rem; }}
  @media (max-width: 800px) {{ .act {{ grid-template-columns: 1fr; }} .act .est {{ text-align: left; }} }}
  .tile .odds {{ font-size: .82rem; margin-top: .35rem; color: {ch.INK};
                font-variant-numeric: tabular-nums; }}
  .tile .odds b {{ white-space: nowrap; }}
  .tile .why {{ color: {ch.MUTED}; font-size: .82rem; margin-top: .5rem; line-height: 1.35; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data
def load_synthetic(seed: int, vol: float, sev: float) -> pd.DataFrame:
    return generate_data(seed, vol, sev)


def default_compat(nodes: list[str]) -> pd.DataFrame:
    """Donor rows x receiver columns. New lines are assumed not yet qualified as donors."""
    allowed = {("SZH-L3", "AUS-L1"), ("SZH-L3", "PEN-L2"), ("SZH-L3", "GDL-L1"),
               ("AUS-L1", "PEN-L2"), ("PEN-L2", "AUS-L1"), ("AUS-L1", "GDL-L1"),
               ("PEN-L2", "GDL-L1")}
    known = set(nodes) <= {"AUS-L1", "PEN-L2", "GDL-L1", "SZH-L3"}
    return pd.DataFrame([[d != r and ((d, r) in allowed if known else True) for r in nodes]
                         for d in nodes], index=nodes, columns=nodes)


with st.sidebar:
    st.header("Scenario")

    with st.expander("Data", expanded=False):
        upload = st.file_uploader("Use your own node-week CSV", type="csv",
                                  help="Same columns as the synthetic data. "
                                       "Download it from the Data tab for a template.")
        seed = st.number_input("Random seed", 0, 9999, 7)
        vol = st.slider("Demand volatility", 0.0, 0.15, 0.04, 0.01,
                        help="Std-dev of weekly multiplicative demand noise.")
        sev = st.slider("Downtime severity", 0.5, 2.0, 1.0, 0.1,
                        help="Multiplier on unplanned, material and quality downtime.")
        carry = st.toggle("Carry unfilled demand forward as backlog", True,
                          help="Off: unfilled demand each week is lost.")

    raw, source = load_synthetic(seed, vol, sev), "synthetic"
    if upload is not None:
        try:
            up_df = pd.read_csv(upload)
            problems = validate(up_df)
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
            problems = [f"Could not read the file: {exc}"]
        if problems:
            st.error("Using synthetic data. Fix the upload:\n\n- " + "\n- ".join(problems))
        else:
            raw, source = up_df, upload.name

    nodes = sorted(raw["node"].unique())
    n_weeks = int(raw["week"].max())
    base_shifts = raw.groupby("node")["base_shifts"].first().to_dict()

    with st.expander("Demand uplift", expanded=True):
        up_pct = st.slider("Uplift", 0, 50, 15, 1, format="%d%%") / 100
        up_start = st.slider("From week", 1, n_weeks, 6)
        up_phase = st.slider("Phase-in (weeks)", 1, 6, 3, help="1 = step change.")
        up_nodes = st.multiselect("Apply to", nodes, default=nodes)

    with st.expander("Recovery shifts", expanded=True):
        sh_on = st.toggle("Enable recovery shifts", True)
        sh_start = st.slider("Decision week", 1, n_weeks, 5, key="sh_start")
        sh_lead = st.slider("Staffing lead time (weeks)", 0, 4, 1)
        sh_eff = st.slider("Recovery shift efficiency", 0.5, 1.0, 0.85, 0.05,
                           help="Output vs a regular shift.")
        sh_cost = st.number_input("Premium per shift ($)", 0, 100_000, 9_000, 500)
        for n in nodes:
            st.session_state.setdefault(f"shift_{n}", {"GDL-L1": 2, "PEN-L2": 1}.get(n, 0))
        shift_slot = st.container()

    with st.expander("Cross-node transfers", expanded=True):
        tr_on = st.toggle("Enable transfers", True)
        tr_start = st.slider("Decision week", 1, n_weeks, 5, key="tr_start")
        tr_lead = st.slider("Qualification lead time (weeks)", 0, 6, 2)
        tr_cap = st.slider("Max share of donor capacity", 0.0, 0.5, 0.25, 0.05)
        tr_eff = st.slider("Transfer efficiency", 0.6, 1.0, 0.92, 0.01,
                           help="Units received per unit of donor capacity.")
        tr_cost = st.number_input("Cost per unit ($)", 0.0, 500.0, 14.0, 1.0)
        st.caption("Qualified pairs. Rows lend, columns receive.")
        compat_df = st.data_editor(default_compat(nodes), key=f"compat_{'_'.join(nodes)}",
                                   width="stretch")

    with st.expander("Readiness thresholds"):
        th = Thresholds(
            red_fill_rate=st.slider("Red below fill rate", 0.80, 1.0, 0.95, 0.01),
            amber_fill_rate=st.slider("Amber below fill rate", 0.80, 1.0, 0.99, 0.01),
            exit_coverage_red=st.slider("Red below exit coverage", 0.80, 1.2, 0.98, 0.01),
            exit_coverage_amber=st.slider("Amber below exit coverage", 0.80, 1.3, 1.05, 0.01),
            high_util=st.slider("High-utilization line", 0.7, 1.0, 0.90, 0.01),
            hot_weeks_amber=st.slider("Amber at hot weeks", 1, n_weeks, 4),
            downtime_amber=st.slider("Amber above downtime share", 0.0, 0.3, 0.12, 0.01),
            min_exit_yield=st.slider("Amber below exit yield", 0.80, 1.0, 0.95, 0.005),
            exit_window=st.slider("Exit window (weeks)", 1, 4, 2),
        )

# Shift inputs come after the uplift settings so "Suggest shifts" can size against them.
uplift = UpliftConfig(up_pct, up_start, up_phase, up_nodes)
shift_probe = ShiftConfig(True, {}, sh_start, sh_lead, sh_eff, sh_cost)
uplift_only, _ = simulate(raw, ScenarioConfig(uplift, carry_backlog=carry))
suggested = suggest_shifts(uplift_only, ScenarioConfig(uplift, shift_probe), th)


def _apply(s: dict) -> None:
    for k, v in s.items():
        st.session_state[f"shift_{k}"] = v


with shift_slot:
    st.caption("Extra shifts per week by node")
    for n in nodes:
        st.number_input(n, 0, shift_headroom(base_shifts[n]), key=f"shift_{n}",
                        help=f"Base {base_shifts[n]} shifts; ceiling {MAX_SHIFTS_PER_WEEK}.")
    st.button("Suggest shifts", on_click=_apply, args=(suggested,),
              help="Size shifts to reach the Amber exit-coverage buffer on each node's own "
                   "capacity, before transfers.", width="stretch")

compat = {(d, r): bool(compat_df.at[d, r]) for d in nodes for r in nodes if d != r}
cfg = ScenarioConfig(
    uplift=uplift,
    shifts=ShiftConfig(sh_on, {n: st.session_state[f"shift_{n}"] for n in nodes},
                       sh_start, sh_lead, sh_eff, sh_cost),
    transfer=TransferConfig(tr_on, tr_start, tr_lead, tr_cap, tr_eff, tr_cost, compat),
    carry_backlog=carry,
)
base_cfg = ScenarioConfig(carry_backlog=carry)

res, flows = simulate(raw, cfg)
base, _ = simulate(raw, base_cfg)
nw, nw_base, nw_up = network_weekly(res), network_weekly(base), network_weekly(uplift_only)
k, kb, ku = kpis(res), kpis(base), kpis(uplift_only)
recovered = ku["shortfall"] - k["shortfall"]
ready = readiness_table(res, th)
ready_base = readiness_table(base, th).set_index("node")
recs = recommendations(res, flows, cfg, th)
colors = ch.node_colors(nodes)


# --------------------------------------------------------------------------- Monte Carlo
@st.cache_data(show_spinner=False, max_entries=6)
def cached_mc(key: str, _raw, _cfg, _th, _settings):
    return run_monte_carlo(_raw, _cfg, _th, _settings)


st.session_state.setdefault("mc_runs", 200)
st.session_state.setdefault("mc_auto", False)
mc_mode = "regenerate" if source == "synthetic" else "perturb"
mc_settings = MCSettings(n_runs=int(st.session_state["mc_runs"]), mode=mc_mode,
                         demand_volatility=vol, downtime_severity=sev)
data_id = ("synthetic" if mc_mode == "regenerate"
           else str(int(pd.util.hash_pandas_object(raw, index=False).sum())))
mc_key = repr((data_id, mc_settings, cfg, th.exit_window, th.high_util))
if st.session_state["mc_auto"]:
    st.session_state["mc_key"] = mc_key
mc = None
if st.session_state.get("mc_key") == mc_key:
    with st.spinner(f"Running {mc_settings.n_runs} simulations"):
        mc = cached_mc(mc_key, raw, cfg, th, mc_settings)
mc_probs = status_probabilities(mc, th) if mc else None

# --------------------------------------------------------------------------- header
st.title("Ramp readiness")
levers = []
if up_pct:
    levers.append(f"+{up_pct:.0%} demand from week {up_start}")
if sh_on and any(cfg.shifts.extra_shifts.values()):
    levers.append(f"{sum(cfg.shifts.extra_shifts.values())} recovery shifts/week "
                  f"from week {sh_start + sh_lead}")
if tr_on:
    levers.append(f"transfers from week {tr_start + tr_lead}")
scenario_text = "; ".join(levers) if levers else "baseline plan, no levers applied"
st.markdown(f"<p class='lede'>{len(nodes)} nodes over a {n_weeks}-week launch ramp "
            f"({'synthetic data' if source == 'synthetic' else source}). "
            f"Active scenario: {scenario_text}.</p>", unsafe_allow_html=True)

cols = st.columns(len(nodes))
for col, (_, r) in zip(cols, ready.iterrows()):
    c = ch.STATUS[r["status"]]
    odds = ""
    if mc_probs is not None:
        pr = mc_probs[(mc_probs["scenario"] == SCENARIOS[2]) & (mc_probs["node"] == r["node"])]
        if len(pr):
            pr = pr.iloc[0]
            parts = [f"<b style='color:{ch.STATUS[st_]}'>{st_} {pr[st_]:.0%}</b>"
                     for st_ in ("Red", "Amber", "Green") if pr[st_] >= 0.005]
            odds = (f"<div class='odds'>Across {mc.settings.n_runs} runs: "
                    f"{', '.join(parts)}</div>")
    was = ready_base.at[r["node"], "status"]
    change = "" if was == r["status"] else f", <em>was {was} at baseline</em>"
    col.markdown(f"""
    <div class="tile" style="--c:{c}">
      <div class="top"><span class="node">{r['node']}</span>
        <span class="status">{r['status']}</span></div>
      <div class="site">{r['site']}{change}</div>
      <div class="figs">
        <span>Fill rate<br><b>{r['fill_rate']:.1%}</b></span>
        <span>Exit coverage<br><b>{r['exit_coverage']:.0%}</b></span>
      </div>
      {odds}
      <div class="why">{r['drivers'][0]}</div>
    </div>""", unsafe_allow_html=True)

st.write("")
m = st.columns(5)
m[0].metric("Demand, 12 weeks", f"{k['demand']:,.0f}", f"{k['demand'] - kb['demand']:+,.0f} vs baseline",
            delta_color="off")
m[1].metric("Fill rate", f"{k['fill_rate']:.1%}", f"{(k['fill_rate'] - kb['fill_rate']) * 100:+.1f} pts")
m[2].metric("Shortfall over horizon", f"{k['shortfall']:,.0f}",
            f"{k['shortfall'] - kb['shortfall']:+,.0f}", delta_color="inverse")
m[3].metric("Utilization", f"{k['utilization']:.1%}",
            f"{(k['utilization'] - kb['utilization']) * 100:+.1f} pts", delta_color="off")
m[4].metric("Mitigation cost", f"${k['mitigation_cost']:,.0f}",
            f"${k['mitigation_cost'] / recovered:,.0f} per unit recovered"
            if k["mitigation_cost"] and recovered > 1 else None, delta_color="off",
            help="Shift premiums plus transfer costs. Units recovered compares against the "
                 "uplift scenario with no levers.")

tabs = st.tabs(["Demand and capacity", "Downtime and utilization", "Shortage",
                "Scenario impact", "Mitigation levers", "Readiness and actions",
                "Uncertainty", "Assumptions", "Data"])

# --------------------------------------------------------------------------- tabs
with tabs[0]:
    st.subheader("Network demand against capacity")
    st.plotly_chart(ch.network_demand_capacity(nw_base, nw), width="stretch")
    st.caption("Nominal capacity is the ramp-limited rate with no losses. Effective capacity "
               "removes downtime and yield loss. Red bars show demand not yet filled.")
    st.subheader("By node")
    pick = st.segmented_control("Node", nodes, default=ready["node"].iloc[0], key="dc_node")
    pick = pick or nodes[0]
    st.plotly_chart(ch.node_demand_capacity(res, pick), width="stretch")
    g = res[res["node"] == pick]
    loss = pd.DataFrame({
        "Loss": ["Ramp immaturity", "Downtime", "Yield"],
        "Units over horizon": [
            (g["nameplate_capacity"] - g["nominal_capacity"]).sum(),
            (g["downtime_hours"] * g["rate_uph"] * g["ramp_factor"]).sum(),
            (g["available_hours"] * g["rate_uph"] * g["ramp_factor"] * (1 - g["yield_pct"])).sum(),
        ]})
    loss["Units over horizon"] = loss["Units over horizon"].round(0)
    st.caption(f"Where {pick}'s nameplate capacity goes")
    st.dataframe(loss, hide_index=True, width="content",
                 column_config={"Units over horizon": st.column_config.NumberColumn(format="localized")})

with tabs[1]:
    a, b = st.columns(2)
    with a:
        st.subheader("Downtime by cause")
        dn = st.selectbox("Scope", ["Whole network", *nodes], key="dt_scope")
        st.plotly_chart(ch.downtime_stacked(res, None if dn == "Whole network" else dn),
                        width="stretch")
    with b:
        st.subheader("Largest downtime losses")
        st.plotly_chart(ch.downtime_pareto(res), width="stretch")
    st.subheader("Downtime share of scheduled hours")
    st.plotly_chart(ch.heatmap(res, "downtime_pct", ch.DOWNTIME_SCALE, 0, 0.35),
                    width="stretch")
    st.subheader("Load on each node")
    view = st.radio("Measure", ["Load ratio (demand + backlog ÷ capacity)",
                                "Utilization (output ÷ capacity)", "OEE proxy"],
                    horizontal=True, key="util_view")
    if view.startswith("Load"):
        st.plotly_chart(ch.heatmap(res, "load_ratio", ch.LOAD_SCALE, 0.5, 1.5, 1.0),
                        width="stretch")
        st.caption("Above 100% the node cannot keep up without help.")
    elif view.startswith("Util"):
        st.plotly_chart(ch.heatmap(res, "utilization", ch.LOAD_SCALE, 0.5, 1.0),
                        width="stretch")
    else:
        st.plotly_chart(ch.heatmap(res, "oee_proxy", ch.OEE_SCALE, 0.3, 1.0), width="stretch")
        st.caption("Availability × ramp factor × yield × utilization. A proxy, not true OEE.")

with tabs[2]:
    st.subheader("Weekly capacity gap and unfilled demand")
    st.plotly_chart(ch.shortage_bars(res, colors), width="stretch")
    st.caption("Bars: demand minus own capacity in that week (structural gap, before transfers "
               "and ignoring backlog). Line: unfilled demand after all levers.")
    a, b = st.columns([3, 2])
    with a:
        st.subheader("Unfilled demand by node")
        st.plotly_chart(ch.backlog_lines(res, colors), width="stretch")
    with b:
        st.subheader("Shortage summary")
        sm = ready[["node", "shortfall", "fill_rate", "ending_backlog", "peak_load_ratio"]].copy()
        sm["weeks_short"] = sm["node"].map(res[res["unfilled"] > 0.5].groupby("node").size()) \
            .fillna(0).astype(int)
        sm["first_short_week"] = sm["node"].map(
            res[res["unfilled"] > 0.5].groupby("node")["week"].min())
        sm[["shortfall", "ending_backlog"]] = sm[["shortfall", "ending_backlog"]].round(0)
        st.dataframe(sm, hide_index=True, width="stretch", column_config={
            "node": "Node",
            "shortfall": st.column_config.NumberColumn("Shortfall", format="localized"),
            "fill_rate": st.column_config.ProgressColumn("Fill rate", min_value=0.5, max_value=1,
                                                         format="percent"),
            "ending_backlog": st.column_config.NumberColumn("Open at end", format="localized"),
            "peak_load_ratio": st.column_config.NumberColumn("Peak load", format="percent"),
            "weeks_short": "Weeks short",
            "first_short_week": st.column_config.NumberColumn("First short week", format="%d"),
        })

with tabs[3]:
    st.subheader("Where the shortfall comes from and where it goes")
    br = shortage_bridge(raw, cfg)
    st.plotly_chart(ch.bridge(br), width="stretch")
    st.caption("Levers are applied in order: uplift, recovery shifts, then transfers. "
               "Transfers only get credit for gaps that shifts left open.")
    series = {"Baseline": nw_base, "Uplift, no levers": nw_up, "Uplift with levers": nw}
    a, b = st.columns(2)
    with a:
        st.markdown("**Unfilled demand**")
        st.plotly_chart(ch.compare_lines(series, "unfilled", "Units"), width="stretch")
    with b:
        st.markdown("**Fulfilled units per week**")
        st.plotly_chart(ch.compare_lines(series, "fulfilled", "Units / week"),
                        width="stretch")
    comp = pd.DataFrame([
        {"Scenario": "Baseline", **kb}, {"Scenario": "Uplift, no levers", **ku},
        {"Scenario": "Uplift with levers", **k}])
    comp[["demand", "fulfilled", "shortfall"]] = comp[["demand", "fulfilled", "shortfall"]].round(0)
    st.dataframe(comp[["Scenario", "demand", "fulfilled", "shortfall", "fill_rate",
                       "utilization", "mitigation_cost"]], hide_index=True,
                 width="stretch", column_config={
                     "demand": st.column_config.NumberColumn("Demand", format="localized"),
                     "fulfilled": st.column_config.NumberColumn("Fulfilled", format="localized"),
                     "shortfall": st.column_config.NumberColumn("Shortfall", format="localized"),
                     "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
                     "utilization": st.column_config.NumberColumn("Utilization", format="percent"),
                     "mitigation_cost": st.column_config.NumberColumn("Lever cost", format="dollar"),
                 })

with tabs[4]:
    a, b = st.columns(2)
    with a:
        st.subheader("Recovery shifts")
        if not sh_on:
            st.info("Recovery shifts are off. Turn them on in the sidebar to model extra crews.")
        else:
            sh = res.groupby("node").agg(
                shifts=("extra_shifts", "max"),
                units=("shift_capacity", "sum"), cost=("shift_cost", "sum")).reset_index()
            sh["headroom"] = sh["node"].map(lambda n: shift_headroom(base_shifts[n])) - sh["shifts"]
            sh["suggested"] = sh["node"].map(suggested)
            sh["cost_per_unit"] = sh["cost"] / sh["units"].where(sh["units"] > 0)
            sh["units"] = sh["units"].round(0)
            sh = sh[["node", "shifts", "suggested", "headroom", "units", "cost", "cost_per_unit"]]
            st.dataframe(sh, hide_index=True, width="stretch", column_config={
                "node": "Node", "shifts": "Extra shifts/wk", "suggested": "Suggested",
                "headroom": "Slots left",
                "units": st.column_config.NumberColumn("Capacity added", format="localized"),
                "cost": st.column_config.NumberColumn("Cost", format="dollar"),
                "cost_per_unit": st.column_config.NumberColumn("$ / unit", format="$%.2f"),
            })
            st.caption(f"Productive from week {sh_start + sh_lead}. One recovery shift adds "
                       f"{HOURS_PER_SHIFT} h at {sh_eff:.0%} efficiency.")
    with b:
        st.subheader("Cross-node transfers")
        if not tr_on:
            st.info("Transfers are off. Turn them on in the sidebar and tick qualified pairs.")
        elif flows.empty:
            st.info("No transfers happened. Either no qualified donor had spare capacity once "
                    "transfers went live, or no node was short.")
        else:
            st.plotly_chart(ch.transfer_sankey(flows, colors), width="stretch")
            fs = flows.groupby(["donor", "receiver"], as_index=False).agg(
                weeks=("week", "nunique"), units=("units_received", "sum"), cost=("cost", "sum"))
            fs["units"] = fs["units"].round(0)
            st.dataframe(fs, hide_index=True, width="stretch", column_config={
                "donor": "Donor", "receiver": "Receiver", "weeks": "Weeks active",
                "units": st.column_config.NumberColumn("Units received", format="localized"),
                "cost": st.column_config.NumberColumn("Cost", format="dollar"),
            })
    st.subheader("Week-by-week lever activity")
    act = res.pivot_table(index="node", columns="week",
                          values=["shift_capacity", "received_units", "sent_capacity"], aggfunc="sum")
    lever_view = st.radio("Show", ["Capacity from recovery shifts", "Units received",
                                   "Capacity lent"], horizontal=True, key="lever_view")
    col = {"Capacity from recovery shifts": "shift_capacity", "Units received": "received_units",
           "Capacity lent": "sent_capacity"}[lever_view]
    st.dataframe(act[col].round(0).rename_axis("Node").rename(columns=lambda w: f"W{w}"),
                 width="stretch")

with tabs[5]:
    st.subheader("Node readiness")
    rt = ready.copy()
    rt["baseline"] = rt["node"].map(ready_base["status"])
    for c in ("fill_rate", "exit_coverage", "downtime_pct", "exit_yield"):
        rt[c] = (rt[c] * 100).round(1)
    rt[["transfer_in", "transfer_out"]] = rt[["transfer_in", "transfer_out"]].round(0)
    st.dataframe(rt[["node", "status", "baseline", "fill_rate", "exit_coverage", "hot_weeks",
                     "downtime_pct", "exit_yield", "transfer_in", "transfer_out"]],
                 hide_index=True, width="stretch", column_config={
                     "node": "Node", "status": "Status", "baseline": "Baseline status",
                     "fill_rate": st.column_config.NumberColumn("Fill rate", format="%.1f%%"),
                     "exit_coverage": st.column_config.NumberColumn("Exit coverage", format="%.0f%%"),
                     "hot_weeks": st.column_config.NumberColumn(
                         "Hot weeks", help=f"Weeks at or above {th.high_util:.0%} utilization"),
                     "downtime_pct": st.column_config.NumberColumn("Downtime", format="%.1f%%"),
                     "exit_yield": st.column_config.NumberColumn("Exit yield", format="%.1f%%"),
                     "transfer_in": st.column_config.NumberColumn("Units received", format="localized"),
                     "transfer_out": st.column_config.NumberColumn("Capacity lent", format="localized"),
                 })

    st.subheader("Recommended actions")
    st.caption("Sized at the exit rate, on top of the levers already in this scenario.")
    show_green = st.toggle("Include Green nodes", False, key="rec_green")
    for _, r in ready.iterrows():
        if r["status"] == "Green" and not show_green:
            continue
        node_recs = recs[recs["node"] == r["node"]]
        with st.container(border=True):
            st.markdown(
                f"<div class='nodehead' style='--c:{ch.STATUS[r['status']]}'>{r['node']}"
                f"<span>{r['status']}</span></div>"
                f"<div class='drivers'>{'; '.join(r['drivers'])}</div>",
                unsafe_allow_html=True)
            if node_recs.empty:
                st.markdown("No action needed.")
                continue
            rows = []
            for _, a in node_recs.iterrows():
                imp = a["impact_units_per_week"]
                cost = a["cost_per_week"]
                est = (f"+{imp:,.0f} units/wk" if pd.notna(imp) else "Not sized")
                est += f"<small>${cost:,.0f}/wk</small>" if pd.notna(cost) else ""
                rows.append(f"<div class='act'><div class='lever'>{a['lever']}"
                            f"<small>{a['priority']} priority</small></div>"
                            f"<div>{a['action']}</div><div class='est'>{est}</div></div>")
            st.markdown("".join(rows), unsafe_allow_html=True)
    if not show_green and (ready["status"] == "Green").any():
        n_g = int((ready["status"] == "Green").sum())
        st.caption(f"{n_g} Green {'node' if n_g == 1 else 'nodes'} hidden.")
    st.download_button("Download actions (CSV)", recs.to_csv(index=False), "ramp_actions.csv",
                       "text/csv")

with tabs[6]:
    st.subheader("How sure are these ratings?")
    what_varies = ("Each run regenerates the synthetic data with a new seed, so weekly demand "
                   "noise, random downtime and the timing of quality and material events all "
                   "change. Ramp curves, yield curves and the built-in incidents stay fixed."
                   if mc_mode == "regenerate" else
                   "Your uploaded data is perturbed in each run: demand gets random weekly noise "
                   f"(std-dev {vol:.0%}) and unplanned, material and quality downtime get "
                   "mean-preserving random scaling. Planned maintenance, ramp and yield stay "
                   "as uploaded.")
    st.markdown(f"<p class='lede'>The other tabs show one random draw. This reruns the "
                f"baseline, the uplift on its own and the active scenario across many draws. "
                f"{what_varies}</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 2], vertical_alignment="bottom")
    c1.select_slider("Runs", [100, 200, 500, 1000], key="mc_runs",
                     help="More runs narrow the uncertainty on each probability. "
                          "1,000 runs take a few seconds.")
    run_clicked = c2.button("Run simulation", type="primary", width="stretch")
    c3.toggle("Rerun automatically when settings change", key="mc_auto",
              help="Keeps the ranges current while you adjust the sidebar. Each change "
                   "costs a second or two.")
    if run_clicked:
        st.session_state["mc_key"] = mc_key
        st.rerun()

    if mc is None:
        if "mc_key" in st.session_state:
            st.info("Settings changed since the last run. Run the simulation again to update "
                    "the ranges.")
        else:
            st.info("Run the simulation to see how often each node lands Red, Amber or Green, "
                    "and the range of shortfall and fill rate.")
    else:
        ns = network_summary(mc, th.amber_fill_rate).set_index("scenario")
        act = ns.loc[SCENARIOS[2]]
        act_runs = mc.network[mc.network["scenario"] == SCENARIOS[2]]
        pctl = percentile_of(k["fill_rate"], act_runs["fill_rate"])
        st.markdown(
            f"With the active scenario, the network reaches the {th.amber_fill_rate:.0%} "
            f"fill-rate target in **{act['p_meets_target']:.0%} of {mc.settings.n_runs} runs**. "
            f"Median fill rate is {act['fill_p50']:.1%}, and 80% of runs fall between "
            f"{act['fill_p10']:.1%} and {act['fill_p90']:.1%}. The single run shown in the "
            f"other tabs sits at the {pctl:.0f}th percentile, so it is "
            + ("about typical." if 25 <= pctl <= 75 else
               "on the optimistic side." if pctl > 75 else "on the pessimistic side."))
        mm = st.columns(4)
        mm[0].metric("Runs meeting fill target", f"{act['p_meets_target']:.0%}",
                     f"{(act['p_meets_target'] - ns.loc[SCENARIOS[1], 'p_meets_target']) * 100:+.0f} pts vs no levers")
        mm[1].metric("Fill rate, P10 to P90", f"{act['fill_p10']:.1%} to {act['fill_p90']:.1%}")
        mm[2].metric("Shortfall, median", f"{act['short_p50']:,.0f}",
                     f"P10 {act['short_p10']:,.0f}, P90 {act['short_p90']:,.0f}", delta_color="off")
        mm[3].metric("Lever cost, median", f"${act['cost_p50']:,.0f}")

        st.subheader("How often each node lands in each status")
        scen_pick = st.segmented_control("Scenario", list(SCENARIOS), default=SCENARIOS[2],
                                         key="mc_scen") or SCENARIOS[2]
        sp = mc_probs[mc_probs["scenario"] == scen_pick]
        st.plotly_chart(ch.status_probability_bars(sp, list(ready["node"])), width="stretch")
        single_status = (ready.set_index("node")["status"] if scen_pick == SCENARIOS[2] else
                         readiness_table(base if scen_pick == SCENARIOS[0] else uplift_only, th)
                         .set_index("node")["status"])
        tbl = sp.set_index("node").reindex(list(ready["node"])).reset_index()
        tbl["single"] = tbl["node"].map(single_status)
        tbl["p_red"] = tbl.apply(lambda r: f"{r['Red']:.0%} ± {1.96 * r['red_se']:.0%}", axis=1)
        tbl["p_amber"] = tbl["Amber"].map(lambda v: f"{v:.0%}")
        tbl["p_green"] = tbl["Green"].map(lambda v: f"{v:.0%}")
        tbl["agrees"] = tbl.apply(
            lambda r: "Yes" if max(("Red", "Amber", "Green"), key=lambda x: r[x]) == r["single"]
            else "No", axis=1)
        st.dataframe(tbl[["node", "single", "p_red", "p_amber", "p_green", "agrees"]],
                     hide_index=True, width="stretch", column_config={
                         "node": "Node", "single": "Single-run status",
                         "p_red": st.column_config.TextColumn(
                             "Red", help="Share of runs, with a 95% confidence margin"),
                         "p_amber": "Amber", "p_green": "Green",
                         "agrees": st.column_config.TextColumn(
                             "Single run matches most likely status"),
                     })
        st.caption("Statuses are re-rated instantly when you change readiness thresholds; "
                   "only the exit window and high-utilization line need a rerun.")

        st.subheader("Unfilled demand, week by week")
        fan_pick = st.multiselect("Scenarios", list(SCENARIOS), default=list(SCENARIOS[1:]),
                                  key="mc_fan")
        if fan_pick:
            single_nw = nw if SCENARIOS[2] in fan_pick else None
            st.plotly_chart(ch.fan_chart({s_: weekly_bands(mc, s_) for s_ in fan_pick}, "Units",
                                         single_nw, "unfilled"), width="stretch")
            st.caption("Shaded bands hold the middle 80% of runs. Open circles are the single "
                       "run from the other tabs.")

        a, b = st.columns(2)
        with a:
            st.markdown("**Fill rate across runs**")
            st.plotly_chart(ch.outcome_histogram(mc.network, "fill_rate", "Fill rate", ".0%",
                                                 marker=k["fill_rate"],
                                                 target=th.amber_fill_rate), width="stretch")
        with b:
            st.markdown("**Shortfall across runs**")
            st.plotly_chart(ch.outcome_histogram(mc.network, "shortfall", "Units unfilled",
                                                 ",.0f", marker=k["shortfall"]), width="stretch")

        st.subheader("What drives the spread")
        corr = driver_correlations(mc, SCENARIOS[2])
        if corr.empty:
            st.info("Shortfall is the same in every run, so there is no spread to explain.")
        else:
            st.plotly_chart(ch.driver_bars(corr), width="stretch")
            top = corr.iloc[-1]
            st.caption(f"Rank correlation between each run's inputs and its shortfall under the "
                       f"active scenario. {top['driver']} is most closely associated with "
                       f"shortfall ({top['rho']:+.2f}). This shows association within the sampled "
                       f"range, not a causal estimate.")

        st.download_button("Download run results (CSV)",
                           mc.network.merge(mc.drivers, on="run").to_csv(index=False),
                           "ramp_monte_carlo_runs.csv", "text/csv")

with tabs[7]:
    st.markdown((Path(__file__).parent / "docs" / "ASSUMPTIONS.md").read_text())

with tabs[8]:
    st.subheader("Input data")
    st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}. Downtime columns are hours "
               f"per week; ramp_factor and yield_pct are fractions.")
    st.dataframe(raw, hide_index=True, width="stretch", height=300)
    a, b = st.columns(2)
    a.download_button("Download input data (CSV)", raw.to_csv(index=False),
                      "ramp_input.csv", "text/csv", width="stretch")
    b.download_button("Download scenario results (CSV)", res.to_csv(index=False),
                      "ramp_results.csv", "text/csv", width="stretch")
    with st.expander("Column reference"):
        ref = [("week", "Week number, 1 to 12"), ("node", "Line identifier"),
               ("site", "Site label"), ("base_shifts", "Scheduled 8-hour shifts per week"),
               ("rate_uph", "Nameplate units per hour at full ramp"),
               ("ramp_factor", "Fraction of nameplate rate achieved this week"),
               ("yield_pct", "Good units / units started"),
               ("demand", "Units required from this node this week"),
               *[(c, f"{v} downtime, hours") for c, v in DOWNTIME_COLS.items()]]
        st.dataframe(pd.DataFrame(ref, columns=["Column", "Meaning"]), hide_index=True,
                     width="stretch")
