"""Plotly figures. All styling flows from the tokens below."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .data import DOWNTIME_COLS

INK = "#17212B"
MUTED = "#5E6B78"
GRID = "#D9E0E6"
SURFACE = "#FFFFFF"
PRIMARY = "#0E6E8C"
STATUS = {"Green": "#2F7D4F", "Amber": "#C98410", "Red": "#B83227"}
NODE_PALETTE = ["#0E6E8C", "#6B4E9B", "#A0522D", "#3F6E5A", "#8A6F2F", "#4A5D8C"]
DT_COLORS = {"dt_planned": "#9AA8B5", "dt_unplanned": "#B83227",
             "dt_material": "#C98410", "dt_quality": "#6B4E9B"}


def node_colors(nodes) -> dict:
    return {n: NODE_PALETTE[i % len(NODE_PALETTE)] for i, n in enumerate(sorted(nodes))}


def _style(fig: go.Figure, height=360, y_title=None, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, size=13),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(title="Week", dtick=1, gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=GRID, zeroline=False,
                     tickformat=",.0f")
    return fig


def network_demand_capacity(nw_base: pd.DataFrame, nw: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=nw["week"], y=nw["unfilled"], name="Open backlog / lost",
                marker_color=STATUS["Red"], opacity=0.25)
    fig.add_scatter(mode="lines", x=nw["week"], y=nw["nominal_capacity"], name="Nominal capacity",
                    line=dict(color=MUTED, dash="dot", width=1.5))
    fig.add_scatter(mode="lines", x=nw["week"], y=nw["effective_capacity"], name="Effective capacity",
                    line=dict(color=PRIMARY, width=2))
    if (nw["own_capacity"] - nw["effective_capacity"]).abs().sum() > 1:
        fig.add_scatter(mode="lines", x=nw["week"], y=nw["own_capacity"], name="With recovery shifts",
                        line=dict(color=PRIMARY, width=2, dash="dash"))
    if (nw["demand"] - nw_base["demand"]).abs().sum() > 1:
        fig.add_scatter(mode="lines", x=nw_base["week"], y=nw_base["demand"], name="Baseline demand",
                        line=dict(color=INK, width=1.5, dash="dot"))
    fig.add_scatter(mode="lines", x=nw["week"], y=nw["demand"], name="Demand",
                    line=dict(color=INK, width=3))
    fig.add_scatter(x=nw["week"], y=nw["fulfilled"], name="Fulfilled",
                    mode="markers", marker=dict(color=STATUS["Green"], size=8, symbol="diamond"))
    return _style(fig, 400, "Units / week")


def node_demand_capacity(res: pd.DataFrame, node: str) -> go.Figure:
    g = res[res["node"] == node]
    fig = go.Figure()
    fig.add_bar(x=g["week"], y=g["backlog_out"] + g["lost_units"], name="Unfilled",
                marker_color=STATUS["Red"], opacity=0.25)
    fig.add_scatter(mode="lines", x=g["week"], y=g["nameplate_capacity"], name="Nameplate",
                    line=dict(color=GRID, width=2))
    fig.add_scatter(mode="lines", x=g["week"], y=g["nominal_capacity"], name="Nominal (ramp)",
                    line=dict(color=MUTED, dash="dot", width=1.5))
    fig.add_scatter(mode="lines", x=g["week"], y=g["effective_capacity"], name="Effective",
                    line=dict(color=PRIMARY, width=2))
    if g["shift_capacity"].sum() > 0:
        fig.add_scatter(mode="lines", x=g["week"], y=g["own_capacity"], name="With shifts",
                        line=dict(color=PRIMARY, width=2, dash="dash"))
    if g["received_units"].sum() > 0:
        fig.add_scatter(mode="lines", x=g["week"], y=g["own_capacity"] + g["received_units"],
                        name="With transfers in", line=dict(color=STATUS["Green"], width=2, dash="dash"))
    fig.add_scatter(mode="lines", x=g["week"], y=g["demand"], name="Demand", line=dict(color=INK, width=3))
    return _style(fig, 380, "Units / week")


def heatmap(res: pd.DataFrame, value: str, colorscale, zmin=None, zmax=None, zmid=None,
            fmt=".0%") -> go.Figure:
    pv = res.pivot(index="node", columns="week", values=value).sort_index(ascending=False)
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index, colorscale=colorscale,
        zmin=zmin, zmax=zmax, zmid=zmid, texttemplate="%{z:" + fmt + "}",
        textfont=dict(size=11), xgap=2, ygap=2, colorbar=dict(thickness=10, tickformat=fmt),
        hovertemplate="%{y}, week %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig = _style(fig, 60 + 52 * len(pv), legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title=None, tickformat=None)
    fig.update_xaxes(showgrid=False)
    return fig


LOAD_SCALE = [[0, "#E8F1F4"], [0.45, "#9CC6D3"], [0.6, "#F2D58A"], [0.75, "#E39A53"],
              [1, "#B83227"]]
OEE_SCALE = [[0, "#B83227"], [0.5, "#F2D58A"], [1, "#2F7D4F"]]
DOWNTIME_SCALE = [[0, "#F3F5F7"], [0.5, "#E3A85A"], [1, "#B83227"]]


def downtime_stacked(res: pd.DataFrame, node: str | None) -> go.Figure:
    g = res if node is None else res[res["node"] == node]
    w = g.groupby("week")[list(DOWNTIME_COLS)].sum().reset_index()
    fig = go.Figure()
    for c, label in DOWNTIME_COLS.items():
        fig.add_bar(x=w["week"], y=w[c], name=label, marker_color=DT_COLORS[c])
    fig.update_layout(barmode="stack")
    fig = _style(fig, 340, "Downtime hours")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def downtime_pareto(res: pd.DataFrame) -> go.Figure:
    long = res.melt(id_vars="node", value_vars=list(DOWNTIME_COLS),
                    var_name="cat", value_name="hours")
    long = long.groupby(["node", "cat"], as_index=False)["hours"].sum()
    long["label"] = long["node"] + ", " + long["cat"].map(DOWNTIME_COLS).str.lower()
    long = long.sort_values("hours", ascending=False).head(10)
    long["cum"] = long["hours"].cumsum() / res[list(DOWNTIME_COLS)].sum().sum()
    fig = go.Figure()
    fig.add_bar(x=long["label"], y=long["hours"], marker_color=long["cat"].map(DT_COLORS),
                name="Hours", showlegend=False,
                hovertemplate="%{x}: %{y:.0f} h<extra></extra>")
    fig.add_scatter(x=long["label"], y=long["cum"], yaxis="y2", name="Cumulative share",
                    line=dict(color=INK, width=2), mode="lines+markers",
                    hovertemplate="%{y:.0%}<extra></extra>")
    fig = _style(fig, 360, "Hours over horizon")
    fig.update_layout(hovermode="closest",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], dtick=0.25,
                                  tickformat=".0%", showgrid=False))
    fig.update_xaxes(title=None, dtick=None, tickangle=-30)
    return fig


def shortage_bars(res: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()
    for n, g in res.groupby("node"):
        fig.add_bar(x=g["week"], y=g["capacity_gap"].clip(lower=0), name=n,
                    marker_color=colors[n])
    tot = res.groupby("week")["unfilled"].sum()
    fig.add_scatter(mode="lines", x=tot.index, y=tot.values, name="Network unfilled (cumulative if carried)",
                    line=dict(color=STATUS["Red"], width=2.5))
    fig.update_layout(barmode="stack")
    return _style(fig, 380, "Units")


def backlog_lines(res: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()
    for n, g in res.groupby("node"):
        fig.add_scatter(mode="lines", x=g["week"], y=g["unfilled"], name=n, line=dict(color=colors[n], width=2.5))
    return _style(fig, 320, "Units unfilled")


def bridge(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=df["step"], y=df["value"], measure=df["measure"],
        text=[f"{v:+,.0f}" if m == "relative" else f"{v:,.0f}"
              for v, m in zip(df["value"], df["measure"])],
        textposition="outside",
        connector=dict(line=dict(color=GRID)),
        increasing=dict(marker=dict(color=STATUS["Red"])),
        decreasing=dict(marker=dict(color=STATUS["Green"])),
        totals=dict(marker=dict(color=INK)),
    ))
    fig = _style(fig, 380, "Units unfilled over horizon", legend=False)
    fig.update_xaxes(title=None, dtick=None)
    fig.update_layout(hovermode="closest")
    running = df["value"].iloc[:-1].cumsum()
    top = max(running.max(), df["value"].iloc[-1], 1.0)
    fig.update_yaxes(range=[min(running.min(), 0), top * 1.18])
    return fig


def compare_lines(series: dict[str, pd.DataFrame], col: str, y_title: str) -> go.Figure:
    styles = [dict(color=MUTED, dash="dot"), dict(color=STATUS["Red"]),
              dict(color=PRIMARY, width=3)]
    fig = go.Figure()
    for (name, df), st in zip(series.items(), styles):
        fig.add_scatter(mode="lines", x=df["week"], y=df[col], name=name, line={"width": 2, **st})
    return _style(fig, 340, y_title)


def transfer_sankey(flows: pd.DataFrame, colors: dict) -> go.Figure | None:
    if flows.empty:
        return None
    f = flows.groupby(["donor", "receiver"], as_index=False)["units_received"].sum()
    donors = sorted(f["donor"].unique())
    receivers = sorted(f["receiver"].unique())
    labels = [f"{d} (donor)" for d in donors] + [f"{r} (receiver)" for r in receivers]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=18, thickness=14,
                  color=[colors[d] for d in donors] + [colors[r] for r in receivers]),
        link=dict(source=[donors.index(d) for d in f["donor"]],
                  target=[len(donors) + receivers.index(r) for r in f["receiver"]],
                  value=f["units_received"], color="rgba(14,110,140,0.25)",
                  hovertemplate="%{source.label} to %{target.label}: %{value:,.0f} units<extra></extra>"),
    ))
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=16, b=8), paper_bgcolor=SURFACE,
                      font=dict(color=INK, size=13))
    return fig


# --------------------------------------------------------------------------- Monte Carlo
SCENARIO_STYLE = {
    "Baseline": dict(color=MUTED, dash="dot"),
    "Uplift, no levers": dict(color=STATUS["Red"], dash="solid"),
    "Uplift with levers": dict(color=PRIMARY, dash="solid"),
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def status_probability_bars(probs: pd.DataFrame, order: list[str]) -> go.Figure:
    """Horizontal 100% bars: share of runs Red / Amber / Green per node."""
    p = probs.set_index("node").reindex(order[::-1])
    fig = go.Figure()
    for s in ("Red", "Amber", "Green"):
        fig.add_bar(y=p.index, x=p[s], name=s, orientation="h", marker_color=STATUS[s],
                    text=[f"{v:.0%}" if v >= 0.06 else "" for v in p[s]],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#fff"),
                    hovertemplate="%{y}: " + s + " in %{x:.1%} of runs<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig = _style(fig, 70 + 56 * len(p), legend=True)
    fig.update_layout(legend=dict(traceorder="normal"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Share of runs", tickformat=".0%", range=[0, 1], dtick=0.25)
    fig.update_yaxes(title=None, tickformat=None, showgrid=False)
    return fig


def fan_chart(bands: dict[str, pd.DataFrame], y_title: str,
              single: pd.DataFrame | None = None, single_col: str | None = None) -> go.Figure:
    fig = go.Figure()
    for name, b in bands.items():
        st = SCENARIO_STYLE[name]
        fig.add_scatter(x=b["week"], y=b["p90"], mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=b["week"], y=b["p10"], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=_rgba(st["color"], 0.16),
                        name=f"{name}, P10 to P90", hoverinfo="skip")
        fig.add_scatter(x=b["week"], y=b["p50"], mode="lines", name=f"{name}, median",
                        line=dict(color=st["color"], dash=st["dash"], width=2.5),
                        customdata=b[["p10", "p90"]].to_numpy(),
                        hovertemplate="%{y:,.0f} (P10 %{customdata[0]:,.0f}, "
                                      "P90 %{customdata[1]:,.0f})")
    if single is not None:
        fig.add_scatter(x=single["week"], y=single[single_col], mode="markers",
                        name="Single run (other tabs)",
                        marker=dict(color=INK, size=7, symbol="circle-open", line=dict(width=2)))
    return _style(fig, 380, y_title)


def outcome_histogram(network: pd.DataFrame, col: str, x_title: str, fmt: str,
                      marker: float | None = None, target: float | None = None) -> go.Figure:
    fig = go.Figure()
    for name, g in network.groupby("scenario", sort=False):
        fig.add_histogram(x=g[col], name=name, opacity=0.6, nbinsx=60, bingroup=1,
                          marker_color=SCENARIO_STYLE[name]["color"],
                          hovertemplate="%{x}: %{y} runs<extra>" + name + "</extra>")
    fig.update_layout(barmode="overlay")
    fig = _style(fig, 340, "Runs")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title=x_title, dtick=None, tickformat=fmt)
    if target is not None:
        fig.add_vline(x=target, line=dict(color=INK, dash="dash", width=1.5),
                      annotation_text="Target", annotation_position="top")
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=INK, width=1.5),
                      annotation_text="Single run", annotation_position="top left")
    return fig


def driver_bars(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=corr["driver"], x=corr["rho"], orientation="h",
        marker_color=[STATUS["Red"] if v > 0 else STATUS["Green"] for v in corr["rho"]],
        text=[f"{v:+.2f}" for v in corr["rho"]], textposition="outside",
        hovertemplate="%{y}: rank correlation %{x:+.2f}<extra></extra>"))
    fig = _style(fig, 60 + 44 * len(corr), legend=False)
    fig.update_layout(hovermode="closest")
    lo = min(corr["rho"].min() - 0.15, -0.05)
    hi = max(corr["rho"].max() + 0.15, 0.05)
    fig.update_xaxes(title="Rank correlation with shortfall", dtick=0.25, range=[lo, hi],
                     tickformat=".2f", zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(title=None, tickformat=None, showgrid=False)
    return fig
