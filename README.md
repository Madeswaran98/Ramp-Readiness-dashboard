# Ramp readiness dashboard

A Streamlit tool for judging whether a multi-site manufacturing ramp will hold up. It takes 12 weeks of node-level demand, capacity, downtime, ramp and yield data, then answers four questions:

1. Where does demand outrun effective capacity, and why (ramp immaturity, downtime or yield)?
2. What happens if demand rises?
3. How much do recovery shifts and cross-node capacity transfers recover, and at what cost?
4. Which nodes are ready (Green / Amber / Red), and what should each one do next?
5. How sure are those ratings? A Monte Carlo layer reruns every scenario across hundreds of random draws and reports each node's chance of landing Red, Amber or Green.

The default data is synthetic. See [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) for every modelling assumption and limitation; the same text appears in the app's Assumptions tab.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q            # engine invariants, plus Monte Carlo agreement with the single-run engine
```

## What's in the app

| Area | Contents |
|---|---|
| Header | One tile per node with status, fill rate, exit coverage and the main reason for its rating; network KPIs with deltas against baseline |
| Demand and capacity | Network demand vs nominal, effective and shift-boosted capacity; per-node view with a nameplate loss breakdown (ramp, downtime, yield) |
| Downtime and utilization | Downtime by cause per week, Pareto of the largest losses, heatmaps of downtime share, load ratio, utilization and an OEE proxy |
| Shortage | Weekly capacity gap by node, unfilled demand over time, shortage summary per node |
| Scenario impact | Waterfall from baseline shortfall, through demand uplift, to the shortfall left after each lever; baseline / uplift / mitigated comparison |
| Mitigation levers | Recovery shift capacity and cost, suggested shift counts, transfer flows (Sankey) and week-by-week lever activity |
| Readiness and actions | Readiness table, plus per-node recommended actions sized in units/week and cost/week; CSV export |
| Uncertainty | Monte Carlo over 100 to 1,000 runs: chance of meeting the fill-rate target, P10 to P90 ranges, per-node status probabilities with confidence margins, fan chart of unfilled demand, outcome histograms, and which inputs drive the spread. Once run, header tiles also show each node's status odds |
| Assumptions | Full method and limitations |
| Data | Input and results tables, CSV downloads, column reference |

Sidebar controls cover the data (seed, demand volatility, downtime severity, backlog carry-over, CSV upload), demand uplift (size, start week, phase-in, nodes), recovery shifts (per node, decision week, staffing lead time, efficiency, premium), transfers (decision week, qualification lead time, lending cap, efficiency, unit cost, qualified-pair matrix) and every readiness threshold. **Suggest shifts** fills in per-node shift counts sized to reach the Amber exit-coverage buffer.

## Using your own data

Upload a CSV in the sidebar's Data section with one row per node per week:

`week, node, site, base_shifts, rate_uph, ramp_factor, yield_pct, demand, dt_planned, dt_unplanned, dt_material, dt_quality`

Download the synthetic input from the Data tab for a ready-made template. Uploads are validated (missing columns, negative values, fractions out of range, downtime exceeding scheduled hours, uneven weeks); if a check fails the app says what to fix and stays on synthetic data.

## Project layout

```
app.py                 Streamlit UI
ramp/data.py           Node definitions, synthetic generator, schema validation
ramp/model.py          Capacity derivation, uplift, recovery shifts, transfers, KPIs, shortage bridge
ramp/readiness.py      Shared readiness metrics, status rules, shift sizing, recommendations
ramp/montecarlo.py     Sampling (regenerate or perturb), vectorised run metrics, summaries
ramp/charts.py         Plotly figures and visual tokens
docs/ASSUMPTIONS.md    Method, assumptions and limitations (rendered in-app)
tests/test_model.py    Engine invariants
tests/test_montecarlo.py  Monte Carlo reproducibility and agreement with the single-run engine
.streamlit/config.toml Theme
```

## Key limitations (short version)

Synthetic, uncalibrated data; one aggregate resource per node; Monte Carlo varies only demand and downtime, with independent draws; greedy rather than optimal transfer allocation; downtime held fixed when shifts are added; flat-rate costs. Details and the rest are in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).
