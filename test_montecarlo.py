"""Monte Carlo layer: reproducibility, agreement with the single-run engine, sane outputs."""
import numpy as np
import pandas as pd
import pytest

from ramp.data import NODES, generate_data
from ramp.model import (ScenarioConfig, ShiftConfig, TransferConfig, UpliftConfig, kpis,
                        simulate)
from ramp.montecarlo import (SCENARIOS, MCSettings, driver_correlations, network_summary,
                             run_monte_carlo, status_probabilities, weekly_bands)
from ramp.readiness import Thresholds, classify, node_metrics, readiness_table, status_array

ALL = {(d.node, r.node): True for d in NODES for r in NODES if d != r}
TH = Thresholds()


def cfg(carry=True):
    return ScenarioConfig(UpliftConfig(0.15, 6, 3), ShiftConfig(True, {"GDL-L1": 2}, 5, 1),
                          TransferConfig(True, 5, 2, compatible=ALL), carry)


@pytest.fixture(scope="module")
def raw():
    return generate_data(7)


@pytest.mark.parametrize("carry", [True, False])
def test_regenerate_run_matches_single_run_engine(raw, carry):
    """Run i of regenerate mode must equal simulate() on generate_data(base_seed + i)."""
    r = run_monte_carlo(raw, cfg(carry), TH, MCSettings(n_runs=3, base_seed=11))
    for i in range(3):
        res, _ = simulate(generate_data(11 + i), cfg(carry))
        k = kpis(res)
        row = r.network[(r.network.run == i) & (r.network.scenario == SCENARIOS[2])].iloc[0]
        assert np.isclose(row.shortfall, k["shortfall"])
        assert np.isclose(row.cost, k["mitigation_cost"])
        nm = node_metrics(res, TH).set_index("node")
        mc = r.node_runs[(r.node_runs.run == i) & (r.node_runs.scenario == SCENARIOS[2])] \
            .set_index("node").loc[nm.index]
        for c in ("fill_rate", "exit_coverage", "hot_weeks", "downtime_pct", "exit_yield"):
            assert np.allclose(mc[c], nm[c]), c


def test_reproducible(raw):
    a = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=20, mode="perturb"))
    b = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=20, mode="perturb"))
    pd.testing.assert_frame_equal(a.network, b.network)


def test_status_array_agrees_with_classify(raw):
    r = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=60))
    for th in (TH, Thresholds(red_fill_rate=0.9, exit_coverage_amber=1.2, hot_weeks_amber=2)):
        nr = r.node_runs
        vec = status_array(nr, th)
        row = nr.apply(lambda m: classify(m, th)[0], axis=1).to_numpy()
        assert (vec == row).all()


def test_probabilities_are_distributions(raw):
    r = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=50))
    p = status_probabilities(r, TH)
    assert np.allclose(p[["Red", "Amber", "Green"]].sum(axis=1), 1.0)
    assert len(p) == len(SCENARIOS) * len(NODES)


def test_summaries_are_ordered(raw):
    r = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=50))
    ns = network_summary(r, 0.99)
    assert (ns.fill_p10 <= ns.fill_p50).all() and (ns.fill_p50 <= ns.fill_p90).all()
    b = weekly_bands(r, SCENARIOS[2])
    assert ((b.p10 <= b.p50) & (b.p50 <= b.p90)).all()
    assert driver_correlations(r)["rho"].between(-1, 1).all()


def test_common_random_numbers_keep_levers_helpful(raw):
    """Same draw for each scenario: levers never make a run worse than uplift alone."""
    r = run_monte_carlo(raw, cfg(), TH, MCSettings(n_runs=80))
    w = r.network.pivot(index="run", columns="scenario", values="shortfall")
    assert (w[SCENARIOS[2]] <= w[SCENARIOS[1]] + 1e-6).all()


def test_perturb_mode_handles_uploaded_shape():
    df = generate_data(3).replace({"node": {"AUS-L1": "Line-A", "SZH-L3": "Line-B"}})
    c = ScenarioConfig(UpliftConfig(0.1, 4, 1), carry_backlog=True)
    r = run_monte_carlo(df, c, TH, MCSettings(n_runs=25, mode="perturb"))
    assert set(r.nodes) == set(df["node"])
    assert r.network["fill_rate"].between(0, 1).all()
    # zero noise reproduces the uploaded data exactly
    z = run_monte_carlo(df, c, TH, MCSettings(n_runs=2, mode="perturb",
                                              demand_volatility=0, downtime_sigma=1e-12))
    res, _ = simulate(df, c)
    assert np.allclose(z.network.query("scenario == @SCENARIOS[2]").shortfall,
                       kpis(res)["shortfall"])
