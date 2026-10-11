"""Invariants for the simulation engine. Run: pytest -q"""
import numpy as np
import pytest

from ramp.data import NODES, generate_data, validate
from ramp.model import (ScenarioConfig, ShiftConfig, TransferConfig, UpliftConfig, kpis,
                        shortage_bridge, simulate)
from ramp.readiness import Thresholds, readiness_table, recommendations

ALL = {(d.node, r.node): True for d in NODES for r in NODES if d != r}


@pytest.fixture(scope="module")
def raw():
    return generate_data(seed=7)


def full_cfg(carry=True):
    return ScenarioConfig(UpliftConfig(0.2, 5, 3),
                          ShiftConfig(True, {"GDL-L1": 3, "AUS-L1": 2}, 4, 1),
                          TransferConfig(True, 4, 1, compatible=ALL), carry)


def test_synthetic_data_is_valid(raw):
    assert validate(raw) == []
    assert len(raw) == 12 * len(NODES)


def test_seed_is_reproducible():
    assert generate_data(3).equals(generate_data(3))


@pytest.mark.parametrize("carry", [True, False])
def test_units_are_conserved(raw, carry):
    res, flows = simulate(raw, full_cfg(carry))
    # demand = fulfilled + unfilled, cumulatively
    assert np.isclose(res["demand"].sum() - res["fulfilled"].sum(),
                      res.loc[res.week == 12, "backlog_out"].sum() + res["lost_units"].sum())
    # transfers: received = sent x efficiency
    assert np.isclose(res["received_units"].sum(), res["sent_capacity"].sum() * 0.92)
    assert np.isclose(flows["units_received"].sum(), res["received_units"].sum())


def test_no_negative_or_overfilled(raw):
    res, _ = simulate(raw, full_cfg())
    for c in ("own_output", "fulfilled", "backlog_out", "sent_capacity", "received_units"):
        assert (res[c] >= -1e-9).all(), c
    assert (res["fulfilled"] <= res["required_load"] + 1e-6).all()
    assert (res["utilization"] <= 1 + 1e-9).all()


def test_donor_never_lends_more_than_spare_or_cap(raw):
    res, _ = simulate(raw, full_cfg())
    spare = (res["own_capacity"] - res["required_load"]).clip(lower=0)
    assert (res["sent_capacity"] <= spare + 1e-6).all()
    assert (res["sent_capacity"] <= 0.25 * res["own_capacity"] + 1e-6).all()


def test_incompatible_pairs_do_not_transfer(raw):
    cfg = full_cfg()
    cfg.transfer.compatible = {("SZH-L3", "GDL-L1"): True}
    _, flows = simulate(raw, cfg)
    assert set(zip(flows.donor, flows.receiver)) <= {("SZH-L3", "GDL-L1")}


def test_levers_respect_lead_times(raw):
    res, flows = simulate(raw, full_cfg())
    assert res.loc[res.week < 5, "extra_shifts"].sum() == 0
    assert flows.empty or flows["week"].min() >= 5


def test_levers_do_not_increase_shortfall(raw):
    up = ScenarioConfig(UpliftConfig(0.2, 5, 3))
    assert kpis(simulate(raw, full_cfg())[0])["shortfall"] <= \
        kpis(simulate(raw, up)[0])["shortfall"] + 1e-6


def test_bridge_reconciles(raw):
    br = shortage_bridge(raw, full_cfg())
    assert np.isclose(br["value"].iloc[:4].sum(), br["value"].iloc[4])


def test_readiness_and_recommendations_run(raw):
    res, flows = simulate(raw, full_cfg())
    table = readiness_table(res, Thresholds())
    assert set(table["status"]) <= {"Red", "Amber", "Green"}
    recs = recommendations(res, flows, full_cfg(), Thresholds())
    assert {"node", "lever", "action"} <= set(recs.columns)
