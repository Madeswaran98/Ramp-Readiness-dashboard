## What the data represents

The default dataset is **synthetic**. It describes a single product family ramping across four nodes (production lines at different sites) over 12 weeks. No figure corresponds to a real factory, supplier or customer. The generator is seeded, so the same seed always reproduces the same data.

| Node | Site | Nameplate (units/wk) | Base shifts/wk | Week-1 rate | Weeks to full rate | Yield start to exit | Demand share |
|---|---|---|---|---|---|---|---|
| AUS-L1 | Austin, US | 4,000 | 10 | 70% | 6 | 93% to 97% | 28% |
| PEN-L2 | Penang, MY | 4,500 | 10 | 55% | 8 | 90% to 96% | 29% |
| GDL-L1 | Guadalajara, MX | 3,200 | 10 | 40% | 10 | 85% to 94% | 23% |
| SZH-L3 | Suzhou, CN | 3,000 | 15 | 80% | 4 | 95% to 98% | 20% |

Built-in events: an equipment failure at GDL-L1 in weeks 6 and 7 (22 h and 10 h), a supplier shortage at PEN-L2 in week 9 (12 h), and 4 h planned maintenance windows on a rotating schedule.

## How capacity is calculated

1. **Scheduled hours** = base shifts × 8 h.
2. **Downtime hours** = planned maintenance + unplanned equipment + material shortage + quality hold. Capped at 60% of scheduled hours.
3. **Nominal capacity** = scheduled hours × nameplate rate × ramp factor. This is what the line could build with no losses at its current maturity.
4. **Effective capacity** = (scheduled − downtime hours) × nameplate rate × ramp factor × yield. These are good units.
5. **Demand** is the requirement placed on each node. The network curve is a logistic launch ramp; each node receives a fixed share plus multiplicative noise.

## How shortage and utilization are measured

- **Required load** = this week's demand + backlog carried in (when backlog carry-over is on).
- **Fulfilled** = units the node builds for its own demand + units received from other nodes.
- **Unfilled** = required load − fulfilled. With carry-over on it becomes next week's backlog; with carry-over off it is counted as lost.
- **Shortfall over horizon** = total demand − total fulfilled. This is identical to week-12 backlog when carry-over is on, and to total lost units when it is off.
- **Capacity gap** = demand − own capacity for a single week. Positive means short. It ignores backlog, so it shows the structural gap rather than the accumulated one.
- **Utilization** = (own output + capacity lent to other nodes) ÷ own capacity. It cannot exceed 100%.
- **Load ratio** = required load ÷ own capacity. It can exceed 100% and shows how overloaded a node is.
- **OEE proxy** = availability × ramp factor × yield × utilization. It is a proxy only; true OEE needs cycle-time and speed-loss data that this model does not carry.

## Scenario levers

**Demand uplift** multiplies demand from a chosen week, phased in linearly over N weeks (1 = step change), for all nodes or a selected subset.

**Recovery shifts** add 8-hour shifts per week to chosen nodes.
- They produce from *decision week + staffing lead time*.
- Each recovery shift runs at a configurable efficiency (default 85%) of a regular shift to reflect new or overtime crews.
- Recovery shifts inherit the node's ramp factor and yield for that week.
- Downtime hours are **not** scaled up with extra shifts. Planned and event downtime are treated as fixed per week, which flatters recovery shifts slightly.
- A node can never exceed 21 shifts per week (3 × 7).
- Cost is a flat premium per shift.

**Cross-node transfers** let a qualified donor build for a short receiver.
- They begin at *decision week + qualification lead time*.
- Only pairs ticked in the compatibility matrix can transfer, and direction matters.
- A donor lends only spare capacity (own capacity − own required load), capped at a set share of its capacity.
- Each unit of donor capacity delivers *transfer efficiency* units to the receiver, covering freight damage, re-test and yield differences.
- Allocation is greedy each week: the most-short receiver is served first, from the donor with the most spare. This is a heuristic, not an optimum.
- Transferred units arrive in the same week; transit time is folded into the qualification lead time.
- Cost is charged per unit of donor capacity used.

**Shortage bridge.** The waterfall applies levers in a fixed order (uplift, then shifts, then transfers). Each lever's credit therefore depends on the levers ahead of it; transfers are only credited with gaps that shifts did not close.

## Readiness status

Each node is rated on the active scenario. **Red** if any Red rule fires; otherwise **Amber** if any Amber rule fires; otherwise **Green**. All thresholds are editable in the sidebar.

| Rule | Red | Amber |
|---|---|---|
| Fill rate over horizon | < 95% | < 99% |
| Exit coverage (avg capacity ÷ demand, last N weeks, after transfers) | < 98% | < 105% |
| Weeks at or above the high-utilization line (default 90%) | | ≥ 4 |
| Downtime share of scheduled hours | | > 12% |
| Exit yield | | < 95% |

Exit coverage carries the most weight for launch readiness because it asks whether the node can hold the run-rate once the ramp is over, regardless of how the early weeks went.

## Mitigation recommendations

Recommendations are rule-based and estimated at the exit rate:
- **Recovery shifts**: shifts needed = exit gap ÷ good units per recovery shift, rounded up and limited to remaining shift headroom.
- **Cross-node transfers**: spare exit capacity at qualified donors, after the lending cap and transfer losses.
- **Downtime reduction**: halving the node's largest downtime category, valued at exit good-units-per-hour.
- **Yield improvement**: value of one yield point at exit availability and rate.
- **Demand shaping**: raised when more than half a week of demand is still open at week 12.
- **Act as donor**: raised for Green nodes holding more than 10% spare at exit.

"Suggest shifts" sizes each node's shifts to reach the Amber exit-coverage buffer on its own capacity, before transfers.

## Uncertainty (Monte Carlo)

The other tabs show a single random draw. The Uncertainty tab reruns three scenarios (baseline, uplift on its own, and the active scenario) across 100 to 1,000 draws.

- **Synthetic data:** each run regenerates the dataset with a new seed (seeds 1000, 1001, …), using the sidebar's demand volatility and downtime severity. Weekly demand noise, random unplanned downtime, and whether and when material and quality events occur all change. Ramp curves, yield curves, planned maintenance and the built-in incidents stay fixed. The sidebar seed does not affect the simulation; it only picks which draw the other tabs show.
- **Uploaded data:** each run perturbs the upload. Demand gets multiplicative normal noise (std-dev = demand volatility). Unplanned, material and quality downtime are scaled by mean-preserving lognormal noise (σ = 0.5) times downtime severity, then capped at 60% of scheduled hours. Planned maintenance, ramp and yield stay as uploaded.
- **Common random numbers:** all three scenarios in a run share the same draw, so differences between them reflect the levers, not sampling luck.
- **Same rules as the single run:** readiness metrics come from the same function in both views, and tests confirm that a simulated run matches the single-run engine on the same seed. Changing readiness thresholds re-rates stored runs instantly; only the exit window and high-utilization line need a rerun.
- **Confidence margins:** the ± on each Red share is a 95% normal-approximation margin, 1.96 × √(p(1 − p)/n). At 200 runs a 50% share carries about ±7 points; 1,000 runs bring that to about ±3.
- **Drivers of spread:** Spearman rank correlation between each run's total demand or each node's total downtime hours and its network shortfall. It shows association within the sampled range, not cause. Drivers that never vary (ramp, yield) cannot appear.

## Limitations

- **Synthetic data.** Patterns are plausible but not calibrated to any real operation. Treat outputs as a demonstration of the method.
- **Single product, single constraint.** Each node is one aggregate resource. Multi-stage routings, bottleneck shifting, changeovers and product mix are not modelled.
- **Only demand and downtime vary.** The Monte Carlo layer samples demand noise and downtime. Ramp speed, yield learning and the size of the uplift are held at their settings, so the ranges understate total uncertainty. Draws are independent week to week and node to node, which ignores correlated shocks such as a network-wide supplier problem.
- **No material constraint beyond downtime.** Component supply is represented only as material-shortage downtime hours.
- **No inventory buffer.** There is no finished-goods stock to absorb early shortfalls, and no build-ahead.
- **Linear economics.** Shift and transfer costs are flat rates; there is no step cost for hiring, no overtime fatigue, and no capital cost for tooling.
- **Heuristic transfer allocation.** The greedy rule is transparent but can leave value on the table compared with an LP solver.
- **Fixed downtime under extra shifts.** Running more hours usually brings more failures; the model does not add them.
- **Readiness rules are judgement calls.** Default thresholds are typical starting points and should be agreed with operations, supply chain and quality before use.
