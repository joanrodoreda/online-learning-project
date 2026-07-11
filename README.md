# Online Learning Applications — Multi-Campaign Bidding under Budget Constraints

Online learning algorithms for an advertising agency that bids on **multiple
campaigns** across repeated **first-price auctions**, under a shared **budget**
and a **conflict graph** of mutually-exclusive campaigns. Competing bids are
unknown and must be learned purely from online feedback.

The project covers four requirements, from a single stochastic campaign to
non-stationary multi-campaign environments.

---

## Problem Setting

At each round `t = 1, …, T`, for every campaign `i`:

- The agency sets a bid `b_i` from a small discrete set `B`.
- A competing bid `m_i` is drawn (unknown distribution). We win iff `b_i ≥ m_i`.
- On a win: utility `v_i − b_i`, cost `b_i`. On a loss: both zero.
- **Feedback** is limited to the auctions we participated in (bandit / semi-bandit).
- A **conflict graph** forbids bidding on some pairs of campaigns in the same round
  (we must choose an independent set).
- A shared **budget** `B = ρ·T` limits total spend.

Each bid `b` behaves as a bandit arm with mean reward `μ(b) = (v − b)·F_D(b)`,
where `F_D` is the CDF of the competing bid — the trade-off between winning
probability and profit per win.

---

## The Four Requirements

| Req | Setting | Algorithm(s) |
|-----|---------|--------------|
| 1 | Single campaign, stochastic | UCB1 · Budget-UCB1 (Lagrangian dual) |
| 2 | Multiple campaigns, stochastic | Combinatorial-UCB over the conflict graph + budget |
| 3 | Best-of-both-worlds (stochastic & non-stationary) | Primal-Dual (Hedge + OGD), full feedback |
| 4 | Slightly non-stationary (piecewise) | Sliding-Window UCB · CUSUM change detector |

---

## File Structure

```
├── config_v4.py         Constants, hyperparameters and distribution specs
├── environment_v4.py    Auction simulators + clairvoyant (LP) benchmarks
├── agents_v4.py         All learning algorithms
├── main_v4.py           Experiment orchestration and plots (all requirements)
├── project.pdf          Project statement
└── README.md            This file
```

Dependency direction: `config_v4` → `environment_v4` / `agents_v4` → `main_v4`.
Everything is implemented on `numpy` + `scipy` only (no bandit/RL libraries).

---

## Algorithms (`agents_v4.py`)

**Baselines** — `RandomBiddingAgent`, `GreedyBiddingAgent`, `ETCBiddingAgent`
(Explore-Then-Commit). Reference points for the linear-regret floor.

**Requirement 1**
- `UCB1BiddingAgent` — UCB1, `UCB(b) = μ̂(b) + √(c·log T / n_b)`.
- `BudgetUCB1BiddingAgent` — UCB1 with a Lagrangian dual `λ` on the budget,
  updated by online gradient descent: `λ ← clip(λ + η(c_t − ρ), 0, 1/ρ)`.

**Requirement 2**
- `CombinatorialUCB1BiddingAgent` — maintains UCB estimates per
  (campaign, bid) for both reward and cost, scores feasible campaign subsets of
  the conflict graph with a Lagrangian budget objective, and plays the best
  admissible subset.

**Requirement 3**
- `PrimalDualHedgeBiddingAgent` — a per-campaign Hedge (Exponential Weights)
  primal regret minimiser combined with an OGD dual variable for the budget.
  Uses full feedback (observes every competing bid). Aims for best-of-both-worlds
  performance in stochastic and non-stationary regimes.

**Requirement 4**
- `SlidingWindowCombinatorialUCBAgent` — Combinatorial-UCB restricted to a
  sliding window of the most recent rounds, so stale pre-change data is forgotten.
- `ChangeDetectorCombinatorialUCBAgent` — Combinatorial-UCB with a CUSUM change
  detector that resets a campaign's statistics when a distribution shift is
  detected.

---

## Environments & Benchmarks (`environment_v4.py`)

| Class | Requirement | Feedback |
|-------|-------------|----------|
| `SingleCampaignEnv` | 1 | bandit |
| `MultiCampaignEnv` | 2 | semi-bandit (selected campaigns) |
| `NonStationaryMultiCampaignEnv` | 3 | full feedback |
| `SlightlyNonStationaryMultiCampaignEnv` | 4 | semi-bandit, piecewise phases |

Clairvoyant baselines are computed with linear programming
(`compute_clairvoyant`, `compute_clairvoyant_mc`, `compute_hindsight_clairvoyant`,
`compute_per_interval_clairvoyant`). The budget constraint uses the **expected
realized cost** `b·F_D(b)` (cost is only paid on a win), not the raw bid.

Competing bids are pre-generated per seed so that every agent in a trial faces
the identical noise sequence — differences in regret reflect the algorithm, not
luck.

---

## How to Run

```bash
pip install numpy scipy matplotlib
python3 main_v4.py
```

`main_v4.py` runs all four requirements in sequence and displays the plots
(cumulative regret, cost vs. budget, dual-variable trajectories, etc.). Close
each plot window to advance to the next.

---

## Theory / Course Notebook Connections

| Concept | Notebook |
|---------|----------|
| UCB1, ETC, regret methodology | 01 |
| Exponential Weights / Hedge | 03 |
| First-price auction clairvoyant (LP, cost `b·F_D(b)`) | 07 |
| Lagrangian relaxation + OGD dual | 08 |
| Combinatorial UCB over a constraint | 09 |
| Sliding-window UCB and CUSUM change detection | 10 |

Regret is estimated over multiple independent trials and reported as
`mean ± standard error`, following the course convention.

---

## Authors

Joan Rodoreda · Gianluca Croce · Davide Bottinelli — *Online Learning Applications*, Politecnico di Milano.
