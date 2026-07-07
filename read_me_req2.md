# Requirement 2 — Multiple Campaigns, Stochastic Environment

Requirement 2 extends the single-campaign auction setting to a **multi-campaign** one.
The agent must choose a set of campaigns and a bid for each selected campaign, while
respecting both the conflict graph and the budget.

## Problem Setting

At each round:

1. The environment samples a joint vector of competing bids, one per campaign.
2. The agent selects an admissible set of campaigns.
3. The agent assigns one bid to each selected campaign.
4. The environment returns semi-bandit feedback:
   - only selected campaigns are observed
   - each selected campaign reveals win/loss, reward, and cost

This is the natural multi-campaign extension of the bandit feedback model used in
Requirement 1.

## Files Added

- [`environment_req2.py`](C:\Users\davbo\Documents\GitHub\online-learning-project\environment_req2.py)
- [`agents_req2.py`](C:\Users\davbo\Documents\GitHub\online-learning-project\agents_req2.py)
- [`main_req2.py`](C:\Users\davbo\Documents\GitHub\online-learning-project\main_req2.py)

The structure mirrors the rest of the project:

- environment file: stochastic simulator
- agents file: learning algorithm
- main file: experiment orchestration

## Theory Connections

This requirement is built from the course material on:

- [`Practical Sessions-20260506/08_constrained_problems.ipynb`](C:\Users\davbo\Documents\GitHub\online-learning-project\Practical%20Sessions-20260506\08_constrained_problems.ipynb)
- [`Practical Sessions-20260506/09_combinatorial_mabs.ipynb`](C:\Users\davbo\Documents\GitHub\online-learning-project\Practical%20Sessions-20260506\09_combinatorial_mabs.ipynb)
- [`Nuova cartella/8-combinatorialBandits.pdf`](C:\Users\davbo\Documents\GitHub\online-learning-project\Nuova%20cartella\8-combinatorialBandits.pdf)
- [`Nuova cartella/5-OGD (1).pdf`](C:\Users\davbo\Documents\GitHub\online-learning-project\Nuova%20cartella\5-OGD%20(1).pdf)

The implementation follows the same ideas seen in class:

1. Use optimistic estimates for each campaign-bid pair.
2. Search over feasible campaign subsets with the conflict graph.
3. Penalize cost through a Lagrangian-style budget term.

## Environment

`MultiCampaignEnv` generates a joint vector of competing bids `m_t` for all campaigns.
The joint draw can be independent or correlated.

Returned feedback is:

```python
win_t, reward_t, cost_t
```

but only for campaigns the agent actually activates.

## Agent

`CombinatorialUCB1BiddingAgent` is the Requirement 2 strategy.
It maintains one UCB table per campaign and bid for both reward and cost, then scores
feasible subsets from the conflict graph using a Lagrangian objective:

```text
UCB_reward(i,b) - λ_t · UCB_cost(i,b)
```

The best subset is selected by explicit enumeration of the admissible campaign sets.

## Clairvoyant Benchmark

The benchmark enumerates feasible independent sets and bid combinations, then keeps
the best action under the budgeted objective. This is practical because Requirement 2
is intended to start small, with a limited number of campaigns.

## Suggested Experiments

- 3 campaigns
- 6 bid levels
- no-conflict graph
- path graph
- clique graph
- uniform competing bids with optional correlation

## Plots

The runner now produces three explanatory plots, in the same spirit as
Requirement 1:

- cumulative regret
- cumulative cost vs budget line
- campaign activity heatmap

These plots help visualize whether the algorithm is learning good bids,
staying inside budget, and selecting campaigns consistently over time.

## Presentation Summary

- Requirement 1 learns a single optimal bid.
- Requirement 2 learns a **portfolio of campaign-bid choices**.
- The feedback becomes semi-bandit because only selected campaigns reveal outcomes.
- The budget is handled with the same primal-dual intuition already used in Requirement 1.
