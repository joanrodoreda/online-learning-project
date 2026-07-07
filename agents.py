"""
agents.py — Requirement 1: Single Campaign, Stochastic Environment
===================================================================
Implements all bidding agents for Requirement 1.

Shared interface (course convention, notebook 01):
    agent.pull_arm()              → int    (arm / bid index)
    agent.update(r_t, c_t=0.0)   → None   (update statistics)

Agents
------
  RandomBiddingAgent     — uniform random selection           (baseline)
  GreedyBiddingAgent     — greedy exploitation after init     (baseline)
  ETCBiddingAgent        — Explore-Then-Commit                (baseline)
  UCB1BiddingAgent       — UCB1, no budget constraint         (Algorithm 1a)
  BudgetUCB1BiddingAgent — UCB1 + Lagrangian dual variable    (Algorithm 1b)

All agents normalise rewards by v before updating statistics so that
empirical means  μ̂(b) ∈ [0, 1]  and the UCB confidence radius
  sqrt(2 log T / n_b)
is directly applicable without range scaling (notebook 01, UCB1Agent).

Theory references are cited inline with notebook/lecture anchors.
"""

from __future__ import annotations

import numpy as np
from config import UCB_EXPLORATION_FACTOR

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Tuple



# ─────────────────────────────────────────────────────────────────────────────
# 0.  Abstract base class  (course convention)
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    Abstract bidding agent.

    All concrete agents inherit from this class and implement
    pull_arm() and update(), matching the pattern used in all
    10 course notebooks.
    """

    def __init__(self):
        pass

    def pull_arm(self) -> int:
        """Return the index of the chosen bid in bid_set."""
        raise NotImplementedError

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        """
        Update internal statistics after observing feedback.

        Parameters
        ----------
        r_t : float   Realized reward  (v − b_t) · 1[win].
        c_t : float   Realized cost    b_t · 1[win].
                      Ignored by non-budget agents.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Random Agent  (baseline — linear regret expected)
# ─────────────────────────────────────────────────────────────────────────────

class RandomBiddingAgent(Agent):
    """
    Baseline: sample a bid uniformly at random every round.

    Expected pseudo-regret:
        R_T = T · (μ(b*) − mean_b μ(b)) = Θ(T)  [linear]

    Provides the lowest bar for comparison — any sensible learning
    algorithm should do strictly better.

    Source: RandomAgent pattern, notebook 01.

    Parameters
    ----------
    bid_set : (K,) array   Discrete bid set B.
    T       : int          Time horizon (stored for logging only).
    v       : float        Campaign value (accepted for uniform interface).
    """

    label: str = "Random"

    def __init__(self, bid_set: np.ndarray, T: int, v: float, **kwargs):
        super().__init__()
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.K       = len(self.bid_set)
        self.T       = T
        self.v       = v
        self.a_t     = None
        self.t       = 0

        # Logging
        self.N_pulls         = np.zeros(self.K, dtype=int)
        self.average_rewards = np.zeros(self.K, dtype=float)

    def pull_arm(self) -> int:
        self.a_t = int(np.random.randint(0, self.K))
        return self.a_t

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        r_norm = r_t / self.v
        self.N_pulls[self.a_t] += 1
        # Incremental mean  (notebook 01 convention)
        self.average_rewards[self.a_t] += (
            r_norm - self.average_rewards[self.a_t]
        ) / self.N_pulls[self.a_t]
        self.t += 1


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Greedy Agent  (baseline — may have linear regret)
# ─────────────────────────────────────────────────────────────────────────────

class GreedyBiddingAgent(Agent):
    """
    Baseline: pull each arm once (initialization), then exploit the arm
    with the highest empirical mean reward forever.

    Notebook 01 (GreedyAgent) demonstrates that greedy can incur linear
    regret when the initialization phase produces misleading estimates —
    it locks on a suboptimal arm and never corrects.

    Parameters
    ----------
    bid_set : (K,) array
    T       : int
    v       : float
    """

    label: str = "Greedy"

    def __init__(self, bid_set: np.ndarray, T: int, v: float, **kwargs):
        super().__init__()
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.K       = len(self.bid_set)
        self.T       = T
        self.v       = v
        self.a_t     = None
        self.t       = 0

        self.N_pulls         = np.zeros(self.K, dtype=float)
        self.average_rewards = np.zeros(self.K, dtype=float)

    def pull_arm(self) -> int:
        if self.t < self.K:
            # Initialization: pull each arm exactly once (round-robin)
            self.a_t = self.t
        else:
            # Pure exploitation: argmax of empirical means
            self.a_t = int(np.argmax(self.average_rewards))
        return self.a_t

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        r_norm = r_t / self.v
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            r_norm - self.average_rewards[self.a_t]
        ) / self.N_pulls[self.a_t]
        self.t += 1


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Explore-Then-Commit (ETC)  (baseline — O(T^{2/3}) regret)
# ─────────────────────────────────────────────────────────────────────────────

class ETCBiddingAgent(Agent):
    """
    Explore-Then-Commit for the bidding problem.

    Phase 1 — Exploration (rounds 1 … K·T₀):
        Pull each arm T₀ times in round-robin order.

    Phase 2 — Commitment (rounds K·T₀+1 … T):
        Always play  b̂* = argmax_b μ̂(b).

    Optimal T₀ (notebook 01, ETCAgent theory):
        T₀ = ⌊ (T/K)^{2/3} · log(T)^{1/3} ⌋

    Regret bound:
        R_T = O( T^{2/3} · log(T)^{1/3} )        [sub-linear, but worse than UCB1]

    Parameters
    ----------
    bid_set : (K,) array
    T       : int
    v       : float
    T0      : int    Exploration rounds per arm (from config.T0_ETC).
    """

    label: str = "ETC"

    def __init__(
        self,
        bid_set: np.ndarray,
        T:       int,
        v:       float,
        T0:      int,
        **kwargs,
    ):
        super().__init__()
        self.bid_set         = np.asarray(bid_set, dtype=float)
        self.K               = len(self.bid_set)
        self.T               = T
        self.v               = v
        self.T0              = T0
        self.exploration_end = self.K * self.T0   # last exploration round
        self.a_t             = None
        self.t               = 0
        self.committed_arm   = None               # set at end of exploration

        self.N_pulls         = np.zeros(self.K, dtype=float)
        self.average_rewards = np.zeros(self.K, dtype=float)

    def pull_arm(self) -> int:
        if self.t < self.exploration_end:
            # Round-robin: arm index cycles through 0, 1, …, K−1
            self.a_t = int(self.t % self.K)
        else:
            # Commit: choose best arm discovered during exploration
            if self.committed_arm is None:
                self.committed_arm = int(np.argmax(self.average_rewards))
            self.a_t = self.committed_arm
        return self.a_t

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        r_norm = r_t / self.v
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            r_norm - self.average_rewards[self.a_t]
        ) / self.N_pulls[self.a_t]
        self.t += 1


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UCB1 Bidding Agent  (Algorithm 1a — no budget)
# ─────────────────────────────────────────────────────────────────────────────

class UCB1BiddingAgent(Agent):
    """
    UCB1 applied to the single-campaign bidding problem — no budget.

    Direct extension of UCB1Agent from notebook 01.
    Each bid  b ∈ B  is treated as an independent MAB arm whose
    unknown mean reward is  μ(b) = (v − b) · F_D(b).

    UCB formula (notebook 01, Hoeffding bound on [0,1]-bounded rewards):

        UCB(b, t) = μ̂(b) + √( 2 log T / n_b )

    Rewards are normalised to [0,1] by dividing by v so the radius is
    valid.  The factor of 2 is the standard Hoeffding constant.

    Selection:
      • Rounds 0 … K−1: pull each arm once (initialisation, avoids ÷0).
      • Rounds K … T−1: b_t = argmax_b UCB(b, t).

    Regret guarantee (notebook 01 / lecture 2):
        R_T ≤ Σ_{b ≠ b*}  8 log T / Δ_b  +  lower-order terms
           = O(log T)     gap-dependent
           = O(√(KT log T))  worst-case (no gap assumption)

    Parameters
    ----------
    bid_set : (K,) array   Discrete bid set B.
    T       : int          Time horizon (used in log T term).
    v       : float        Campaign value (reward normalisation).
    """

    label: str = "UCB1 (no budget)"

    def __init__(self, bid_set: np.ndarray, T: int, v: float, **kwargs):
        super().__init__()
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.K       = len(self.bid_set)
        self.T       = T
        self.v       = v
        self.a_t     = None
        self.t       = 0

        # Statistics  (names mirror notebook 01 UCB1Agent exactly)
        self.N_pulls         = np.zeros(self.K, dtype=float)
        self.average_rewards = np.zeros(self.K, dtype=float)   # μ̂(b), normalised

    def pull_arm(self) -> int:
        """
        Return arm index with highest UCB value.
        Initialisation phase guarantees N_pulls[a] ≥ 1 before UCB is used.
        """
        if self.t < self.K:
            # Initialisation: pull arm t (= 0, 1, …, K−1)
            self.a_t = self.t
        else:
            # UCB selection  (notebook 01, cell UCB1Agent.pull_arm)
            ucbs = self.average_rewards + np.sqrt(
                UCB_EXPLORATION_FACTOR * np.log(self.T) / self.N_pulls
            )
            self.a_t = int(np.argmax(ucbs))
        return self.a_t

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        """Incremental empirical mean update (notebook 01 convention)."""
        r_norm = r_t / self.v                   # normalise to [0,1]
        self.N_pulls[self.a_t] += 1
        # μ̂_new = μ̂_old + (r − μ̂_old) / n   (online mean)
        self.average_rewards[self.a_t] += (
            r_norm - self.average_rewards[self.a_t]
        ) / self.N_pulls[self.a_t]
        self.t += 1


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Budget-Constrained UCB1 Agent  (Algorithm 1b — Lagrangian dual)
# ─────────────────────────────────────────────────────────────────────────────

class BudgetUCB1BiddingAgent(Agent):
    """
    UCB1 bidding strategy with a budget constraint via Lagrangian duality.

    Extends UCB1BiddingAgent using the primal-dual structure from notebook 08
    (OGDHedgeSingleKnapsackAgent).

    ── Lagrangian Formulation (notebook 08) ────────────────────────────────
    Constrained LP:
        OPT^S = max_{x ∈ Δ(B)}  Σ_b x(b) μ(b)
                s.t.             Σ_b x(b) b  ≤  ρ

    Lagrangian:
        L(x, λ) = Σ_b x(b) [ μ(b) − λ · b ]

    ── Algorithm ────────────────────────────────────────────────────────────
    At each round t:

      PRIMAL step — modified UCB selection:
        modified_UCB(b) = UCB(b) − λ_t · (b / v)
        b_t = argmax_b  modified_UCB(b)

      DUAL step — OGD on λ  (notebook 08, OGDHedgeSingleKnapsackAgent.update):
        λ_{t+1} = clip( λ_t + η · (c_t − ρ),  0,  1/ρ )

    Interpretation of λ_t:
      • λ_t ↑ when c_t > ρ  (over budget) → penalises expensive bids
      • λ_t ↓ when c_t < ρ  (under budget) → allows expensive bids
      • Equilibrium: λ* is the optimal dual variable from the LP.

    ── Reward/cost normalisation ────────────────────────────────────────────
    Rewards are normalised by v  →  μ̂(b) ∈ [0,1].
    Bids    are normalised by v  →  penalty  λ·(b/v) ∈ [0, λ_max].
    Since v = 1 in our experiments this is a no-op, but keeps the code
    general for v ≠ 1.

    ── Budget guard ─────────────────────────────────────────────────────────
    When budget_remaining < ε_guard, the agent returns arm index 0
    (bid = 0.0), incurring zero cost and essentially zero reward.
    This prevents overspend after budget exhaustion.

    Parameters
    ----------
    bid_set : (K,) array   Discrete bid set B.
    T       : int          Time horizon.
    v       : float        Campaign value.
    rho     : float        Per-round budget ρ = B/T.
    eta     : float|None   OGD step size η.  Default = 1/√T  (theory).
    """

    def __init__(
        self,
        bid_set: np.ndarray,
        T:       int,
        v:       float,
        rho:     float,
        eta:     float = None,
        **kwargs,
    ):
        super().__init__()
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.K       = len(self.bid_set)
        self.T       = T
        self.v       = v
        self.rho     = float(rho)
        self.eta     = float(eta) if eta is not None else 1.0 / np.sqrt(T)

        # Label dynamically encodes ρ for plot legend
        self.label = f"Budget-UCB1 (ρ={rho})"

        self.a_t = None
        self.t   = 0

        # UCB statistics  (same as UCB1BiddingAgent)
        self.N_pulls         = np.zeros(self.K, dtype=float)
        self.average_rewards = np.zeros(self.K, dtype=float)   # μ̂(b), normalised

        # Dual variable  λ_t  and budget tracker
        self.lmbd             = 0.0            # λ_0 = 0 (start unconstrained)
        self.budget_remaining = self.rho * T   # B_remaining = B − Σ_s c_s
        self._budget_guard    = 1e-6           # threshold for "exhausted"

        # History arrays (pre-allocated for efficiency; used in plotting)
        self.lmbd_history    = np.zeros(T, dtype=float)
        self.budget_history  = np.zeros(T, dtype=float)
        self._history_filled = 0

    # ── UCB selection ───────────────────────────────────────────────────────

    def pull_arm(self) -> int:
        """
        Return the arm index maximising the budget-penalised UCB:

            modified_UCB(b) = UCB(b) − λ_t · (b/v)

        Steps (in order):
          1. Log current λ_t and budget_remaining.
          2. If budget exhausted → opt-out (arm 0).
          3. If t < K → initialisation phase (round-robin).
          4. Otherwise → modified UCB argmax.
        """
        # ── Log λ_t BEFORE the decision ─────────────────────────────────────
        idx = self._history_filled
        self.lmbd_history[idx]   = self.lmbd
        self.budget_history[idx] = self.budget_remaining
        self._history_filled += 1

        # ── Budget guard ─────────────────────────────────────────────────────
        if self.budget_remaining < self._budget_guard:
            self.a_t = 0    # bid = 0.0, cost ≈ 0, reward ≈ 0
            return self.a_t

        # ── Initialisation phase ─────────────────────────────────────────────
        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        # ── Modified UCB selection ───────────────────────────────────────────
        ucbs = self.average_rewards + np.sqrt(
            UCB_EXPLORATION_FACTOR * np.log(self.T) / self.N_pulls
        )
        # Lagrangian penalty: normalise bid by v so penalty lives in [0, 1]
        penalty  = self.lmbd * (self.bid_set / self.v)
        modified = ucbs - penalty
        self.a_t = int(np.argmax(modified))
        return self.a_t

    # ── Update step ─────────────────────────────────────────────────────────

    def update(self, r_t: float, c_t: float = 0.0) -> None:
        """
        Primal update: incremental mean for UCB statistics.
        Dual   update: OGD step on λ  (notebook 08 formula).
        Budget update: subtract realized cost from remaining budget.

        Dual OGD step (notebook 08, OGDHedgeSingleKnapsackAgent.update):
            λ_{t+1} = clip( λ_t + η·(c_t − ρ),  0,  1/ρ )

        Note: clip upper bound is 1/ρ, not ∞ — ensures λ is bounded and
        the Lagrangian  L(x,λ)  remains well-behaved  (notebook 08).
        """
        # ── Primal: UCB statistics ────────────────────────────────────────────
        r_norm = r_t / self.v
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            r_norm - self.average_rewards[self.a_t]
        ) / self.N_pulls[self.a_t]

        # ── Dual: OGD on λ  (notebook 08) ────────────────────────────────────
        self.lmbd = float(np.clip(
            self.lmbd + self.eta * (c_t - self.rho),
            a_min=0.0,
            a_max=1.0 / self.rho,
        ))

        # ── Budget accounting ─────────────────────────────────────────────────
        self.budget_remaining -= c_t

        self.t += 1

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def lmbd_trajectory(self) -> np.ndarray:
        """Return only the filled portion of lmbd_history."""
        return self.lmbd_history[: self._history_filled]

    @property
    def budget_trajectory(self) -> np.ndarray:
        """Return only the filled portion of budget_history."""
        return self.budget_history[: self._history_filled]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Registry  — maps string names to constructors
#     Used by main.py to build experiment configurations declaratively.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: dict = {
    "Random":    RandomBiddingAgent,
    "Greedy":    GreedyBiddingAgent,
    "ETC":       ETCBiddingAgent,
    "UCB1":      UCB1BiddingAgent,
    "BudgetUCB1": BudgetUCB1BiddingAgent,
}
"""
agents_req2.py — Requirement 2: Multiple Campaigns, Combinatorial UCB
======================================================================
Implements a small, readable Combinatorial-UCB agent for multiple campaigns
with a budget constraint.
"""


class CombinatorialAgent:
    def __init__(self):
        pass

    def pull_action(self):
        raise NotImplementedError

    def update(self, reward_vec, cost_vec):
        raise NotImplementedError


def enumerate_independent_sets(conflict_graph: np.ndarray) -> List[Tuple[int, ...]]:
    """
    Return every subset of campaigns that does not contain a conflict.

    This is the "combinatorial" part of the problem: we do not choose a
    single arm, but a whole set of campaigns that can be played together.
    For small N, brute-force enumeration is the clearest way to do it.
    """
    n = conflict_graph.shape[0]
    feasible = [tuple()]
    for r in range(1, n + 1):
        # itertools.combinations gives all subsets of size r.
        for subset in itertools.combinations(range(n), r):
            ok = True
            # Check that no pair inside the subset is connected by an edge.
            for i, u in enumerate(subset):
                for v in subset[i + 1 :]:
                    if conflict_graph[u, v] or conflict_graph[v, u]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                feasible.append(subset)
    return feasible


class CombinatorialUCB1BiddingAgent(CombinatorialAgent):
    """
    Combinatorial UCB with a budget constraint.

    Action:
      - choose an independent set of campaigns
      - choose one bid per selected campaign
      - unselected campaigns receive bid 0
    """

    label: str = "Combinatorial-UCB"

    def __init__(
        self,
        bid_set: np.ndarray,
        values: Sequence[float],
        conflict_graph: np.ndarray,
        T: int,
        rho: float,
        eta: float = None,
    ):
        super().__init__()
        # Store the problem instance in numpy form for vectorized operations.
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.n_campaigns = len(self.values)
        self.K = len(self.bid_set)
        self.conflict_graph = np.asarray(conflict_graph, dtype=int)
        self.T = int(T)
        self.rho = float(rho)
        self.eta = float(eta) if eta is not None else 1.0 / np.sqrt(T)

        # λ_t is the dual variable: if we overspend, it goes up and makes
        # expensive actions less attractive.
        self.lambda_t = 0.0
        # Remaining budget in "real" units, used as a guardrail.
        self.budget_remaining = self.rho * T
        self.t = 0

        # N[i, b] = how many times campaign i was played with bid b.
        self.N = np.zeros((self.n_campaigns, self.K), dtype=float)
        # Empirical mean reward for each campaign-bid pair.
        self.mean_rewards = np.zeros((self.n_campaigns, self.K), dtype=float)
        # Empirical mean cost for each campaign-bid pair.
        self.mean_costs = np.zeros((self.n_campaigns, self.K), dtype=float)
        # Save the last action so update() knows what to update.
        self.last_bids = np.zeros(self.n_campaigns, dtype=float)
        self.last_active = np.zeros(self.n_campaigns, dtype=bool)
        # Precompute all feasible subsets once.
        self.independent_sets = enumerate_independent_sets(self.conflict_graph)

        # Trajectories useful for debugging and plotting.
        self.lambda_history = np.zeros(self.T, dtype=float)
        self.budget_history = np.zeros(self.T, dtype=float)

    def _ucb_reward(self) -> np.ndarray:
        """
        Optimistic estimate of the reward.

        mean + confidence_radius.
        This is the standard UCB trick: be optimistic about uncertain pairs.
        """
        return self.mean_rewards + np.sqrt(
            2.0 * np.log(max(self.T, 2)) / np.maximum(self.N, 1.0)
        )

    def _ucb_cost(self) -> np.ndarray:
        """
        Optimistic estimate of the cost.

        We keep a UCB for cost as well, because the budget penalty should be
        applied to an optimistic estimate of spending, not only to reward.
        """
        return self.mean_costs + np.sqrt(
            2.0 * np.log(max(self.T, 2)) / np.maximum(self.N, 1.0)
        )

    def pull_action(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Choose the next action.

        Returns:
            bids:   one bid per campaign
            active:  True for campaigns we actually play
        """
        # Store the state before choosing the next action.
        if self.t < self.T:
            self.lambda_history[self.t] = self.lambda_t
            self.budget_history[self.t] = self.budget_remaining

        # If budget is gone, stop bidding altogether.
        if self.budget_remaining <= 1e-9:
            self.last_bids[:] = 0.0
            self.last_active[:] = False
            return self.last_bids.copy(), self.last_active.copy()

        # Very early rounds: play each campaign once so we collect an initial
        # sample before trusting UCB.
        if self.t < self.n_campaigns:
            bids = np.zeros(self.n_campaigns, dtype=float)
            active = np.zeros(self.n_campaigns, dtype=bool)
            # We start with the second bid in the grid when possible, just to
            # avoid the trivial zero-bid opt-out during initialization.
            bids[self.t] = self.bid_set[min(1, self.K - 1)]
            active[self.t] = True
            self.last_bids = bids
            self.last_active = active
            return bids.copy(), active.copy()

        # Build optimistic tables for reward and cost.
        ucb_r = self._ucb_reward()
        ucb_c = self._ucb_cost()

        # We search over all independent sets and keep the one with best score.
        best_score = -np.inf
        best_bids = np.zeros(self.n_campaigns, dtype=float)
        best_active = np.zeros(self.n_campaigns, dtype=bool)

        for subset in self.independent_sets:
            # Start from an empty action and fill only the campaigns in subset.
            bids = np.zeros(self.n_campaigns, dtype=float)#bid to assign to each campaign
            active = np.zeros(self.n_campaigns, dtype=bool)
            score = 0.0
            for i in subset:
                # For campaign i, evaluate every bid:
                # optimistic reward - λ * optimistic cost.
                bid_scores = ucb_r[i] - self.lambda_t * ucb_c[i]#ignoring term+lambda rho, since its constant
                b_idx = int(np.argmax(bid_scores))
                bids[i] = self.bid_set[b_idx]
                active[i] = True
                score += float(bid_scores[b_idx])

            # Keep the highest-scoring feasible subset.
            if score > best_score:
                best_score = score
                best_bids = bids
                best_active = active

        self.last_bids = best_bids
        self.last_active = best_active
        return best_bids.copy(), best_active.copy()

    def update(self, reward_vec, cost_vec):
        """
        Update empirical means after observing the realized feedback.

        Only the campaigns that were active in the chosen subset are updated.
        This is exactly the semi-bandit idea: we learn from the campaigns we
        actually played, not from the ones we skipped.
        """
        reward_vec = np.asarray(reward_vec, dtype=float)
        cost_vec = np.asarray(cost_vec, dtype=float)

        # Update the statistics of the selected campaign-bid pairs.
        for i in np.where(self.last_active)[0]:
            b = self.last_bids[i]
            # Map the floating bid back to an index in bid_set.
            b_idx = int(np.argmin(np.abs(self.bid_set - b)))
            self.N[i, b_idx] += 1.0

            # Normalize by campaign value so rewards live roughly in [0,1].
            self.mean_rewards[i, b_idx] += (
                reward_vec[i] / max(self.values[i], 1e-12) - self.mean_rewards[i, b_idx]
            ) / self.N[i, b_idx]

            # Same normalization idea for cost, so reward and cost are on
            # comparable scales inside the Lagrangian.
            self.mean_costs[i, b_idx] += (
                cost_vec[i] / max(self.values[i], 1e-12) - self.mean_costs[i, b_idx]
            ) / self.N[i, b_idx]

        # Budget bookkeeping: subtract the realized cost from the remaining budget.
        total_cost = float(cost_vec.sum())
        self.budget_remaining -= total_cost

        # Dual update:
        # - if we spent more than the per-round budget rho, λ increases
        # - otherwise λ decreases or stays small
        self.lambda_t = float(np.clip(
            self.lambda_t + self.eta * (total_cost - self.rho),
            0.0,
            1.0 / max(self.rho, 1e-12),
        ))
        self.t += 1

    def get_histories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the stored lambda/budget trajectories for the current trial."""
        return self.lambda_history.copy(), self.budget_history.copy()
