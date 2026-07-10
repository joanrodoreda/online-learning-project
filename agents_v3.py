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
from config_v3 import UCB_EXPLORATION_FACTOR

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

# "EMPTY" FATHER CLASS 
# the class exists in order to esnure that any class that inherits from it needs to have the pull_action and update method.
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


# ═════════════════════════════════════════════════════════════════════════════
# REQUIREMENT 3 — Best-of-both-worlds primal-dual agent (Hedge + OGD dual)
# ═════════════════════════════════════════════════════════════════════════════
# Everything below is ADDITIVE.  Requirement 1 and 2 code above is untouched.


class PrimalDualHedgeBiddingAgent(CombinatorialAgent): #CHILD CLASS OF CombinatorialAgent
    """
    Best-of-both-worlds primal-dual bidding strategy (Requirement 3).

    ###### MOTIVATING HEDGE TO SOLVE PRIMAL PROBLEM FOR EACH CAMPAIGN
    UCB's guarantees need i.i.d. rewards.  Under FULL FEEDBACK 
    we observe every campaign's highest competing bid m_{i,t}, so we can
    compute what reward/cost EVERY bid would have produced:
        f_i(b) = (v_i − b)·1[b ≥ m_{i,t}]      c_i(b) = b·1[b ≥ m_{i,t}]
    That turns the primal problem into a full-information expert problem, hence
    the natural regret minimizer is Hedge (multiplicative weights), which is
    O(√(T log K)) against ANY sequence (adversarial or stochastic). Combined
    the OGD on the dual on the budget: on the lambda of the Lagrangian game 
    that controls how strongly the per-round budget constraint needs to be respected.
    This is the primal-dual template (agent) that is repated for each campaign
    in the problem. 
    #IN SHORT: YOU HAVE A LAGRANGIAN GAME (PRIMAL-DUAL PROBLEM) FOR EACH CAMPAIGN, 
               WHERE THE DUAL IS IN COMMON TO ALL CAMPAIGNS, SINCE SINGLE BUDGET FOR
               ALL CAMPAIGNS. YOU INSTANTIATE A HEDGE AGENT TO SOLVE THE PRIMAL FOR 
               EACH CAMPAIGN (WORKING WITH FULL FEEDBACK) AND THE DUAL PROBLEM IS SOLVED
               BY AN "ADVERSARY" GRADIENT DESCENT THAT ADJUST LAMBDA BY LOOKING AT THE COSTS
               INCURRED BY ALL CAMPAIGNS COMPARING IT TO THE PER-ROUND BUDGET. 

    ###### Algorithm 
    State: one Hedge weight vector w_i ∈ R^K per campaign, shared dual λ_t.
    - where K is the number of possible bids (discretized set)

    Each round t:
      PRIMAL —
        1. p_i = w_i / Σ w_i for each campaign (Hedge distributions).
        2. Score each campaign by its expected Lagrangian gain under p_i
           (vector of running average of each bid x probability distribution p_i); 
           pick the independent set S of the conflict graph maximizing Σ_{i∈S} score_i, provided
           score_i > opt-out value (opt-out value = 0, we consider a campaign 
           if its score is better than not participating in the campaign).
        3. For each i ∈ S sample b_i ~ p_i  (randomization is REQUIRED
           against an adversary — deterministic play is exploitable).
      OBSERVE — full feedback m_t (all campaigns, active or not).
      DUAL / UPDATE —
        4. Compute gain of every (i, b):
              g_i(b) = [ f_i(b) − λ_t · c_i(b) ]  normalized to [0, 1]
        5. Hedge update for ALL campaigns:  w_i(b) ← w_i(b)·exp(γ·g_i(b)).
        6. OGD dual:  λ ← clip( λ + η·(Σ_i c_{i,t} − ρ), 0, λ_max ).
        7. Hard budget guard: once budget is exhausted, opt out entirely from all campaigns

    ##### Normalisation note 
    Theory allows λ_max = 1/ρ, but the Hedge gains must be mapped into [0,1];
    a large λ_max compresses the reward signal in that mapping and slows
    learning.  EXPLAINED IN config.py FILE FOR THE HYPERPARAM CONFIG FOR REQ3.

    ##### INPUT PARAMETERS
    ----------
    bid_set        : (K,) array   Discrete bid set (must contain 0.0 = opt-out).
    values         : (N,) array   Campaign values v_i.
    conflict_graph : (N, N) array 0/1 adjacency of non-compatible campaigns.
    T              : int          Horizon.
    rho            : float        Per-round budget ρ = B/T (shared, global).
    eta_dual       : float|None   OGD step. Default 1/√T (theory).
    eta_hedge      : float|None   Hedge rate. Default √(log K / regime_length) (theory).
    lambda_max     : float        Cap on λ (see note). Default 1.0.
    regime_length  : int|None     Length of regime for non-stationary environments. 
                                   If None, defaults to T (full horizon).
    """
    
    #class attribute: a label attached to every instance of agent used purely for labelling
    label: str = "Primal-Dual (Hedge)"

    def __init__( 
        self,
        bid_set: np.ndarray,
        values: Sequence[float],
        conflict_graph: np.ndarray,
        T: int,
        rho: float,
        eta_dual: float = None,
        eta_hedge: float = None,
        lambda_max: float = 1.0,
        regime_length: int = None,
    ): #CONSTRUCTOR
        
        #call the constructor of the parent class (CombinatorialAgent) which is empty
        #create one attribute per passed input parameter
        super().__init__()
        self.bid_set = np.asarray(bid_set, dtype=float) #set of possible bids (discretized)
        self.values = np.asarray(values, dtype=float) #array containing value of each campaign
        self.n_campaigns = len(self.values) #number of campaigns (N)
        self.K = len(self.bid_set) #number of possible bids (K)
        self.conflict_graph = np.asarray(conflict_graph, dtype=int) #conflict graph indicating non-compatible campaigns #2d array of shape (N, N) with 0/1 entries
        self.T = int(T) #time horizon (number of rounds)
        self.rho = float(rho) #per-round budget (B/T)
        self.regime_length = int(regime_length) if regime_length is not None else self.T #regime length for non-stationary environments, defaults to T
        
        self.eta_dual = float(eta_dual) if eta_dual is not None else 1.0 / np.sqrt(T) #use passed eta, otherwise default to 1/sqrt(T)
        self.eta_hedge = (float(eta_hedge) if eta_hedge is not None
                          else np.sqrt(np.log(self.K) / self.regime_length)) #use passed eta, otherwise default to sqrt(log K / regime_length) - uses regime length instead of full time horizon T
        self.lambda_max = float(lambda_max) #cap on the dual variable λ (see note)

        # PRIMAL PROBLEM STATE: 
        # initialize Hedge weights table of shape (N, K) - one entry corresponds to the log weight for a campaign and an available bid
        self.log_weights = np.zeros((self.n_campaigns, self.K), dtype=float) #all initialized to 0, so initial distribution over bids is uniform for all campaigns (softmax of 0 is uniform)
        # initialize second table of shape (N, K) to store the running average (sample average on observed rounds) of the Lagrangian gain (reward) for each campaign and bid
        self.avg_gain = np.zeros((self.n_campaigns, self.K), dtype=float) #initialized all to 0, so initially the average gain is 0 for all bids with respect to all campaigns

        # DUAL PROBLEM STATE:
        self.lambda_t = 0.0 #initialize dual variable λ_t to 0 (Lagrangian game of each campaign starts unconstrained)
        #we start with no penalty on over-spending, since we haven't spent anything yet. 
        # The dual variable will be updated in the update() method based on the observed costs and the per-round budget rho.
        self.budget_remaining = self.rho * T #initialize budget remaining to total budget (B = ρ·T). This will be decremented in update() as costs are incurred.

        self.t = 0 #initialize round counter to 0 (will be incremented in update() after each round)
        self.last_bids = np.zeros(self.n_campaigns, dtype=float) #initialize last selected bids array (it contains last bid for each campaign)
        self.last_active = np.zeros(self.n_campaigns, dtype=bool)#initialize last active campaigns array (it contains True for campaigns that were selected in the last round, False otherwise)

        #Precompute feasible subsets (same helper as Requirement 2).
        self.independent_sets = enumerate_independent_sets(self.conflict_graph)

        #Trajectories for plotting.
        self.lambda_history = np.zeros(self.T, dtype=float) #array with one entry per round to store the value of λ_t at each round
        self.budget_history = np.zeros(self.T, dtype=float) #array with one entry per round to store the value of budget_remaining at each round

    #### HELPER METHOD
    # this method has the goal to turn the hedge weights into probabilities
    def _distributions(self) -> np.ndarray: #it returns a matrix of probabilities of shape (N, K) where each row represents the probability distribution over bids for a campaign
        """Row-wise softmax of log-weights → (N, K) Hedge distributions."""
        #numerical stability trick that subtracts the max log weight from the log weights of each row to avoid overflow in the exponentiation
        #this doesn't change the resulting probability distribution since softmax is invariant to constant shifts in the input
        #the biggest number in eeach row becomes exactly 0.
        lw = self.log_weights - self.log_weights.max(axis=1, keepdims=True) 
        w = np.exp(lw) #converts the log weights into actual weights by exponentiating them
        #compute the row-wise softmax by dividing each weight by the sum of weights in its row, 
        #resulting in a valid probability distribution for each campaign over the available bids
        return w / w.sum(axis=1, keepdims=True) 

    
    #### METHOD THAT DETERMINS WHICH ARM IS PULLED IN THE CURRENT ROUND
    def pull_action(self) -> Tuple[np.ndarray, np.ndarray]:
        #verifies if within the time horizon
        if self.t < self.T:
            self.lambda_history[self.t] = self.lambda_t #store the current value of the dual variable λ_t in the history array for plotting 
            self.budget_history[self.t] = self.budget_remaining #store the current value of the remaining budget in the history array for plotting 

        # Hard budget guard: opt out entirely once the budget is exhausted.
        # in the case the budget is under the indicated thrshold below we opt-out of all campaigns
        if self.budget_remaining <= 1e-9:
            self.last_bids[:] = 0.0
            self.last_active[:] = False
            return self.last_bids.copy(), self.last_active.copy()

        p = self._distributions() # use distribution helper method to compute the probability distribution over the bids for each campagin - matrix (N, K)

        # Expected Lagrangian gain of activating each campaign under its own
        # Hedge distribution.  Baseline 0 = opting out (bid 0 → no gain).
        # multiply every row of p by the corresponding row of avg_gain. Then sum the elements on the row of the resulting matrix
        #this get the expected Lagrangian gain (reward) for each campaign under its own current Hedge distribution.
        camp_score = (p * self.avg_gain).sum(axis=1)    # (N,) - vector of N expected Lagrangian gains- one for each campaign

        # Choose the best compatible set of arms to be activated, based on the expected Lagrangian gain. 
        clipped = np.maximum(camp_score, 0.0) #campaigns with negative expected gain are clipped to 0, we only consider campaigns with non-negative expected gain for selection
        best_subset, best_val = tuple(), -np.inf #we initialize the two variables that will store the best independent (compatible) set of campaigns and its corresponding expected gain value
        for subset in self.independent_sets: #iterate over all feasible independent sets of campaigns (compatible sets)
            val = float(sum(clipped[i] for i in subset)) #we sum the expected gains of the campaigns in the current subset to get the total expected gain for that subset
            if val > best_val: #if the total expected gain of the current subset is better than the best found so far, we update the best found values  
                best_val = val 
                best_subset = subset

        # Once we have decided the best set of campaigns to activate, we sample a bid for each of them according to their own Hedge distribution.
        # campaigns that are not in the best subset will have bid 0 and will be inactive
        bids = np.zeros(self.n_campaigns, dtype=float) #initialize vector containing bids we will choose
        active = np.zeros(self.n_campaigns, dtype=bool) #initialize vector containing which campaigns will be active (True) or inactive (False)
        
        for i in best_subset: #for each campaign in the best independent set previously indentified
            if camp_score[i] <= 0.0 and self.t >= self.K:
                continue                                #additional secruity check controlling that we are not considering a campagin with negative expected gain (should not happen)
            b_idx = int(np.random.choice(self.K, p=p[i])) #sample a bid index from the probability distribution over bids for campaign i (row i of p) using numpy's random choice function
            bids[i] = self.bid_set[b_idx] #store the bid
            active[i] = True #set active to true for that campagin

        self.last_bids = bids #store the vector of selected vids 
        self.last_active = active #store the vector of active campaigns
        return bids.copy(), active.copy() #return a copy of both

    
    ##### METHOD THAT UPDATES THE AGENT'S INTERNAL STATE AFTER OBSERVING THE REALIZED REWARDS AND COSTS
    def update(self, reward_vec, cost_vec, m_t) -> None:
        """
        Full-feedback update.

        Parameters
        ----------
        reward_vec : (N,) realized rewards (only active campaigns non-zero).
        cost_vec   : (N,) realized costs.
        m_t        : (N,) OBSERVED highest competing bids — full feedback,
                     available for every campaign 
        """
        #we convert the vector of highest competing bids at round t of each campaign into a numpy array of floats
        m_t = np.asarray(m_t, dtype=float)

        ##### Compute the Lagrangian gain for every bid under every campaign, using the full feedback m_t.
        # f_i(b) = (v_i − b)·1[b ≥ m_{i,t}]      c_i(b) = b·1[b ≥ m_{i,t}]
        # g_i(b) = f_i(b) − λ_t · c_i(b)
        #we compute a boolean matrix of shape (N, K) where each entry indicates whether 
        #the bid b is greater than or equal to the observed highest competing bid m_t for campaign i
        #Therefore position i,j is True if bid j is greater than or equal to the observed highest competing bid for campaign i, and False otherwise
        #broadcasting is used
        wins = self.bid_set[None, :] >= m_t[:, None]                # (N, K) #compare the set of possible bids [1xK] with the highest competing bids of each campagin [Nx1]
        f = (self.values[:, None] - self.bid_set[None, :]) * wins   # reward - compute reward for each possible bid for each campaign and then multiply by wins to obtain a (N,K) reward matrix
        c = self.bid_set[None, :] * wins                            # cost - compute cost for each possible bid for each campaign and then multiply by wins to obtain a (N,K) cost matrix
        lagr = f - self.lambda_t * c                                # gain - compute the Lagrangian gain for each possible bid for each campaign by subtracting the cost term 
        #    weighted by the dual variable λ_t from the reward term

        ####### Normalize to [0, 1] since hedge requires gains (rewards) in [0, 1]. 
        #Use the following fomrula to map the Lagrangian gain to [0,1] range (Hedge normalized Gains)
        #    lagr ∈ [−λ_max·v_i,  v_i]  →  (lagr + λ_max·v_i) / (v_i(1+λ_max))
        v = self.values[:, None]
        gain = (lagr + self.lambda_max * v) / (v * (1.0 + self.lambda_max))
        gain = np.clip(gain, 0.0, 1.0)

        ####### Primal update — update of Hedge weights for ALL campaigns (this is the full-feedback benefit).
        self.log_weights += self.eta_hedge * gain #we are updating the log weights hence we have a sum and not multiplication

        ### Upate the Running average gain of each bid for each campaign, centered around the zero-gain level.
        # In this way, the average gain is always in [-1, 1] and Hedge sees a "balanced" signal.
        # if average gain is positive, it means that the bid is performing better than the zero-gain level, better than not bidding.
        zero_level = self.lambda_max / (1.0 + self.lambda_max)
        self.avg_gain += ((gain - zero_level) - self.avg_gain) / (self.t + 1) #classical incremental update formula of the average (centered on the zero-gain level)

        ###### DUAL OGD STEP  - update lambda shared by all campaigns.  This is the "adversary" in the primal-dual game.
        # λ_{t+1} = clip( λ_t + η·(Σ_i c_{i,t} − ρ), 0, λ_max )
        total_cost = float(np.asarray(cost_vec, dtype=float).sum()) #total cost incurred in the current round by all campaigns (sum cost of all campaigns)
        self.lambda_t = float(np.clip(
            self.lambda_t + self.eta_dual * (total_cost - self.rho),
            0.0,
            self.lambda_max,
        )) #update lambda and clip between 0 and lambda_max (1), already explained why.

        ###### update budget remaining and increment round counter
        self.budget_remaining -= total_cost
        self.t += 1

    ###### METHOD THAT RETURNS THE HISTORIES OF LAMBDA AND BUDGET FOR PLOTTING PURPOSES
    def get_histories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the stored λ / budget trajectories (same API as Req 2)."""
        return self.lambda_history.copy(), self.budget_history.copy()
