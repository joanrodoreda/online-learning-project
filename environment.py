"""
environment.py — Requirement 1: Single Campaign, Stochastic Environment
========================================================================
Provides:
  • SingleCampaignEnv      — first-price auction simulator (stochastic)
  • compute_true_arm_means — analytical μ(b) = (v−b)·F_D(b)  (clairvoyant only)
  • compute_clairvoyant    — optimal value via LP  (notebook 08 pattern)

Design principles (enforced throughout):
  • All competing bids are PRE-GENERATED at construction time using whatever
    seed was set by the caller.  This mirrors BernoulliEnvironment in notebook
    01 and guarantees that every agent sees the SAME noise sequence within a
    trial, making regret comparisons fair.
  • The environment is stateless between trials; create a fresh instance per
    trial rather than resetting.
  • rewards ∈ [0, v],  costs ∈ [0, v].
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as beta_dist, norm as norm_dist


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Abstract base class  (course convention, notebook 01)
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """Minimal abstract base.  Subclasses implement round()."""

    def __init__(self):
        pass

    def round(self, action):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single-Campaign Stochastic Environment
# ─────────────────────────────────────────────────────────────────────────────

class SingleCampaignEnv(Environment):
    """
    Stochastic first-price auction for a SINGLE campaign.

    At each round t ∈ [T]:
      1. The competing bid  m_t ~ D  is drawn i.i.d. from distribution D.
      2. The agency places bid  b_t ∈ B.
      3. Win condition: b_t ≥ m_t.
      4. Reward:  r_t = (v − b_t) · 1[win]   ∈ [0, v]
         Cost:    c_t =       b_t  · 1[win]   ∈ [0, v]

    All T competing bids are pre-generated in __init__ so that multiple
    agents can be evaluated on IDENTICAL noise within the same trial
    (same design as BernoulliEnvironment, notebook 01).

    Parameters
    ----------
    v           : float  Campaign value (utility of winning the slot).
    bid_set     : array  Discrete bid set  B ⊆ [0, v].
    T           : int    Time horizon.
    dist_config : dict   Competing bid distribution specification.
                         {'type': 'uniform'|'beta'|'normal', ...params}
    """

    def __init__(
        self,
        v:           float,
        bid_set:     np.ndarray,
        T:           int,
        dist_config: dict,
    ):
        super().__init__()
        self.v           = float(v)
        self.bid_set     = np.asarray(bid_set, dtype=float)
        self.K           = len(self.bid_set)
        self.T           = int(T)
        self.dist_config = dist_config
        self.t           = 0  # current round counter

        # Pre-generate ALL competing bids for this trial.
        # np.random.seed() MUST be called by the caller beforehand.
        self.competing_bids: np.ndarray = self._sample_competing_bids(self.T)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _sample_competing_bids(self, n: int) -> np.ndarray:
        """
        Draw n i.i.d. samples from the competing bid distribution D.
        Results are clipped to [0, v] to handle truncated Normal.
        """
        cfg = self.dist_config
        kind = cfg["type"]

        if kind == "uniform":
            samples = np.random.uniform(cfg["low"], cfg["high"], size=n)

        elif kind == "beta":
            # m_t ~ Beta(a, b) scaled to [0, v]
            samples = np.random.beta(cfg["a"], cfg["b"], size=n) * self.v

        elif kind == "normal":
            samples = np.random.normal(cfg["loc"], cfg["scale"], size=n)

        else:
            raise ValueError(
                f"Unknown distribution type '{kind}'. "
                "Supported: 'uniform', 'beta', 'normal'."
            )

        return np.clip(samples, 0.0, self.v).astype(float)

    # ── Public API ──────────────────────────────────────────────────────────

    def round(self, b_t: float):
        """
        Execute one round of the first-price auction.

        Parameters
        ----------
        b_t : float   The bid placed by the agency this round (bid VALUE,
                      not index).  The experiment runner maps arm index →
                      bid value using bid_set before calling this method.

        Returns
        -------
        win_t : bool    True iff b_t ≥ m_t  (won the auction).
        r_t   : float   Reward = (v − b_t) · win_t   ∈ [0, v].
        c_t   : float   Cost   =        b_t · win_t   ∈ [0, v].

        Note: the competing bid m_t is NOT returned.  Per the project
        specification (slide 6), agents only observe the set of won auctions,
        not the exact competing bid value.
        """
        if self.t >= self.T:
            raise RuntimeError(
                f"Environment exhausted: round() called {self.t + 1} times "
                f"but T = {self.T}."
            )

        m_t   = float(self.competing_bids[self.t])
        win_t = bool(b_t >= m_t)
        r_t   = float((self.v - b_t) * win_t)
        c_t   = float(b_t            * win_t)
        self.t += 1
        # Return only bandit feedback (project spec, slide 6: "set of won
        # auctions").  m_t is intentionally withheld from callers — it is not
        # observable in a first-price auction.
        return win_t, r_t, c_t

    def reset(self):
        """Reset round counter.  Does NOT re-sample competing bids."""
        self.t = 0

    @property
    def rounds_remaining(self) -> int:
        return self.T - self.t


# ─────────────────────────────────────────────────────────────────────────────
# 2.  True arm means  μ(b) = (v − b) · F_D(b)
#     Only used by the CLAIRVOYANT — agents never access this.
# ─────────────────────────────────────────────────────────────────────────────

def compute_true_arm_means(
    bid_set:     np.ndarray,
    v:           float,
    dist_config: dict,
) -> np.ndarray:
    """
    Compute the true expected reward per bid analytically:

        μ(b) = (v − b) · F_D(b)

    where F_D is the exact CDF of the competing bid distribution D.

    This is the KEY quantity in the MAB reduction: each bid b ∈ B is an
    arm whose mean reward μ(b) is unknown to the agent but derivable by
    a clairvoyant that knows D.

    The function  μ(b)  is NOT monotone — it is zero at b = 0  (win
    probability zero) and at b = v  (zero profit), with a maximum
    somewhere in between.  That maximum defines the optimal bid b*.

    Parameters
    ----------
    bid_set     : (K,) array   Discrete bid set.
    v           : float        Campaign value.
    dist_config : dict         Same spec dict used by SingleCampaignEnv.

    Returns
    -------
    mu : (K,) array   True expected reward for each bid in bid_set.
    """
    bids = np.asarray(bid_set, dtype=float)
    cfg  = dist_config
    kind = cfg["type"]

    if kind == "uniform":
        lo, hi = cfg["low"], cfg["high"]
        # F_D(b) = (b − lo) / (hi − lo),  clipped to [0, 1]
        F = np.clip((bids - lo) / (hi - lo), 0.0, 1.0)

    elif kind == "beta":
        # m_t = Beta(a, b) · v  →  P(m_t ≤ b) = P(Beta(a,b) ≤ b/v)
        F = beta_dist.cdf(bids / v, cfg["a"], cfg["b"])

    elif kind == "normal":
        F = norm_dist.cdf(bids, loc=cfg["loc"], scale=cfg["scale"])
        F = np.clip(F, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown distribution type '{kind}'.")

    mu = (v - bids) * F
    return mu.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Clairvoyant computation via Linear Programming
#     Replicates compute_clairvoyant() from notebook 08 exactly.
# ─────────────────────────────────────────────────────────────────────────────

def compute_clairvoyant(
    bid_set:     np.ndarray,
    v:           float,
    dist_config: dict,
    rho:         float = None,
):
    """
    Compute the optimal bidding strategy for a single campaign.

    ── Without budget (rho = None) ─────────────────────────────────────────
        OPT = max_{b ∈ B}  μ(b)          (best pure strategy)
        Returns the arm index of b* and μ(b*).

    ── With budget (rho given) ──────────────────────────────────────────────
        OPT^S = max_{x ∈ Δ(B)}   Σ_b x(b) μ(b)
                s.t.              Σ_b x(b) b  ≤  ρ

        Solved as an LP using scipy.optimize.linprog with the HiGHS backend.
        This is the EXACT same pattern as compute_clairvoyant() in notebook 08:

            res = optimize.linprog(-f, A_ub=A_ub, b_ub=b_ub,
                                   A_eq=A_eq, b_eq=b_eq, bounds=(0,1))

        The LP cost vector uses the bid value b itself (not the realized cost
        b · F_D(b)) — this matches the project formulation  Σ x(b)·b ≤ ρ.

    Parameters
    ----------
    bid_set     : (K,) array   Discrete bid set B.
    v           : float        Campaign value.
    dist_config : dict         Distribution specification.
    rho         : float|None   Per-round budget target ρ = B/T.

    Returns
    -------
    x_opt     : (K,) array   Optimal mixed strategy x*(b).
    opt_value : float        OPT (or OPT^S) — expected per-round reward.
    opt_cost  : float        Expected per-round cost Σ_b x*(b)·b·F_D(b).
    """
    mu  = compute_true_arm_means(bid_set, v, dist_config)
    K   = len(bid_set)
    bids = np.asarray(bid_set, dtype=float)

    # ── Unconstrained: pure strategy ────────────────────────────────────────
    if rho is None:
        best = int(np.argmax(mu))
        x_opt = np.zeros(K, dtype=float)
        x_opt[best] = 1.0
        return x_opt, float(mu[best]), float(bids[best])

    # ── Budget-constrained: LP  (notebook 08 pattern) ───────────────────────
    # Per the project spec (slide 6): cost is incurred ONLY when winning.
    # Expected realized cost of bid b = b · P(win with b) = b · F_D(b).
    # The LP constraint must be:  Σ x(b)·b·F_D(b) ≤ ρ
    # Using raw bid b (as in notebook 08's generic bandit) would assume you
    # always pay b regardless of winning — overly pessimistic for auctions.
    F        = _compute_cdf(bid_set, v, dist_config)   # F_D(b) per bid
    cost_vec = bids * F                                 # E[cost | bid b]

    # scipy.optimize.linprog minimises → pass objective as  −μ
    res = linprog(
        c       = -mu,                     # minimise  −Σ x(b)·μ(b)
        A_ub    = [cost_vec],              # Σ x(b)·b·F_D(b) ≤ ρ
        b_ub    = [rho],
        A_eq    = [np.ones(K)],            # Σ x(b) = 1  (simplex)
        b_eq    = [1.0],
        bounds  = [(0.0, 1.0)] * K,
        method  = "highs",
    )

    if not res.success:
        raise RuntimeError(
            f"LP solver failed (rho={rho}): {res.message}\n"
            "Try reducing rho or checking that bid_set spans [0, v]."
        )

    x_opt     = np.asarray(res.x, dtype=float)
    opt_value = float(-res.fun)
    opt_cost  = float(cost_vec @ x_opt)   # E[realized cost per round]
    return x_opt, opt_value, opt_cost


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Quick environment validation  (call once before experiments)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cdf(bid_set: np.ndarray, v: float, dist_config: dict) -> np.ndarray:
    """
    Compute F_D(b) = P(m ≤ b) directly from the distribution spec.
    Used internally by validate_environment.
    """
    bids = np.asarray(bid_set, dtype=float)
    cfg  = dist_config
    kind = cfg["type"]

    if kind == "uniform":
        lo, hi = cfg["low"], cfg["high"]
        return np.clip((bids - lo) / (hi - lo), 0.0, 1.0)
    elif kind == "beta":
        return beta_dist.cdf(bids / v, cfg["a"], cfg["b"])
    elif kind == "normal":
        return np.clip(norm_dist.cdf(bids, cfg["loc"], cfg["scale"]), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown distribution type '{kind}'.")


def validate_environment(v: float, bid_set: np.ndarray, dist_config: dict,
                          n_rounds: int = 10_000) -> None:
    """
    Sanity-check the environment against the analytical win rate F_D(b).

    For each bid b in bid_set:
      • Runs the environment for n_rounds with that fixed bid.
      • Computes empirical win rate = wins / n_rounds.
      • Compares against theoretical F_D(b) = P(m ≤ b).
      • Asserts |empirical − theoretical| < 0.05.

    Usage:
        validate_environment(V, BID_SET, DIST_CONFIGS["uniform"])
    """
    np.random.seed(42)
    cdf_true = _compute_cdf(bid_set, v, dist_config)

    max_err = 0.0
    for idx, b in enumerate(bid_set):
        env = SingleCampaignEnv(v, bid_set, n_rounds, dist_config)
        wins = 0
        for _ in range(n_rounds):
            win_t, r_t, c_t = env.round(b)
            wins += int(win_t)

        win_rate_empirical = wins / n_rounds
        win_rate_true      = float(cdf_true[idx])
        err = abs(win_rate_empirical - win_rate_true)
        max_err = max(max_err, err)

        assert err < 0.05, (
            f"Validation failed for bid {b:.2f}: "
            f"empirical win rate = {win_rate_empirical:.4f}, "
            f"theoretical F_D(b) = {win_rate_true:.4f}, "
            f"error = {err:.4f} > 0.05"
        )

    print(
        f"[validate_environment] PASSED — {len(bid_set)} bids checked, "
        f"max error = {max_err:.4f}  (< 0.05)  over {n_rounds:,} rounds per bid."
    )
"""
environment_req2.py — Requirement 2: Multiple Campaigns, Stochastic Environment
===============================================================================
Provides:
  • MultiCampaignEnv        — stochastic environment for multiple campaigns
  • compute_campaign_means  — analytical mean reward per campaign/bid
  • compute_clairvoyant_mc  — brute-force clairvoyant via Monte Carlo or
                               exact marginal evaluation when independent

Design principles:
  • One round returns a vector of feedback, one entry per campaign.
  • Competing bids are pre-generated at construction time for fair comparison.
  • The joint distribution can be independent or correlated across campaigns.
  • Only campaigns actually selected by the agent are observed (semi-bandit).
"""



import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_dist, norm as norm_dist


@dataclass(frozen=True)
class MultiCampaignSpec:
    n_campaigns: int
    bid_set: np.ndarray
    values: np.ndarray
    dist_configs: List[dict]
    conflict_graph: np.ndarray
    rho: float
    correlation: float = 0.0


class MultiCampaignEnv:
    """
    Stochastic first-price auction environment for multiple campaigns.

    At each round t:
      1. A joint vector m_t of competing bids is drawn.
      2. The agent chooses a bid vector b_t and an activity mask.
      3. Each selected campaign i wins iff b_{i,t} >= m_{i,t}.
      4. Reward and cost are returned per campaign.

    Feedback is semi-bandit:
      - selected campaigns reveal win/reward/cost
      - unselected campaigns remain unobserved
    """

    def __init__(
        self,
        bid_set: np.ndarray,
        values: Sequence[float],
        dist_configs: Sequence[dict],
        T: int,
        conflict_graph: np.ndarray,
        rho: float,
        correlation: float = 0.0,
    ):
        # Keep the experiment configuration around for the whole run.
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.dist_configs = list(dist_configs)
        self.n_campaigns = len(self.values)
        self.T = int(T)
        self.conflict_graph = np.asarray(conflict_graph, dtype=int)
        self.rho = float(rho)
        self.correlation = float(correlation)
        self.t = 0
        # We sample the whole sequence up front so all agents face the same
        # randomness during a trial.
        self.competing_bids = self._pre_generate_joint_bids(self.T)

    def _pre_generate_joint_bids(self, n: int) -> np.ndarray:
        # Each row is one round, each column is one campaign.
        samples = np.zeros((n, self.n_campaigns), dtype=float)
        latent = None
        if self.correlation > 0:
            # If correlation is requested, use a latent Gaussian vector.
            cov = np.full((self.n_campaigns, self.n_campaigns), self.correlation)
            np.fill_diagonal(cov, 1.0)
            latent = np.random.multivariate_normal(
                mean=np.zeros(self.n_campaigns),
                cov=cov,
                size=n,
            )

        for i, cfg in enumerate(self.dist_configs):
            kind = cfg["type"]
            if kind == "uniform":
                low, high = cfg["low"], cfg["high"]
                if latent is None:
                    # Independent Uniform draws.
                    samples[:, i] = np.random.uniform(low, high, size=n)
                else:
                    # Transform latent Gaussian samples into uniforms.
                    u = norm_dist.cdf(latent[:, i])
                    samples[:, i] = low + u * (high - low)
            elif kind == "beta":
                a, b = cfg["a"], cfg["b"]
                if latent is None:
                    # Independent Beta draws.
                    samples[:, i] = beta_dist.rvs(a, b, size=n) * self.values[i]
                else:
                    # Correlated Beta draws via inverse CDF.
                    u = norm_dist.cdf(latent[:, i])
                    samples[:, i] = beta_dist.ppf(u, a, b) * self.values[i]
            elif kind == "normal":
                loc, scale = cfg["loc"], cfg["scale"]
                if latent is None:
                    # Independent Normal draws.
                    samples[:, i] = np.random.normal(loc, scale, size=n)
                else:
                    # Correlated Normal draws from the latent vector.
                    samples[:, i] = loc + scale * latent[:, i]
            else:
                raise ValueError(f"Unknown distribution type '{kind}'.")

        # Force bids back into the valid interval [0, campaign value].
        return np.clip(samples, 0.0, self.values).astype(float)

    def round(self, bid_vector: Sequence[float], active_mask: Sequence[bool]):
        # One round = one joint auction vector.
        if self.t >= self.T:
            raise RuntimeError("Environment exhausted.")

        bid_vector = np.asarray(bid_vector, dtype=float)
        active_mask = np.asarray(active_mask, dtype=bool)
        m_t = self.competing_bids[self.t]
        # Only active campaigns can win.
        win_t = (bid_vector >= m_t) & active_mask
        # Reward is utility only when we win.
        reward_t = (self.values - bid_vector) * win_t
        # Cost is actually paid only when we win.
        cost_t = bid_vector * win_t
        self.t += 1
        return win_t, reward_t, cost_t

    def reset(self) -> None:
        self.t = 0


def compute_campaign_means(
    bid_set: np.ndarray,
    values: Sequence[float],
    dist_configs: Sequence[dict],
) -> np.ndarray:
    bids = np.asarray(bid_set, dtype=float)
    values = np.asarray(values, dtype=float)
    K = len(bids)
    N = len(values)
    means = np.zeros((N, K), dtype=float)

    for i, cfg in enumerate(dist_configs):
        kind = cfg["type"]
        if kind == "uniform":
            lo, hi = cfg["low"], cfg["high"]
            F = np.clip((bids - lo) / (hi - lo), 0.0, 1.0)
        elif kind == "beta":
            F = beta_dist.cdf(bids / values[i], cfg["a"], cfg["b"])
        elif kind == "normal":
            F = norm_dist.cdf(bids, loc=cfg["loc"], scale=cfg["scale"])
            F = np.clip(F, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown distribution type '{kind}'.")
        means[i] = (values[i] - bids) * F
    return means


def _campaign_cdf(bids: np.ndarray, value: float, cfg: dict) -> np.ndarray:
    kind = cfg["type"]
    if kind == "uniform":
        lo, hi = cfg["low"], cfg["high"]
        return np.clip((bids - lo) / (hi - lo), 0.0, 1.0)
    if kind == "beta":
        return beta_dist.cdf(bids / value, cfg["a"], cfg["b"])
    if kind == "normal":
        return np.clip(norm_dist.cdf(bids, loc=cfg["loc"], scale=cfg["scale"]), 0.0, 1.0)
    raise ValueError(f"Unknown distribution type '{kind}'.")


def _is_independent(subset: Tuple[int, ...], conflict_graph: np.ndarray) -> bool:
    for i, u in enumerate(subset):
        for v in subset[i + 1 :]:
            if conflict_graph[u, v] or conflict_graph[v, u]:
                return False
    return True


def enumerate_feasible_sets(conflict_graph: np.ndarray) -> List[Tuple[int, ...]]:
    n = conflict_graph.shape[0]
    feasible = [tuple()]
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            if _is_independent(subset, conflict_graph):
                feasible.append(subset)
    return feasible


def compute_clairvoyant_mc(
    bid_set: np.ndarray,
    values: Sequence[float],
    dist_configs: Sequence[dict],
    conflict_graph: np.ndarray,
    rho: float,
    correlation: float = 0.0,
    mc_samples: int = 5000,
) -> Tuple[Tuple[int, ...], np.ndarray, float]:
    """
    Return the best feasible campaign subset, its bid indices, and value.

    The action space is:
      - choose an independent set of campaigns
      - choose one bid per selected campaign
      - total expected cost <= rho
    """
    bids = np.asarray(bid_set, dtype=float)
    values = np.asarray(values, dtype=float)
    means = compute_campaign_means(bid_set, values, dist_configs)
    feasible_sets = enumerate_feasible_sets(np.asarray(conflict_graph, dtype=int))

    best_subset: Tuple[int, ...] = tuple()
    best_bid_idx = np.zeros(len(values), dtype=int)
    best_value = -np.inf

    for subset in feasible_sets:
        if not subset:
            if best_value < 0:
                best_value = 0.0
            continue

        choices = [range(len(bids)) for _ in subset]
        for bid_choice in itertools.product(*choices):
            cost = 0.0
            reward = 0.0
            for local_pos, i in enumerate(subset):
                b_idx = bid_choice[local_pos]
                win_prob = float(_campaign_cdf(np.array([bids[b_idx]]), values[i], dist_configs[i])[0])
                cost += bids[b_idx] * win_prob
                reward += means[i, b_idx]
            if cost <= rho and reward > best_value:
                best_value = float(reward)
                best_subset = tuple(subset)
                best_bid_idx = np.zeros(len(values), dtype=int)
                for local_pos, i in enumerate(subset):
                    best_bid_idx[i] = int(bid_choice[local_pos])

    if best_value == -np.inf:
        best_value = 0.0
    return best_subset, best_bid_idx, best_value
