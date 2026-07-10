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
        The activity mask indicates in which campaigns the agent (bidder) will participate
      3. Each selected campaign i wins iff b_{i,t} >= m_{i,t}.
      4. Reward and cost are returned per campaign.

    Feedback is semi-bandit:
      - selected campaigns reveal win/reward/cost
      - unselected campaigns remain unobserved
    Full-Feedback extension has been added in order to make this class usable also for Requirement 3
    """

    def __init__( #CONSTRUCTOR
        self,
        bid_set: np.ndarray, #DISCRETE LIST OF POSSIBLE BID VALUES
        values: Sequence[float], #VECTOR OF VALUATION OF EACH CAMPAIGN
        #each element in "values" tells how much the agent gains when winning that campaign
        dist_configs: Sequence[dict], #VECTOR OF LENGTH OF NUMBER OF CAMPAIGNS CONTAINING THE 
        #TYPE OF DISTRIBUTION FROM WHICH THE HIGHEST COMPETING BIDS OF EACH CAMPAIGN ARE SAMPLED
        T: int, #TIME HORIZON OF THE PROBLEM
        conflict_graph: np.ndarray, #GRAPH TELLING WHICH CAMPAIGNS CANNOT BE RUN TOGETHER
        rho: float, #PER-ROUND BUDGET
        correlation: float = 0.0, #it is used to determine how much competition levels of the different campaigns are similar to each other
        #the higher is the correlation the more similar are the competition levels of the different campaigns.
    ):
        # DEFINE THE ATTRIBUTES OF THE ENVIRONMENT
        # create on attribute per parameter that was passed in input to the constructor
        self.bid_set = np.asarray(bid_set, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.dist_configs = list(dist_configs)
        self.n_campaigns = len(self.values)
        self.T = int(T)
        self.conflict_graph = np.asarray(conflict_graph, dtype=int)
        self.rho = float(rho)
        self.correlation = float(correlation)
        self.t = 0 #ADDITIONAL ATTRIBUTE - ROUND COUNTER
        # We sample the whole sequence up front so all agents face the same
        # randomness during a trial.
        self.competing_bids = self._pre_generate_joint_bids(self.T) #DDITIONAL ATTRIBUTE - MATRIX OF PRE-COMPUTED HIGHEST COMPETING BIDS
        # it is a metrix that has shape (T, N) where T is the number of rounds and N the number of campaigns. 
        # it stores the pre-computed sequence of highest competing bids for each campaign.
        # REMEMBER: THE SEQUENCES OF HIGHEST COMPETING BIDS ARE PRE-COMPUTED SO THAT WE CAN COMPARE TWO AGENTS ON THE SAME ENVIRONMENT
        # if you want to compare two agents fairly, you can create two environments with the same random seed and both agents will face 
        # exactly the same sequence of competing bids. 
    
    # METHOD USED TO PRE-COMPUTED THE SEQUENCE OF HIGHEST COMPETING BIDS FOR THE DIFFERENT CAMPAIGNS (INVOKED IN CONSTRUCTOR)
    def _pre_generate_joint_bids(self, n: int) -> np.ndarray: #INPUT PARAMETER OF THE METHOD n = NUMBER OF ROUNDS
        # Each row is one round, each column is one campaign - SHAPE OF MATRIX (T, N)
        samples = np.zeros((n, self.n_campaigns), dtype=float) #CREATES AN EMPTY (FULL OF ZEROES) MATRIX OF THE DESIRED SHAP 
        
        ####### WITHOUT THE BLOCK BELOW THE HIGHEST COMPETING BIDS OF EACH CAMPAIGN ARE DRAWN COMPLETELY INDEPENDENT FROM ONE ANOTHER
        # RUN ONLY IF WE WANT CORRELATION BETWEEN THE DIFFERENT CAMPAINGS (to have similar competition levels)
        latent = None #latent = hidden helper variable that if we use will drive all the campaign's randomness in a coordinate way.
        if self.correlation > 0:
            # If correlation is requested, use a latent Gaussian vector.
            cov = np.full((self.n_campaigns, self.n_campaigns), self.correlation) # BUILDS COVARIANCE MATRIX (NUM_CAMPAIGNS X NUM_CAMPAINGS)
            # the line above fills the correlation matrix with the value of the correlation attribute.
            np.fill_diagonal(cov, 1.0) # we overwrite the diagonal placing all elements to 1 (variances = 1)
            # we are sampling a latent matrix (TxN) from a multivariate gaussian distribution - that will be used to keep the campaign's random generation of the highest competing bids correlated.
            latent = np.random.multivariate_normal(
                mean=np.zeros(self.n_campaigns), #MEAN  = 0 VECTOR
                cov=cov, #COVARIANCE = MATRIX DEFINED ABOVE
                size=n, #SAMPLE n=T VECTORS OF DIM N (CREATES TxN MATRIX)
            )

        ###### FILLING THE MATRIX OF PRE-COMPUTED HIGHEST COMPETING BIDS 
        for i, cfg in enumerate(self.dist_configs): #iterate over campaigns
            kind = cfg["type"] #store the distribution type of highest competing bids for the campaign
            
            #IF TYPE = UNIFORM DISTRIBUTUON
            if kind == "uniform": 
                low, high = cfg["low"], cfg["high"] #read the lower and upper bounds of the uniform distribution
                if latent is None: #IF NO CORRELATION BETWEEN CAMAPIGNS
                    # uniformly sample the [low,high] range n times, in order to sample the entire sequence of highest competing bids for the campaign
                    samples[:, i] = np.random.uniform(low, high, size=n) 
                else: #IF CORRELATION = YES
                    # get the corresponding column to the campaign in the latent matrix and convert it into a number between 0 and 1, using normal dist CDF.
                    u = norm_dist.cdf(latent[:, i]) 
                    samples[:, i] = low + u * (high - low) #COPULA TRICK
                    # this stretches the number u in [0,1] creating samples that come from a marginally uniform distirbution, but correlated by a Gaussian mood
            
            # IF TYPE = BETA DISTRIBUTION
            elif kind == "beta":
                a, b = cfg["a"], cfg["b"] #read the alpha and beta coefficient of the beta distribution
                if latent is None: # IF NO CORRELATION 
                    # sample the beta distribution n times and multiply each sample by the value of the campaign to have highest comp bids in [0, camp_value]
                    samples[:, i] = beta_dist.rvs(a, b, size=n) * self.values[i] 
                else: # IF CORRELATION YES
                    # get the corresponding column to the campaign in the latent matrix and convert it into a number between 0 and 1, using beta dist CDF.
                    u = norm_dist.cdf(latent[:, i])
                    samples[:, i] = beta_dist.ppf(u, a, b) * self.values[i] #COPULA TRICK
                    #same trick to stretch the value of u and generate samples that come from a beta distribution, but correlated by a Gaussian mood
            
            #IF TYPE = NORMAL DISTRIBUTION
            elif kind == "normal":
                loc, scale = cfg["loc"], cfg["scale"] #we read the loc (center of the distribution) param, and the scale (spread of distribution) param
                if latent is None: # IF NO CORRELATION 
                    # sample the normal distribution n=T times in order to generate the sequence of highest competing bids
                    samples[:, i] = np.random.normal(loc, scale, size=n)
                else: # IF CORRELATION YES
                    #in this case no CDF trick (or copula trick) is needed since we are trying to add Gaussian mood to a gaussian distribution
                    #we just rescale and recenter the gaussian distribution, using the gaussian mood (another gaussian distribution)
                    samples[:, i] = loc + scale * latent[:, i]
           
            #IF OTHER UNSUPPORTED TYPE IS SPECIFIED
            else:
                raise ValueError(f"Unknown distribution type '{kind}'.")

        # FRETURN THE MATRIX OF HIGHEST COMEPTING BIDS FOR EACH CAMPAIGN ENSURING THEY ARE IN THE VALID RANGE [0, VALUE OF THE CAMPAIGN]
        return np.clip(samples, 0.0, self.values).astype(float) #converted to float
    
    
    #METHOD THAT SIMULATES PLAYING ONE ROUND (returns wins, rewards and costs at round t for all the campaigns)
    def round(self, bid_vector: Sequence[float], active_mask: Sequence[bool],
              full_feedback: bool = False):
        # CHECK CURRENT ROUND WITHIN TIME HORIZON
        if self.t >= self.T:
            raise RuntimeError("Environment exhausted.")

        # convert the vector of played bids (one per campaign) and the active mask vector in arrays so we can do vectorized maths upon them
        bid_vector = np.asarray(bid_vector, dtype=float)
        active_mask = np.asarray(active_mask, dtype=bool)
        #we retrieve the highest competing bid vector of the current round - one highest competing bid per campaign
        m_t = self.competing_bids[self.t]
        # determine the campaings that we can win in the current round.
        # WE WIN ALL THE CAMPAIGNS FOR WHICH OUR BID IS GREATER OR EQUAL THAN THE HIGHEST COMPETING BIDS AND IN WHICH WE PARTICIPATE
        win_t = (bid_vector >= m_t) & active_mask # we compute the and with the active mask to consider only the campaigns in which we participates
        #win_t is a boolean vector
        # Reward = value - price (= bid since first price auction)
        reward_t = (self.values - bid_vector) * win_t #multiply by win vector since we gain reward only when we win
        # Cost is actually paid only when we win - so again multiply by win vector
        cost_t = bid_vector * win_t
        # increment round counter
        self.t += 1 
        ####### REQ 3 EXTENSION FOR FULL FEEDBACK
        # With full_feedback=True the environment also reveals the vector of highest competing bids m_t 
        if full_feedback:
            return win_t, reward_t, cost_t, m_t.copy() #we also return the vector of highest competing bids of the current round
            # having full feedback is equivalent to observing vector of highest competing bids, 
            # since from it we can compute all Lagrangian rewards.
        
        return win_t, reward_t, cost_t #return the win vector, reward vector and cost vector for the current round (each one has one element per campaign)
    
    #HELPER METHOD TO REST THE ROUND COUNTER
    def reset(self) -> None:
        self.t = 0 #set round counter back to zero


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


# ═════════════════════════════════════════════════════════════════════════════
# REQUIREMENT 3 — Best-of-both-worlds: highly non-stationary environment
# ═════════════════════════════════════════════════════════════════════════════
# Everything below is ADDITIVE.  Requirement 1 and 2 code above is untouched.
#
# Provides:
#   • NonStationaryMultiCampaignEnv — competing-bid distributions that change
#     rapidly over time (project slide 15: "a non-stochastic sequence of
#     highest competing bids ... sampled from a distribution that changes
#     quickly over time").
#   • compute_hindsight_clairvoyant — best FIXED feasible action in hindsight
#     on the realized bid sequence (the correct benchmark in the adversarial /
#     non-stationary world, where expectations are meaningless).


class NonStationaryMultiCampaignEnv(MultiCampaignEnv): #SUBCLASS OF MultiCampaignEnv
    # since SUBCLASS it automatically inherits round(), reset() and all the attributes of MultiCampaignEnv
    """
    Highly non-stationary multi-campaign first-price auction environment.

    Identical interface to MultiCampaignEnv, but the per-campaign competing-bid distribution 
    is RE-RANDOMIZED every `change_every` rounds:

    Example of how change_every parameter works: 
      • change_every = 1  → new distribution parameters every single round
                            (essentially an oblivious adversary)
      • change_every = k  → piecewise regime of length k, but with many
                            regimes (T/k changes) — still "highly"
                            non-stationary, unlike Requirement 4's few
                            long intervals.

    The `dist_configs` passed at construction only fix the FAMILY of each
    campaign's distribution (uniform / beta / normal); the parameters are
    drawn fresh at every change point from `param_ranges`. EACH CAMPAGIN IS 
    ALWAYS CHARACTERIZED BY THE SAME TYPE OF HIGHEST COMPETING BID DISTRIBUTION.

    The whole sequence is still pre-generated in __init__, preserving the
    fair-comparison design: every agent evaluated on the same seed faces the
    IDENTICAL sequence of competing bids. Therefore the sequence of highest competing bids of the 
    different campaigns is pre-generated at the start so that each agent 
    acts on the same environment and we have a fiar comparison betweena agents. 
    """

    def __init__(
        self,
        bid_set: np.ndarray,
        values: Sequence[float],
        dist_configs: Sequence[dict],
        T: int,
        conflict_graph: np.ndarray,
        rho: float,
        change_every: int = 1,
        param_ranges: Optional[dict] = None,
        correlation: float = 0.0,
    ):  # CONSTRUCTOR OF THE CLASS - it has the same input parameters of the father class MultiCampaignEnv + two additional input params 
        # that will be converted into two new attributes: change_every and param_ranges.
        
        #NEW ATTRIBUTE - this attributes defines how much each regime last (by default = 1), max ensures a "regime" lasts at least 1 round
        self.change_every = max(1, int(change_every)) 
        
        # Default parameter ranges (mirrors config.NS_PARAM_RANGES; duplicated
        # here so environment.py stays importable without config).
        # NEW ATTRIBUTE to store the dictionary that defines the ranges of the params of the various distributions
        # we duplicate it in order to make environment runnable also without config file. 
        # the param ranges defined below mirror the ones in the config file.
        self.param_ranges = param_ranges or {
            "uniform": {"low": (0.0, 0.3), "high": (0.4, 1.0)},
            "beta":    {"a": (1.0, 6.0), "b": (1.0, 6.0)},
            "normal":  {"loc": (0.2, 0.8), "scale": (0.05, 0.30)},
        }
        
        #last steo is calling the parent constructor to istantiate all the other attributes, after we have instantiated the two new ones.
        super().__init__( 
            bid_set=bid_set,
            values=values,
            dist_configs=dist_configs,
            T=T,
            conflict_graph=conflict_graph,
            rho=rho,
            correlation=correlation,   # we will always use correlation = 0 un requirement 3 - independent sampling of highest competing bids for each campaign.
        )            
    
    #IMPORTANT: the __init__ method of the super-class also invokes _pre_generate_joint_bids and pre-computes the highest competing bids for each campaign over the
    # time horizon T. The super-class considers a stochastic stationary environment in which the sequence of highest competing bids is drawn from a fixed distribution.
    # In this case we are considering a non-stationary environment and therefore the pre-computation of the highest competing bids needs to happen by dividing the time
    # horizon in "regimes" and change the distribution parameters of each campaign at the start of each regime. FOR THIS REASON WE IMPLEMENT BELOW THE METHOD AGAIN, 
    # OVER-RIDING THE ONE OF THE FATHER CLASS.      

    # ── ADVERSARY HELPER METHODS (the "adversary uses them" to generate the highest competing bids) ────────────────────────────────────────────────────
    
    #METHOD THAT IS USED TO PICK FRESH "REGIME", FRESH PARAMETERS FOR "ONE" CAMPAIGN (invoked at the start of each regime)
    def _random_params(self, kind: str) -> dict: #INPUT PARAMETER: the type of the distribution of the considered campaign
        """The method's job is the following: given a distribution type, randomly generate one new set of parameters for it, considering the allowed ranges for the params."""
        # retrieve the params of the type (kind) of distribution considered together with their ranges
        r = self.param_ranges[kind]
        if kind == "uniform": #IF UNIFORM DISTRIBUTION
            low  = np.random.uniform(*r["low"]) #uniformly sample the lower bound param
            high = np.random.uniform(*r["high"]) #uniformly sample the upper bound param
            return {"low": low, "high": max(high, low + 1e-3)} #return the sampled low and the max between the sampled high and low + 1e - 3 (to ensure the two are not too close to each other)
        if kind == "beta": #IF BETA DISTRIBUTION
            return {"a": np.random.uniform(*r["a"]), #uniformly sample in the range allowed for the alpha param
                    "b": np.random.uniform(*r["b"])} #uniformly sample in the range allowed for the beta param and return them
        if kind == "normal": #IF NORMAL DISTRIBUTION
            return {"loc":   np.random.uniform(*r["loc"]), #uniformly sample in the range allowed for the loc param
                    "scale": np.random.uniform(*r["scale"])} #uniformly sample in the range allowed for the scale param
        #ERROR IF DISTRIBUTION TYPE IS NOT SUPPORTED
        raise ValueError(f"Unknown distribution type '{kind}'.")

    # METHOD THAT GENERATES BID VALUES FOR A STRETCH (BLOCK) OF ROUNDS
    def _draw_block(self, kind: str, params: dict, value: float,
                    n: int) -> np.ndarray: 
        """
        INPUT PARAMETERS: type (kind) of the distribution of highest competing bids of the considered campaign, the randomly selected params of the 
                          current regime, the value of the considered campaign and the number of rounds n=length of regime.
        OUTPUT: it returns the highest competing bid sequence for a stretch of rounds corresponding to the regime for the considered campaign
        IMPLEMENTED OPERATION: Given a distribution family, a specific set of parameters (already chosen by _random_params), a campaign's value and how many rounds (n)
                               the considered regime should cover, we produce n random highest competing bid values. 
        """
        # we sample from disributions that are characterized by the params chosen via _random_params
        if kind == "uniform": #IF UNIFORM DISTRIBUTION
            # sample n times from the uniform distribution to generate sequence of highest competing bids for the regime
            s = np.random.uniform(params["low"], params["high"], size=n) 
        elif kind == "beta": #IF BETA DISTRIBUTION
            # sample n times from the beta distribution and multiply by the value of the campaign to generate the sequence of highest competing bids for the regime
            s = np.random.beta(params["a"], params["b"], size=n) * value
        elif kind == "normal": #IF NORMAL DISTRIBUTION
            s = np.random.normal(params["loc"], params["scale"], size=n)
        else: #ERROR IN CASE OF TYPE OF DISTRIBUTION NOT SUPPORTED
            raise ValueError(f"Unknown distribution type '{kind}'.")
        return np.clip(s, 0.0, value) #we return the sequence of highest compting bids for the considered regime and considered campaign ensuring they are in the valid range [0,value]
 
    # ── override: non-stationary pre-generation ─────────────────────────────
    #METHOD GENERATES HIGHEST COMPETING BID MATRIX (TxN) FOR ALL CAMPAIGNS OVER THE CONSIDERE TIME HORIZON
    def _pre_generate_joint_bids(self, n: int) -> np.ndarray: #we pass as input parameter the time horizon of the problem n=T
        samples = np.zeros((n, self.n_campaigns), dtype=float) #initialize empty (filled with zeroes) highest competing bid matrix
        t = 0 #initialize round index of the method to zero
        while t < n: #while considered round within conisdered time horizon of the problem
            block = min(self.change_every, n - t) #length of the regime is the minimum between the fixed length of regime and remaining rounds till the end of time horizon
            for i, cfg in enumerate(self.dist_configs):#for every campaign
                kind = cfg["type"] #retrieve type of campaign
                params = self._random_params(kind) # fresh regime (fresh params)
                samples[t:t + block, i] = self._draw_block(
                    kind, params, self.values[i], block
                ) #invoke draw_block function and generate the sequence of highest competing bids for the considered campaign and the considered block (regime)
            t += block #increment round counter adding length of just considered regime
        return samples #(TxN) MATRIX

# METHOD TO IMPLEMENT THE CLAIRVOYANT AGENT ACTING IN THE NON-STATIONARY ENVIRONMENT, AGAINST WHICH WE WILL NEED TO COMPETE
# REGRET WILL BE COMPUTED WITH RESPECT TO THE UTILITY GAINED BY THIS CLAIRVOYANT AGENT
"""
CLAIROVANT AGENT: 
If i could see the entire future in advance, what single fixed strategy would have earn me the most money? This strategy is the one chosen by the 
clairvoyant agent
"""
def compute_hindsight_clairvoyant(
    competing_bids: np.ndarray,
    bid_set: np.ndarray,
    values: Sequence[float],
    conflict_graph: np.ndarray,
    rho: float,
) -> Tuple[Tuple[int, ...], np.ndarray, float]:
    """
    Best FIXED feasible action in hindsight on a realized bid sequence.

    In the non-stationary / adversarial world, the benchmark (with respect to which we compute regret) 
    is no longer an expectation over a known distribution: it is the best fixed
    (independent set, bid vector) evaluated on the ACTUAL sequence m_{1..T}, that respect the budget constraint: "total realized cost ≤ ρ·T".
    
    Parameters
    ----------
    competing_bids : (T, N) array - MATRIX OF PRECOMPUTED HIGHEST COMPETING BIDS
    bid_set        : (K,) array - LIST OF ALLOWED BIDS
    values         : (N,) array - VECTOR STORING THE VALUE OF EACH CAMPAIGN
    conflict_graph : (N, N) 0/1 2d array - MATRIX INDICATING WHICH CAMPAIGNS ARE COMPATIBLE AND WHICH NOT
    rho            : float - PER-ROUND BUDGET

    Returns
    -------
    best_subset  : tuple of campaign indices
    best_bid_idx : (N,) int array (bid index per campaign; 0 outside subset)
    best_value   : float — PER-ROUND value of the best fixed action
                   (total hindsight reward / T), directly comparable with the
                   per-round clairvoyant used in the stochastic experiments.
    """
    m = np.asarray(competing_bids, dtype=float)  #CONVERT HIGHEST COMPETING BID MATRIX TO NUMPY ARRAY AND RENAME TO m. 
    bids = np.asarray(bid_set, dtype=float) #CONVERT LIST OF POSSIBLE BIDS, IN A NUMPY ARRAY AND RENAME TO bids. 
    values = np.asarray(values, dtype=float) #CONVER VECTOR OF CAMPAIGN VALUES TO NUMYP ARRAY AND RENAME TO values.
    T_len = m.shape[0] #store the time horizon of the problem
    total_budget = rho * T_len #compute TOTAL BUDGET = TIME HORIZON * PER-ROUND BUDGET

    #FIND ALL VALID CAMPAIGN COMBINATIONS
    #enumerate_feasible_sets DEFINED IN THE REQUIREMENT 2 PART IN THIS .py FILE
    """
    REMAINDER: enumerate_feasible_sets looks athe conflict graph and returns every possible group of compatible campaigns. All groups of campaigns
    such that no pair of campaigns in the same group are conflicting with each other.
    """
    feasible_sets = enumerate_feasible_sets(np.asarray(conflict_graph, dtype=int))
    #it returns something like: [(), (0,), (1,), (0,1), (0,2), ...]

    # INITIALIZE THE THREE VARIABLES THAT STORE THE DETAILS OF THE BEST SOLUTION (STRATEGY) FOUND SO FAR
    # these will be updated as the best solution (strategy) is researched.
    best_subset: Tuple[int, ...] = tuple() #initialize to empty subset of campaigns
    best_bid_idx = np.zeros(len(values), dtype=int) #initialize the best bid for each campaign to zero
    best_value = 0.0 # per-round value of the current best solution (strategy)

    ########## FINDING THE BEST STRATEGY OF THE CLAIRVOYANT AGENT THAT KNOWS THE ENTIRE HIGHEST COMPETING BID MATRIX
    K = len(bids) #number of possible bids 
    N = len(values) #number of campaigns
    # initialize two empty matrices with zeroes (each shaped N,K)
    #row = campaign, column = bid choice
    tot_reward = np.zeros((N, K)) #in position i,j it stores the total reward (utility) in playing bid j in a fixed manner for campaign i
    tot_cost   = np.zeros((N, K)) #in position i,j it stores the total cost in playing bid j in a fixed manner for campaign i
    
    # COMPUTING THE TOTAL COSTS AND TOTAL REWARDS FOR EACH POSSIBLE FIXED STRATEGY, FOR EACH CAMPAIGN
    #iterate over the campaigns
    for i in range(N):
        #EXPLAINING NEXT LINE
        """
        - we consider the column i (corresponding to the considered campaign) of the highest competing bid matrix
        - we reshape it adding an extra dimension so that the 1d array of T entries becomes a 2d array [T,1] (using None)
        - we reshape the array of available bids from a 1d array of k entries to a 2d array [1,K] (using None)
        - numpy uses broadcasting: it compares every avaiable bid against each rounds highest competing bid. 
        - the result is a (T,K) matrix owith True and False entries.
        - if in position i,j we have true: it means that avaiable bid j is greater than the highest competing bid at round i
        """
        wins = bids[None, :] >= m[:, i][:, None] 
        
        #EXPLAINING NEXT LINE: 
        """
        - it gets the value of campaign and we calculate for each available bid the reward you get in winnin an auction in that campaign with that bid
        - the results is a vector of dimension [1,K], where for each avaiable bid we report the reward of winning an auction of the campaign with that bid
        - we multiply this vector by each row of the wins matrix and this mutliplication zeroes out the per-round reward each time we do not win. 
        - we then sum the elements column wise and we have for each available bid, what is the total reward for the considered campaign
        - this is stored as a row in the tot_reward matrix
        """
        tot_reward[i] = ((values[i] - bids)[None, :] * wins).sum(axis=0)
        """
        - similar as above you get the vectors od possible bids and add a dimension becoming (1,K)
        - this because in first price auctions cost = bid.
        - multiply it against each row of win matrix, the matrix zeroes out the cost for each pair bid, round in which we do not win. 
        - we sum the elements in the matrix column wise and we obtain the total cost of playing each possible bid in a fixed manner w.r.t the considered campaign
        - stored as row to the tot_cost matrix
        """
        tot_cost[i]   = (bids[None, :] * wins).sum(axis=0)

    # FIND THE BEST STRATEGY THAT CAN BE FOLLOWED BY THE CLAIRVOYANT AGENT (CONSIDERING COMPATIBLE SETS OF CAMPAIGNS)
    for subset in feasible_sets: #iterating over compatible sets of auctions
        if not subset: #skips the empty set
            continue
        
        #for the current subset of compatible campaigns we need to try every possible combination of fixed bis for each of the campaigns in it. 
        #all possible combination of bids is generated by iteratools.product(range(K), repeat = len(subset))
        #if we have two campaigns in the subset and 6 possible bids, then we need to try 6^2 = 36 combinations
        for bid_choice in itertools.product(range(K), repeat=len(subset)):
            #bid choice is a vector of the same size of the campaigns in the subset
            #array with one bid choice per campaign
            cost = sum(tot_cost[i, bid_choice[k]] for k, i in enumerate(subset)) #sum the total cost for each bid, campagin pair for the current bid choice. 
            if cost > total_budget: #verify if the budget is satisfied
                continue #if not satisfied go to next bid combination (bid choice)
            reward = sum(tot_reward[i, bid_choice[k]] for k, i in enumerate(subset)) #sum the total reward for each bid, campaign pair for the current bid choice
            if reward > best_value * T_len: #if the total reward across all auctions is greater than the current best total reward - update best strategy (bid choice)
                best_value = float(reward) / T_len # commpute new best value which is the per-round reward of the current best bid choice
                best_subset = tuple(subset) #store the subset of campaigns
                best_bid_idx = np.zeros(N, dtype=int) #re-initialize the best bid choice for each campaign
                for k, i in enumerate(subset): #store the best bid for each campaign (bid = 0) for campaigns in which we do not participate
                    best_bid_idx[i] = int(bid_choice[k])

    return best_subset, best_bid_idx, best_value #return best strategy - best set of compatible campaigns and bid choice of those campaigns to play in a fixed manner over the entire time horizon 
    #return also the average per-round reward of this strategy across all the campaigns
