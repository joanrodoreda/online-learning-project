"""
config.py — Requirement 1: Single Campaign, Stochastic Environment
============================================================
Central registry for every constant, hyperparameter, and distribution
specification used across environment.py, agents.py, and main.py.

All values are grounded in:
  • Project statement (project.pdf, slides 5–10)
  • Notebook 01 — stochastic MABs (UCB1, ETC, regret methodology)
  • Notebook 08 — constrained problems (Lagrangian, OGD dual, budget)
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CAMPAIGN PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

V: float = 1.0
"""Campaign value v.  Winning a slot yields utility  v - b_t  (project, slide 6)."""

BID_SET: np.ndarray = np.round(np.linspace(0.0, V, 11), 2)
"""Discrete bid set  B = {0.00, 0.10, …, 1.00}.
Project statement: 'Set of possible bids B (small and discrete set)' (slide 5).
K = 11 arms gives UCB1 clean convergence within T = 5 000 rounds."""

K: int = len(BID_SET)
"""Number of arms / possible bids  (K = 11)."""

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPERIMENT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

T: int = 5_000
"""Time horizon T.  Enough to observe O(log T) regret shape clearly."""

N_TRIALS: int = 50
"""Number of independent trials for pseudo-regret estimation.
Notebook 01 uses 10–150 trials; 50 balances stability and runtime."""

RANDOM_SEED_START: int = 0
"""First seed.  Trial i uses  np.random.seed(RANDOM_SEED_START + i)
to guarantee reproducibility across agents on the same noise sequence."""

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUDGET PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

RHO_MODERATE: float = 0.4
"""Per-round budget ρ = B/T (moderate constraint).
Budget  B_moderate = 0.4 × 5 000 = 2 000 total spend units."""

RHO_TIGHT: float = 0.2
"""Per-round budget ρ = B/T (tight constraint).
Budget  B_tight = 0.2 × 5 000 = 1 000 total spend units."""

BUDGET_MODERATE: float = RHO_MODERATE * T   # 2 000.0
BUDGET_TIGHT: float    = RHO_TIGHT    * T   # 1 000.0

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ALGORITHM HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# UCB1 confidence radius  (notebook 01, UCB1Agent):
#   UCB(b) = μ̂(b) + sqrt( 2 log T / n_b )
# The factor of 2 comes from Hoeffding's inequality on [0,1]-bounded rewards.
UCB_EXPLORATION_FACTOR: float = 2.0

# OGD dual step size  η  (notebook 08, theory: η = 1/√T)
#   λ_{t+1} = clip( λ_t + η·(c_t − ρ),  0,  1/ρ )
ETA_DUAL: float = 1.0 / np.sqrt(T)   # ≈ 0.01414 for T = 5 000

# ETC exploration rounds per arm  (notebook 01, ETCAgent)
#   T₀ = ⌊ (T/K)^(2/3) · log(T)^(1/3) ⌋
T0_ETC: int = max(1, int((T / K) ** (2.0 / 3.0) * np.log(T) ** (1.0 / 3.0)))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPETING BID DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a self-contained spec dict consumed by SingleCampaignEnv
# and compute_true_arm_means().  Competing bids are always clipped to [0, V].

DIST_CONFIGS: dict = {
    "uniform": {
        "type":  "uniform",
        "low":   0.0,
        "high":  1.0,
        "label": "Uniform(0, 1)",
    },
    "beta": {
        "type":  "beta",
        "a":     2,
        "b":     5,
        "label": "Beta(2, 5)",
        # Competing bids ~ Beta(2,5) × v.
        # F_D(b) = P(Beta(2,5) ≤ b/v).  Skewed toward low values.
    },
    "normal": {
        "type":  "normal",
        "loc":   0.5,
        "scale": 0.2,
        "label": "Truncated Normal(0.5, 0.2)",
        # Normal samples clipped to [0, v].
    },
}

DEFAULT_DIST: str = "uniform"
"""Distribution used in the primary experiment (simplest, closed-form CDF)."""

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_SIZE: tuple       = (10, 6)
UNCERTAINTY_ALPHA: float = 0.3
"""Transparency of fill_between uncertainty bands (notebook 01 convention)."""

# Consistent colour assignment across all plots.
# Keys match the 'label' strings returned by agents.
AGENT_COLORS: dict = {
    "Random":               "#808080",   # grey
    "Greedy":               "#9467bd",   # purple
    "ETC":                  "#8c564b",   # brown
    "UCB1 (no budget)":     "#1f77b4",   # steelblue
    "Budget-UCB1 (ρ=0.4)":  "#ff7f0e",   # orange
    "Budget-UCB1 (ρ=0.2)":  "#d62728",   # crimson
    "Clairvoyant":          "#2ca02c",   # forest green
}

# ─────────────────────────────────────────────────────────────────────────────
# 7.  DERIVED / CONVENIENCE VALUES
# ─────────────────────────────────────────────────────────────────────────────

OPT_OUT_ARM_IDX: int = 0
"""Index of the opt-out arm (bid = 0.0).
When budget is exhausted the agent returns this arm — cost = 0, reward ≈ 0."""

assert BID_SET[OPT_OUT_ARM_IDX] == 0.0, "Opt-out arm must be bid = 0.0"
assert K == 11,                          "Bid set must contain exactly 11 bids"
assert T0_ETC >= 1,                      "ETC exploration length must be ≥ 1"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  REQUIREMENT 3 — Best-of-both-worlds, multiple campaigns  (additions)
# ─────────────────────────────────────────────────────────────────────────────

# Hedge learning rate to update the weights of the arms (bids)— theory-optimal learning rate for T rounds,
# With K experts (arms/bids):  γ = sqrt(log K / T)   (in full-feedback it esnures a regret O(sqrt(T log K))).
ETA_HEDGE: float = float(np.sqrt(np.log(K) / T))

# Hyperparameter to create highly non-stationary environment necessary for requirement 3
# it indicates the number of rounds in which a "regime" lasts in the non-stationary environment before the competitor's behavior is randomized again
#25 is a design choice that is small enough that the world visibly shits many times over T=2000 rounds, large neought so that each regime isn't just noise
#stationary ennvironment with abrupt changes every 25 rounds
####### ENVIORNEMENT'S NON STATIONARY BEHAVIOR ##########
"""
Every 25 rounds each campaign's competing bid dsiteribution parameters are instantly and completely re-drawn from scratch,
not drifting gradually, not smoothly interpolated. This is why we define it as a stationary enviornment with abrupt changes.
Within each 25 round block the distribution reamins fixed and stationary (bids are i.i.d w.r.t the randomly chosen distribution).
"""
NS_CHANGE_EVERY: int = 25


# LOOK UP TABLE FOR PARAMETER DISTRIBUTIONS
"""
This is the a look up table that tells the environment, for each distribution family, in which range the parameters of that distribution can 
be extracted from when the regime changes. The design choice was to have three different possible distributions from which the highest competing
bids of a campaign can be sampled and to randomize even more the behavior of the environment, we randomly select one of the three distributions
and then we randomly sample the parameters of the distribution within the specified ranges in the look up table.
"""
NS_PARAM_RANGES: dict = {
    "uniform": {"low": (0.0, 0.3), "high": (0.4, 1.0)},
    "beta":    {"a": (1.0, 6.0), "b": (1.0, 6.0)},
    "normal":  {"loc": (0.2, 0.8), "scale": (0.05, 0.30)},
}

# Cap on the dual variable λ used by the Req-3 primal-dual agent.
# It puts a ceiling on how harshly overspending can get penalized (remember: λ grows when we cost > per-round budget)
# Theory allows 1/ρ, but a smaller cap keeps the Hedge gain normalisation
# from compressing the reward signal when ρ is tight.
# In simpler term caping lambda to 1 and not 1/p allows to have greater difference between the normalized reward of actions when the budget is small. 
# Without this with a very small budget the normalized Lagrangian rewards of a very good action (bid), might be very close to the one of a bad action (bid)
LAMBDA_MAX_REQ3: float = 1.0

###### HEDGE GAIN NORMALIZATION #########
"""
The Hedge Algorithm is an algorithm that works with rewards (losses) that need to be in the range [0,1]. Raw Lagrangian rewards are not natively in this range, 
therefore we need to normalize them into [0,1]. To do so we introduce the Hedge Gain normlization formula: gain = (lagr + λ_max·v) / (v·(1 + λ_max)) - this returns
essentially a "normalized Lagrangian reward" in [0,1] that can be fed to the Hedge algorithm. When lambda is capped at 1/p these normalized rewards get compressed in
a very tight interval between in [0,1] meaning that the rewards of the different bids become very similar to each other. More difficult to understand what is the best bid.
"""

# Adds the plotting color for the two agents for the previously defined color dictionary. 
# NOT ADDED DIRECTLY TO DICTIONARY TO SHOW INCREMENTAL NATURE OF PROJECT
AGENT_COLORS["Primal-Dual (Hedge)"] = "#e377c2"   # pink
AGENT_COLORS["Combinatorial-UCB"]   = "#17becf"   # teal
