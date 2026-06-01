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
