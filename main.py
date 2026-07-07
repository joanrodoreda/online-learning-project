"""
main.py — Requirement 1: Single Campaign, Stochastic Environment
=================================================================
Orchestrates all experiments for Requirement 1:

  Experiment A — Algorithm comparison (no budget)
      Random vs. Greedy vs. ETC vs. UCB1 (no budget) vs. Clairvoyant
      → verifies O(log T) regret for UCB1

  Experiment B — Budget constraint effect
      UCB1 (no budget) vs. Budget-UCB1 (ρ=0.4) vs. Budget-UCB1 (ρ=0.2)
      → verifies budget satisfaction + regret degradation under tighter ρ

Plots generated:
  1. Cumulative pseudo-regret — Experiment A
  2. Cumulative pseudo-regret — Experiment B
  3. Cumulative cost trajectory vs. budget lines (Experiment B)
  4. Bid selection histogram — UCB1 and Budget-UCB1 (ρ=0.4)
  5. Lambda (λ_t) trajectory — Budget-UCB1 both ρ values
  6. Regret vs T (log-log) — UCB1 and Budget-UCB1 scalability check

Run:
    python main.py

All plots are shown inline.  Set SAVE_FIGURES = True to write PNGs.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type


from dataclasses import dataclass
from typing import Dict, Sequence

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from agents import CombinatorialUCB1BiddingAgent
from environment import MultiCampaignEnv, compute_clairvoyant_mc

# ── Project modules ──────────────────────────────────────────────────────────
from config import (
    V, BID_SET, K, T, N_TRIALS, RANDOM_SEED_START,
    RHO_MODERATE, RHO_TIGHT, BUDGET_MODERATE, BUDGET_TIGHT,
    ETA_DUAL, DIST_CONFIGS, DEFAULT_DIST, T0_ETC,
    UCB_EXPLORATION_FACTOR,
    FIGURE_SIZE, UNCERTAINTY_ALPHA, AGENT_COLORS,
    OPT_OUT_ARM_IDX,
)
from environment import (
    SingleCampaignEnv,
    compute_true_arm_means,
    compute_clairvoyant,
    validate_environment,
)
from agents import (
    Agent,
    RandomBiddingAgent,
    GreedyBiddingAgent,
    ETCBiddingAgent,
    UCB1BiddingAgent,
    BudgetUCB1BiddingAgent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Global flag — set to True to save every figure as a PNG
# ─────────────────────────────────────────────────────────────────────────────
SAVE_FIGURES: bool = False
FIGURES_DIR:  str  = "figures_req1"


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Raw results for a single trial of a single agent."""
    rewards:   np.ndarray   # shape (T,) — per-round reward
    costs:     np.ndarray   # shape (T,) — per-round cost
    actions:   np.ndarray   # shape (T,) — arm index chosen
    lambdas:   np.ndarray   # shape (T,) — λ_t  (zeros for non-budget agents)


@dataclass
class AgentResults:
    """Aggregated results across N_TRIALS trials for one agent."""
    label:          str
    rewards:        np.ndarray   # (N_TRIALS, T)
    costs:          np.ndarray   # (N_TRIALS, T)
    actions:        np.ndarray   # (N_TRIALS, T)
    lambdas:        np.ndarray   # (N_TRIALS, T)
    cum_regrets:    np.ndarray   # (N_TRIALS, T) — cumulative pseudo-regret
    mean_regret:    np.ndarray   # (T,)
    se_regret:      np.ndarray   # (T,) — standard error = std / sqrt(n)
    mean_cum_cost:  np.ndarray   # (T,)
    se_cum_cost:    np.ndarray   # (T,)
    mean_lambda:    np.ndarray   # (T,)
    se_lambda:      np.ndarray   # (T,)
    final_regret_mean: float
    final_regret_std:  float


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single-trial runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_trial(
    agent_class:  Type[Agent],
    agent_kwargs: dict,
    env_kwargs:   dict,
    bid_set:      np.ndarray,
    T:            int,
) -> TrialResult:
    """
    Run ONE trial: create environment + agent, loop T rounds, collect feedback.

    The caller is responsible for setting  np.random.seed()  before this
    function is invoked (ensures reproducibility across agents on the same
    noise sequence — notebook 01 convention).

    Returns
    -------
    TrialResult with per-round arrays of shape (T,).
    """
    env   = SingleCampaignEnv(**env_kwargs)
    agent = agent_class(**agent_kwargs)

    rewards = np.empty(T, dtype=float)
    costs   = np.empty(T, dtype=float)
    actions = np.empty(T, dtype=int)
    lambdas = np.zeros(T, dtype=float)

    for t in range(T):
        # 1. Agent picks arm index
        arm_idx = agent.pull_arm()

        # 2. Map arm index → bid value, execute auction round.
        #    round() returns only (win_t, r_t, c_t) — bandit feedback per
        #    project spec slide 6 ("set of won auctions").
        b_t = bid_set[arm_idx]
        win_t, r_t, c_t = env.round(b_t)

        # 3. Agent updates internal statistics
        agent.update(r_t, c_t)

        # 4. Log
        rewards[t] = r_t
        costs[t]   = c_t
        actions[t] = arm_idx

        # 5. Log λ_t for budget agent (already appended inside pull_arm)
        if isinstance(agent, BudgetUCB1BiddingAgent):
            lambdas[t] = agent.lmbd_history[t]

    return TrialResult(
        rewards=rewards,
        costs=costs,
        actions=actions,
        lambdas=lambdas,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-trial runner + aggregation
# ─────────────────────────────────────────────────────────────────────────────

def run_all_trials(
    agent_class:          Type[Agent],
    agent_kwargs:         dict,
    env_kwargs:           dict,
    bid_set:              np.ndarray,
    T:                    int,
    n_trials:             int,
    clairvoyant_per_round: float,
    label:                str,
    seed_start:           int = RANDOM_SEED_START,
    verbose:              bool = True,
) -> AgentResults:
    """
    Run N independent trials for one agent and aggregate statistics.

    Pseudo-regret per trial:
        R_t = Σ_{s=1}^t  ( clairvoyant_per_round − r_s )

    which equals the cumulative sum of the per-round regret
        δ_t = clairvoyant_per_round − r_t

    This matches the regret computation in notebook 01:
        cumulative_regret = np.cumsum(expected_clairvoyant_rewards − agent_rewards)

    Uncertainty quantification:
        uncertainty band = mean ± std / sqrt(n_trials)   (standard error)
    This is the convention used in EVERY notebook in the course.

    Parameters
    ----------
    clairvoyant_per_round : float
        OPT^S (or OPT if no budget) — the per-round reward of the best
        feasible policy.  Computed once by compute_clairvoyant() before
        calling this function.

    Returns
    -------
    AgentResults with pre-computed statistics ready for plotting.
    """
    if verbose:
        print(f"  [{label}]  running {n_trials} trials ...", end="", flush=True)

    all_rewards  = np.empty((n_trials, T), dtype=float)
    all_costs    = np.empty((n_trials, T), dtype=float)
    all_actions  = np.empty((n_trials, T), dtype=int)
    all_lambdas  = np.zeros((n_trials, T), dtype=float)

    for seed in range(seed_start, seed_start + n_trials):
        np.random.seed(seed)   # reproducibility — notebook 01 convention
        trial = run_single_trial(
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
            env_kwargs=env_kwargs,
            bid_set=bid_set,
            T=T,
        )
        i = seed - seed_start
        all_rewards[i]  = trial.rewards
        all_costs[i]    = trial.costs
        all_actions[i]  = trial.actions
        all_lambdas[i]  = trial.lambdas

    if verbose:
        print(" done.")

    # ── Regret  (notebook 01 pattern: cumsum of per-round gap) ───────────────
    per_round_regret = clairvoyant_per_round - all_rewards   # (n_trials, T)
    cum_regrets      = np.cumsum(per_round_regret, axis=1)   # (n_trials, T)

    # ── Aggregate: mean and standard error ───────────────────────────────────
    def _stats(arr2d):
        mean = arr2d.mean(axis=0)
        se   = arr2d.std(axis=0) / np.sqrt(n_trials)
        return mean, se

    mean_regret,   se_regret   = _stats(cum_regrets)
    mean_cum_cost, se_cum_cost = _stats(np.cumsum(all_costs, axis=1))
    mean_lambda,   se_lambda   = _stats(all_lambdas)

    return AgentResults(
        label           = label,
        rewards         = all_rewards,
        costs           = all_costs,
        actions         = all_actions,
        lambdas         = all_lambdas,
        cum_regrets     = cum_regrets,
        mean_regret     = mean_regret,
        se_regret       = se_regret,
        mean_cum_cost   = mean_cum_cost,
        se_cum_cost     = se_cum_cost,
        mean_lambda     = mean_lambda,
        se_lambda       = se_lambda,
        final_regret_mean = float(cum_regrets[:, -1].mean()),
        final_regret_std  = float(cum_regrets[:, -1].std()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_save(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG if SAVE_FIGURES is True."""
    if SAVE_FIGURES:
        import os
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {path}")


def plot_cumulative_regret(
    results:  Dict[str, AgentResults],
    T:        int,
    title:    str = "Cumulative Pseudo-Regret",
    filename: str = "regret",
) -> None:
    """
    Plot mean cumulative pseudo-regret with ±SE uncertainty bands.

    Follows the exact style used throughout all 10 course notebooks:
        plt.plot(...)
        plt.fill_between(..., alpha=UNCERTAINTY_ALPHA)
    Uncertainty = mean ± std/sqrt(n_trials)  (standard error).
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    t_axis = np.arange(T)

    for label, res in results.items():
        colour = AGENT_COLORS.get(label, None)
        ax.plot(t_axis, res.mean_regret, label=label, color=colour)
        ax.fill_between(
            t_axis,
            res.mean_regret - res.se_regret,
            res.mean_regret + res.se_regret,
            alpha=UNCERTAINTY_ALPHA,
            color=colour,
        )

    ax.set_xlabel("$t$",    fontsize=13)
    ax.set_ylabel("$R_t$",  fontsize=13)
    ax.set_title(title,     fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, filename)
    plt.show()


def plot_cumulative_cost(
    results:       Dict[str, AgentResults],
    budget_lines:  Dict[str, float],
    T:             int,
    title:         str = "Cumulative Cost vs. Budget",
    filename:      str = "cost",
) -> None:
    """
    Plot mean cumulative cost with budget constraint lines.

    budget_lines: dict mapping legend label → budget value B = ρT.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    t_axis = np.arange(T)

    for label, res in results.items():
        colour = AGENT_COLORS.get(label, None)
        ax.plot(t_axis, res.mean_cum_cost, label=label, color=colour)
        ax.fill_between(
            t_axis,
            res.mean_cum_cost - res.se_cum_cost,
            res.mean_cum_cost + res.se_cum_cost,
            alpha=UNCERTAINTY_ALPHA,
            color=colour,
        )

    # ── Budget reference lines ────────────────────────────────────────────────
    budget_colors = {"ρ=0.4  (B=2000)": "#ff7f0e", "ρ=0.2  (B=1000)": "#d62728"}
    for bl_label, bl_val in budget_lines.items():
        c = budget_colors.get(bl_label, "black")
        ax.axhline(bl_val, color=c, linestyle="--", linewidth=1.5,
                   label=bl_label)

    ax.set_xlabel("$t$",                     fontsize=13)
    ax.set_ylabel("Cumulative cost",         fontsize=13)
    ax.set_title(title,                      fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, filename)
    plt.show()


def plot_bid_histogram(
    results:  Dict[str, AgentResults],
    bid_set:  np.ndarray,
    mu_true:  np.ndarray,
    title:    str = "Bid Selection Frequency",
    filename: str = "bid_histogram",
) -> None:
    """
    Bar chart of arm selection frequency averaged over all trials.
    Overlays true μ(b) as a line to show convergence to b*.
    """
    n_agents = len(results)
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5),
                             sharey=False)
    if n_agents == 1:
        axes = [axes]

    for ax, (label, res) in zip(axes, results.items()):
        # Average pull count per bid across trials
        avg_pulls = res.actions.reshape(-1).astype(int)
        counts    = np.bincount(avg_pulls, minlength=len(bid_set))
        freq      = counts / counts.sum()

        colour = AGENT_COLORS.get(label, "steelblue")
        ax.bar(bid_set, freq, width=0.07, color=colour, alpha=0.75,
               label="Selection freq.")

        # Overlay μ(b) (normalised to same scale)
        mu_norm = mu_true / mu_true.max() if mu_true.max() > 0 else mu_true
        ax2 = ax.twinx()
        ax2.plot(bid_set, mu_true, "k--", linewidth=1.5,
                 label="True $\\mu(b)$")
        ax2.set_ylabel("$\\mu(b)$", fontsize=11)

        best_bid = bid_set[np.argmax(mu_true)]
        ax.axvline(best_bid, color="green", linestyle=":", linewidth=2,
                   label=f"$b^*={best_bid:.2f}$")

        ax.set_xlabel("Bid $b$",   fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(label,        fontsize=12)

        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=9)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    _maybe_save(fig, filename)
    plt.show()


def plot_lambda_trajectory(
    results:  Dict[str, AgentResults],
    T:        int,
    rho_map:  Dict[str, float],
    title:    str = "Lagrange Multiplier $\\lambda_t$ Trajectory",
    filename: str = "lambda",
) -> None:
    """
    Plot the evolution of the dual variable λ_t over rounds.

    λ_t ↑ when the agent overspends  (c_t > ρ)
    λ_t ↓ when the agent underspends (c_t < ρ)
    Should converge toward the optimal λ* = dual value of the LP.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    t_axis = np.arange(T)

    for label, res in results.items():
        if np.all(res.mean_lambda == 0.0):
            continue   # non-budget agent: skip
        colour = AGENT_COLORS.get(label, None)
        ax.plot(t_axis, res.mean_lambda, label=label, color=colour)
        ax.fill_between(
            t_axis,
            res.mean_lambda - res.se_lambda,
            res.mean_lambda + res.se_lambda,
            alpha=UNCERTAINTY_ALPHA,
            color=colour,
        )
        # Upper bound reference: 1/ρ
        rho = rho_map.get(label, None)
        if rho:
            ax.axhline(1.0 / rho, color=colour, linestyle=":",
                       linewidth=1.0, alpha=0.6,
                       label=f"$1/\\rho = {1/rho:.1f}$ ({label})")

    ax.set_xlabel("$t$",              fontsize=13)
    ax.set_ylabel("$\\lambda_t$",     fontsize=13)
    ax.set_title(title,               fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, filename)
    plt.show()


def plot_regret_scaling(
    agent_class:    Type[Agent],
    agent_kwargs_fn,          # callable: T_val → agent_kwargs
    env_kwargs_fn,            # callable: T_val → env_kwargs
    clairvoyant_fn,           # callable: T_val → clairvoyant_per_round
    T_values:       List[int],
    n_trials:       int,
    label:          str,
    title:          str = "Regret Scaling vs. T  (log-log)",
    filename:       str = "regret_scaling",
) -> None:
    """
    Plot final cumulative regret R_T vs. T on a log-log scale.
    A slope of 1 on this scale confirms O(T) growth.
    A slope of ~0 (flat) confirms O(log T) growth.

    Used to empirically verify the theoretical regret bound of UCB1.
    """
    final_regrets = []
    for T_val in T_values:
        res = run_all_trials(
            agent_class           = agent_class,
            agent_kwargs          = agent_kwargs_fn(T_val),
            env_kwargs            = env_kwargs_fn(T_val),
            bid_set               = BID_SET,
            T                     = T_val,
            n_trials              = n_trials,
            clairvoyant_per_round = clairvoyant_fn(T_val),
            label                 = f"{label} T={T_val}",
            verbose               = False,
        )
        final_regrets.append(res.final_regret_mean)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(T_values, final_regrets, "o-", label=f"Empirical $R_T$ — {label}",
              color=AGENT_COLORS.get(label, "steelblue"))

    # Reference lines: O(log T) and O(T)
    T_arr = np.array(T_values, dtype=float)
    c_log = final_regrets[0] / np.log(T_values[0])
    c_lin = final_regrets[0] / T_values[0]
    ax.loglog(T_arr, c_log * np.log(T_arr), "k--", alpha=0.6,
              label="$\\mathcal{O}(\\log T)$")
    ax.loglog(T_arr, c_lin * T_arr,         "r--", alpha=0.4,
              label="$\\mathcal{O}(T)$")

    ax.set_xlabel("$T$",       fontsize=13)
    ax.set_ylabel("$R_T$",     fontsize=13)
    ax.set_title(title,        fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, filename)
    plt.show()


def print_summary_table(
    results: Dict[str, AgentResults],
    clairvoyant_value: float,
    budget_map: Dict[str, float] = None,
) -> None:
    """
    Print a formatted table of final regret and budget compliance.
    """
    print("\n" + "=" * 70)
    print(f"  {'Agent':<30}  {'R_T mean':>10}  {'R_T std':>8}  {'Budget OK':>10}")
    print("=" * 70)
    for label, res in results.items():
        budget_ok = "N/A"
        if budget_map and label in budget_map:
            B        = budget_map[label]
            # Check strict budget compliance across all trials.
            # Use the MAX over all trials (worst-case), not the mean.
            max_cost = res.costs.cumsum(axis=1)[:, -1].max()
            overshoot = max(0.0, max_cost - B)
            budget_ok = (
                f"YES"               if overshoot == 0.0   else
                f"~YES (+{overshoot:.1f})" if overshoot < 1.0   else
                f"NO  (+{overshoot:.1f})"
            )
        print(
            f"  {label:<30}  {res.final_regret_mean:>10.2f}  "
            f"{res.final_regret_std:>8.2f}  {budget_ok:>10}"
        )
    print(f"\n  Clairvoyant per-round value: {clairvoyant_value:.4f}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Experiment A — Algorithm comparison (no budget)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_A(
    dist_config: dict = None,
    T:           int  = T,
    n_trials:    int  = N_TRIALS,
    verbose:     bool = True,
) -> Dict[str, AgentResults]:
    """
    Compare Random / Greedy / ETC / UCB1 (no budget) against the
    unconstrained clairvoyant (best pure strategy).

    Purpose: verify that UCB1 achieves O(log T) regret while baselines
    have linear or worse regret.
    """
    if dist_config is None:
        dist_config = DIST_CONFIGS[DEFAULT_DIST]

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  EXPERIMENT A — Algorithm comparison (no budget)")
        print(f"  T={T}, n_trials={n_trials}, dist={dist_config['label']}")
        print(f"{'─'*60}")

    # ── Clairvoyant ───────────────────────────────────────────────────────────
    x_opt, opt_value, opt_cost = compute_clairvoyant(
        BID_SET, V, dist_config, rho=None
    )
    mu_true  = compute_true_arm_means(BID_SET, V, dist_config)
    best_bid = BID_SET[int(np.argmax(mu_true))]

    if verbose:
        print(f"\n  Clairvoyant:  b* = {best_bid:.2f},  OPT = {opt_value:.4f}")
        print(f"  True μ(b):    {np.round(mu_true, 4)}\n")

    # ── Shared environment kwargs ─────────────────────────────────────────────
    env_kwargs = dict(v=V, bid_set=BID_SET, T=T, dist_config=dist_config)

    # ── Agent configurations ──────────────────────────────────────────────────
    agents_cfg = {
        "Random":          (RandomBiddingAgent,  dict(bid_set=BID_SET, T=T, v=V)),
        "Greedy":          (GreedyBiddingAgent,  dict(bid_set=BID_SET, T=T, v=V)),
        "ETC":             (ETCBiddingAgent,     dict(bid_set=BID_SET, T=T, v=V, T0=T0_ETC)),
        "UCB1 (no budget)":(UCB1BiddingAgent,   dict(bid_set=BID_SET, T=T, v=V)),
    }

    results: Dict[str, AgentResults] = {}
    for label, (AgentClass, kwargs) in agents_cfg.items():
        results[label] = run_all_trials(
            agent_class           = AgentClass,
            agent_kwargs          = kwargs,
            env_kwargs            = env_kwargs,
            bid_set               = BID_SET,
            T                     = T,
            n_trials              = n_trials,
            clairvoyant_per_round = opt_value,
            label                 = label,
            verbose               = verbose,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cumulative_regret(
        results,
        T,
        title    = f"Req 1A — Cumulative Pseudo-Regret  [{dist_config['label']}]",
        filename = "exp_A_regret",
    )

    plot_bid_histogram(
        {"UCB1 (no budget)": results["UCB1 (no budget)"]},
        BID_SET,
        mu_true,
        title    = "Req 1A — Bid Selection: UCB1 (no budget)",
        filename = "exp_A_bid_hist",
    )

    print_summary_table(results, opt_value)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Experiment B — Budget constraint effect
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_B(
    dist_config: dict = None,
    T:           int  = T,
    n_trials:    int  = N_TRIALS,
    verbose:     bool = True,
) -> Dict[str, AgentResults]:
    """
    Compare UCB1 (no budget) vs. Budget-UCB1 at two budget levels.

    Purpose:
      • Budget-UCB1 should keep cumulative cost ≤ B = ρT (budget satisfied).
      • Regret increases as ρ decreases (tighter budget → worse performance).
      • λ_t should rise when over budget and fall when under budget.
    """
    if dist_config is None:
        dist_config = DIST_CONFIGS[DEFAULT_DIST]

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  EXPERIMENT B — Budget constraint effect")
        print(f"  T={T}, n_trials={n_trials}, dist={dist_config['label']}")
        print(f"  ρ values: {RHO_MODERATE} (moderate), {RHO_TIGHT} (tight)")
        print(f"{'─'*60}")

    env_kwargs = dict(v=V, bid_set=BID_SET, T=T, dist_config=dist_config)

    # ── Clairvoyants (one per budget level + unconstrained) ───────────────────
    _, opt_no_budget, _ = compute_clairvoyant(BID_SET, V, dist_config, rho=None)
    _, opt_mod,       _ = compute_clairvoyant(BID_SET, V, dist_config, rho=RHO_MODERATE)
    _, opt_tight,     _ = compute_clairvoyant(BID_SET, V, dist_config, rho=RHO_TIGHT)
    mu_true              = compute_true_arm_means(BID_SET, V, dist_config)

    if verbose:
        print(f"\n  OPT (no budget):   {opt_no_budget:.4f}")
        print(f"  OPT^S (ρ=0.4):     {opt_mod:.4f}")
        print(f"  OPT^S (ρ=0.2):     {opt_tight:.4f}\n")

    # ── Agent configurations ──────────────────────────────────────────────────
    # Each budget agent is evaluated against its OWN clairvoyant OPT^S
    agents_cfg = {
        "UCB1 (no budget)": (
            UCB1BiddingAgent,
            dict(bid_set=BID_SET, T=T, v=V),
            opt_no_budget,
        ),
        "Budget-UCB1 (ρ=0.4)": (
            BudgetUCB1BiddingAgent,
            dict(bid_set=BID_SET, T=T, v=V, rho=RHO_MODERATE, eta=ETA_DUAL),
            opt_mod,
        ),
        "Budget-UCB1 (ρ=0.2)": (
            BudgetUCB1BiddingAgent,
            dict(bid_set=BID_SET, T=T, v=V, rho=RHO_TIGHT, eta=ETA_DUAL),
            opt_tight,
        ),
    }

    results: Dict[str, AgentResults] = {}
    for label, (AgentClass, kwargs, clv_val) in agents_cfg.items():
        results[label] = run_all_trials(
            agent_class           = AgentClass,
            agent_kwargs          = kwargs,
            env_kwargs            = env_kwargs,
            bid_set               = BID_SET,
            T                     = T,
            n_trials              = n_trials,
            clairvoyant_per_round = clv_val,
            label                 = label,
            verbose               = verbose,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cumulative_regret(
        results,
        T,
        title    = "Req 1B — Cumulative Pseudo-Regret: Budget Effect",
        filename = "exp_B_regret",
    )

    plot_cumulative_cost(
        {k: v for k, v in results.items() if "Budget" in k},
        budget_lines={
            "ρ=0.4  (B=2000)": BUDGET_MODERATE,
            "ρ=0.2  (B=1000)": BUDGET_TIGHT,
        },
        T        = T,
        title    = "Req 1B — Cumulative Cost vs. Budget Constraint",
        filename = "exp_B_cost",
    )

    plot_lambda_trajectory(
        {k: v for k, v in results.items() if "Budget" in k},
        T        = T,
        rho_map  = {
            "Budget-UCB1 (ρ=0.4)": RHO_MODERATE,
            "Budget-UCB1 (ρ=0.2)": RHO_TIGHT,
        },
        title    = "Req 1B — Lagrange Multiplier $\\lambda_t$ Evolution",
        filename = "exp_B_lambda",
    )

    budget_map = {
        "Budget-UCB1 (ρ=0.4)": BUDGET_MODERATE,
        "Budget-UCB1 (ρ=0.2)": BUDGET_TIGHT,
    }
    print_summary_table(results, opt_mod, budget_map)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Experiment C — Regret vs. T scaling  (log-log verification)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_C(
    dist_config: dict = None,
    T_values:    List[int] = None,
    n_trials:    int = 20,
    verbose:     bool = True,
) -> None:
    """
    Verify O(log T) regret growth for UCB1 by plotting R_T vs. T on a
    log-log scale.  A flat line (slope 0 on log-log) confirms O(log T).

    Uses fewer trials (n_trials=20) for speed; log-log slope is robust.
    """
    if dist_config is None:
        dist_config = DIST_CONFIGS[DEFAULT_DIST]
    if T_values is None:
        T_values = [500, 1_000, 2_000, 5_000, 10_000]

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  EXPERIMENT C — Regret scaling vs. T  (log-log)")
        print(f"  T values: {T_values}")
        print(f"{'─'*60}")

    def _env_kw(T_val):
        return dict(v=V, bid_set=BID_SET, T=T_val, dist_config=dist_config)

    def _ucb_kw(T_val):
        return dict(bid_set=BID_SET, T=T_val, v=V)

    def _budget_kw(T_val):
        return dict(bid_set=BID_SET, T=T_val, v=V, rho=RHO_MODERATE,
                    eta=1.0 / np.sqrt(T_val))

    def _clv_no_budget(T_val):
        _, val, _ = compute_clairvoyant(BID_SET, V, dist_config, rho=None)
        return val

    def _clv_budget(T_val):
        _, val, _ = compute_clairvoyant(BID_SET, V, dist_config, rho=RHO_MODERATE)
        return val

    plot_regret_scaling(
        agent_class    = UCB1BiddingAgent,
        agent_kwargs_fn= _ucb_kw,
        env_kwargs_fn  = _env_kw,
        clairvoyant_fn = _clv_no_budget,
        T_values       = T_values,
        n_trials       = n_trials,
        label          = "UCB1 (no budget)",
        title          = "Req 1C — Regret Scaling: UCB1 vs. T",
        filename       = "exp_C_scaling_ucb1",
    )

    plot_regret_scaling(
        agent_class    = BudgetUCB1BiddingAgent,
        agent_kwargs_fn= _budget_kw,
        env_kwargs_fn  = _env_kw,
        clairvoyant_fn = _clv_budget,
        T_values       = T_values,
        n_trials       = n_trials,
        label          = "Budget-UCB1 (ρ=0.4)",
        title          = "Req 1C — Regret Scaling: Budget-UCB1 vs. T",
        filename       = "exp_C_scaling_budget",
    )

"""
main_req2.py — Requirement 2 orchestration
==========================================
Standalone runner for the multiple-campaign stochastic environment.
"""


@dataclass
class TrialResult2:
    rewards: np.ndarray
    costs: np.ndarray
    regrets: np.ndarray
    actions: np.ndarray
    lambda_history: np.ndarray
    budget_history: np.ndarray


def run_single_trial2(
    env: MultiCampaignEnv,
    agent: CombinatorialUCB1BiddingAgent,
    clairvoyant_value: float,
) -> TrialResult2:
    # Store the per-round outcomes for one complete trial.
    rewards = np.zeros((env.T, env.n_campaigns), dtype=float)
    costs = np.zeros((env.T, env.n_campaigns), dtype=float)
    regrets = np.zeros(env.T, dtype=float)
    actions = np.zeros((env.T, env.n_campaigns), dtype=bool)

    for t in range(env.T):
        # 1) Agent chooses the campaigns and bids.
        bids, active = agent.pull_action()
        # 2) Environment returns the realized feedback.
        win_t, r_t, c_t = env.round(bids, active)
        # 3) Agent updates its estimates.
        agent.update(r_t, c_t)
        rewards[t] = r_t
        costs[t] = c_t
        actions[t] = active
        # Regret is computed against the clairvoyant benchmark.
        regrets[t] = clairvoyant_value - float(r_t.sum())

    return TrialResult2(
        rewards=rewards,
        costs=costs,
        regrets=np.cumsum(regrets),
        actions=actions,
        lambda_history=agent.get_histories()[0],
        budget_history=agent.get_histories()[1],
    )


def _plot_cumulative_regret(results: Sequence[TrialResult2], title: str) -> None:
    """Plot mean cumulative regret with a simple standard-error band."""
    regret_mat = np.stack([r.regrets for r in results], axis=0)
    mean_regret = regret_mat.mean(axis=0)
    se_regret = regret_mat.std(axis=0) / np.sqrt(regret_mat.shape[0])
    t_axis = np.arange(regret_mat.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, mean_regret, color="steelblue", label="Mean cumulative regret")
    ax.fill_between(
        t_axis,
        mean_regret - se_regret,
        mean_regret + se_regret,
        color="steelblue",
        alpha=0.25,
        label="± SE",
    )
    ax.set_title(title)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative regret")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_cumulative_cost(results: Sequence[TrialResult2], budget_line: float, title: str) -> None:
    """Plot mean cumulative cost and the budget reference line."""
    cost_mat = np.stack([r.costs.sum(axis=1).cumsum() for r in results], axis=0)
    mean_cost = cost_mat.mean(axis=0)
    se_cost = cost_mat.std(axis=0) / np.sqrt(cost_mat.shape[0])
    t_axis = np.arange(cost_mat.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, mean_cost, color="darkorange", label="Mean cumulative cost")
    ax.fill_between(
        t_axis,
        mean_cost - se_cost,
        mean_cost + se_cost,
        color="darkorange",
        alpha=0.25,
        label="± SE",
    )
    ax.axhline(budget_line, color="crimson", linestyle="--", label="Budget line")
    ax.set_title(title)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_campaign_activity(results: Sequence[TrialResult2], title: str) -> None:
    """Plot how often each campaign is selected over time."""
    action_mat = np.stack([r.actions.astype(float) for r in results], axis=0)
    mean_action = action_mat.mean(axis=0)  # (T, N)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mean_action.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Campaign i")
    fig.colorbar(im, ax=ax, label="Selection frequency")
    plt.tight_layout()
    plt.show()


def run_experiment_req2(
    T: int = 2000,
    n_trials: int = 20,
    rho: float = 0.6,
):
    # Small toy instance, deliberately kept simple so the combinatorial search
    # stays readable and can be inspected by hand.
    bid_set = np.round(np.linspace(0.0, 1.0, 6), 2)
    values = np.array([1.0, 0.9, 1.1,1.0], dtype=float)
    dist_configs = [
        {"type": "uniform", "low": 0.0, "high": 1.0},
        {"type": "beta", "a": 2, "b": 5},
        {"type": "normal",     "loc": 0.5, "scale": 0.2},
        {"type": "normal", "loc": 0.5, "scale": 0.2},
    ]
    conflict_graph = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    )

    results = []
    for seed in range(n_trials):
        # Same seed => same stochastic environment for repeatable trials.
        np.random.seed(seed)
        env = MultiCampaignEnv(
            bid_set=bid_set,
            values=values,
            dist_configs=dist_configs,
            T=T,
            conflict_graph=conflict_graph,
            rho=rho,
            correlation=0.25,
        )
        _, _, clairvoyant_value = compute_clairvoyant_mc(
            bid_set=bid_set,
            values=values,
            dist_configs=dist_configs,
            conflict_graph=conflict_graph,
            rho=rho,
            correlation=0.25,
        )
        agent = CombinatorialUCB1BiddingAgent(
            bid_set=bid_set,
            values=values,
            conflict_graph=conflict_graph,
            T=T,
            rho=rho,
        )
        # Run one trial and store the trajectory.
        results.append(run_single_trial2(env, agent, clairvoyant_value))

    # Report one simple summary number so we can sanity-check the run.
    mean_final_regret = float(np.mean([r.regrets[-1] for r in results]))
    print(f"Requirement 2 complete. Mean final regret: {mean_final_regret:.2f}")


    # Plots: regret, cost, and campaign activity.
    _plot_cumulative_regret(results, title="Req 2 — Cumulative Regret")
    _plot_cumulative_cost(
        results,
        budget_line=rho * T,
        title="Req 2 — Cumulative Cost vs Budget",
    )
    _plot_campaign_activity(results, title="Req 2 — Campaign Activity Heatmap")
    return results
# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # """
    # Run all Requirement 1 experiments in sequence.
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 0: Environment validation                                     │
    # │  Step 1: Experiment A — UCB1 vs. baselines (no budget)             │
    # │  Step 2: Experiment B — budget constraint effect                    │
    # │  Step 3: Experiment C — regret vs. T scaling (log-log)             │
    # └─────────────────────────────────────────────────────────────────────┘
    # """
    # print("\n" + "=" * 60)
    # print("  REQUIREMENT 1 — Single Campaign, Stochastic Environment")
    # print("=" * 60)
    #
    # dist_config = DIST_CONFIGS[DEFAULT_DIST]
    # print(f"\n  Distribution : {dist_config['label']}")
    # print(f"  T            : {T}")
    # print(f"  N_TRIALS     : {N_TRIALS}")
    # print(f"  K (bids)     : {K},  B = {BID_SET}")
    # print(f"  V            : {V}")
    # print(f"  RHO_MODERATE : {RHO_MODERATE}  (B = {BUDGET_MODERATE:.0f})")
    # print(f"  RHO_TIGHT    : {RHO_TIGHT}  (B = {BUDGET_TIGHT:.0f})")
    # print(f"  ETA_DUAL     : {ETA_DUAL:.5f}")
    # print(f"  T0_ETC       : {T0_ETC}")
    #
    # # ── Step 0: Validate environment ──────────────────────────────────────────
    # print("\n[0] Validating environment ...")
    # validate_environment(V, BID_SET, dist_config, n_rounds=5_000)
    #
    # # ── Step 1: Experiment A — all three distributions ───────────────────────
    # print("\n[1a] Experiment A — Uniform(0,1) distribution")
    # run_experiment_A(dist_config=DIST_CONFIGS["uniform"], T=T, n_trials=N_TRIALS)
    #
    # print("\n[1b] Experiment A — Beta(2,5) distribution")
    # run_experiment_A(dist_config=DIST_CONFIGS["beta"], T=T, n_trials=N_TRIALS)
    #
    # print("\n[1c] Experiment A — Truncated Normal(0.5, 0.2) distribution")
    # run_experiment_A(dist_config=DIST_CONFIGS["normal"], T=T, n_trials=N_TRIALS)
    #
    # # ── Step 2: Experiment B — budget constraint (uniform, canonical case) ───
    # print("\n[2] Experiment B — Budget constraint effect  [Uniform(0,1)]")
    # run_experiment_B(dist_config=DIST_CONFIGS["uniform"], T=T, n_trials=N_TRIALS)
    #
    # # ── Step 3: Experiment C — regret scaling ────────────────────────────────
    # print("\n[3] Experiment C — Regret scaling vs. T  [Uniform(0,1)]")
    # run_experiment_C(
    #     dist_config = DIST_CONFIGS["uniform"],
    #     T_values    = [500, 1_000, 2_000, 5_000, 10_000],
    #     n_trials    = 20,
    # )
    #
    # print("\n" + "=" * 60)
    # print("  Requirement 1 complete.")
    # print("=" * 60 + "\n")

    # ============================================================
    # REQUIREMENT 2
    # ============================================================
    print("\n" + "#" * 60)
    print("  REQUIREMENT 2 — Multiple Campaigns, Stochastic Environment")
    print("#" * 60)
    print("\n[1] Running Requirement 2 experiment ...")
    run_experiment_req2(T=2_000, n_trials=20, rho=0.6)

    print("\n" + "#" * 60)
    print("  Requirement 2 complete.")
    print("#" * 60 + "\n")





# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
