# Online Learning Applications - Project Roadmap

**Project Title:** Online Learning Algorithms for Multiple Advertising Campaigns Under Budget Constraints

**Evaluation:** 16 points (2nd part of course grading alongside written exam)

**Format:** Groups of 4-5 students | 20 min presentation + 10 min Q&A | Three submission dates (July, Sept, Dec)

---

## PROJECT BREAKDOWN

### REQUIREMENT 1: Single Campaign, Stochastic Environment

#### 1.1 Goal
Design and evaluate bidding algorithms for a **single advertising campaign** in a **stochastic environment** with competing bids drawn from a distribution.

#### 1.2 Problem Formulation

**Setting:**
- Single campaign with value $v$ (utility from winning the slot)
- Stochastic competing bid: $m_t \sim D$ at each round $t$
- Advertiser bids $b_t \in B$ (discrete set of possible bids)
- Win condition: $b_t \geq m_t$
- Utility if win: $v - b_t$
- Cost if win: $b_t$

**Decision at round t:**
- Choose bid $b_t \in B$
- Observe feedback: win ($b_t \geq m_t$) or lose ($b_t < m_t$)
- If win, observe utility $v - b_t$ and cost $b_t$

**Regret Notion:**
- **Pseudo-Regret (with budget B):** $R_T(B) = T \cdot \text{OPT}_B - \sum_{t=1}^T r_t(b_t)$
- Where $\text{OPT}_B = \max_{b} (v - b) \cdot \mathbb{P}(b \geq m) \cdot T$ subject to budget constraint
- Alternatively, can measure simple regret: $\max_b (v-b)\mathbb{P}(b \geq m_t) - (v - b_t)\mathbb{1}[b_t \geq m_t]$

#### 1.3 Environment Implementation

**Class Name:** `StochasticSingleCampaignEnvironment`

**Parameters:**
- `v` (float): Campaign value
- `bid_set` (list/array): Discrete set of possible bids $B$
- `distribution_params` (dict): Parameters for competing bid distribution $D$
  - Example: `{"type": "uniform", "low": 0, "high": v}` or `{"type": "lognormal", "mean": ..., "std": ...}`
- `T` (int): Total rounds

**Methods:**
- `round(b_t)` → returns `(m_t, win_t, utility_t, cost_t)` where:
  - `m_t`: competing bid
  - `win_t`: boolean whether $b_t \geq m_t$
  - `utility_t`: $v - b_t$ if win, 0 otherwise
  - `cost_t`: $b_t$ if win, 0 otherwise

**Assumptions:**
- Competing bids are i.i.d. across rounds
- Distribution is fixed (no changes within this requirement)
- Bids are from a known discrete set

#### 1.4 Algorithm 1a: UCB1 for Single Campaign (No Budget)

**Algorithm Name:** `UCB1SingleCampaignAgent`

**Theory Connection:**
- Extends notebook **01_stochastic_mabs.ipynb** UCB1 class
- Adapts from Bernoulli MAB to continuous bid space with stochastic feedback
- Key insight: Each bid $b \in B$ is an "arm" with unknown reward $r(b) = (v - b) \cdot \mathbb{P}(b \geq m)$

**Mathematical Formulation:**

At each round $t$:
1. Maintain empirical means: $\hat{r}(b) = \frac{1}{n_b} \sum_{s: b_s = b} r_s$
2. Compute UCB for each bid: $\text{UCB}(b) = \hat{r}(b) + \sqrt{\frac{2\log T}{n_b}}$
3. Select: $b_t = \arg\max_b \text{UCB}(b)$
4. Observe reward $r_t = (v - b_t) \cdot \mathbb{1}[b_t \geq m_t]$ and update $\hat{r}(b_t)$, $n_{b_t}$

**Inputs/Outputs:**
- **Input:** Environment, T, bid_set, exploration bonus scaling (default 1)
- **Output:** Sequence of bids $b_1, ..., b_T$, sequence of rewards, cumulative regret trajectory

**Expected Behavior:**
- Regret should be sublinear: $\mathcal{O}(\log T)$ if rewards are well-separated
- Algorithm should explore different bids initially, then exploit the best one
- Performance degrades gracefully if bid values are very close

**Potential Difficulties:**
- Determining which bid is actually "best" depends on the distribution
- If distribution concentrates at certain values, some bids may be dominated
- Need to handle discrete bid set appropriately

#### 1.5 Algorithm 1b: UCB1 with Budget Constraint

**Algorithm Name:** `BudgetConstrainedUCB1Agent`

**Theory Connection:**
- Extends notebook **08_constrained_problems.ipynb** concepts
- Formulates as constrained online problem: maximize reward subject to cost constraint
- Uses Lagrangian relaxation approach shown in lecture 9 (OGD)

**Mathematical Formulation:**

**Unconstrained problem at each round:**
$$\max_{x \in \Delta(B)} \sum_{b \in B} x(b) \cdot (v - b) \cdot \mathbb{P}(m \geq b)$$
$$\text{s.t.} \sum_{b \in B} x(b) \cdot b \leq \rho$$

Where $\rho = B/T$ is per-round budget, $B$ is total budget, $x(b)$ is probability of choosing bid $b$.

**Lagrangian:**
$$L(x, \lambda) = \sum_{b \in B} x(b)[(v - b)\mathbb{P}(m \geq b) - \lambda b]$$

**Algorithm:**
1. Maintain empirical estimates: $\hat{r}(b)$, $\hat{c}(b) = b$ (cost is deterministic)
2. Estimate optimal Lagrange multiplier: $\hat{\lambda}_t$
3. At round t, compute modified utilities: $\tilde{r}(b, \lambda_t) = \hat{r}(b) - \lambda_t \cdot b$
4. Apply UCB to modified utilities: $\text{UCB}(b, \lambda_t) = \tilde{r}(b, \lambda_t) + \sqrt{\frac{2\log T}{n_b}}$
5. Select: $b_t = \arg\max_b \text{UCB}(b, \lambda_t)$
6. Update $\lambda_t$ based on cumulative cost tracking

**Inputs/Outputs:**
- **Input:** Environment, T, bid_set, budget B
- **Output:** Bid sequence, rewards, costs, cumulative regret, cumulative cost trajectory

**Expected Behavior:**
- Should maintain budget constraint: $\sum_t \text{cost}_t \leq B + O(\sqrt{BT})$
- Regret should increase (worse than unconstrained) due to budget restrictions
- Should balance between high-utility bids (expensive) and low-utility bids (cheap)

**Potential Difficulties:**
- Need to estimate good Lagrange multiplier online
- Budget constraint violation at early rounds (need correction mechanism)
- Non-stationary nature of constraint satisfaction

#### 1.6 Required Plots for Requirement 1

1. **Cumulative Reward Trajectory** (multi-trial average with uncertainty bands)
   - x-axis: Round $t$
   - y-axis: $\sum_{s=1}^t r_s$
   - Show: Random agent, UCB1 (no budget), UCB1 (with budget), clairvoyant (best bid in hindsight)

2. **Cumulative Cost Trajectory** (for budget-constrained version)
   - x-axis: Round $t$
   - y-axis: $\sum_{s=1}^t c_s$
   - Show horizontal line at budget B
   - Should not exceed budget (or show violation for analysis)

3. **Bid Selection Histogram**
   - x-axis: Bid values
   - y-axis: Number of times selected across all rounds
   - Show convergence to best bid

4. **Cumulative Regret over Trials**
   - x-axis: Trial number
   - y-axis: Regret at end of trial
   - Show regret distribution and mean

#### 1.7 Experimental Methodology

**Parameters:**
- **T** (rounds): 1000, 5000 (scale to see asymptotic behavior)
- **Number of trials:** 50-100 (to estimate uncertainty)
- **Bid set B:** e.g., `[0, 0.1, 0.2, ..., 1.0]` (11 bids)
- **Campaign value v:** e.g., 1.0
- **Competing bid distribution:** 
  - Test 1: Uniform(0, 1)
  - Test 2: Normal($\mu=0.5$, $\sigma=0.2$) truncated to [0, 1]
  - Test 3: Beta(2, 5) scaled to [0, 1]
- **Budget B:** 0.4T (allows room for exploration and exploitation)

**Baseline algorithms:**
- Random: Select bid uniformly at random
- Greedy: Always select bid that maximizes last-observed reward
- Clairvoyant: Knows distribution, plays optimal bid always
- ETC: Explore-then-commit with equal pulls, then best

**Evaluation metrics:**
- Cumulative regret (compared to clairvoyant)
- Cumulative cost (must stay ≤ B)
- Convergence speed (when does exploration stop?)
- Concentration (do all trials converge to same bid?)

#### 1.8 Expected Results

**For UCB1 (no budget):**
- Logarithmic regret: $R_T = \mathcal{O}(\log T)$
- Converges to best bid with high probability
- Efficient exploration balances reward and information gain

**For Budget-Constrained UCB1:**
- Budget constraint approximately satisfied (within $O(\sqrt{BT})$)
- Higher regret than unconstrained (budget is limiting factor)
- May not explore all bids (budget prevents it)
- If budget is very tight, may get stuck with suboptimal bid

---

### REQUIREMENT 2: Multiple Campaigns, Stochastic Environment

#### 2.1 Goal
Design algorithms that simultaneously bid on **N competing campaigns** in a **stochastic multi-dimensional environment** with:
- Joint distribution of competing bids for all campaigns
- Budget constraint shared across all campaigns
- Constraint: Cannot bid on conflicting campaigns in same round (conflict graph)

#### 2.2 Problem Formulation

**Setting:**
- N campaigns, each with value $v_i$ and competing bid distribution $D_i$
- Conflict graph $G = (V, E)$ where vertices are campaigns and edges represent conflicts
- Discrete bid set $B$ (same for all campaigns)
- Total budget B spread across all campaigns and rounds

**Decision at round t:**
1. Choose bids $b_{i,t}$ for each campaign $i \in [N]$ from bid set B
2. Constraint: If $(i, j) \in E$ (campaigns conflict), can bid on at most one of them
3. Observe feedback:
   - Competing bids: $m_{i,t}$ for each campaign i
   - Wins: $w_{i,t} = \mathbb{1}[b_{i,t} \geq m_{i,t}]$ for each campaign
4. Rewards and costs:
   - Reward: $r_{i,t} = (v_i - b_{i,t}) \cdot w_{i,t}$ if won
   - Cost: $c_{i,t} = b_{i,t} \cdot w_{i,t}$ if won

**Regret Notion:**
- **Pseudo-Regret:** $R_T = T \cdot \text{OPT}_{\text{stoch}} - \sum_{t=1}^T \sum_i r_{i,t}$
- Where $\text{OPT}_{\text{stoch}}$ is optimal expected reward per round subject to:
  - Budget constraint: $\mathbb{E}[\sum_i c_{i,t}] \leq \rho$ per round
  - Conflict constraint: Choose set of campaigns that form independent set in $G$

**Optimization problem at each round (oracle):**
$$\text{OPT}_{\text{stoch}} = \max_{S \text{ independent in } G} \sum_{i \in S} (v_i - b_i^*) \mathbb{P}(b_i^* \geq m_i)$$
$$\text{s.t.} \sum_{i \in S} b_i^* \leq \rho$$

#### 2.3 Environment Implementation

**Class Name:** `StochasticMultiCampaignEnvironment`

**Parameters:**
- `v` (array, length N): Campaign values
- `bid_set` (list): Discrete set of bids
- `conflict_graph` (adjacency matrix or networkx graph): Conflict relationships
- `distribution_params` (list of dicts): Parameters for each campaign's competing bid distribution
  - Can be per-campaign or shared
- `T` (int): Total rounds
- `correlation` (float, 0-1): Whether competing bids are correlated across campaigns

**Methods:**
- `round(bid_vector)` → returns `(m_vector, w_vector, r_vector, c_vector)` where:
  - `m_vector`: Array of competing bids for each campaign
  - `w_vector`: Array of wins (boolean)
  - `r_vector`: Array of rewards
  - `c_vector`: Array of costs

**Assumptions:**
- Conflict graph is static (given in advance)
- Competing bids can be correlated or independent
- Distributions are fixed

#### 2.4 Algorithm 2a: Combinatorial UCB with Budget Constraint

**Algorithm Name:** `CombinatorialUCB1Agent`

**Theory Connection:**
- Extends notebook **09_combinatorial_mabs.ipynb** concepts
- Combines:
  - UCB confidence sets (from **02_stochastic_mabs_2.ipynb**)
  - Budget constraints (from **08_constrained_problems.ipynb**)
  - Combinatorial optimization (matching/independent set, from **09_combinatorial_mabs.ipynb**)

**Mathematical Intuition:**

The key idea is to lift the single-campaign UCB to combinatorial setting:
1. Maintain UCB upper bounds for each campaign and each bid pair: $\text{UCB}_{i,b}$
2. At each round, solve the constrained combinatorial optimization:
   $$\max_{S \text{ independent in } G} \sum_{i \in S} \max_{b \in B} \text{UCB}_{i,b}$$
   $$\text{s.t.} \sum_{i \in S} b_i^* \leq \rho \text{ (where } b_i^* = \arg\max_b \text{UCB}_{i,b})$$

**Algorithm:**

1. **Initialization:** For each campaign $i$ and bid $b$:
   - $n_{i,b} = 0$ (number of times campaign i played with bid b)
   - $\hat{r}_{i,b} = 0$ (empirical mean reward)
   - $\hat{c}_{i,b} = b$ (cost is deterministic)

2. **At round t:**
   
   a. Compute optimistic estimates for each campaign-bid pair:
   $$\text{UCB}_{i,b} = \hat{r}_{i,b} + \sqrt{\frac{2\log T}{n_{i,b} + 1}}$$
   
   b. Compute optimistic utility for each campaign-bid pair:
   $$u_{i,b} = \text{UCB}_{i,b} - \lambda_t \cdot b$$
   where $\lambda_t$ is Lagrange multiplier for budget constraint
   
   c. For each campaign i, find best bid:
   $$b_i^* = \arg\max_b u_{i,b}$$
   
   d. Solve combinatorial optimization: Find independent set $S \subseteq [N]$ and bids $\{b_i^* : i \in S\}$ that maximize:
   $$\sum_{i \in S} u_{i, b_i^*}$$
   subject to:
   - $(i, j) \notin E$ for all $i \neq j \in S$ (independent set constraint)
   - $\sum_{i \in S} b_i^* \leq \rho$ (budget constraint)
   
   e. Output bids: $b_{i,t} = b_i^*$ if $i \in S$, else $b_{i,t} = 0$ (don't bid)

3. **Observe feedback:** $(m_t, w_t, r_t, c_t)$

4. **Update statistics:**
   For each campaign $i \in S$ (those we bid on):
   - $n_{i, b_i^*} += 1$
   - $\hat{r}_{i, b_i^*} = \frac{(n_{i,b_i^*}-1) \hat{r}_{i, b_i^*} + r_{i,t}}{n_{i,b_i^*}}$

5. **Update Lagrange multiplier:** Use gradient descent on dual variable
   $$\lambda_t \leftarrow \lambda_t + \eta \cdot (\sum_{i \in S} c_{i,t} - \rho)$$

**Inputs/Outputs:**
- **Input:** Environment, T, bid_set, conflict_graph, budget B
- **Output:** Bid sequences for each campaign, rewards/costs per campaign, cumulative regret

**Expected Behavior:**
- Should respect conflict constraint: No conflicting campaigns bid in same round
- Should maintain budget constraint
- Should have sublinear regret in stochastic setting
- Regret degradation compared to single campaign due to combinatorial complexity

**Potential Difficulties:**
- Solving combinatorial optimization exactly is NP-hard in general (need approximation or special structure)
- Feedback is partial: Only observe feedback for campaigns we bid on (partial feedback)
- Estimating quality of bids not tried yet (exploration-exploitation)
- Coupling between campaigns makes regret analysis harder

#### 2.5 Combinatorial Optimization Subroutine

**Problem:** Maximize total utility subject to conflict and budget constraints

$$\max_{S \text{ independent}} \sum_{i \in S} u_{i, b_i^*}$$
$$\text{s.t.} \sum_{i \in S} b_i^* \leq \rho$$

**Solution strategies:**

1. **Brute Force** (if N is small, say N ≤ 12):
   - Enumerate all independent sets in conflict graph
   - For each set, solve the budget-constrained knapsack
   - Return best

2. **Greedy Approximation** (if N is large):
   - Sort campaigns by utility per cost ratio: $u_i / b_i^*$
   - Greedily add campaigns that don't conflict, respecting budget
   - Approximation factor: $\mathcal{O}(1)$ for some instances

3. **Dynamic Programming** (if conflict graph has special structure):
   - If graph is a forest: Can use tree DP
   - If graph is bipartite: Can use flow-based methods

**Implementation:** Start with brute force for clarity, optimize if needed

#### 2.6 Required Plots for Requirement 2

1. **Per-Campaign Cumulative Reward**
   - x-axis: Round t
   - y-axis: Cumulative reward for campaign i
   - Show multiple subplots (one per campaign) or separate lines

2. **Cumulative Cost Trajectory** (aggregate across campaigns)
   - x-axis: Round t
   - y-axis: $\sum_{i} \sum_{s=1}^t c_{i,s}$
   - Horizontal line at total budget B
   - Should respect constraint

3. **Campaign Selection Pattern**
   - x-axis: Round t
   - y-axis: Binary indicator for each campaign
   - Heat map or bar chart showing which campaigns were active each round

4. **Cumulative Regret Comparison**
   - Combinatorial-UCB vs. Random, ETC, Clairvoyant
   - With uncertainty bands

5. **Bid Distribution per Campaign**
   - Histogram of chosen bids for each campaign
   - Show concentration at preferred bids

#### 2.7 Experimental Methodology

**Parameters:**
- **T** (rounds): 2000-5000
- **Number of trials:** 30-50
- **N** (campaigns): 3-5 campaigns (start small)
- **Bid set B:** Same as Requirement 1 (e.g., 0 to 1 in steps of 0.1)
- **Campaign values:** $v_i$ uniform or diverse
- **Conflict graph types:**
  - Test 1: No conflicts (independent set = all campaigns)
  - Test 2: Path graph (campaigns arranged in line, adjacent ones conflict)
  - Test 3: Clique (all campaigns conflict - can only choose 1 per round)
  - Test 4: Random (60% probability of edge between each pair)
- **Competing bid distributions:** Same per campaign (symmetric case) or different
- **Correlation:** Test uncorrelated and correlated versions
- **Budget B:** 0.4T - 0.6T depending on test

**Baseline algorithms:**
- Random: Random valid selection and bids
- Single-campaign UCB applied to each independently (ignoring conflict constraint)
- ETC applied to combinatorial setting
- Clairvoyant (knows distributions, plays optimal)

#### 2.8 Expected Results

**For Combinatorial-UCB with Budget:**
- Regret should be sublinear: $\mathcal{O}(\log T \cdot d)$ where $d$ is problem-dependent dimension
- Regret worse than single-campaign due to combinatorial complexity
- Should efficiently learn which campaign sets are viable (conflict checking)
- Should adapt bids per campaign based on individual distributions

---

### REQUIREMENT 3: Best-of-Both-Worlds with Multiple Campaigns

#### 3.1 Goal
Design algorithm that performs well in **both stochastic AND highly adversarial settings** using **primal-dual method** with full feedback.

"Best-of-both-worlds" means:
- In stochastic setting: Sublinear regret $\mathcal{O}(\log T)$
- In adversarial setting: Sublinear regret $\mathcal{O}(\sqrt{T})$
- Without needing to know which regime applies!

#### 3.2 Problem Formulation

**Two environments to consider:**

**Environment 1 (Stochastic):** Same as Requirement 2
- Competing bids drawn i.i.d. from fixed distributions

**Environment 2 (Adversarial Non-Stationary):** Worst-case sequence
- Competing bids are arbitrary sequence $(m_{i,t})_{t,i}$
- Advertiser must adapt online without knowing future
- Compared to best fixed strategy in hindsight (full feedback available post-hoc)

**Feedback Assumption:** Full feedback
- Observe competing bids $m_{i,t}$ for ALL campaigns, regardless of what we bid
- This is stronger than Requirement 2 where we only see results of our bids
- Allows better estimation of "value" of each campaign-bid combination

**Regret Notion:**
- Against best fixed bid distribution: $R_T = \max_{x \in \Delta(B)^N} \sum_{t=1}^T (\sum_i \text{reward}_t(x_i)) - \sum_{t=1}^T (\sum_i \text{reward}_t(b_{i,t}))$
- Regret bound should depend only on problem structure, not specific rewards

#### 3.3 Environment Implementation

**Class Name:** `AdversarialMultiCampaignEnvironment`

**Parameters:**
- Same as `StochasticMultiCampaignEnvironment` but can also specify adversarial sequences
- Mode: "stochastic" or "adversarial"
- If adversarial: Sequence of competing bids $(m_{i,t})_{t,i}$ pre-generated or provided

**Methods:**
- `round(bid_vector)` with full feedback option:
  - Returns: `(m_vector, w_vector, r_vector, c_vector, m_all_campaigns)` 
  - `m_all_campaigns`: competing bids for ALL campaigns (even those not bid on)

#### 3.4 Algorithm 3: Primal-Dual Method with Budget Constraint

**Algorithm Name:** `PrimalDualMultiCampaignAgent`

**Theory Connection:**
- Based on notebook **08_constrained_problems.ipynb** Lagrangian approach
- Extends to multi-round, multi-campaign, with full feedback
- Uses Online Gradient Descent (OGD) intuition from lecture 9
- Adapts to adversarial setting using regret minimizer

**Mathematical Foundation:**

At each round $t$, the constrained optimization problem is:
$$\max_{x \in \Delta(B)^N, S \text{ indep}} \sum_{i \in S} x_i(b_i) \cdot \mathbb{E}[(v_i - b_i) \mathbb{1}[b_i \geq m_i]]$$
$$\text{s.t.} \sum_{i \in S} b_i \leq \rho$$

Using Lagrangian duality, the optimal policy solves:
$$\max_{x} \min_{\lambda \geq 0} L(x, \lambda) = \sum_{i \in S} \sum_{b} x_i(b) [(v_i - b) \mathbb{P}(b \geq m_i) - \lambda b]$$

**Primal-Dual Algorithm:**

1. **Initialization:**
   - Dual variable: $\lambda_0 = 0.5$ (or tuned based on problem)
   - Primal variables: $x_{i,b,0} = 1/|B|$ for all $i, b$ (uniform)
   - Momentum parameter: $\alpha \in (0,1)$ (e.g., 0.01)

2. **At round t:**
   
   a. **Primal step** (update bid distribution):
      - Receive feedback: for all campaigns, for all bids, estimate value
      $$v_{i,b,t} = (v_i - b) \mathbb{1}[\text{win against } m_{i,t}] - \lambda_{t-1} \cdot b$$
      
      - Apply Online Gradient Ascent on primal:
      $$x_{i,b,t} \propto \exp(\alpha \sum_{s=1}^t v_{i,b,s})$$
      
      - Select bids: $b_{i,t}$ from distribution $x_{i,t}$ (draw from current distribution)
      - Add independent set constraint: if campaigns conflict, re-sample to ensure independence

   b. **Dual step** (update Lagrange multiplier):
      - Observe total cost: $C_t = \sum_{i \in S_t} b_{i,t}$ (where $S_t$ is set of campaigns bid on)
      - Update Lagrange multiplier:
      $$\lambda_t \leftarrow \max(0, \lambda_{t-1} + \beta \cdot (C_t - \rho))$$
      where $\beta$ is step size (learning rate for dual, e.g., 0.1)

3. **Output:** Bids for round $t$ based on current distribution

**Key Differences from Requirement 2:**
- Uses **full feedback** (see ALL competing bids)
- Uses **exponential weights** (regret minimizer) instead of UCB with confidence intervals
- Handles **adversarial sequences** naturally (not assuming stochastic)
- Converges to **minimax** solution (min over strategies, max over adversary)

**Inputs/Outputs:**
- **Input:** Adversarial environment, T, bid_set, conflict_graph, budget B
- **Output:** Bid sequences (possibly randomized), realized rewards, cumulative regret vs. OPT

**Expected Behavior:**
- In stochastic setting: Regret $\approx \mathcal{O}(\log T)$ (learns optimal strategy)
- In adversarial setting: Regret $\approx \mathcal{O}(\sqrt{T})$ (no better possible without knowing future)
- Should work reasonably in both settings without environment-specific tuning

**Potential Difficulties:**
- Exponential weights can be numerically unstable (need careful implementation with log-sum-exp trick)
- Hyperparameter tuning critical: $\alpha$, $\beta$, initial $\lambda_0$
- Full feedback assumption is strong (may not hold in real auctions)
- Randomized strategies harder to analyze empirically (need averaging)

#### 3.5 Required Plots for Requirement 3

1. **Comparison: Stochastic vs. Adversarial**
   - x-axis: Round t
   - y-axis: Cumulative regret
   - Show separate curves for same algorithm in both environments
   - Illustrate different regret rates

2. **Dual Variable Evolution**
   - x-axis: Round t
   - y-axis: $\lambda_t$
   - Show how Lagrange multiplier adapts to maintain budget

3. **Primal Weights Evolution** (for one campaign-bid pair)
   - x-axis: Round t
   - y-axis: $x_{i,b,t}$ (probability of choosing that bid)
   - Show convergence pattern

4. **Cumulative Cost Trajectory**
   - Verify budget constraint maintenance
   - Compare to non-constrained version

5. **Regret Comparison: Primal-Dual vs. Combinatorial-UCB**
   - Both algorithms in both environments
   - Show theoretical predictions (log T vs sqrt T)

#### 3.6 Experimental Methodology

**Stochastic Test:**
- Use same environment as Requirement 2
- Run Primal-Dual and compare to Combinatorial-UCB

**Adversarial Test:**
- Generate highly non-stationary sequence: Competing bids change every few rounds
- Example: $m_{i,t} = \sin(2\pi t / 500) / 2 + 0.5$ (sinusoidal pattern)
- Or: $m_{i,t}$ drawn uniformly at random each round

**Parameters:**
- **T:** 5000
- **Trials:** 20-30 (regret minimizer is randomized, need more averaging)
- **N:** 3-5 campaigns
- **Hyperparameters:** Tune $\alpha, \beta$ on small instance, then use same for all tests

**Metrics:**
- Cumulative regret
- Regret at end: $R_T / T$ (regret rate)
- Budget violation magnitude
- Convergence speed of dual variable

#### 3.7 Expected Results

**In stochastic setting:**
- Should match or beat Combinatorial-UCB
- Regret should show $\mathcal{O}(\log T)$ behavior
- May be slightly worse due to randomization (need more samples to estimate)

**In adversarial setting:**
- Combinatorial-UCB fails (linear regret as it assumes stochastic)
- Primal-Dual shows $\mathcal{O}(\sqrt{T})$ behavior
- Budget constraint harder to maintain (adversary can force expensive bids)

---

### REQUIREMENT 4: Non-Stationary Environments with Change Detection

#### 4.1 Goal
Handle **piecewise-stationary environments** where competing bids have different distributions in different time intervals, using change detection and sliding window techniques.

#### 4.2 Problem Formulation

**Setting:**
- Same multi-campaign, budget-constrained setting as Requirement 2
- Time is divided into intervals: $[1, \tau_1], [\tau_1+1, \tau_2], ..., [\tau_K, T]$
- In each interval $j$, competing bid distributions are fixed: $D_j = (D_{1,j}, ..., D_{N,j})$
- BUT: Distributions change between intervals
- Advertiser doesn't know intervals in advance (online setting)

**Regret Notion:**
- Compared to best fixed policy within each interval
- Policy regret: $R_T = \sum_{j=1}^K [\tau_j \cdot \text{OPT}_j - \sum_{t=\tau_{j-1}+1}^{\tau_j} r_t]$
- Where $\text{OPT}_j$ is optimal expected reward for interval $j$

#### 4.3 Environment Implementation

**Class Name:** `PiecewiseStationaryEnvironment`

**Parameters:**
- Interval structure: `intervals = [tau_1, tau_2, ..., tau_K]` (list of interval boundaries)
- Distributions per interval: `distributions = [(D_{1,1}, ..., D_{N,1}), (D_{1,2}, ..., D_{N,2}), ...]`
- Can be constructed from Requirement 2 environment with time-dependent distribution

**Methods:**
- `round(bid_vector)` returns feedback as before, internally switches distributions at interval boundaries

#### 4.4 Algorithm 4a: Combinatorial-UCB with Sliding Window

**Algorithm Name:** `SlidingWindowCombinatoricalUCB1Agent`

**Theory Connection:**
- Extends Combinatorial-UCB from Requirement 2
- Adapts sliding window technique from notebook **10_nonstationary_bandits.ipynb**
- Idea: Only use recent samples (within window) to estimate rewards, discard old potentially stale data

**Mathematical Formulation:**

At round $t$, only use data from rounds in window $[t - W, t]$ where $W$ is window size.

$$\hat{r}_{i,b,t} = \frac{1}{n_{i,b,t}} \sum_{s \in [t-W, t]: a_s = (i,b)} r_s$$

Where $n_{i,b,t} = |\{s \in [t-W, t]: a_s = (i,b)\}|$

UCB computation uses this windowed empirical mean:
$$\text{UCB}_{i,b,t} = \hat{r}_{i,b,t} + \sqrt{\frac{2\log t}{n_{i,b,t}}}$$

Rest of algorithm unchanged from Requirement 2.

**Hyperparameter Selection:**
- Window size $W$ should be large enough to have confident estimates
- But small enough to discard old data before distribution change
- Typical choice: $W = \sqrt{T}$ or $W = T^{2/3}$
- Can be tuned via parameter sweep in experiments

**Inputs/Outputs:**
- Same as Requirement 2, but with window size $W$ as parameter

**Expected Behavior:**
- Regret increases immediately after distribution change (older samples become less informative)
- Once window fully adapts to new distribution, should recover good performance
- Cumulative regret should have "steps" at change points

**Potential Difficulties:**
- Choosing $W$ is critical but problem-dependent
- Too small: Poor estimation, high variance
- Too large: Slow adaptation, misses changes
- Need to detect changes separately (else algorithm doesn't know when to "reset" mentally)

#### 4.5 Algorithm 4b: Combinatorial-UCB with Change Detector

**Algorithm Name:** `ChangeDetectorCombinatoricalUCB1Agent`

**Theory Connection:**
- Combines Combinatorial-UCB with change detection technique from **10_nonstationary_bandits.ipynb**
- Uses hypothesis testing to detect when distribution has changed significantly

**Mathematical Formulation:**

**Change Detection Test:**

Maintain running statistics for each campaign-bid pair:
- Recent samples: $r_{i,b}^{(1)}, ..., r_{i,b}^{(n_1)}$ from recent "phase"
- Previous phase samples: $r_{i,b}^{(n_1+1)}, ..., r_{i,b}^{(n_1+n_2)}$

Test if means are equal using confidence intervals or t-test:
$$\text{Difference} = |\hat{\mu}_1 - \hat{\mu}_2|$$
$$\text{Threshold}(\alpha, n_1, n_2) = c \sqrt{\frac{\log T}{n_1} + \frac{\log T}{n_2}}$$

If $\text{Difference} > \text{Threshold}$, declare change detected.

**Algorithm:**

1. Maintain two phases:
   - **Active phase:** Currently collecting data and making decisions
   - **Historical phase:** Data from before potential change point

2. At round $t$:
   - If phase has seen $M$ rounds: Check all campaign-bid pairs for change
   - If any pair signals change significantly: **Reset** - start new active phase
   - Otherwise: Continue as Combinatorial-UCB

3. Resetting means:
   - Forget all historical data
   - Keep only active phase data
   - Increased exploration (lower confidence) to re-learn distribution

**Inputs/Outputs:**
- Same as Requirement 2, with change detection threshold parameter

**Expected Behavior:**
- Adapts more quickly to changes than sliding window (detects change vs. gradual forgetting)
- May have false positives (declare change when variance is high)
- After detection: Temporary regret increase due to re-exploration
- Should perform better than sliding window in sharp changes

**Potential Difficulties:**
- Change detection threshold is hyperparameter (controls false positive rate)
- High threshold: Misses real changes
- Low threshold: Too many false alarms, wastes resources re-exploring
- Need enough samples per phase to reliable detect (tradeoff between exploration and detection)

#### 4.6 Algorithm 4c: Primal-Dual (from Requirement 3) as Baseline

The Primal-Dual method from Requirement 3 with full feedback should also be evaluated on non-stationary setting as baseline.

#### 4.7 Required Plots for Requirement 4

1. **Competing Bid Distributions Over Time** (for one campaign)
   - x-axis: Round t
   - y-axis: Mean of competing bid distribution
   - Show step changes at interval boundaries

2. **Cumulative Regret Comparison**
   - Three curves: Sliding Window, Change Detector, Primal-Dual
   - x-axis: Round t
   - y-axis: Cumulative regret
   - Should show different behaviors at change points

3. **Per-Interval Regret**
   - x-axis: Interval number
   - y-axis: Regret within that interval
   - Show regret distribution across intervals

4. **Detection Events** (for change detector version)
   - x-axis: Round t
   - y-axis: Binary indicator whether change was detected
   - Overlay with actual change points
   - Show false positives / false negatives

5. **Window Size Sensitivity** (for sliding window version)
   - x-axis: Window size W
   - y-axis: Total cumulative regret
   - Show optimal W value

#### 4.8 Experimental Methodology

**Environment Construction:**

Partition $T = 5000$ rounds into intervals:
- Example: 4 intervals of 1250 rounds each
- $\tau_1 = 1250, \tau_2 = 2500, \tau_3 = 3750, \tau_4 = 5000$

For each interval, use different distributions:
- **Interval 1:** Distributions from Requirement 2 (baseline)
- **Interval 2:** Different means (e.g., shift competitors' bids up by 0.1)
- **Interval 3:** Back to baseline
- **Interval 4:** Different again

**Parameters:**
- **T:** 5000 (4 intervals × 1250 rounds each)
- **Trials:** 30-50
- **N:** 3-5 campaigns
- **Window sizes to test:** $\sqrt{T} \approx 70$, $T^{2/3} \approx 370$, $T^{3/4} \approx 840$
- **Change detection threshold:** Tune to balance false positives/negatives

**Baselines:**
- Sliding Window with different W values
- Change Detector with different thresholds
- Primal-Dual from Requirement 3
- Non-adaptive Combinatorial-UCB (ignores non-stationarity)
- Clairvoyant (knows distribution for each interval)

#### 4.9 Expected Results

**Non-Adaptive Combinatorial-UCB (baseline):**
- Linear regret: $R_T = \Theta(T)$ (as distributional drift accumulates)

**Sliding Window:**
- Regret: $\tilde{\mathcal{O}}(\sqrt{T} \log T)$ if $W \propto \sqrt{T}$
- Regret spikes at change points but recovers
- Smooth regret trajectory

**Change Detector:**
- Regret: $\tilde{\mathcal{O}}(\log T + \#\text{changes} \cdot \log T)$
- Fewer spikes than sliding window (detects changes)
- Faster adaptation after detection

**Primal-Dual:**
- Should show similar or better performance
- May adapt faster due to full feedback

---

## CODE ARCHITECTURE

### Directory Structure

```
project/
├── README.md                          # Project overview, setup, running instructions
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── 01_single_campaign_stochastic.ipynb        # Requirement 1
│   ├── 02_multi_campaign_stochastic.ipynb         # Requirement 2
│   ├── 03_primal_dual_best_of_both_worlds.ipynb   # Requirement 3
│   └── 04_non_stationary_environments.ipynb       # Requirement 4
├── src/
│   ├── __init__.py
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base Environment class
│   │   ├── single_campaign.py        # StochasticSingleCampaignEnvironment
│   │   ├── multi_campaign.py         # StochasticMultiCampaignEnvironment
│   │   ├── adversarial.py            # AdversarialMultiCampaignEnvironment
│   │   └── piecewise_stationary.py   # PiecewiseStationaryEnvironment
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base Agent class
│   │   ├── bandit_agents.py          # RandomAgent, GreedyAgent (baselines)
│   │   ├── single_campaign_ucb.py    # UCB1SingleCampaignAgent, BudgetConstrainedUCB1Agent
│   │   ├── combinatorial_ucb.py      # CombinatorialUCB1Agent
│   │   ├── sliding_window.py         # SlidingWindowCombinatorialUCB1Agent
│   │   ├── change_detector.py        # ChangeDetectorCombinatorialUCB1Agent
│   │   └── primal_dual.py            # PrimalDualMultiCampaignAgent
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── combinatorial_solver.py   # Solve independent set + knapsack problem
│   │   ├── lagrangian.py             # Lagrangian-based utilities
│   │   └── change_detection.py       # Change detection tests
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── regret_plots.py           # Cumulative regret, regret per trial
│   │   ├── bid_plots.py              # Bid distributions, selections
│   │   ├── resource_plots.py         # Cost/budget trajectories
│   │   └── comparison_plots.py       # Multi-algorithm comparisons
│   └── utils/
│       ├── __init__.py
│       ├── experiment.py             # Experiment running framework
│       ├── metrics.py                # Regret computation, performance metrics
│       └── data_handling.py          # Save/load results
└── results/
    ├── requirement_1/                # Results for Req 1
    ├── requirement_2/                # Results for Req 2
    ├── requirement_3/                # Results for Req 3
    └── requirement_4/                # Results for Req 4
```

### File Responsibilities

#### `environments/base.py`
**Purpose:** Define base `Environment` class following course conventions

**Key components:**
```python
class Environment:
    def __init__(self, ...):
        pass
    
    def round(self, action) -> (feedback):
        """Execute one round, return feedback."""
        pass
```

#### `environments/single_campaign.py`
**Purpose:** Stochastic environment for single campaign

**Key components:**
- `StochasticSingleCampaignEnvironment`: Manages campaign value, bid set, competing bid distribution
- Supports multiple distribution types (Uniform, Normal, Beta, etc.)
- Tracks round counter, returns feedback with competing bid, win/loss, utility

#### `environments/multi_campaign.py`
**Purpose:** Stochastic environment for multiple campaigns with conflict graph

**Key components:**
- `StochasticMultiCampaignEnvironment`: N campaigns, joint distributions, conflict constraints
- Stores conflict graph (adjacency matrix or networkx)
- Supports correlated/uncorrelated competing bids
- Returns vector feedback for all campaigns

#### `environments/adversarial.py`
**Purpose:** Adversarial and full-feedback environment

**Key components:**
- `AdversarialMultiCampaignEnvironment`: Worst-case competing bids
- Can accept pre-generated sequences or generate adversarially
- Provides full feedback (all competing bids) for Requirement 3

#### `environments/piecewise_stationary.py`
**Purpose:** Non-stationary environment with intervals

**Key components:**
- `PiecewiseStationaryEnvironment`: Extends multi-campaign environment
- Maintains list of intervals and corresponding distributions
- Internally switches distributions at interval boundaries
- Useful for Requirement 4

#### `agents/base.py`
**Purpose:** Define base `Agent` class

**Key components:**
```python
class Agent:
    def __init__(self, ...):
        pass
    
    def pull_arm(self) -> action:
        """Choose action (bid for campaign(s))."""
        pass
    
    def update(self, reward):
        """Update internal state based on observed reward."""
        pass
```

#### `agents/bandit_agents.py`
**Purpose:** Simple baseline agents (random, greedy, ETC)

**Key components:**
- `RandomAgent`: Selects actions uniformly at random
- `GreedyAgent`: Plays empirical best arm
- `ETCAgent`: Explore-then-commit
- These are baselines for all requirements

#### `agents/single_campaign_ucb.py`
**Purpose:** UCB agents for single campaign (Requirement 1)

**Key components:**
- `UCB1SingleCampaignAgent`: UCB without budget constraint
- `BudgetConstrainedUCB1Agent`: With budget constraint and Lagrange multiplier
- Maintains empirical means per bid, computes UCB confidence radii
- Tracks cumulative cost for budget constraint

#### `agents/combinatorial_ucb.py`
**Purpose:** Combinatorial UCB for multiple campaigns (Requirement 2)

**Key components:**
- `CombinatorialUCB1Agent`: Main algorithm
- Manages per-campaign, per-bid statistics
- Integrates with combinatorial optimizer to select campaigns + bids
- Implements Lagrange multiplier updates for budget

#### `agents/sliding_window.py`
**Purpose:** Sliding window variant (Requirement 4a)

**Key components:**
- `SlidingWindowCombinatorialUCB1Agent`: Extends CombinatorialUCB with window mechanism
- Maintains only recent samples (within window W)
- Discards old data progressively
- Configurable window size

#### `agents/change_detector.py`
**Purpose:** Change detection variant (Requirement 4b)

**Key components:**
- `ChangeDetectorCombinatorialUCB1Agent`: Extends CombinatorialUCB with detection
- Maintains hypothesis test statistics per campaign-bid
- Detects distributional shifts
- Resets data upon detection

#### `agents/primal_dual.py`
**Purpose:** Primal-dual method (Requirement 3)

**Key components:**
- `PrimalDualMultiCampaignAgent`: Full feedback, randomized actions
- Maintains exponential weight distribution over bids
- Updates using regret minimizer (exp weights) and dual variable
- Handles both stochastic and adversarial feedback

#### `optimization/combinatorial_solver.py`
**Purpose:** Solve constrained combinatorial optimization

**Key components:**
- `solve_max_weighted_independent_set_knapsack(utilities, weights, graph, budget)`: 
  - Find independent set in conflict graph
  - Maximize total utility
  - Subject to weight (cost) constraint
- Multiple implementations:
  - Brute force for small instances (N ≤ 12)
  - Greedy approximation for larger instances
  - Dynamic programming for special graph structures

#### `optimization/lagrangian.py`
**Purpose:** Utility functions for Lagrangian optimization

**Key components:**
- `compute_modified_utilities(utilities, costs, lambda_multiplier)`: Compute $u(b) - \lambda \cdot c(b)$
- `update_lagrange_multiplier(cumulative_cost, budget_per_round, learning_rate)`: Update dual variable
- `project_lambda(lambda_val, min_val, max_val)`: Keep Lagrange multiplier in valid range

#### `optimization/change_detection.py`
**Purpose:** Change detection utilities

**Key components:**
- `detect_change(recent_samples, historical_samples, threshold, t)`: Hypothesis test for distribution shift
- `compute_change_threshold(n1, n2, t, confidence_param)`: Compute detection threshold
- `track_phase_statistics(samples, phase_id)`: Maintain statistics per phase

#### `plotting/regret_plots.py`
**Purpose:** Visualize cumulative regret trajectories

**Key functions:**
- `plot_cumulative_regret(regret_per_trial, labels, title)`: Multi-algorithm regret comparison with uncertainty bands
- `plot_regret_vs_theory(regret_trajectory, theoretical_bound, label)`: Compare empirical to theoretical regret
- `plot_regret_distribution(regret_per_trial, labels)`: Box plots of final regrets across trials

#### `plotting/bid_plots.py`
**Purpose:** Visualize bid choices and distributions

**Key functions:**
- `plot_bid_histogram(bid_sequence, bid_set, label)`: Show which bids were chosen how often
- `plot_bid_trajectory(bid_sequence, label)`: Time series of bid choices
- `plot_campaign_selection(campaign_selections, campaign_ids, title)`: Heat map of campaign selections over time

#### `plotting/resource_plots.py`
**Purpose:** Visualize cost, budget, and resource usage

**Key functions:**
- `plot_cumulative_cost(cost_sequence, budget, label)`: Cost trajectory with budget line
- `plot_budget_usage_per_campaign(costs, campaign_ids, budget)`: Cost breakdown by campaign
- `plot_cost_vs_reward(costs, rewards, label)`: Efficiency analysis (reward per cost)

#### `plotting/comparison_plots.py`
**Purpose:** Multi-algorithm and multi-setting comparisons

**Key functions:**
- `plot_algorithm_comparison(results_dict, setting_name, title)`: Compare multiple algorithms on same setting
- `plot_stochastic_vs_adversarial(stochastic_regret, adversarial_regret, algorithm_name)`: Show best-of-both-worlds effect
- `plot_non_stationary_analysis(regrets_per_interval, change_times, labels)`: Non-stationary performance breakdown

#### `utils/experiment.py`
**Purpose:** Framework for running multi-trial experiments

**Key functions:**
```python
def run_experiment(env_class, env_params, agent_class, agent_params, 
                   n_trials, T, seed_start=0) -> ExperimentResults:
    """Run agent on environment for multiple trials."""
    
class ExperimentResults:
    """Container for experiment outcomes."""
    regrets: List[np.array]  # Per-trial cumulative regret
    rewards: List[np.array]  # Per-trial cumulative reward
    costs: List[np.array]    # Per-trial cumulative cost
    actions: List[np.array]  # Per-trial action sequence
    # ... other fields
```

#### `utils/metrics.py`
**Purpose:** Compute regret and performance metrics

**Key functions:**
- `compute_pseudo_regret(rewards, clairvoyant_value, T)`: Compute pseudo-regret
- `compute_actual_regret(rewards, optimal_rewards_per_trial)`: Actual regret from observed data
- `compute_regret_trajectory(reward_sequence, clairvoyant_value)`: Cumulative regret over time
- `compute_cost_budget_violation(cost_sequence, budget)`: Budget constraint violation

#### `utils/data_handling.py`
**Purpose:** Save and load experimental results

**Key functions:**
- `save_results(results, filename)`: Pickle or JSON save
- `load_results(filename)`: Load saved results
- `aggregate_results(results_list, label)`: Combine results from multiple runs

---

## IMPLEMENTATION ORDER

### Phase 1: Foundation (Requirements 1)

**Checkpoint Goal:** Implement and validate single-campaign bidding with UCB

**Step 1.1:** Environment infrastructure
- [ ] Implement `Environment` base class in `environments/base.py`
- [ ] Implement `StochasticSingleCampaignEnvironment` in `environments/single_campaign.py`
- [ ] Test environment with manual bid sequences

**Step 1.2:** Baseline agents
- [ ] Implement `Agent` base class
- [ ] Implement `RandomAgent` (bid uniformly from set B)
- [ ] Test: Run random agent, verify reward distribution

**Step 1.3:** Single-campaign UCB (no budget)
- [ ] Implement `UCB1SingleCampaignAgent` in `agents/single_campaign_ucb.py`
- [ ] Test: Verify UCB values decrease with arm pulls
- [ ] Test: Verify agent converges to best bid

**Step 1.4:** Single-campaign UCB with budget
- [ ] Implement `BudgetConstrainedUCB1Agent`
- [ ] Implement Lagrangian utilities in `optimization/lagrangian.py`
- [ ] Test: Verify budget constraint satisfied
- [ ] Test: Verify regret increases compared to unconstrained

**Step 1.5:** Experiment framework and plotting
- [ ] Implement basic experiment runner in `utils/experiment.py`
- [ ] Implement `compute_pseudo_regret()` in `utils/metrics.py`
- [ ] Implement regret plotting in `plotting/regret_plots.py`
- [ ] Implement bid histogram plotting in `plotting/bid_plots.py`

**Step 1.6:** Requirement 1 evaluation
- [ ] Create notebook `notebooks/01_single_campaign_stochastic.ipynb`
- [ ] Run experiments with various settings (3 distribution types, 2 budget levels)
- [ ] Generate all 4 required plots
- [ ] Document results

**Deliverables:**
- Working single-campaign bidding agents
- Verified budget constraint maintenance
- Plots showing regret, cost, bid selection
- Discussion of exploration-exploitation tradeoff

---

### Phase 2: Combinatorial Optimization (Requirements 2)

**Checkpoint Goal:** Implement multi-campaign bidding with conflict constraints

**Step 2.1:** Combinatorial optimization solver
- [ ] Implement `solve_max_weighted_independent_set_knapsack()` in `optimization/combinatorial_solver.py`
- [ ] Start with brute-force enumeration (for N ≤ 12)
- [ ] Test: Verify independent set constraint
- [ ] Test: Verify budget constraint
- [ ] Test: Compare to known optimal solutions

**Step 2.2:** Multi-campaign environment
- [ ] Implement `StochasticMultiCampaignEnvironment` in `environments/multi_campaign.py`
- [ ] Support networkx graphs for conflict representation
- [ ] Implement feedback for multiple campaigns
- [ ] Test: Verify conflict constraint in round() method

**Step 2.3:** Combinatorial UCB agent
- [ ] Implement `CombinatorialUCB1Agent` in `agents/combinatorial_ucb.py`
- [ ] Maintain per-campaign, per-bid statistics
- [ ] Integrate combinatorial solver for action selection
- [ ] Implement Lagrange multiplier updates

**Step 2.4:** Multi-campaign plotting
- [ ] Extend plotting functions for per-campaign rewards
- [ ] Implement campaign selection visualization
- [ ] Implement bid distribution plots per campaign

**Step 2.5:** Requirement 2 evaluation
- [ ] Create notebook `notebooks/02_multi_campaign_stochastic.ipynb`
- [ ] Design experiments with 3-5 campaigns, various conflict graphs
- [ ] Run Combinatorial-UCB vs. baselines
- [ ] Generate required plots
- [ ] Document results and insights

**Deliverables:**
- Working multi-campaign bidding with conflict constraints
- Combinatorial solver (at least brute force)
- Plots showing per-campaign performance, campaign selection patterns
- Discussion of how conflicts and budget affect exploration

---

### Phase 3: Advanced Algorithms (Requirements 3 & 4)

**Step 3.1:** Primal-dual agent
- [ ] Implement `PrimalDualMultiCampaignAgent` in `agents/primal_dual.py`
- [ ] Exponential weight distribution over bids
- [ ] Full feedback processing
- [ ] Dual variable updates for budget constraint
- [ ] Test: Verify budget constraint with randomized actions

**Step 3.2:** Adversarial environment
- [ ] Implement `AdversarialMultiCampaignEnvironment` in `environments/adversarial.py`
- [ ] Support full feedback option
- [ ] Support worst-case sequences (e.g., sinusoidal patterns)

**Step 3.3:** Requirement 3 evaluation
- [ ] Create notebook `notebooks/03_primal_dual_best_of_both_worlds.ipynb`
- [ ] Test Primal-Dual on stochastic environment (compare to Combinatorial-UCB)
- [ ] Test Primal-Dual on adversarial environment (show sqrt(T) regret)
- [ ] Generate comparison plots
- [ ] Document best-of-both-worlds behavior

**Step 3.4:** Non-stationary environment
- [ ] Implement `PiecewiseStationaryEnvironment` in `environments/piecewise_stationary.py`
- [ ] Support interval-based distribution switching
- [ ] Test: Verify distributions change at boundaries

**Step 3.5:** Sliding window agent
- [ ] Implement `SlidingWindowCombinatorialUCB1Agent` in `agents/sliding_window.py`
- [ ] Maintain only recent samples in window
- [ ] Discard old data gradually
- [ ] Support configurable window size

**Step 3.6:** Change detection agent and utilities
- [ ] Implement `ChangeDetectorCombinatorialUCB1Agent` in `agents/change_detector.py`
- [ ] Implement change detection functions in `optimization/change_detection.py`
- [ ] Test: Verify detection accuracy on synthetic changes

**Step 3.7:** Requirement 4 evaluation
- [ ] Create notebook `notebooks/04_non_stationary_environments.ipynb`
- [ ] Design piecewise-stationary environment (4 intervals)
- [ ] Compare: Sliding Window vs. Change Detector vs. Primal-Dual vs. Non-adaptive
- [ ] Generate required plots (distributions, regret per interval, detection events)
- [ ] Analyze performance on sharp vs. gradual changes

**Deliverables:**
- Working primal-dual algorithm for best-of-both-worlds
- Sliding window and change detection variants
- Plots showing performance across all settings
- Discussion of adaptation mechanisms

---

### Phase 4: Comprehensive Experiments and Presentation

**Step 4.1:** Consolidate all experiments
- [ ] Run all algorithms on all four requirements
- [ ] Generate final result files
- [ ] Create aggregate comparison plots

**Step 4.2:** Create presentation materials
- [ ] Generate slides with algorithm descriptions
- [ ] Include all required plots
- [ ] Prepare for 20 min presentation

**Step 4.3:** Prepare GitHub repository
- [ ] Organize code following architecture
- [ ] Write comprehensive README.md
- [ ] Include setup instructions and data requirements
- [ ] Document all algorithms with references to theory

**Step 4.4:** Final report and discussion
- [ ] Write analysis of unexpected results
- [ ] Discuss exploration-exploitation tradeoffs in each setting
- [ ] Compare theoretical vs. empirical regret bounds
- [ ] Suggest improvements and extensions

---

## NOTEBOOK MAPPING

### How Each Course Notebook Informs the Project

#### **01_stochastic_mabs.ipynb** → Requirement 1
**Relevant Concepts:**
- `Agent` base class and `pull_arm()`/`update()` pattern
- `Environment` class for feedback simulation
- `UCB1Agent` implementation with confidence intervals
- Multi-trial experiment structure with uncertainty bands
- Cumulative regret plotting methodology

**Reuse:**
- Copy `BernoulliEnvironment` as template for `StochasticSingleCampaignEnvironment`
- Adapt `UCB1Agent.pull_arm()` to work with continuous bid set B
- Use same plotting pattern for regret trajectories
- Same multi-trial loop structure (run with different random seeds)

**Adaptation Needed:**
- Rewards are now deterministic given bid and competing bid: $r = (v - b) \cdot \mathbb{1}[b \geq m]$
- Instead of Bernoulli sampling per arm, competing bid is drawn from distribution
- Bid set B is discrete but larger than typical MAB problems

---

#### **02_stochastic_mabs_2.ipynb** → Requirement 1 & 2
**Relevant Concepts:**
- Thompson Sampling background (informational, not used in project)
- Beta distribution for prior beliefs
- Confidence interval shrinking visualization
- Extending single-arm concepts to structured problems

**Reuse:**
- General structure of Agent-based bidding
- Multi-trial evaluation pattern
- Visualization of learning dynamics

---

#### **08_constrained_problems.ipynb** → Requirements 1, 2, 3
**Relevant Concepts:**
- Budget constraints in online learning
- Lagrangian relaxation: $L(x, \lambda) = \sum_a x(a) f(a) - \lambda(\sum_a x(a) c(a) - \rho)$
- Linear program solver for computing clairvoyant: `scipy.optimize.linprog()`
- Online gradient ascent for primal, gradient descent for dual
- Pseudo-regret vs. actual regret definitions

**Reuse:**
- `SingleConstraintEnvironment` class as template for multi-campaign environment
- `compute_clairvoyant(f, c, rho)` using linprog directly for single-campaign OPT
- Lagrangian structure and dual variable updates
- Budget tracking code

**Adaptation Needed:**
- Extend from single constraint to multiple campaigns
- Extend linprog-based clairvoyant computation to handle multiple campaigns
- Adapt Lagrangian to work with combinatorial action space

---

#### **09_combinatorial_mabs.ipynb** → Requirement 2 & 3
**Relevant Concepts:**
- Weighted matching problem formulation
- Maximum weighted independent set concept
- Graph representation of constraints
- `scipy.optimize.linear_sum_assignment()` for matching
- NetworkX for graph visualization and manipulation

**Reuse:**
- Use independent set formulation instead of matching (different combinatorial structure)
- Graph representation of conflict constraints
- `scipy.optimize` for optimization subroutine
- Visualization patterns for bipartite graphs

**Adaptation Needed:**
- Adapt from matching (each worker matched to at most one task) to independent set (select subset of campaigns)
- Add knapsack constraint (budget) to independent set selection
- Extend to randomized solution selection for Primal-Dual

---

#### **10_nonstationary_bandits.ipynb** → Requirement 4
**Relevant Concepts:**
- Piecewise-constant reward functions $\mu_a(t)$
- Policy regret vs. pseudo-regret definitions
- Sliding window technique: keep only recent samples
- Change detection based on confidence intervals
- Visual representation of non-stationarity

**Reuse:**
- Piecewise-constant environment construction
- Sliding window mechanism for forgetting old data
- Change detection test structure
- Visualization of distribution changes over time
- Per-interval regret computation

**Adaptation Needed:**
- Apply to multi-campaign setting (multiple arms changing simultaneously)
- Combine sliding window with Combinatorial-UCB
- Adapt change detection to campaign-bid pairs instead of individual arms

---

### Direct Code References

| Course File | Location | Project Usage |
|------------|----------|-----------------|
| `01_stochastic_mabs.ipynb` | UCB1Agent | Model for SingleCampaignUCBAgent |
| | BernoulliEnvironment | Template for StochasticEnvironments |
| | Multi-trial loop | Experiment runner framework |
| `08_constrained_problems.ipynb` | Lagrangian utilities | Budget constraint handling |
| | compute_clairvoyant() | OPT computation |
| | linprog usage | Solver for optimization |
| `09_combinatorial_mabs.ipynb` | Matching problem | Independent set formulation |
| | linear_sum_assignment | Solver idea (adapt to our problem) |
| | Graph construction | Conflict graph representation |
| `10_nonstationary_bandits.ipynb` | Sliding window | Keep only last W samples |
| | Change detection | Hypothesis test for shifts |
| | Piecewise-constant means | Non-stationary environment |

---

## THEORY CONNECTIONS

### Requirement 1: Single Campaign

#### Exploration-Exploitation Challenge
**Tradeoff:**
- **Explore:** Try different bids to learn which has highest winning probability and net reward
- **Exploit:** Use currently-best bid to maximize immediate reward

**Why it matters:**
- If bid too high: Almost always win but pay too much, low profit
- If bid too low: Cheap but rarely win, zero reward
- Optimal bid depends on competing bid distribution (unknown)

#### Regret Objective & Theory

**Key Insight:** Each bid $b \in B$ is like an "arm" in a bandit problem
- Reward of arm $b$: $r(b) = (v - b) \mathbb{P}(\text{win with } b) = (v - b) \mathbb{P}(m < b)$
- Problem: Both $(v - b)$ and $\mathbb{P}(m < b)$ depend on $b$, creating complex landscape

**Regret Bound (UCB1):**
$$R_T = \mathcal{O}(\log T) \sum_b \frac{\log T}{\Delta_b}$$

where $\Delta_b = \max_b r(b) - r(b)$ is the gap of bid $b$.

**Why UCB Works:**
1. Confidence intervals shrink at rate $\sqrt{\frac{\log T}{n_b}}$ (optimistic)
2. Can prove: Sub-optimal arms played $\mathcal{O}(\log T)$ times
3. Best arm played $T - \mathcal{O}(\log T)$ times
4. Total regret accumulates logarithmically

#### Budget Constraint Effect

**Problem Formulation:**
$$\max_x \sum_b x(b) r(b) \quad \text{s.t.} \sum_b x(b) \cdot b \leq \rho$$

**Lagrangian:**
$$L(x, \lambda) = \sum_b x(b) [r(b) - \lambda b]$$

**Effect of constraint:**
- Even if bid $b_1$ has high reward $r(b_1)$, if it's expensive, Lagrange multiplier $\lambda$ will discourage it
- Algorithm must balance high-reward (expensive) bids with low-cost bids
- Results in equilibrium where marginal reward per cost is equal across bids

**Regret Degradation:**
- Constrained regret typically worse than unconstrained
- Tight budget forces exploration of cheap bids even if mediocre
- Empirical regret should show $R_T^{\text{constrained}} > R_T^{\text{unconstrained}}$

### Requirement 2: Multiple Campaigns with Combinatorial Constraints

#### Exploration-Exploitation with Conflict Constraints

**New Challenge:** Decisions are coupled
- Cannot choose campaigns $i$ and $j$ simultaneously if they conflict
- Advertiser must understand conflict graph structure
- Each subset of campaigns is one "super-arm" (exponentially many)

**Combinatorial Explosion:**
- N campaigns → $2^N$ possible subsets
- Only some subsets are valid (independent sets in conflict graph)
- Number of independent sets can still be exponential

#### Regret with Combinatorial Actions

**Decomposition Insight:**
$$R_T^{\text{comb}} = R_T^{\text{exploration}} + R_T^{\text{conflict}} + R_T^{\text{budget}}$$

1. **Exploration regret:** Learning individual campaign rewards
2. **Conflict regret:** Learning which campaigns are compatible
3. **Budget regret:** Balancing total cost across campaigns

**Regret Bound (Combinatorial-UCB):**
$$R_T = \mathcal{O}(\log T) \cdot \text{poly}(N, |B|)$$

Polynomial factors come from:
- N campaigns to explore
- |B| bids per campaign
- Structure of conflict graph (affects number of valid subsets)

#### Partial Feedback Limitation

**In Requirement 2:** Only observe outcomes for campaigns we actually bid on
- If we don't bid on campaign $i$, we don't learn about distribution $D_i$
- This forces exploration: Must occasionally bid on campaigns we think are bad, to keep learning

**Information Acquisition:**
- Bidding on campaign $i$ with bid $b$ gives one sample from $(v_i - b) \mathbb{1}[b \geq m_i]$
- Must balance between:
  - Exploiting known good campaigns/bids
  - Exploring unknown campaigns to learn distributions
  - Respecting budget and conflict constraints during exploration

### Requirement 3: Best-of-Both-Worlds with Primal-Dual

#### Why Primal-Dual is "Best-of-Both-Worlds"

**Two Guarantees:**
1. **Stochastic setting:** Regret $\mathcal{O}(\log T)$ - learns optimal policy
2. **Adversarial setting:** Regret $\mathcal{O}(\sqrt{T})$ - adapts without knowing future

**Impossible without algorithm adaptation:**
- MAB algorithm like UCB: Linear regret in adversarial
- Pure regret minimizer: Sublinear in adversarial but poor in stochastic

**Primal-Dual Key:**
- Uses **regret minimizer on primal** (exponential weights → plays well in both stochastic and adversarial)
- Uses **dual update** to track constraint satisfaction
- No explicit stochasticity assumption, so naturally handles adversarial

#### Full Feedback Advantage

**In Requirement 3:** Observe all competing bids (full feedback)
- Compare to Requirement 2 where only observe results of our bids
- Full feedback allows better value estimation
- But still uncertain about future (adversarial setting)

**Regret Analysis:**
$$R_T = \mathcal{O}(\sqrt{T \log T}) \text{ (adversarial)}$$

Regret comes from:
- Exponential weights need time to adapt: $\sqrt{T}$ term
- Logarithmic factors from confidence/drift
- Constraint satisfaction (dual variable) adds complexity

#### Primal-Dual Mechanics

**Primal Update (Strategy Adaptation):**
$$x_t \propto \exp(\alpha \sum_{s=1}^t \text{reward}_s(\text{bid}))$$

Exponential weights give:
- Zero cost to exploration (no explicit exploration bonus)
- Automatic focus on high-reward actions
- Handles both stochastic and adversarial naturally

**Dual Update (Constraint Enforcement):**
$$\lambda_t = \lambda_{t-1} + \beta (\text{cumulative cost} - \text{budget allowance})$$

Gradient ascent on Lagrangian:
- Increases $\lambda$ if over-budget (penalizes expensive actions)
- Decreases $\lambda$ if under-budget (encourages spending)
- Converges to optimal Lagrange multiplier

### Requirement 4: Non-Stationary with Change Detection & Sliding Window

#### Exploration-Exploitation vs. Adaptation-Exploitation

**Traditional MAB:** Exploration decreases over time (trust empirical estimates)

**Non-Stationary MAB:** Must always maintain some exploration
- Distributions change, so old empirical estimates become stale
- Question: When to discard old data? How to detect changes?

#### Sliding Window Approach

**Mechanism:** Use only most recent W rounds of data

**Regret Analysis:**
- Within each stationary interval: $\mathcal{O}(\log T)$ regret
- Transition cost at each change point: $\mathcal{O}(1)$ regret
- Total: $R_T = K \cdot \mathcal{O}(\log T) + \#\text{changes} \cdot \mathcal{O}(1)$

where K is number of intervals.

**Optimal Window Size:**
- Too small ($W = \log T$): High variance, poor estimates
- Too large ($W = T$): Slow adaptation to changes
- Sweet spot: $W = \sqrt{T}$ or $W = T^{2/3}$ (depends on problem)

#### Change Detection Approach

**Mechanism:** Run hypothesis test to detect when distribution shifts

**Test Statistic:**
$$Z = \frac{|\hat{\mu}_{\text{recent}} - \hat{\mu}_{\text{historical}}|}{\sqrt{\text{Var}(\hat{\mu}_{\text{recent}}) + \text{Var}(\hat{\mu}_{\text{historical}})}}$$

**Decision Rule:**
- If $Z > \text{Threshold}(t, \alpha)$: Declare change
- Reset: Start new "phase" (forget history, reset UCB)
- Otherwise: Continue as before

**Advantages over sliding window:**
- More responsive to sharp changes
- Doesn't gradually forget data (avoids information loss)
- Can achieve regret $\mathcal{O}(\log T + \#\text{changes} \log T)$ (better constant factors)

**Disadvantages:**
- Hyperparameter tuning (threshold selection)
- Risk of false positives/negatives
- Requires more structure in problem (needs clear detection mechanism)

#### Non-Stationary Regret Notion

**Policy Regret:** Compared to best sequence of actions
$$R_T = \sum_{t=1}^T [\max_b r_t(b) - r_t(b_t)]$$

vs. **Pseudo-Regret:** Compared to best fixed strategy
$$R_T = T \max_b \bar{r}(b) - \sum_{t=1}^T r_t(b_t)$$

where $\bar{r}(b) = \frac{1}{T} \sum_t r_t(b)$ (average over time).

**Why policy regret is right:**
- Fixed-strategy regret is misleading (sequences can dominate all fixed policies)
- Policy regret measures ability to adapt to changes

**Achievable Rates:**
- Stochastic non-stationary: $\mathcal{O}(\log T + \#\text{changes} \log T)$ with change detection
- General adversarial: $\mathcal{O}(\sqrt{T})$ (cannot do better in worst case)

---

## EXPERIMENT DESIGN

### General Framework for All Requirements

#### Template Experiment Structure

```python
def run_full_experiment(req_number):
    """
    Run all baselines and proposed algorithms on environment for Requirement X.
    Save results for plotting.
    """
    
    # 1. SETUP ENVIRONMENT
    env_params = {...}  # Define campaign values, bid set, distributions, etc.
    env_class = {...}   # StochasticSingleCampaignEnvironment, etc.
    
    # 2. DEFINE AGENTS
    agents = {
        'Random': (RandomAgent, {}),
        'Greedy': (GreedyAgent, {}),
        'ETC': (ETCAgent, {'T0': ...}),
        'UCB1': (UCB1Agent, {'bonus_scale': 1}),
        'UCB1-Budget': (BudgetConstrainedUCB1Agent, {'budget': B}),
        # ... other algorithms
    }
    
    # 3. RUN MULTI-TRIAL EXPERIMENTS
    results = {}
    for agent_name, (agent_class, agent_params) in agents.items():
        regrets, rewards, costs, actions = run_experiment(
            env_class=env_class,
            env_params=env_params,
            agent_class=agent_class,
            agent_params=agent_params,
            n_trials=50,
            T=T,
            seed_start=0
        )
        results[agent_name] = {
            'regrets': regrets,
            'rewards': rewards,
            'costs': costs,
            'actions': actions
        }
    
    # 4. COMPUTE STATISTICS
    statistics = {}
    for agent_name, data in results.items():
        regrets = data['regrets']  # Shape: (n_trials, T)
        mean_regret = regrets.mean(axis=0)  # Shape: (T,)
        std_regret = regrets.std(axis=0)    # Shape: (T,)
        ci = std_regret / np.sqrt(len(regrets))  # Confidence interval
        
        statistics[agent_name] = {
            'mean': mean_regret,
            'std': std_regret,
            'ci': ci,
            'final_regret': mean_regret[-1],
        }
    
    # 5. GENERATE PLOTS
    plot_cumulative_regret(statistics)
    plot_bid_histograms(results)
    plot_cost_trajectories(results, budget=B)
    
    # 6. SAVE RESULTS
    save_results(results, f'requirement_{req_number}_results.pkl')
    
    return results, statistics
```

### Requirement 1: Single Campaign Experiments

#### Experiment 1.1: Basic Performance

**Purpose:** Establish baseline performance of UCB vs. budget-constrained variant

**Environment:**
- T = 3000 rounds
- Campaign value v = 1.0
- Bid set B = [0, 0.1, 0.2, ..., 1.0] (11 bids)
- Competing bid distribution: Uniform(0, 1)
- n_trials = 100

**Agents:**
- Random (baseline)
- Greedy (baseline)
- UCB1 without budget
- UCB1 with budget B = 0.4T = 1200
- UCB1 with budget B = 0.2T = 600 (tight budget)
- Clairvoyant (oracle)

**Metrics:**
- Cumulative regret at end
- Final cumulative cost (check budget satisfaction)
- Convergence speed (when does regret growth slow down?)

**Expected Results:**
- UCB regret $\approx \mathcal{O}(\log T)$ (should be ~100-200 at T=3000)
- Budget-UCB regret higher, but cost exactly within budget
- Random regret linear: $R_T \approx \Theta(T)$ (3000)
- Clairvoyant regret ≈ 0

#### Experiment 1.2: Distribution Sensitivity

**Purpose:** Show how algorithm performance depends on competing bid distribution

**Environment Parameters (one per sub-experiment):**
1. Uniform(0, 1) - baseline
2. Normal($\mu=0.5$, $\sigma=0.2$) - concentrated around 0.5
3. Beta(2, 5) - skewed towards low values
4. Bimodal: 0.5 * Uniform(0, 0.3) + 0.5 * Uniform(0.7, 1.0)

**Other params same as Exp 1.1**

**Expected Results:**
- Performance varies by distribution (some distributions easier than others)
- Algorithm should adapt to each distribution's structure
- Tight distributions (normal) should have lower regret (easier to identify best bid)
- Bimodal distributions may confuse learning (multiple local optima)

#### Experiment 1.3: Budget Tightness

**Purpose:** Understand effect of budget constraint severity

**Budget levels:**
- No budget (B = ∞)
- B = 0.6T (loose)
- B = 0.4T (medium)
- B = 0.2T (tight)
- B = 0.1T (very tight)

**Other params same as Exp 1.1**

**Expected Results:**
- Regret increases monotonically with budget tightness
- Very tight budget (0.1T) forces algorithm to play only cheap bids
- May see phase transition: At some budget level, algorithm can't afford good exploration
- Illustrate exploration-exploitation tradeoff with budget constraint

#### Experiment 1.4: Scalability with T

**Purpose:** Verify logarithmic regret growth

**Horizon lengths:**
- T = 500, 1000, 3000, 10000, 30000

**Other params same as Exp 1.1**

**Analysis:**
- Plot log T vs. final regret on log-log scale
- Should see approximately linear relationship (slope = 1)
- Fitted line: $R_T \approx c \log T$, estimate constant $c$

**Expected Results:**
- UCB should show $\log T$ scaling
- Random agent shows $T$ scaling
- Budget-UCB shows slightly worse $\log T$ with larger constant

---

### Requirement 2: Multiple Campaigns Experiments

#### Experiment 2.1: Basic Combinatorial Performance

**Purpose:** Verify Combinatorial-UCB works for multiple campaigns

**Environment:**
- T = 5000 rounds
- N = 3 campaigns
- Campaign values: v = [1.0, 1.0, 1.0]
- Bid set B = [0, 0.2, 0.4, 0.6, 0.8, 1.0] (6 bids)
- Competing bids: Uniform(0, 1) per campaign, uncorrelated
- Conflict graph: NO edges (all campaigns compatible)
- Budget: B = 2.0T (total budget for all campaigns)
- n_trials = 30

**Agents:**
- Random (select random subset and bids)
- Combinatorial-UCB without budget
- Combinatorial-UCB with budget
- Single-campaign UCB applied independently (ignoring conflicts) - what if no coordination?
- Clairvoyant (knows all distributions)

**Metrics:**
- Per-campaign cumulative regret
- Total cumulative regret
- Total cumulative cost (check budget)
- Campaign selection patterns

**Expected Results:**
- Combinatorial-UCB should perform well (learns all three campaigns)
- Without budget: Can explore each campaign freely
- With budget: Must trade-off between campaigns
- Independent UCB: Should perform similarly when no conflicts (no coordination needed)

#### Experiment 2.2: Conflict Graph Effect

**Purpose:** Understand how conflict structure affects learning

**Conflict graphs (3 campaigns):**
1. No edges (independent)
2. Complete graph (only 1 campaign per round)
3. Path: (1-2), (2-3) [campaign 2 conflicts with both]
4. Triangle: (1-2), (2-3), (1-3) [all conflict pairwise]

**Other params same as Exp 2.1**

**Analysis:**
- Count number of valid independent sets per graph
- No edges: $2^3 = 8$ sets (can choose any subset)
- Complete graph: 4 sets (only singletons and empty set)
- Path: 6 sets
- Triangle: 4 sets

**Expected Results:**
- Regret increases with graph density (fewer valid actions)
- Complete graph: Highest regret (limited flexibility)
- No edges: Lowest regret (full flexibility)
- Quantify tradeoff between exploration and conflict constraints

#### Experiment 2.3: Campaign Value Heterogeneity

**Purpose:** Show algorithm adapts to different campaign values

**Campaign value sets:**
1. Symmetric: v = [1.0, 1.0, 1.0]
2. Heterogeneous: v = [2.0, 1.0, 0.5]
3. Dominated: v = [1.0, 0.5, 0.3] (one campaign much better)

**Conflict graph:** NO conflicts (all compatible)

**Other params same as Exp 2.1**

**Expected Results:**
- Algorithm learns to prioritize high-value campaigns
- With heterogeneous values, should spend more on campaign 1 (v=2.0)
- Regret analysis should show which campaigns account for most regret

#### Experiment 2.4: Scalability with N

**Purpose:** Understand how number of campaigns affects performance

**Number of campaigns:**
- N = 2, 3, 4, 5, 6

**Other params:**
- Conflict graph: NO edges
- Campaign values: All equal v_i = 1.0
- Budget per campaign: 0.3T (loose)
- T = 2000 (fixed)

**Analysis:**
- As N increases, more arms to learn (higher-dimensional problem)
- But each campaign independent, so expected regret should scale roughly as N times single-campaign regret

**Expected Results:**
- Regret scales approximately linearly with N
- Convergence speed slower for more campaigns (need more samples to learn all)
- Algorithm should still handle up to 6 campaigns efficiently (within combinatorial solver capacity)

---

### Requirement 3: Primal-Dual Best-of-Both-Worlds

#### Experiment 3.1: Stochastic Setting

**Purpose:** Verify primal-dual works well in stochastic setting

**Environment:**
- Same as Requirement 2.1 (3 campaigns, no conflicts, stochastic)
- T = 5000
- n_trials = 30

**Agents:**
- Combinatorial-UCB (Requirement 2)
- Primal-Dual (Requirement 3)
- Clairvoyant

**Metrics:**
- Cumulative regret trajectory
- Final regret comparison
- Cost trajectory (budget satisfaction)

**Expected Results:**
- Primal-Dual should be comparable to Combinatorial-UCB in stochastic setting
- Both should have sublinear $\mathcal{O}(\log T)$ regret
- Primal-Dual might have higher constants (randomization overhead)

#### Experiment 3.2: Adversarial Setting

**Purpose:** Show Primal-Dual handles adversarial sequences

**Environment:**
- Same 3 campaigns, no conflicts
- Competing bids generated adversarially: $m_{i,t} = 0.3 + 0.4 \sin(2\pi t / 1000)$ (sinusoidal pattern)
- T = 5000
- n_trials = 20

**Agents:**
- Combinatorial-UCB (should fail)
- Primal-Dual with full feedback
- Primal-Dual with only observed feedback (compare)

**Metrics:**
- Cumulative regret vs. best fixed policy in hindsight
- Regret scaling: should see $\sqrt{T}$ behavior

**Expected Results:**
- Combinatorial-UCB: Linear or superlinear regret (assumes stochasticity)
- Primal-Dual: Sublinear $\mathcal{O}(\sqrt{T})$ (adapts to adversarial)
- Full feedback vs. observed: Full feedback better (lower regret constant)

#### Experiment 3.3: Hybrid Stochastic-Adversarial

**Purpose:** Test robustness to partially adversarial settings

**Environment:**
- Two phases:
  - Phase 1 (t=1 to 2500): Stochastic (Uniform(0,1))
  - Phase 2 (t=2501 to 5000): Adversarial (sinusoidal, deterministic)

**Agents:**
- Combinatorial-UCB
- Primal-Dual

**Expected Results:**
- Combinatorial-UCB: Linear regret in phase 2 (assumes stochasticity)
- Primal-Dual: Sublinear overall (adapts to phase 2)
- Illustrate "best-of-both-worlds" benefit

---

### Requirement 4: Non-Stationary Environments

#### Experiment 4.1: Sliding Window Parameter Tuning

**Purpose:** Find optimal window size

**Environment:**
- 3 campaigns, no conflicts
- 4 intervals of 1250 rounds each
- Each interval has different distribution:
  - Interval 1: Uniform(0, 1)
  - Interval 2: Uniform(0.2, 1.0) [shifted up]
  - Interval 3: Uniform(0, 0.8) [shifted down]
  - Interval 4: Back to Uniform(0, 1)

**Window sizes to test:**
- W = 50, 100, 300, 700, 1000, 2000 (no window)

**Agents:**
- Sliding Window Combinatorial-UCB with each W
- Non-adaptive Combinatorial-UCB (baseline)

**Metrics:**
- Total cumulative regret
- Per-interval regret breakdown
- Adaptation speed (how fast to learn new distribution)

**Analysis:**
- Plot regret vs. W
- Identify optimal W
- Check if optimal W matches theory: $W \propto \sqrt{T}$ or $W \propto T^{2/3}$?

**Expected Results:**
- Very small W: High regret (poor estimation)
- Very large W: High regret in new intervals (slow adaptation)
- Optimal W somewhere in middle
- Theoretical prediction $W = \sqrt{T} \approx 70$ should be in range

#### Experiment 4.2: Change Detector Threshold Tuning

**Purpose:** Tune change detection sensitivity

**Environment:**
- Same as Exp 4.1

**Thresholds to test:**
- C = 1.0, 2.0, 3.0, 4.0, 5.0 (multiplicative factor on standard deviation)

**Agents:**
- Change Detector Combinatorial-UCB with each threshold
- Non-adaptive baseline

**Metrics:**
- True positives: Detected actual changes
- False positives: Declared change when distribution unchanged
- Total regret
- Per-interval regret

**Expected Results:**
- Low threshold: Many false positives (reset often, waste exploration)
- High threshold: Miss real changes
- Sweet spot: Detects most changes with few false positives
- Regret vs. threshold curve shows optimal point

#### Experiment 4.3: Comparison of Adaptation Methods

**Purpose:** Compare sliding window, change detection, and primal-dual

**Environment:**
- Same piecewise-stationary with 4 intervals

**Agents:**
- Sliding Window (tuned from Exp 4.1)
- Change Detector (tuned from Exp 4.2)
- Primal-Dual (Requirement 3)
- Non-adaptive Combinatorial-UCB (baseline)
- Clairvoyant (oracle per interval)

**Metrics:**
- Cumulative regret
- Per-interval breakdown
- Cumulative cost (verify budget)

**Expected Results:**
- Primal-Dual: Consistent good performance (full feedback advantage)
- Sliding Window: Smooth adaptation, good on gradual changes
- Change Detector: Faster response to sharp changes
- Non-adaptive: Linear regret (fails completely)
- Show tradeoffs between methods

#### Experiment 4.4: Sharp vs. Gradual Changes

**Purpose:** Understand when each algorithm excels

**Environment 1 (Sharp):**
- Distributions swap completely: $D_1, D_2, D_1, D_2$ every 1250 rounds

**Environment 2 (Gradual):**
- Distributions transition smoothly: Distribution gradually morphs from $D_1$ to $D_2$ over 1250 rounds

**Agents:**
- Sliding Window
- Change Detector
- Primal-Dual

**Expected Results:**
- Sharp changes: Change Detector excels (detects instantaneous shift)
- Gradual changes: Sliding Window excels (smoothly forgets old data)
- Primal-Dual: Robust to both (but may not be optimal for either)

---

## FINAL DELIVERABLES

### 1. GitHub Repository Structure

```
github.com/group-name/ola-advertising-campaigns/

├── README.md                    # Comprehensive overview and setup guide
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignore __pycache__, results/, etc.
│
├── src/                         # Implementation code (described in Architecture section)
│   ├── __init__.py
│   ├── environments/
│   ├── agents/
│   ├── optimization/
│   ├── plotting/
│   └── utils/
│
├── notebooks/                   # Jupyter notebooks for each requirement
│   ├── 01_single_campaign_stochastic.ipynb
│   ├── 02_multi_campaign_stochastic.ipynb
│   ├── 03_primal_dual_best_of_both_worlds.ipynb
│   └── 04_non_stationary_environments.ipynb
│
├── results/                     # Experiment outputs (generated, not in repo)
│   ├── requirement_1/
│   ├── requirement_2/
│   ├── requirement_3/
│   └── requirement_4/
│
├── analysis/                    # Post-hoc analysis scripts
│   ├── analyze_results.py      # Load results, compute statistics
│   ├── compare_algorithms.py   # Generate comparison plots
│   └── sensitivity_analysis.py # Hyperparameter tuning analysis
│
└── docs/                        # Documentation
    ├── algorithm_descriptions.md
    ├── mathematical_background.md
    └── experiment_setup.md
```

### 2. README.md Contents

**Sections:**
1. **Overview:** Project goal, requirements summary
2. **Installation:** Python version, venv setup, `pip install -r requirements.txt`
3. **Quick Start:** How to run a single experiment
4. **Project Structure:** Explanation of directories
5. **Algorithms Implemented:** Brief description of each
6. **Experiments:** How to reproduce all experiments
7. **Results:** Links to plots and discussion
8. **References:** Relevant papers and course materials

**Key point:** Make it easy for others to run the code and reproduce results

### 3. Presentation Slides (20 min)

**Slide Breakdown (estimate 3-4 min per requirement):**

**Slides 1-3: Introduction & Problem Setting**
- Advertising campaigns motivation
- Budget constraints & conflicts
- Comparison to theory coursework

**Slides 4-8: Requirement 1 (Single Campaign, Stochastic)**
- Problem formulation
- Algorithm: UCB1 (no budget) and Budget-Constrained UCB1
- Empirical results: Cumulative regret plot
- Key insights: Exploration-exploitation tradeoff

**Slides 9-13: Requirement 2 (Multiple Campaigns, Stochastic)**
- Problem formulation: Conflict graph, combinatorial action space
- Algorithm: Combinatorial-UCB
- Combinatorial solver (brute force for small N)
- Results: Multi-campaign performance, conflict effects
- Key insights: Coupling and information acquisition

**Slides 14-17: Requirement 3 (Best-of-Both-Worlds, Primal-Dual)**
- Problem: Handle both stochastic and adversarial settings
- Algorithm: Primal-Dual with exponential weights
- Results: Stochastic vs. adversarial regret comparison
- Key insights: Why full feedback helps, regret rates

**Slides 18-21: Requirement 4 (Non-Stationary)**
- Problem: Piecewise-stationary distributions
- Algorithms: Sliding Window vs. Change Detection
- Results: Per-interval regret, adaptation speed
- Key insights: Tradeoffs between methods

**Slides 22-25: Discussion & Conclusions**
- Summary of regret bounds achieved
- Unexpected results (if any) and explanations
- Future extensions: Real auction data, other constraints
- Group contributions (who did what)

**Visual Guidelines:**
- Use consistent color scheme for algorithms
- Include sample plots (cumulative regret, bid selections)
- Keep text minimal (talk through content)
- Use clear legends and axis labels

### 4. Empirical Analysis Report

**Sections:**

#### 4.1 Executive Summary
- High-level findings: Which algorithms work best in which settings
- Key surprises or unexpected behaviors
- Recommendations

#### 4.2 Requirement-by-Requirement Analysis

**For each requirement:**
- Problem statement and motivation
- Algorithms tested and why
- Key hyperparameters and their effects
- Experimental setup (T, n_trials, budget levels, etc.)
- Empirical results:
  - Cumulative regret plots with uncertainty bands
  - Final regret statistics (mean ± std)
  - Cost/budget compliance
  - Convergence behavior
- Comparison to theoretical predictions
- Discussion of results:
  - Does empirical regret match $O(\log T)$ or $O(\sqrt{T})$ predictions?
  - Why might it deviate?
  - Sensitivity to hyperparameters
  - Robustness across different environments

#### 4.3 Cross-Requirement Insights
- How does performance scale from single to multiple campaigns?
- Primal-dual advantages in non-stationary settings
- Budget constraints impact across all requirements
- Conflict graph complexity effects

#### 4.4 Unexpected Results & Discussion
- If any algorithm significantly underperforms expectations, explain why
- If any result contradicts theory, discuss possible causes:
  - Small sample size / finite T effects?
  - Hyperparameter choices not optimal?
  - Theoretical analysis loose or problem-specific?
- Suggest fixes or adaptations if problems found

#### 4.5 Sensitivity Analysis
- How robust are conclusions to:
  - Number of trials n_trials
  - Horizon length T
  - Budget tightness
  - Distributions used
  - Algorithm hyperparameters
- Show plots of regret vs. key parameters

#### 4.6 Computational Complexity
- Wall-clock time for each algorithm
- Scalability: How does runtime grow with N, |B|, T?
- Practical feasibility for real auctions

### 5. Code Quality Standards

**Checked before submission:**

- [ ] All code runs without errors
- [ ] Functions have docstrings explaining inputs/outputs
- [ ] Comments explain non-obvious algorithmic steps
- [ ] Consistent variable naming aligned with mathematical notation
- [ ] Proper error handling (no silent failures)
- [ ] Results are reproducible (fixed random seeds)
- [ ] Plots are publication-ready (clear labels, legends, units)
- [ ] README provides clear setup and running instructions
- [ ] No hardcoded paths (use relative paths or config files)
- [ ] All dependencies documented in requirements.txt

### 6. Discussion of Unexpected Results

**Framework:**

For each unexpected result (if any), discuss:

1. **What was expected?** (Theory or intuition)
   - Cite specific theoretical result or paper
   - Explain reasoning

2. **What was observed?** (Empirical result)
   - Show relevant plot/number
   - Quantify deviation from expectation

3. **Why might this happen?** (Hypotheses)
   - Small sample effects ($n_{trials}$ too small)?
   - Finite T effects (asymptotics don't apply)?
   - Hyperparameter suboptimality?
   - Bug in implementation?
   - Theoretical analysis is loose?
   - Problem-specific structure invalidates theory?

4. **Evidence for/against each hypothesis**
   - How would we test each hypothesis?
   - What experiments would clarify?

5. **Proposed fix or explanation** (if applicable)
   - How to resolve the discrepancy?
   - Any code changes needed?
   - New insight gained?

**Common issues to expect:**

- **UCB explores too much early on** → Confidence bonus scaling may be loose; try tuning
- **Budget constraint violations** → Dual update rate wrong; try different step size
- **Combinatorial algorithms slow** → Solver not optimized; pre-compute independent sets
- **Change detector false positives** → Threshold too low; increase sensitivity parameter
- **Primal-dual regret not $\sqrt{T}$ in adversarial** → May need more trials to see asymptotics

### 7. Presentation Tips (from project instructions)

**From project requirements:**
- "A detailed discussion of unexpected results is appreciated"
- Prepare to answer questions about why algorithms perform as they do
- Be ready to discuss trade-offs (e.g., budget tightness vs. regret)

**Suggestions:**
- Practice explaining algorithms concisely (3-5 min per algorithm)
- Have detailed plots ready to show during Q&A
- Understand your own code (be ready to explain implementation details)
- Anticipate questions: "Why not try algorithm X?" Have answer ready
- Discuss real-world applicability (could this work on real ad campaigns?)

---

## SUMMARY

This roadmap provides a complete decomposition of the Online Learning Applications project:

**Requirements Breakdown:** 4 increasingly complex scenarios from single-campaign stochastic to multi-campaign non-stationary settings

**Implementation Roadmap:** 4 phases with specific deliverables, building incrementally from foundations to advanced techniques

**Theoretical Understanding:** How exploration-exploitation, constraints, combinatorial structure, and non-stationarity affect learning and regret

**Experimental Validation:** Detailed experiments to verify algorithms achieve theoretical regret bounds and understand practical tradeoffs

**Deliverables:** Clean repository, presentation slides, empirical analysis, and code quality ready for academic evaluation

The project successfully integrates all 10 course notebooks and applies their concepts to a realistic and challenging problem:  advertising campaign bidding under multiple constraints.

