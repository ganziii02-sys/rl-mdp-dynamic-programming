# Finite-Horizon MDPs with Dynamic Programming

> **UTS 43008 Reinforcement Learning** (Spring 2026) · Assignment 1

Two real-world control problems formulated as finite-horizon Markov Decision Processes and solved with **Value Iteration** (backward induction) and **Policy Iteration**. Each notebook simulates the optimal policies, tests sensitivity to the model parameters and compares the two algorithms.

| Notebook | Problem | MDP |
|---|---|---|
| [`Scenario1_RideSharing_DynamicPricing.ipynb`](Scenario1_RideSharing_DynamicPricing.ipynb) | **Dynamic pricing for a ride-sharing platform**: one zone, 2-hour window, a decision every 10 minutes to balance rider demand and driver supply | T = 12, 9 states, 5 actions (hold, raise/lower price, driver incentive, pause requests) |
| [`Scenario3_SmartGrid_EnergyManagement.ipynb`](Scenario3_SmartGrid_EnergyManagement.ipynb) | **Energy management in a smart grid**: hourly decisions over a 24-hour day to distribute, store, import or curtail energy | T = 24, 10 factored states (grid situation × battery level + critical), 6 actions, time-varying tariffs |

Each notebook covers:

1. MDP formulation (states, actions, rewards, time-varying transitions)
2. Finite-horizon DP solvers
3. Optimal policy and value function
4. Robustness / sensitivity analysis and Value Iteration vs Policy Iteration
5. Policy simulation
6. Results summary and key findings

**Tech stack:** Python · NumPy · pandas · Matplotlib

**Report:** [`docs/Assignment1_Report.pdf`](docs/Assignment1_Report.pdf)

## How to run

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook
```
