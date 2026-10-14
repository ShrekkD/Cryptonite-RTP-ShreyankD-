# Task 2 — Decision Trees, Forests & Gradient Boosters

Mathematical expressions used:

- **Gini impurity:** `Gini(S) = 1 − Σ pᵢ²`
- **Entropy:** `H(S) = − Σ pᵢ log₂(pᵢ)`
- **Information Gain:** `IG(S, A) = Impurity(S) − Σᵥ (|Sᵥ|/|S|)·Impurity(Sᵥ)`
- **Random Forest:** bagging + random feature subsets (`d ≈ √m`) to decorrelate trees.
- **Gradient Boosting:** `Fₘ(x) = Fₘ₋₁(x) + ν·hₘ(x)`, each tree fits the loss's negative gradient.
