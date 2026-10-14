# Task 2 — Decision Trees, Forests & Gradient Boosters

**Cryptonite Research Task · Phase 1 · Module 2**
Cybersecurity **Intrusion Detection** — a binary-classification problem
(`attack_detected`: `1` = intrusion detected, `0` = normal).

This folder contains a complete solution: EDA, a baseline **Random Forest** benchmarked against
the reference targets, an incremental **feature-engineering** study with per-method score deltas,
hyper-parameter **tuning**, and the **bonus** gradient-boosting models (XGBoost / LightGBM /
CatBoost) plus a reusable **Kaggle** submission pipeline.

---

## 📁 Contents

| File | Description |
|------|-------------|
| `Model_Tuning.ipynb` | Task 1–3: EDA → Baseline Random Forest → Feature Engineering (deltas) → Tuning |
| `Kaggle_Competition_and_Boosting.ipynb` | Bonus: XGBoost / LightGBM / CatBoost, ensemble, and a Kaggle submission pipeline |
| `Decision_Trees_and_Boosters_Report.pdf` | 2-page structured report (tree mathematics, comparative analysis, boosting details) |
| `README.md` | This file |
| `cybersecurity_intrusion_data.csv` | The dataset (auto-downloaded from Kaggle if absent) |
| `lib/` | macOS `libomp.dylib` (only needed to run the boosting notebook on Apple Silicon/Intel without Homebrew) |

---

## 📊 Dataset

**Cybersecurity Intrusion Detection Dataset** (Kaggle): `dnkumars/cybersecurity-intrusion-detection-dataset`

- **9,537 rows × 10 usable features** (after dropping the unique `session_id` identifier)
- 6 numeric, 3 categorical (`protocol_type`, `encryption_used`, `browser_type`), 1 binary flag
- **Target** `attack_detected` is **balanced** (~55% normal / 45% attack; imbalance ratio 1.24)
- **Missing values:** only `encryption_used` (~20%, 1,966 cells) — informative missingness

The notebooks look for a local copy first, then download the public Kaggle archive automatically,
so **no Kaggle account or API key is required** to run them.

---

## 🚀 How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm catboost      # for the bonus notebook
jupyter notebook                           # or VS Code / any Jupyter front-end
```

Then open and run `Model_Tuning.ipynb`, followed by `Kaggle_Competition_and_Boosting.ipynb`.

> **macOS note (boosters only).** The official `xgboost`/`lightgbm` macOS wheels need the OpenMP
> runtime. If `import xgboost` fails with *"libomp.dylib not loaded"*, run `brew install libomp`,
> **or** a pre-bundled copy is already in `lib/` and the notebook tries to load it automatically.
> To execute from a terminal on this repo:
> ```bash
> cd Task-2 && DYLD_LIBRARY_PATH="$PWD/lib" jupyter nbconvert --to notebook --execute --inplace \
>     Kaggle_Competition_and_Boosting.ipynb
> ```

---

## ✅ Results (stratified 80/20 split, `random_state = 42`)

Reference baseline targets: **Accuracy ≥ 0.8831**, **F1 ≥ 0.8508**, **ROC-AUC ≥ 0.8762**.

### Baseline vs. targets
| Metric | Target | Baseline RF | Status |
|--------|--------|-------------|--------|
| Accuracy | ≥ 0.8831 | **0.8831** | ✅ met |
| F1 (binary) | ≥ 0.8508 | **0.8512** | ✅ met |
| ROC-AUC | ≥ 0.8762 | **0.8787** | ✅ met |

### Tuned Random Forest (best overall)
| Metric | Tuned RF | Δ vs baseline |
|--------|----------|---------------|
| Accuracy | **0.8863** | +0.0032 |
| F1 | **0.8545** | +0.0033 |
| ROC-AUC | **0.8824** | +0.0037 |

### Bonus boosters
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Random Forest (ref.) | 0.8836 | 0.8514 | **0.8820** |
| XGBoost | 0.8831 | 0.8514 | 0.8791 |
| LightGBM | 0.8826 | 0.8511 | 0.8778 |
| **CatBoost** | **0.8868** | **0.8550** | 0.8786 |
| Ensemble (XGB+LGB) | 0.8826 | 0.8509 | 0.8783 |

**Every reference target is cleared.** CatBoost gives the highest Accuracy/F1; the tuned forest
gives the highest ROC-AUC (~0.882 — the data's own ceiling for these models).

---

## 🔍 Key findings

- **EDA:** balanced target; `ip_reputation_score`, `failed_logins`, `unusual_time_access` and
  `login_attempts` are the strongest single predictors; `encryption_used = DES` is riskier than `AES`.
- **Feature engineering deltas** (cumulative): frequency/target encoding alone slightly overfit for
  bagged trees (small AUC drop), while **interaction features** gave the largest single gain
  (ΔAUC +0.0026) and **tuning** the best overall model. All six families are reported step-by-step.
- **Boosting:** each successive tree fits the residual errors, which helps on this dataset's
  overlapping class region; **CatBoost** handles categoricals natively with ordered target
  statistics (no target leakage) and leads on Accuracy/F1.

---

## 🏆 Kaggle competition

`Kaggle_Competition_and_Boosting.ipynb` ships a **reusable submission pipeline**: it fits on the
training data, applies the same feature engineering to any competition test file, and writes
`submission.csv`. To use it on the latest Playground competition, set the configuration variables
(`COMP_TRAIN_CSV`, `COMP_TEST_CSV`, `ID_COL`, target/submission column names) to match that
competition's schema, re-run the submission cell, and upload the file. Remember to **share your
Kaggle username** so the leaderboard can be tracked.

---

## 🧮 Conceptual reference (full detail in the report PDF)

- **Gini impurity:** `Gini(S) = 1 − Σ pᵢ²`
- **Entropy:** `H(S) = − Σ pᵢ log₂(pᵢ)`
- **Information Gain:** `IG(S, A) = Impurity(S) − Σᵥ (|Sᵥ|/|S|)·Impurity(Sᵥ)`
- **Random Forest:** bagging + random feature subsets (`d ≈ √m`) to decorrelate trees.
- **Gradient Boosting:** `Fₘ(x) = Fₘ₋₁(x) + ν·hₘ(x)`, each tree fits the loss's negative gradient.
