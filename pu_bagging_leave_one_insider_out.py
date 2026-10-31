"""Bagging PU on wallet_features.csv, evaluated with LEAVE-ONE-INSIDER-OUT, with two base learners:
depth-4 decision trees and logistic regression. Both use the same folds, test sets and bags.

Each of the 19 confirmed insiders is held out once; the other 18 stay in training.
Caution: several insiders have a look-alike insider (probably the same event or operator) that stays
in training when one of them is held out, so this evaluation is optimistic. Compare with
pu_bagging_leave_one_group_out.py.

Default run reproduces:
    trees:    top 1% = 9/19, top 5% = 14/19, top 10% = 16/19, AUC = 0.872 (ties against) / 0.913 (ties split)
    logistic: top 1% = 10/19, top 5% = 17/19, top 10% = 18/19, AUC = 0.972 (no ties)
Robustness runs used --split-seed 0/1 and --pu-seed 0/1/2.

Also prints the bagged logistic coefficients of a final model fit on all 19 insiders. They describe what
separates confirmed wallets from a random draw of unconfirmed wallets; they are not causal effects, and
the size features are strongly correlated with each other.

Assumptions: confirmed=True wallets are insiders; confirmed=False wallets are UNLABELED (a mix), not
non-insiders; no bot filter; settings K=360, T=200, depth 4 fixed in advance, never tuned. Logistic
regression uses sklearn defaults (L2 penalty, C=1) on the log features standardized over all wallets.

    python3 pu_bagging_leave_one_wallet_out.py
    python3 pu_bagging_leave_one_wallet_out.py --split-seed 1 --pu-seed 2

Needs numpy, pandas, scikit-learn (Python 3.9+).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

DATA = ("wallet_features.csv")
FEATS = ["total_notional", "max_single_buy", "trade_count", "market_notional_ratio",                   # size
         "avg_buy_price", "concentration", "position_anomaly_ratio", "wallet_age_days", "unique_markets"]  # behavior
NOT_LOGGED = ["avg_buy_price", "concentration"]   # already between 0 and 1
K, T, DEPTH = 360, 200, 4                          # unconfirmed wallets per tree, trees, tree depth


def load(path):
    """Steps 0-1: read the wallet table and log-transform the heavy-tailed features."""
    W = pd.read_csv(path, index_col="wallet")
    X = W[FEATS].to_numpy(float).copy()
    for c, name in enumerate(FEATS):
        if name not in NOT_LOGGED:
            X[:, c] = np.log1p(X[:, c])
    return W, X, W.confirmed.to_numpy(bool)


def standardize(X):
    """Step 1b (logistic only): rescale each feature to mean 0, SD 1 over all wallets (uses no labels)."""
    return (X - X.mean(0)) / X.std(0, ddof=1)


def make_folds(pos, unl, split_seed):
    """Step 2: one fold per insider. Test set = that insider + a random 20% of the unconfirmed wallets."""
    rng = np.random.default_rng(split_seed)
    return [(i, rng.choice(unl, size=len(unl) // 5, replace=False)) for i in pos]


def bagging_pu(X_pos, X_unl, X_score, pu_seed, base="tree"):
    """Steps 3-4: T base models (base="tree": depth-4 tree; base="logistic": logistic regression), each
    fit on ALL training insiders (label 1) + K unconfirmed wallets drawn at random with replacement
    (label 0). The score is the models' average vote. The same pu_seed gives the same bags for both.
    Also returns each logistic model's coefficients (T rows x 9 features; empty for trees)."""
    rng = np.random.default_rng(pu_seed)
    score = np.zeros(len(X_score))
    coefs = []
    for t in range(T):
        pick = rng.choice(len(X_unl), size=K, replace=True)
        X = np.vstack([X_pos, X_unl[pick]])
        y = np.r_[np.ones(len(X_pos)), np.zeros(K)]
        if base == "tree":
            model = DecisionTreeClassifier(max_depth=DEPTH, random_state=pu_seed * 1000 + t)
        else:
            model = LogisticRegression(C=1.0, max_iter=1000)
        score += model.fit(X, y).predict_proba(X_score)[:, 1]
        if base == "logistic":
            coefs.append(model.coef_[0])
    return score / T, np.array(coefs)


def percentiles(s_unl, s_insider):
    """Step 5: share of test unconfirmed wallets ranked at or above the insider (0 = top of the list)."""
    against = np.mean(s_unl >= s_insider)                                     # ties count against the insider
    split = np.mean(s_unl > s_insider) + 0.5 * np.mean(s_unl == s_insider)    # ties split evenly
    return against, split


def summary(name, pct):
    """Step 7: top-k% counts, AUC = 1 - mean percentile, median percentile."""
    n = len(pct)
    return (f"{name:36s} top1={int((pct <= .01).sum())}/{n}  top5={int((pct <= .05).sum())}/{n}  "
            f"top10={int((pct <= .10).sum())}/{n}  AUC={1 - pct.mean():.3f}  median={np.median(pct):.1%}")


def coef_summary(C, fold_means):
    """Step 8: bagged logistic coefficients, on the standardized log scale (change in log-odds per 1 SD).
    mean, OR_per_SD = average over the T bags and its odds ratio; bag_5%, bag_95%, bags_>0 = spread and
    sign agreement across bags; fold_min, fold_max = range of the per-fold averages over the 19 folds."""
    S = pd.DataFrame({"mean": C.mean(0), "OR_per_SD": np.exp(C.mean(0)),
                      "bag_5%": np.percentile(C, 5, 0), "bag_95%": np.percentile(C, 95, 0),
                      "bags_>0": (C > 0).mean(0),
                      "fold_min": fold_means.min(0), "fold_max": fold_means.max(0)}, index=FEATS)
    fmt = {c: "{:+.2f}".format for c in S.columns}
    fmt.update({"OR_per_SD": "{:.2f}".format, "bags_>0": "{:.0%}".format})
    return S.sort_values("mean", ascending=False).to_string(formatters=fmt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DATA)
    p.add_argument("--split-seed", type=int, default=0, help="seed for the random 20%% test sets")
    p.add_argument("--pu-seed", type=int, default=0, help="seed for the PU sampling and the trees")
    a = p.parse_args()

    W, X, y = load(a.data)
    Z = standardize(X)                                               # input for the logistic model
    pos, unl = np.flatnonzero(y), np.flatnonzero(~y)                # 19 insiders, 55,933 unconfirmed
    spend = W.total_notional.to_numpy()
    folds = make_folds(pos, unl, a.split_seed)

    rows, fold_coefs = [], []
    for i, test_unl in folds:                                        # Step 6: one fold per insider
        train_pos = np.setdiff1d(pos, [i])                           # the other 18 insiders
        train_unl = np.setdiff1d(unl, test_unl)                      # the other 80% of unconfirmed wallets
        test = np.r_[test_unl, i]
        row = dict(insider=W.index[i])
        for base, F in [("tree", X), ("logistic", Z)]:               # same bags for both base learners
            s, coefs = bagging_pu(F[train_pos], F[train_unl], F[test], a.pu_seed, base)
            row[f"{base}_pct"], row[f"{base}_pct_ties_split"] = percentiles(s[:-1], s[-1])
            if base == "logistic":
                fold_coefs.append(coefs.mean(0))
        row["spend_pct"] = np.mean(spend[test_unl] >= spend[i])
        rows.append(row)
    R = pd.DataFrame(rows)

    print(f"leave-one-wallet-out | split seed {a.split_seed} | PU seed {a.pu_seed} | K={K} T={T} depth={DEPTH}\n")
    print(summary("bagging PU trees (ties against)", R.tree_pct.to_numpy()))
    # print(summary("bagging PU trees (ties split)", R.tree_pct_ties_split.to_numpy()))
    print(summary("bagging PU logistic (ties against)", R.logistic_pct.to_numpy()))
    # print(summary("bagging PU logistic (ties split)", R.logistic_pct_ties_split.to_numpy()))
    print(summary("spend ranking, same test sets", R.spend_pct.to_numpy()))
    print("\nper insider (percentile, 0% = top of the list):")
    # print(R.sort_values("logistic_pct").to_string(index=False, formatters={c: "{:.1%}".format for c in R.columns[1:]}))

    # Step 8: final logistic model on all 19 insiders and all unconfirmed wallets (its scores are not used)
    _, C = bagging_pu(Z[pos], Z[unl], Z[pos], a.pu_seed, "logistic")
    print(f"\nbagging PU logistic coefficients | all 19 insiders | {T} bags | per 1 SD of the (log) feature:")
    print(coef_summary(C, np.array(fold_coefs)))
