"""
Motor third-party liability claim frequency modelling on freMTPL2freq
(678,013 French MTPL policies).

Python re-implementation and extension of course Assignment 2 (original R
version in assignment2_*.R):
  Q1  Poisson regression tree, cp = 0.00005, min. 1,500 policies per leaf,
      pruned by 10-fold cross-validation
  Q2  Poisson boosting machine, M = 50 trees of depth J = 3,
      min. 10,000 policies per leaf, no shrinkage
Extensions (not in the original assignment):
  - homogeneous (intercept-only) model and a Poisson GLM as benchmarks
  - boosting with shrinkage, depth / learning-rate selected on a validation
    split with early stopping
  - split-gain variable importance of the tuned boosting machine
  - paired bootstrap on the test policies for the gaps between models

All losses are average Poisson deviance in units of 10^-2 (same scale as the
course code).

Usage:  python claim_frequency.py freMTPL2freq.rda .
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 100
CATS = ["Area", "VehBrand", "VehGas", "Region"]
NUMS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
FEATS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
         "VehBrand", "VehGas", "Density", "Region"]


def load(path):
    import rdata
    d = rdata.read_rda(path)["freMTPL2freq"].reset_index(drop=True)
    for c in CATS:
        d[c] = d[c].astype(str)
    for c in NUMS + ["ClaimNb", "Exposure"]:
        d[c] = d[c].astype(float)
    return d


def unit_deviance(mu, n):
    """Poisson deviance of each policy x 100."""
    mu = np.asarray(mu, float)
    n = np.asarray(n, float)
    t = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)
    return 200.0 * (mu - n + t)


def deviance(mu, n):
    """Average Poisson deviance x 100 (in 10^-2)."""
    return float(np.mean(unit_deviance(mu, n)))


class FreqEncoder:
    """Order each categorical level by its observed claim frequency on the
    fitting sample (what a Poisson tree does with unordered factors)."""

    def fit(self, d):
        self.maps = {}
        for c in CATS:
            g = d.groupby(c)[["ClaimNb", "Exposure"]].sum()
            order = (g.ClaimNb / g.Exposure).sort_values().index
            self.maps[c] = {k: i for i, k in enumerate(order)}
        return self

    def transform(self, d):
        X = d[FEATS].copy()
        for c in CATS:
            X[c] = X[c].map(self.maps[c]).fillna(-1)
        return X.to_numpy(float)


# ----------------------------------------------------------------- GLM
def glm_design(d, cols=None):
    X = pd.DataFrame(index=d.index)
    X["AreaNum"] = d["Area"].map({k: i for i, k in enumerate("ABCDEF", 1)}).astype(float)
    X["BonusMalus"] = np.minimum(d["BonusMalus"], 150.0)
    X["logDensity"] = np.log(d["Density"])
    X["Diesel"] = (d["VehGas"] == "Diesel").astype(float)
    cat = pd.DataFrame({
        "VehPower": np.minimum(d["VehPower"], 9).astype(int).astype(str),
        "VehAge": pd.cut(d["VehAge"], [-1, 0, 10, 1e3], labels=["0", "1-10", "11+"]).astype(str),
        "DrivAge": pd.cut(d["DrivAge"], [17, 20, 25, 30, 40, 50, 70, 1e3],
                          labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]).astype(str),
        "VehBrand": d["VehBrand"], "Region": d["Region"]}, index=d.index)
    X = pd.concat([X, pd.get_dummies(cat, drop_first=True, dtype=float)], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


# ----------------------------------------------------------------- trees
def fit_tree(X, n, w, **kw):
    t = DecisionTreeRegressor(criterion="poisson", random_state=SEED, **kw)
    t.fit(X, n / w, sample_weight=w)
    return t


def boosting(Xl, nl, el, evals, M, depth, min_leaf, shrink=1.0, patience=None, init_rate=1.0):
    """Poisson boosting machine: each tree is fitted to the current working
    exposure mu (expected claims so far) and updates it multiplicatively.
    evals = list of (X, n, exposure) on which the loss path is tracked;
    with `patience`, stop when the loss on evals[0] stops improving.
    init_rate = 1 reproduces the course algorithm (start from the exposure);
    with shrinkage the start value is the portfolio frequency instead."""
    mu_l = el * init_rate
    mus = [e * init_rate for _, _, e in evals]
    path_l, paths = [deviance(mu_l, nl)], [[deviance(m, n)] for m, (_, n, _) in zip(mus, evals)]
    gain = np.zeros(Xl.shape[1])
    best, best_m = np.inf, 0
    for m in range(1, M + 1):
        t = fit_tree(Xl, nl, mu_l, max_depth=depth, min_samples_leaf=min_leaf)
        tr = t.tree_
        for i in range(tr.node_count):          # split-gain importance
            l, r = tr.children_left[i], tr.children_right[i]
            if l != -1:
                gain[tr.feature[i]] += (tr.weighted_n_node_samples[i] * tr.impurity[i]
                                        - tr.weighted_n_node_samples[l] * tr.impurity[l]
                                        - tr.weighted_n_node_samples[r] * tr.impurity[r])
        mu_l = mu_l * t.predict(Xl) ** shrink
        path_l.append(deviance(mu_l, nl))
        for k, (X, n, _) in enumerate(evals):
            mus[k] = mus[k] * t.predict(X) ** shrink
            paths[k].append(deviance(mus[k], n))
        if patience is not None:
            if paths[0][-1] < best - 1e-6:
                best, best_m = paths[0][-1], m
            elif m - best_m >= patience:
                break
    return dict(in_path=path_l, eval_paths=paths, best_m=best_m, gain=gain, eval_mu=mus)


def main(rda, out):
    t0 = time.time()
    res = {}
    d = load(rda)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(d))
    n_learn = round(0.9 * len(d))
    learn, test = d.iloc[perm[:n_learn]].reset_index(drop=True), d.iloc[perm[n_learn:]].reset_index(drop=True)
    nl, el, nt, et = learn.ClaimNb.values, learn.Exposure.values, test.ClaimNb.values, test.Exposure.values
    res["data"] = dict(policies=len(d), learn=len(learn), test=len(test),
                       claims=float(d.ClaimNb.sum()), exposure_years=float(d.Exposure.sum()),
                       portfolio_frequency=float(d.ClaimNb.sum() / d.Exposure.sum()),
                       share_claim_free=float((d.ClaimNb == 0).mean()))
    models, pred = {}, {}

    # 0. homogeneous model
    lam = nl.sum() / el.sum()
    models["Homogeneous"] = dict(in_sample=deviance(lam * el, nl), out_sample=deviance(lam * et, nt))
    pred["Homogeneous"] = lam * et

    # 1. Poisson GLM
    Xg = glm_design(learn)
    glm = PoissonRegressor(alpha=0.0, solver="newton-cholesky", max_iter=300)
    glm.fit(Xg, nl / el, sample_weight=el)
    pred["Poisson GLM"] = glm.predict(glm_design(test, Xg.columns)) * et
    models["Poisson GLM"] = dict(
        in_sample=deviance(glm.predict(Xg) * el, nl),
        out_sample=deviance(pred["Poisson GLM"], nt),
        parameters=int(Xg.shape[1] + 1))
    print("GLM", models["Poisson GLM"], round(time.time() - t0), flush=True)

    enc = FreqEncoder().fit(learn)
    Xl, Xt = enc.transform(learn), enc.transform(test)

    # 2. regression tree (assignment Q1): large tree + 10-fold CV pruning
    big = fit_tree(Xl, nl, el, min_samples_leaf=1500, min_impurity_decrease=0.0)
    alphas_all = big.cost_complexity_pruning_path(Xl, nl / el, sample_weight=el).ccp_alphas
    root_imp = big.tree_.impurity[0]
    alphas_all = alphas_all[alphas_all >= 0.00005 * root_imp]       # rpart cp = 0.00005
    idx = np.unique(np.round(np.linspace(0, len(alphas_all) - 1, 25)).astype(int))
    alphas = alphas_all[idx]
    K = 10
    cv = np.zeros((K, len(alphas)))
    for k, (tr, va) in enumerate(KFold(K, shuffle=True, random_state=SEED).split(Xl)):
        e_k = FreqEncoder().fit(learn.iloc[tr])
        Xtr, Xva = e_k.transform(learn.iloc[tr]), e_k.transform(learn.iloc[va])
        for j, a in enumerate(alphas):
            t = fit_tree(Xtr, nl[tr], el[tr], min_samples_leaf=1500, ccp_alpha=a)
            cv[k, j] = deviance(t.predict(Xva) * el[va], nl[va])
        print("tree CV fold", k, round(time.time() - t0), flush=True)
    cv_mean, cv_sd = cv.mean(0), cv.std(0, ddof=1) / np.sqrt(K)
    jmin = int(cv_mean.argmin())
    j1sd = int(max(j for j in range(len(alphas)) if cv_mean[j] <= cv_mean[jmin] + cv_sd[jmin]))
    for name, j in [("Regression tree (min-CV pruned)", jmin), ("Regression tree (1-SE rule)", j1sd)]:
        t = fit_tree(Xl, nl, el, min_samples_leaf=1500, ccp_alpha=alphas[j])
        pred[name] = t.predict(Xt) * et
        models[name] = dict(in_sample=deviance(t.predict(Xl) * el, nl),
                            out_sample=deviance(t.predict(Xt) * et, nt),
                            leaves=int(t.get_n_leaves()), cv_error=float(cv_mean[j]))
    t = fit_tree(Xl, nl, el, min_samples_leaf=1500, ccp_alpha=alphas[0])
    res["tree_cv"] = dict(
        largest_tree_leaves=int(t.get_n_leaves()),
        cp_relative=[float(a / root_imp) for a in alphas],
        cv_mean=cv_mean.tolist(), cv_se=cv_sd.tolist(), folds=K)
    print("tree", models["Regression tree (min-CV pruned)"], round(time.time() - t0), flush=True)

    # 3. Poisson boosting machine with the assignment settings (Q2)
    b = boosting(Xl, nl, el, [(Xt, nt, et)], M=50, depth=3, min_leaf=10000)
    models["Boosting, assignment setting (M=50, J=3, no shrinkage)"] = dict(
        in_sample=b["in_path"][-1], out_sample=b["eval_paths"][0][-1], trees=50)
    pred["Boosting, assignment setting (M=50, J=3, no shrinkage)"] = b["eval_mu"][0]
    res["boosting_assignment_path"] = dict(note="index = number of trees; index 0 is the start value (rate 1)",
                                           in_sample=b["in_path"], out_sample=b["eval_paths"][0])
    print("PBM", b["in_path"][-1], b["eval_paths"][0][-1], round(time.time() - t0), flush=True)

    # 4. tuned boosting: shrinkage, hyper-parameters chosen on a validation
    #    split of the learning sample (the test sample is never used for tuning)
    r2 = np.random.default_rng(SEED + 1).permutation(len(learn))
    n_tr = round(0.8 * len(learn))
    tr, va = r2[:n_tr], r2[n_tr:]
    e_k = FreqEncoder().fit(learn.iloc[tr])
    Xtr, Xva = e_k.transform(learn.iloc[tr]), e_k.transform(learn.iloc[va])
    grid, best = [], None
    lam_tr = nl[tr].sum() / el[tr].sum()
    for depth, shrink, min_leaf in [(3, 0.3, 10000), (3, 0.3, 2000), (4, 0.3, 2000), (5, 0.3, 2000), (5, 0.1, 2000)]:
        g = boosting(Xtr, nl[tr], el[tr], [(Xva, nl[va], el[va])], M=400 if shrink < 0.2 else 200,
                     depth=depth, min_leaf=min_leaf, shrink=shrink, patience=15, init_rate=lam_tr)
        row = dict(depth=depth, shrinkage=shrink, min_leaf=min_leaf, trees=g["best_m"],
                   validation_loss=g["eval_paths"][0][g["best_m"]])
        grid.append(row)
        print("grid", row, round(time.time() - t0), flush=True)
        if best is None or row["validation_loss"] < best["validation_loss"]:
            best = row
    res["boosting_grid"] = grid
    b2 = boosting(Xl, nl, el, [(Xt, nt, et)], M=best["trees"], depth=best["depth"],
                  min_leaf=best["min_leaf"], shrink=best["shrinkage"], init_rate=lam)
    pred["Boosting, tuned (shrinkage + early stopping)"] = b2["eval_mu"][0]
    models["Boosting, tuned (shrinkage + early stopping)"] = dict(
        in_sample=b2["in_path"][-1], out_sample=b2["eval_paths"][0][-1], **{k: best[k] for k in
                                                                         ["depth", "shrinkage", "min_leaf", "trees"]})
    res["boosting_tuned_path"] = dict(note="index = number of trees; index 0 is the homogeneous model",
                                      in_sample=b2["in_path"], out_sample=b2["eval_paths"][0])
    imp = b2["gain"] / b2["gain"].sum()
    res["boosting_tuned_importance"] = {f: float(v) for f, v in sorted(zip(FEATS, imp), key=lambda x: -x[1])}

    res["models"] = models
    h, g = models["Homogeneous"]["out_sample"], models["Poisson GLM"]["out_sample"]
    res["summary"] = {k: dict(out_sample=v["out_sample"],
                              gain_vs_homogeneous_pct=100 * (h - v["out_sample"]) / h,
                              explained_vs_glm=(h - v["out_sample"]) / (h - g))
                      for k, v in models.items()}
    # paired bootstrap on the test policies: is the gap between two models
    # larger than the sampling noise of a 10% hold-out?
    ud = {k: unit_deviance(v, nt) for k, v in pred.items()}
    rb = np.random.default_rng(SEED + 2)
    ix = rb.integers(0, len(nt), size=(2000, len(nt)))
    tuned = "Boosting, tuned (shrinkage + early stopping)"
    res["bootstrap_difference"] = {}
    for other in [k for k in ud if k != tuned]:
        diff = (ud[tuned] - ud[other])[ix].mean(1)
        res["bootstrap_difference"][f"tuned boosting minus {other}"] = dict(
            mean=float((ud[tuned] - ud[other]).mean()), ci95=[float(np.quantile(diff, 0.025)),
                                                             float(np.quantile(diff, 0.975))],
            share_of_resamples_tuned_better=float((diff < 0).mean()))
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(f"{out}/claim_frequency_results.json", "w"), indent=2)
    print(json.dumps(res["models"], indent=1))
    print(json.dumps(res["boosting_tuned_importance"], indent=1))
    print(json.dumps(res["bootstrap_difference"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
