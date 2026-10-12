"""
gate_experiment.py — defect-injection validation of the cooperation-mode quality gates.

Every replication draws one synthetic exposure-level log from a known generative
process, then asks two kinds of certification procedure whether the log supports
a supervised label:

  naive          certify if 5-fold CV AUC on all available features > tau (0.70)
  practitioner   naive + the checks a careful practitioner would add without an
                 activity baseline or exclusion list: n_pos/n_neg >= 30 and drop
                 any feature with univariate AUC > 0.95
  gates          G1 exclusion -> G2 minimum support -> G3 leakage (metadata +
                 tripwire) -> G4 model must beat an activity-only baseline
                 (bootstrap CI on the AUC difference AND permutation test)

Usage
  python3 gate_experiment.py all            # every experiment, writes results/*.csv + results/summary.json
  python3 gate_experiment.py one clean 50   # quick check of one condition
"""
from __future__ import annotations
import json, os, sys, zlib
from dataclasses import dataclass, field
from multiprocessing import Pool
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

N_MIN, TAU_LEAK, TAU_NAIVE = 30, 0.95, 0.70
N_BOOT, N_PERM = 200, 200
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------------------------------------------------ helpers
def auc(y, s):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cv_scores(X, y, seed):
    """Out-of-fold predicted probabilities from a 5-fold logistic model."""
    p = np.zeros(len(y))
    k = int(min(5, y.sum(), len(y) - y.sum()))
    if k < 2:
        return None
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# ------------------------------------------------------------------ generator
@dataclass
class Log:
    X: np.ndarray                 # behavioural features (+ any injected feature)
    names: list
    post_event: list              # metadata: feature recorded at/after the labelled event
    y: np.ndarray
    member: np.ndarray
    activity: np.ndarray          # exposures per member (the baseline feature)
    excluded: set = field(default_factory=set)


def simulate(cond, rng, n_members=400, signal=1.0, leak_noise=0.35, leak_declared=True):
    """Draw one log. Base processes: 'clean' (features drive the outcome) or
    'null' (outcome independent of features). Defects are injected on top."""
    base_signal = cond in ("clean", "scarcity", "operator_sig", "leak_sig", "confound_sig", "power")
    act = np.maximum(1, rng.lognormal(1.6, 0.7, n_members).astype(int))       # exposures per member
    member = np.repeat(np.arange(n_members), act)
    a = act[member].astype(float)
    za = zs(np.log(a))
    n = len(member)
    dwell = rng.normal(0, 1, n); scroll = rng.normal(0, 1, n)
    visits = rng.normal(0, 1, n); via = rng.binomial(1, 0.3, n).astype(float)
    confounded = cond.startswith("confound")
    if confounded:                                   # features inflated by activity level
        dwell += 0.9 * za; scroll += 0.7 * za; visits += 0.8 * za
    logit = np.full(n, -2.0 if cond != "scarcity" else -6.4)   # scarcity: rare outcome
    if base_signal:
        logit += signal * (0.9 * dwell + 0.6 * scroll + 0.4 * visits + 0.5 * via)
    if confounded:
        logit += 1.4 * za                            # outcome driven by activity
    if cond.startswith("leak_confound"):
        logit += 1.4 * za
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    X = np.column_stack([dwell, scroll, visits, via])
    names = ["dwell", "scroll", "n_visits", "via_share"]
    post = [False] * 4
    excluded = set()

    if cond == "testaccts":                          # 150 test/family accounts, ordinary activity
        n_acc = 150
        per = np.maximum(1, rng.lognormal(1.6, 0.7, n_acc).astype(int))
        om = n_members + np.repeat(np.arange(n_acc), per); k = len(om)
        oX = np.column_stack([rng.normal(1.2, 0.8, k), rng.normal(1.0, 0.8, k),
                              rng.normal(1.0, 0.8, k), rng.binomial(1, 0.7, k)])
        oy = rng.binomial(1, 0.6, k)
        X = np.vstack([X, oX]); y = np.concatenate([y, oy]); member = np.concatenate([member, om])
        excluded = set(range(n_members, n_members + n_acc))
    if cond.startswith("operator"):                  # 3 admin accounts, ~450 exposures
        k = 450
        om = n_members + rng.integers(0, 3, k)
        oX = np.column_stack([rng.normal(1.8, 0.6, k), rng.normal(1.6, 0.6, k),
                              rng.normal(1.5, 0.6, k), rng.binomial(1, 0.8, k)])
        oy = rng.binomial(1, 0.85, k)
        X = np.vstack([X, oX]); y = np.concatenate([y, oy]); member = np.concatenate([member, om])
        excluded = {n_members, n_members + 1, n_members + 2}
    if cond.startswith("leak") or cond == "operator_leak":   # feature recorded after the outcome
        leak = y + rng.normal(0, leak_noise, len(y))
        X = np.column_stack([X, leak]); names = names + ["thanks_recv"]
        post = post + [bool(leak_declared)]
    if cond == "leak_confound":
        pass
    activity = np.bincount(member)[member].astype(float)
    return Log(X, names, post, y, member, activity, excluded)


# ------------------------------------------------------------------ procedures
def naive(L, seed, tau=TAU_NAIVE):
    p = cv_scores(L.X, L.y, seed)
    a = np.nan if p is None else auc(L.y, p)
    return {"certified": bool(a > tau) if not np.isnan(a) else False, "auc": a}


def practitioner(L, seed):
    y = L.y
    if y.sum() < N_MIN or len(y) - y.sum() < N_MIN:
        return {"certified": False, "stop": "n"}
    keep = [j for j in range(L.X.shape[1]) if max(auc(y, L.X[:, j]), 1 - auc(y, L.X[:, j])) <= TAU_LEAK]
    if not keep:
        return {"certified": False, "stop": "leak"}
    p = cv_scores(L.X[:, keep], y, seed)
    return {"certified": bool(auc(y, p) > TAU_NAIVE), "stop": None}


def gates(L, seed, skip=()):
    """Returns certified flag and the first gate that declined."""
    rng = np.random.default_rng(seed + 99991)
    X, y, member, activity = L.X, L.y, L.member, L.activity
    if "G1" not in skip and L.excluded:
        keep = ~np.isin(member, list(L.excluded))
        X, y, member = X[keep], y[keep], member[keep]
        activity = np.bincount(member)[member].astype(float)
    if "G2" not in skip and (y.sum() < N_MIN or len(y) - y.sum() < N_MIN):
        return {"certified": False, "gate": "G2"}
    cols = list(range(X.shape[1]))
    if "G3" not in skip:
        cols = [j for j in cols if not L.post_event[j]]
        cols = [j for j in cols if max(auc(y, X[:, j]), 1 - auc(y, X[:, j])) <= TAU_LEAK]
        if not cols:
            return {"certified": False, "gate": "G3"}
    p = cv_scores(X[:, cols], y, seed)
    if p is None:
        return {"certified": False, "gate": "G2"}
    if "G4" in skip:
        return {"certified": bool(auc(y, p) > TAU_NAIVE), "gate": None if auc(y, p) > TAU_NAIVE else "auc"}
    a_m, a_b = auc(y, p), auc(y, activity)
    n = len(y); diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs[b] = auc(y[i], p[i]) - auc(y[i], activity[i])
    ci_lo = np.nanpercentile(diffs, 2.5)
    null = np.array([auc(rng.permutation(y), p) for _ in range(N_PERM)])
    p_perm = (1 + (null >= a_m).sum()) / (1 + N_PERM)
    ok = bool(ci_lo > 0 and p_perm < 0.05)
    return {"certified": ok, "gate": None if ok else "G4", "auc": a_m, "auc_base": a_b}


# ------------------------------------------------------------------ experiments
def resolve(name):
    """Procedures are passed by name so they can cross process boundaries."""
    if name == "naive":
        return lambda L, s: naive(L, s)
    if name.startswith("naive@"):
        t = float(name.split("@")[1]); return lambda L, s: naive(L, s, t)
    if name == "practitioner":
        return practitioner
    if name == "gates":
        return gates
    if name.startswith("no_"):
        g = name[3:]; return lambda L, s: gates(L, s, skip=(g,))
    raise ValueError(name)


def rep(args):
    cond, r, kw, procs = args
    seed = zlib.crc32(f"{cond}|{r}|{sorted(kw.items())}".encode())
    L = simulate(cond, np.random.default_rng(seed), **kw)
    out = {"cond": cond, "rep": r, **kw, "n_pos": int(L.y.sum())}
    for name in procs:
        res = resolve(name)(L, r)
        out[name] = res["certified"]
        if "gate" in res:
            out[name + "_gate"] = res["gate"]
        if name.startswith("naive"):
            out[name + "_auc"] = res["auc"]
    return out


def run(conds, reps, procs, kw=None, tag="main"):
    kw = kw or {}
    jobs = [(c, r, kw, procs) for c in conds for r in range(reps)]
    with Pool() as pool:
        rows = pool.map(rep, jobs, chunksize=8)
    return rows


def summarise(rows, procs):
    import pandas as pd
    df = pd.DataFrame(rows)
    keys = [k for k in ["cond", "signal", "n_members", "leak_noise", "leak_declared", "tau"] if k in df.columns]
    agg = df.groupby(keys).agg(**{p: (p, "mean") for p in procs}, n=("rep", "size"),
                               n_pos_median=("n_pos", "median")).reset_index()
    return df, agg


PROCS_MAIN = ["naive", "practitioner", "gates"]
ABL = ["gates", "no_G1", "no_G2", "no_G3", "no_G4"]


def main():
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "extra":                              # added after the first full run
        rows = run(["testaccts"], 500, PROCS_MAIN)
        df, agg = summarise(rows, PROCS_MAIN); df.to_csv(f"{OUT}/E1b_testaccts.csv", index=False)
        rows = run(["testaccts"], 200, ABL)
        df, agg2 = summarise(rows, ABL); df.to_csv(f"{OUT}/E3b_ablation_testaccts.csv", index=False)
        print(agg.to_string()); print(agg2.to_string()); return
    if mode == "one":
        rows = run([sys.argv[2]], int(sys.argv[3]), PROCS_MAIN)
        print(summarise(rows, PROCS_MAIN)[1].to_string()); return
    summ = {}
    # E1 main: defects on a null base (no genuine signal) + clean/null/scarcity
    E1 = ["clean", "null", "operator", "leak", "confound", "scarcity"]
    rows = run(E1, 500, PROCS_MAIN)
    df, agg = summarise(rows, ["naive", "practitioner", "gates"])
    gate_fired = df.groupby("cond")["gates_gate"].value_counts(normalize=True).rename("share").reset_index()
    df.to_csv(f"{OUT}/E1_main.csv", index=False)
    summ["E1"] = {"rates": agg.to_dict("records"), "gate_fired": gate_fired.to_dict("records"),
                  "naive_auc": df.groupby("cond")["naive_auc"].agg(["mean", "std"]).round(3).reset_index().to_dict("records")}
    # E2 defects on top of genuine signal: the gates should still certify
    E2 = ["operator_sig", "leak_sig", "confound_sig"]
    rows = run(E2, 300, PROCS_MAIN)
    df, agg = summarise(rows, ["naive", "practitioner", "gates"]); df.to_csv(f"{OUT}/E2_defect_on_signal.csv", index=False)
    summ["E2"] = agg.to_dict("records")
    # E3 ablation: drop one gate at a time
    rows = run(E1, 200, ABL)
    df, agg = summarise(rows, ABL); df.to_csv(f"{OUT}/E3_ablation.csv", index=False)
    summ["E3"] = agg.to_dict("records")
    # E4 undeclared leakage: metadata missing, tripwire only; sweep leak strength
    rows = []
    for s in [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5]:
        rows += run(["leak"], 200, PROCS_MAIN, {"leak_noise": s, "leak_declared": False})
    df, agg = summarise(rows, ["naive", "practitioner", "gates"])
    df.to_csv(f"{OUT}/E4_undeclared_leak.csv", index=False)
    # univariate AUC of the leak at each noise level (analytic check by simulation)
    la = []
    for s in [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5]:
        v = []
        for r in range(50):
            L = simulate("leak", np.random.default_rng(r), leak_noise=s)
            v.append(auc(L.y, L.X[:, -1]))
        la.append({"leak_noise": s, "leak_univariate_auc": round(float(np.mean(v)), 3)})
    summ["E4"] = {"rates": agg.to_dict("records"), "leak_auc": la}
    # E5 combined defects (null base)
    rows = run(["leak_confound", "operator_leak"], 300, PROCS_MAIN)
    df, agg = summarise(rows, ["naive", "practitioner", "gates"]); df.to_csv(f"{OUT}/E5_combined.csv", index=False)
    summ["E5"] = agg.to_dict("records")
    # E6 power: certification of genuine signal vs effect size and population
    rows = []
    for sig in [0.25, 0.5, 0.75, 1.0]:
        for nm in [100, 200, 400, 800]:
            rows += run(["power"], 100, ["gates"], {"signal": sig, "n_members": nm})
    df, agg = summarise(rows, ["gates"]); df.to_csv(f"{OUT}/E6_power.csv", index=False)
    summ["E6"] = agg.to_dict("records")
    # E7 naive threshold sensitivity
    rows = run(E1, 200, ["naive@0.6", "naive@0.8"])
    df, agg = summarise(rows, ["naive@0.6", "naive@0.8"]); df.to_csv(f"{OUT}/E7_tau.csv", index=False)
    summ["E7"] = agg.to_dict("records")
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
