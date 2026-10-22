"""
raop_ext.py -- G1 and G2 on the Random Acts of Pizza log.

The leakage test in raop_case.py covers G3/G4 only. This script adds:

(a) Scarcity (G2) on real data. Random subsamples of RAOP at fixed base rate,
    request-time features only, so no leakage is present. The full-data
    request-time model has AUC ~0.65, below the naive threshold of 0.70, so any
    naive certification on a subsample is a chance certification.

(b) Contamination (G1) on a real base. Operator accounts are injected into the
    real log: 3 account IDs, each posting many requests with elevated
    request-time activity and an 85% success rate (the pattern of staff
    posting test requests that they then mark fulfilled). Request-time
    features only. The operator IDs are on the exclusion list.

Usage: python3 raop_ext.py /path/to/pizza_request_dataset.json
"""
import json, os, sys, zlib
import numpy as np
from gate_experiment import Log, naive, practitioner, gates, auc, cv_scores
from raop_case import build

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPS = 50


def request_only(X, names, kind):
    j = [i for i, k in enumerate(kind) if k == "request"]
    return X[:, j], [names[i] for i in j]


def scarcity(Xr, names, y, activity):
    rows = []
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rate = y.mean()
    for n_pos in [10, 20, 30, 50, 100, 200]:
        n_neg = int(round(n_pos * (1 - rate) / rate))
        res = {"naive": [], "practitioner": [], "gates": [], "naive_auc": []}
        for r in range(REPS):
            rng = np.random.default_rng(zlib.crc32(f"scar|{n_pos}|{r}".encode()))
            idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
            L = Log(Xr[idx], names, [False] * len(names), y[idx], np.arange(len(idx)), activity[idx])
            a = naive(L, r); res["naive"].append(a["certified"]); res["naive_auc"].append(a["auc"])
            res["practitioner"].append(practitioner(L, r)["certified"])
            res["gates"].append(gates(L, r)["certified"])
        rows.append({"n_pos": n_pos, "n_neg": n_neg,
                     **{k: float(np.mean(v)) for k, v in res.items()}})
        print("scarcity", rows[-1])
    return rows


def contamination(Xr, names, y, activity, n_op_rows=450, n_acc=3, shift=1.5):
    rows = []
    n = len(y)
    for r in range(REPS):
        rng = np.random.default_rng(zlib.crc32(f"op|{r}".encode()))
        # operator rows: request-time features shifted up (heavy, long-standing accounts)
        Xo = rng.normal(0, 1, (n_op_rows, Xr.shape[1])) + shift
        yo = rng.binomial(1, 0.85, n_op_rows)
        mo = n + rng.integers(0, n_acc, n_op_rows)
        X = np.vstack([Xr, Xo]); yy = np.concatenate([y, yo])
        member = np.concatenate([np.arange(n), mo])
        act = np.concatenate([activity, np.full(n_op_rows, np.percentile(activity, 99))])
        L = Log(X, names, [False] * len(names), yy, member, act)
        a = naive(L, r)
        g_no = gates(L, r)                       # no exclusion list supplied (G1 inactive)
        keep = member < n                        # G1 with the exclusion list; the activity
        Lg = Log(X[keep], names, [False] * len(names), yy[keep], member[keep], act[keep])  # baseline stays Reddit activity
        g = gates(Lg, r)
        rows.append({"naive": a["certified"], "naive_auc": a["auc"],
                     "practitioner": practitioner(L, r)["certified"],
                     "gates": g["certified"], "gates_auc": g.get("auc"), "gates_auc_base": g.get("auc_base"),
                     "gates_noG1": g_no["certified"], "gates_noG1_auc": g_no.get("auc")})
    keys = rows[0].keys()
    summ = {k: float(np.nanmean([np.nan if row[k] is None else float(row[k]) for row in rows])) for k in keys}
    print("contamination", summ)
    return summ


def clean_reference(Xr, names, y, activity):
    L = Log(Xr, names, [False] * len(names), y, np.arange(len(y)), activity)
    out = {"naive_cert": float(np.mean([naive(L, r)["certified"] for r in range(20)])),
           "gates_cert": float(np.mean([gates(L, r)["certified"] for r in range(20)])),
           "auc": float(auc(y, cv_scores(Xr, y, 0))), "activity_auc": float(auc(y, activity))}
    print("clean", out)
    return out


def main(path):
    os.makedirs(OUT, exist_ok=True)
    X, y, names, kind, activity = build(path)
    Xr, rn = request_only(X, names, kind)
    prev = json.load(open(os.path.join(OUT, "raop_ext.json"))) if os.path.exists(os.path.join(OUT, "raop_ext.json")) else {}
    res = {"clean": clean_reference(Xr, rn, y, activity),
           "scarcity": prev.get("scarcity") or scarcity(Xr, rn, y, activity),
           "contamination": contamination(Xr, rn, y, activity)}
    json.dump(res, open(os.path.join(OUT, "raop_ext.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
