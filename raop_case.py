"""
raop_case.py — the quality gates applied to a public help-request log with
documented post-outcome fields: the Random Acts of Pizza dataset
(Althoff, Danescu-Niculescu-Mizil & Jurafsky, ICWSM 2014).

Label: requester_received_pizza. Every row is one requester's first request,
so the "member" of the gate procedure is the requester and the activity
baseline is the requester's Reddit activity at request time.

Ground truth for leakage comes from the dataset documentation, not from us:
fields suffixed "_at_retrieval", the user flair, and the giver name are
recorded after the outcome. Two further fields carry post-outcome information
without being documented as such: post_was_edited and the length difference
between request_text and request_text_edit_aware (successful requesters often
edit their post to thank the giver).

Usage: python3 raop_case.py /path/to/pizza_request_dataset.json
Writes results/raop_*.csv|json.
"""
import json, os, sys
import numpy as np
from gate_experiment import Log, naive, practitioner, gates, auc, cv_scores, N_MIN, TAU_LEAK, TAU_NAIVE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEEDS = range(20)

AT_REQUEST = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]
AT_RETRIEVAL = [
    "number_of_downvotes_of_request_at_retrieval",
    "number_of_upvotes_of_request_at_retrieval",
    "request_number_of_comments_at_retrieval",
    "requester_account_age_in_days_at_retrieval",
    "requester_days_since_first_post_on_raop_at_retrieval",
    "requester_number_of_comments_at_retrieval",
    "requester_number_of_comments_in_raop_at_retrieval",
    "requester_number_of_posts_at_retrieval",
    "requester_number_of_posts_on_raop_at_retrieval",
    "requester_upvotes_minus_downvotes_at_retrieval",
    "requester_upvotes_plus_downvotes_at_retrieval",
]


def slog(v):
    v = np.asarray(v, float)
    return np.sign(v) * np.log1p(np.abs(v))


def build(path):
    d = json.load(open(path))
    y = np.array([int(r["requester_received_pizza"]) for r in d])
    cols, names, kind = [], [], []   # kind: request | documented_post | undocumented_post
    for f in AT_REQUEST:
        cols.append(slog([r[f] for r in d])); names.append(f); kind.append("request")
    for f in AT_RETRIEVAL:
        cols.append(slog([r[f] for r in d])); names.append(f); kind.append("documented_post")
    cols.append(np.array([float(r["requester_user_flair"] is not None) for r in d]))
    names.append("has_user_flair"); kind.append("documented_post")
    cols.append(np.array([float(r["giver_username_if_known"] not in (None, "N/A")) for r in d]))
    names.append("giver_known"); kind.append("documented_post")
    cols.append(np.array([float(bool(r["post_was_edited"])) for r in d]))
    names.append("post_was_edited"); kind.append("undocumented_post")
    cols.append(slog([len(r["request_text"]) - len(r["request_text_edit_aware"]) for r in d]))
    names.append("edit_length"); kind.append("undocumented_post")
    cols.append(slog([len(r["request_text_edit_aware"]) for r in d]))
    names.append("text_length"); kind.append("request")
    cols.append(slog([len(r["request_title"]) for r in d]))
    names.append("title_length"); kind.append("request")
    X = np.column_stack(cols)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    activity = np.array([r["requester_number_of_comments_at_request"] + r["requester_number_of_posts_at_request"]
                         for r in d], float)
    return X, y, names, kind, activity


def make_log(X, y, names, kind, activity, declared):
    post = [k in declared for k in kind]
    L = Log(X, list(names), post, y, np.arange(len(y)), activity)
    return L


def features_used(L, procedure):
    """Which features each procedure lets through to the final model."""
    y = L.y
    allj = list(range(L.X.shape[1]))
    trip = [j for j in allj if max(auc(y, L.X[:, j]), 1 - auc(y, L.X[:, j])) <= TAU_LEAK]
    if procedure == "naive":
        return allj
    if procedure == "practitioner":
        return trip
    return [j for j in trip if not L.post_event[j]]


def main(path):
    os.makedirs(OUT, exist_ok=True)
    X, y, names, kind, activity = build(path)
    summary = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
               "activity_auc": float(auc(y, activity))}
    # univariate AUCs (direction-free)
    uni = []
    for j, n in enumerate(names):
        a = auc(y, X[:, j]); uni.append({"feature": n, "kind": kind[j], "auc": float(max(a, 1 - a))})
    json.dump(uni, open(os.path.join(OUT, "raop_univariate.json"), "w"), indent=1)

    scenarios = {
        "documented": {"documented_post"},                 # metadata follows the documentation
        "undeclared": set(),                               # no feature metadata at all
    }
    rows = []
    for scen, declared in scenarios.items():
        L = make_log(X, y, names, kind, activity, declared)
        for proc in ["naive", "practitioner", "gates"]:
            res = []
            for s in SEEDS:
                if proc == "naive":
                    r = naive(L, s)
                elif proc == "practitioner":
                    r = practitioner(L, s)
                else:
                    r = gates(L, s)
                res.append(r)
            used = features_used(L, proc)
            p = cv_scores(L.X[:, used], y, 0)
            leaks_in = [names[j] for j in used if kind[j] != "request"]
            rows.append({
                "scenario": scen, "procedure": proc,
                "certified_rate": float(np.mean([r["certified"] for r in res])),
                "model_auc": float(auc(y, p)),
                "n_features": len(used),
                "post_outcome_features_used": leaks_in,
                "declined_at": sorted({str(r.get("gate") or r.get("stop")) for r in res if not r["certified"]}),
                "gate_auc": float(np.mean([r["auc"] for r in res if "auc" in r and r["auc"] == r["auc"]])) if any("auc" in r for r in res) else None,
                "gate_auc_base": float(np.mean([r["auc_base"] for r in res if "auc_base" in r])) if any("auc_base" in r for r in res) else None,
            })
    # request-time-only reference model
    req = [j for j in range(len(names)) if kind[j] == "request"]
    summary["request_only_auc"] = float(auc(y, cv_scores(X[:, req], y, 0)))
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(OUT, "raop_results.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["scenario"], r["procedure"], "cert", r["certified_rate"], "auc", round(r["model_auc"], 3),
              "nfeat", r["n_features"], "leaks", r["post_outcome_features_used"], "declined", r["declined_at"],
              "gateAUC", r["gate_auc"], "base", r["gate_auc_base"])
    for u in sorted(uni, key=lambda u: -u["auc"])[:10]:
        print(u)


if __name__ == "__main__":
    main(sys.argv[1])
