"""pipeline_v2.py — cooperation-mode labels from the platform logs, with quality gates.
Unit: (member, request) exposure by a NON-excluded, NON-author member.
Modes (mutually exclusive, priority chain_origin > broker > direct_help > pass):
  direct_help : member posted a non-deleted reply to the request
  broker      : member created a share link for someone else's request AND a different visitor arrived via it
  chain_origin: broker whose share has >=1 child share (parent_share_id) by another member
  pass        : member visited the request and did none of the above
Gates: G1 exclusion (operator/family/test), G2 per-mode n_pos and n_neg >= MIN_POS, G3 leakage
(drop features declared as recorded at/after the labelled event, and any feature whose univariate AUC
exceeds TAU_LEAK), G4 model must beat an activity-only baseline: bootstrap 95% CI of the AUC difference
excludes zero AND permutation p < 0.05 (only if G2 passes).
"""
import json, sys, argparse, hashlib, datetime as dt
import numpy as np, pandas as pd

MIN_POS = 30
TAU_LEAK = 0.95
N_BOOT, N_PERM = 1000, 1000
FEATURES = ["dwell", "scroll", "n_visits"]
POST_EVENT = set()             # features recorded at/after the labelled event (none of the three above)
OPERATOR_IDS = set()           # set to the operator account id(s) of your platform
EXCLUDE_EMAIL_PATTERNS = ["@<platform-domain>"]   # set to the operator e-mail domain
FAMILY_TEST_IDS = set()        # fill in: family / test accounts
MODES = ["pass", "broker", "chain_origin", "direct_help"]
COLMAP = {  # canonical -> candidate names in exports
  "visit_member": ["visitor_user_id", "member_id"], "visit_created": ["created_at", "created"],
  "share_owner": ["owner_user_id", "member_id"], "share_parent": ["parent_share_id"],
  "post_author": ["member_id"], "post_id": ["id"], "reply_member": ["member_id"],
}
def col(df, key):
    for c in COLMAP[key]:
        if c in df.columns: return c
    return None

def load(a):
    r = lambda p: pd.read_csv(p) if p else None
    return dict(members=r(a.members), posts=r(a.posts), visits=r(a.visits), shares=r(a.shares),
                replies=r(a.replies), meta=r(a.meta))

def excluded_ids(d, extra):
    ids = set(OPERATOR_IDS) | set(FAMILY_TEST_IDS) | set(extra)
    m = d["members"]
    for pat in EXCLUDE_EMAIL_PATTERNS:
        ids |= set(m.loc[m.email.fillna("").str.contains(pat, regex=False), "id"])
    return ids

def build(d, excl, notes):
    posts, v, s, rp, m = d["posts"], d["visits"], d["shares"], d["replies"], d["members"]
    author = posts.set_index(col(posts, "post_id"))[col(posts, "post_author")].to_dict()
    vm = col(v, "visit_member")
    ex = v[v[vm].notna()].copy(); ex["member"] = ex[vm].astype(int)
    ex = ex[~ex.member.isin(excl)]
    ex = ex[ex.apply(lambda r: author.get(r.request_id) != r.member, axis=1)]
    feats = ex.groupby(["member", "request_id"]).agg(
        n_visits=("id", "size"), dwell=("dwell_time_seconds", "mean"),
        scroll=("scrolled_percentage", "mean")).reset_index()
    # reply -> member (member_id column, else email join)
    rp = rp[rp.is_deleted.fillna(0) == 0].copy()
    if rp["member_id"].isna().all():
        e2id = m.dropna(subset=["email"]).drop_duplicates("email").set_index("email").id.to_dict()
        rp["member_id"] = rp.email.map(e2id); notes.append("replies.member_id empty -> joined by email")
    helped = set(zip(rp.member_id.dropna().astype(int), rp.request_id))
    so = col(s, "share_owner"); sp = col(s, "share_parent")
    if sp is None: notes.append("share_links has NO parent_share_id column -> chain_origin not identifiable")
    ss = s[s[so].notna()].copy(); ss["owner"] = ss[so].astype(int)
    used = v[v.share_id.notna()].groupby("share_id")[vm].apply(lambda x: set(x.dropna().astype(int)))
    if v.share_id.notna().sum() == 0: notes.append("request_visits.share_id is all null -> share->visit attribution impossible")
    broker, chain = set(), set()
    for _, r in ss.iterrows():
        if r.owner in excl or author.get(r.request_id) == r.owner: continue
        if len(used.get(r.id, set()) - {r.owner}) > 0: broker.add((r.owner, r.request_id))
        if sp and ((s[sp] == r.id) & (s[so] != r.owner)).any(): chain.add((r.owner, r.request_id))
    def lab(k):
        return "chain_origin" if k in chain else "broker" if k in broker else "direct_help" if k in helped else "pass"
    # include helpers/brokers who never logged a visit
    keys = set(zip(feats.member, feats.request_id)) | {k for k in helped | broker | chain
            if k[0] not in excl and author.get(k[1]) != k[0]}
    df = pd.DataFrame(sorted(keys), columns=["member", "request_id"]).merge(feats, how="left")
    df["mode"] = [lab(k) for k in zip(df.member, df.request_id)]
    df["activity"] = df.groupby("member").request_id.transform("size")  # baseline feature
    return df


BOT_RE = "bot|Bot|crawl|spider|facebookexternalhit|Twitterbot|Slack|Discord|Hatena|preview|Googlebot|Headless|python|curl"
def build_session(d, excl, notes):
    """Unit = anonymous human session x request. Most third-party traffic is logged-out, so this is the realistic unit.
    Exclusions: bots (UA), any IP/session ever seen logged in as a member (operator logged-out leakage),
    request_ids not in posts (test fixtures such as 99999)."""
    v, posts = d["visits"], d["posts"]
    bot = v.user_agent.str.contains(BOT_RE, na=False)
    mem = v[v.visitor_user_id.notna()]
    a = v[~bot & v.visitor_user_id.isna() & ~v.ip_hash.isin(set(mem.ip_hash.dropna()))
          & ~v.session_token.isin(set(mem.session_token.dropna())) & v.request_id.isin(set(posts.id))]
    notes.append(f"session unit: {int(bot.sum())} bot visits, {int(v.request_id.isin(set(posts.id)).__invert__().sum())} test-fixture visits removed")
    notes.append("session unit: broker = share button CLICK (not verified share); chain_origin not attributable (share_links has no session/ip)")
    g = a.groupby(["session_token", "request_id"]).agg(
        n_visits=("id", "size"), dwell=("dwell_time_seconds", "mean"), scroll=("scrolled_percentage", "mean"),
        rep=("action_taken", lambda x: (x == "reply").any()), sh=("clicked_share_button", "max"),
        ip=("ip_hash", "first")).reset_index()
    g["mode"] = np.where(g.rep, "direct_help", np.where(g.sh.fillna(0) > 0, "broker", "pass"))
    g["activity"] = g.groupby("ip").request_id.transform("size")
    return g.rename(columns={"session_token": "member"}).drop(columns=["rep", "sh"])

def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def evaluate(df, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for mode in MODES:
        y = (df["mode"] == mode).astype(int).values; npos = int(y.sum())
        res = {"n_pos": npos, "n_neg": int(len(y) - npos)}
        if npos < MIN_POS or len(y) - npos < MIN_POS:                       # G2
            res["verdict"] = f"DATA_INSUFFICIENT (need >= {MIN_POS} pos and neg)"
            out[mode] = res; continue
        X = df[FEATURES].fillna(df[FEATURES].median())
        keep = [f for f in FEATURES if f not in POST_EVENT]                  # G3a metadata
        tripped = [f for f in keep if max(auc(y, X[f].values), 1 - auc(y, X[f].values)) > TAU_LEAK]
        keep = [f for f in keep if f not in tripped]                         # G3b tripwire
        res["G3_dropped"] = sorted(POST_EVENT & set(FEATURES)) + tripped
        if not keep:
            res["verdict"] = "FAIL (G3: every feature dropped as leakage)"; out[mode] = res; continue
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        p = cross_val_predict(LogisticRegression(max_iter=1000), X[keep], y, cv=5, method="predict_proba")[:, 1]
        act = df.activity.values
        a, b = auc(y, p), auc(y, act)                                         # G4
        n = len(y); diffs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            if 0 < y[i].sum() < n: diffs.append(auc(y[i], p[i]) - auc(y[i], act[i]))
        ci_lo = float(np.percentile(diffs, 2.5))
        null = np.array([auc(rng.permutation(y), p) for _ in range(N_PERM)])
        perm_p = float((1 + (null >= a).sum()) / (1 + N_PERM))
        ok = ci_lo > 0 and perm_p < 0.05
        res.update(auc=a, auc_activity_baseline=b, auc_diff_ci_low=ci_lo, perm_p=perm_p,
                   verdict="PASS" if ok else "FAIL (no gain over activity / not significant)")
        out[mode] = res
    return out

def main():
    ap = argparse.ArgumentParser()
    for k in ["members", "posts", "visits", "shares", "replies", "meta"]: ap.add_argument(f"--{k}")
    ap.add_argument("--exclude", default=""); ap.add_argument("--unit", default="member", choices=["member","session"]); ap.add_argument("--out", default="out")
    a = ap.parse_args(); d = load(a); notes = []
    excl = excluded_ids(d, [int(x) for x in a.exclude.split(",") if x])
    df = (build_session if a.unit == "session" else build)(d, excl, notes); ev = evaluate(df)
    raw = {k: int(len(v)) for k, v in d.items() if v is not None}
    card = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
            "source_files": {k: getattr(a, k) for k in raw}, "raw_row_counts": raw,
            "source_sha256": {k: hashlib.sha256(open(getattr(a, k), "rb").read()).hexdigest()[:16] for k in raw},
            "excluded_member_ids": sorted(int(x) for x in excl),
            "unit": a.unit,
            "n_units": int(len(df)), "mode_counts": df["mode"].value_counts().to_dict(),
            "gates": {"G1_exclusion": "applied", "G2_min_positives": MIN_POS,
                      "G3_leakage": {"features": FEATURES, "declared_post_event": sorted(POST_EVENT), "tripwire_auc": TAU_LEAK,
                                     "note": "label-defining columns (replies, shares) are never features"},
                      "G4": ev}, "structural_notes": notes,
            "usable_as_supervised_data": all(r["verdict"] == "PASS" for r in ev.values())}
    import os; os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/labels.csv", index=False)
    json.dump(card, open(f"{a.out}/data_card.json", "w"), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(card, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
