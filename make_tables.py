"""Builds every results table and number in the paper from results/*.csv, so the text cannot drift from the runs."""
import os, json
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "tables")
os.makedirs(T, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(R, f), keep_default_na=False, na_values=[""])
pct = lambda x: f"{100 * x:.1f}\\%"
NAMES = {"clean": "clean", "null": "null", "operator": "operator", "leak": "leakage", "confound": "confound",
         "scarcity": "scarcity", "testaccts": "test accounts", "operator_sig": "operator + signal", "leak_sig": "leakage + signal",
         "confound_sig": "confound + signal", "leak_confound": "leakage + confound", "operator_leak": "operator + leakage"}
ORDER = ["clean", "null", "operator", "testaccts", "leak", "confound", "scarcity"]
macros = {}

# ---- Table 1: main
d = pd.concat([rd("E1_main.csv"), rd("E1b_testaccts.csv")])
g = d.groupby("cond")
truth = {"clean": "yes", "null": "no", "operator": "no", "testaccts": "no", "leak": "no", "confound": "no", "scarcity": "yes (rare)"}
rows = []
for c in ORDER:
    x = g.get_group(c)
    fired = x["gates_gate"].replace("", np.nan).dropna().value_counts(normalize=True)
    fired_s = ", ".join(f"{k} {100*v:.0f}\\%" for k, v in fired.items()) or "---"
    rows.append(f"{NAMES[c]} & {truth[c]} & {int(x.n_pos.median())} & {x.naive_auc.mean():.3f} ({x.naive_auc.std():.3f}) & "
                f"{pct(x.naive.mean())} & {pct(x.practitioner.mean())} & \\textbf{{{pct(x.gates.mean())}}} & {fired_s} \\\\")
    macros[f"gates{c}"] = pct(x.gates.mean()); macros[f"naive{c}"] = pct(x.naive.mean())
    macros[f"prac{c}"] = pct(x.practitioner.mean()); macros[f"auc{c}"] = f"{x.naive_auc.mean():.3f}"
    macros[f"aucsd{c}"] = f"{x.naive_auc.std():.3f}"; macros[f"npos{c}"] = str(int(x.n_pos.median()))
open(f"{T}/t1_main.tex", "w").write("\n".join(rows) + "\n")
macros["nrepmain"] = str(int(g.size().iloc[0]))

# ---- Table 2: defects on top of genuine signal
d = rd("E2_defect_on_signal.csv"); rows = []
for c in ["operator_sig", "leak_sig", "confound_sig"]:
    x = d[d.cond == c]
    rows.append(f"{NAMES[c]} & {pct(x.naive.mean())} & {pct(x.practitioner.mean())} & \\textbf{{{pct(x.gates.mean())}}} \\\\")
    macros[f"gates{c.replace('_','')}"] = pct(x.gates.mean())
open(f"{T}/t2_signal.tex", "w").write("\n".join(rows) + "\n")
macros["nrepsig"] = str(int(d.groupby("cond").size().iloc[0]))

# ---- Table 3: ablation
d = pd.concat([rd("E3_ablation.csv"), rd("E3b_ablation_testaccts.csv")]); rows = []
for c in ORDER:
    x = d[d.cond == c]
    rows.append(f"{NAMES[c]} & " + " & ".join(pct(x[k].mean()) for k in ["gates", "no_G1", "no_G2", "no_G3", "no_G4"]) + " \\\\")
    for k in ["no_G1", "no_G2", "no_G3", "no_G4"]:
        macros[f"abl{k.replace('_','')}{c}"] = pct(x[k].mean())
open(f"{T}/t3_ablation.tex", "w").write("\n".join(rows) + "\n")
macros["nrepabl"] = str(int(d.groupby("cond").size().iloc[0]))

# ---- Table 4: undeclared leakage sweep
d = rd("E4_undeclared_leak.csv")
la = json.load(open(os.path.join(R, "summary.json")))["E4"]["leak_auc"]
la = {round(x["leak_noise"], 2): x["leak_univariate_auc"] for x in la}
rows = []
for s, x in d.groupby("leak_noise"):
    rows.append(f"{s:.2f} & {la[round(s,2)]:.3f} & {pct(x.naive.mean())} & {pct(x.practitioner.mean())} & \\textbf{{{pct(x.gates.mean())}}} \\\\")
open(f"{T}/t4_undeclared.tex", "w").write("\n".join(rows) + "\n")
macros["nrepleak"] = str(int(d.groupby("leak_noise").size().iloc[0]))

# ---- Table 5: combined defects
d = rd("E5_combined.csv"); rows = []
for c in ["leak_confound", "operator_leak"]:
    x = d[d.cond == c]
    rows.append(f"{NAMES[c]} & {pct(x.naive.mean())} & {pct(x.practitioner.mean())} & \\textbf{{{pct(x.gates.mean())}}} \\\\")
    macros[f"gates{c.replace('_','')}"] = pct(x.gates.mean())
open(f"{T}/t5_combined.tex", "w").write("\n".join(rows) + "\n")
macros["nrepcomb"] = str(int(d.groupby("cond").size().iloc[0]))

# ---- Table 6: power
d = rd("E6_power.csv")
p = d.groupby(["signal", "n_members"]).agg(rate=("gates", "mean"), npos=("n_pos", "median")).reset_index()
ns = sorted(p.n_members.unique()); rows = []
for sig, x in p.groupby("signal"):
    cells = [f"{pct(x[x.n_members == n].rate.iloc[0])} ({int(x[x.n_members == n].npos.iloc[0])})" for n in ns]
    rows.append(f"{sig:.2f} & " + " & ".join(cells) + " \\\\")
open(f"{T}/t6_power.tex", "w").write("\n".join(rows) + "\n")
open(f"{T}/t6_head.tex", "w").write(" & ".join(f"{n} members" for n in ns))
macros["nreppow"] = str(int(d.groupby(["signal", "n_members"]).size().iloc[0]))

# ---- Table 7: naive threshold
d = rd("E7_tau.csv"); m = rd("E1_main.csv"); rows = []
for c in [c for c in ORDER if c != "testaccts"]:
    x = d[d.cond == c]; y = m[m.cond == c]
    rows.append(f"{NAMES[c]} & {pct(x['naive@0.6'].mean())} & {pct(y.naive.mean())} & {pct(x['naive@0.8'].mean())} & {pct(y.gates.mean())} \\\\")
open(f"{T}/t7_tau.tex", "w").write("\n".join(rows) + "\n")
macros["nreptau"] = str(int(d.groupby("cond").size().iloc[0]))

with open(f"{T}/macros.tex", "w") as f:
    for k, v in macros.items():
        w = k
        for dgt, word in zip("1234567890", ["one","two","three","four","five","six","seven","eight","nine","zero"]):
            w = w.replace(dgt, word)
        f.write(f"\\newcommand{{\\R{''.join(ch for ch in w if ch.isalpha())}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and 7 tables")
