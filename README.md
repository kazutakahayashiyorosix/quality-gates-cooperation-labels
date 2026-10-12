# Quality Gates for Cooperation-Mode Labels

Code and results for the paper **"Quality Gates for Cooperation-Mode Labels"** (Kazutaka Hayashi, YOROSIX, Osaka, Japan).
The preprint is in `paper.pdf`.

The paper proposes four gates that a cooperation-mode label (pass, broker, chain-origin, direct-help) must clear before it is used as training data — exclusion of operator and test accounts, minimum positives and negatives, leakage checks, and a model-versus-activity-baseline test — and evaluates them by injecting known defects into synthetic data.

## Contents
| File | What it does |
|---|---|
| `gate_experiment.py` | Synthetic generator, naive and practitioner baselines, the four gates, every experiment |
| `make_tables.py` | Builds every table and every number in the text from `results/` |
| `pipeline_v2.py` | Mode labels, gates and data card from raw platform exports (member or session unit) |
| `data_card_platform.json` | Data card from the operating platform (aggregate counts only; raw exports are not released) |
| `results/` | Result files used in the paper |

## Reproduce
```bash
pip install -r requirements.txt
python gate_experiment.py all     # Tables 1–6 and 8 (fixed seeds; about 45 minutes on 2 cores)
python gate_experiment.py extra   # adds the test-account condition (Tables 1 and 3)
python make_tables.py             # writes the LaTeX tables and macros
```

All experimental results come from synthetic data. The platform data card reports aggregate counts only; no personal data is included.

## License
Code: MIT. See `LICENSE`.
