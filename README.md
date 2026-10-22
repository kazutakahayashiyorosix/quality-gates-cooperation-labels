# Quality Gates for Cooperation-Mode Labels

Code and results for the paper **"Quality Gates for Cooperation-Mode Labels"** (Kazutaka Hayashi, YOROSIX, Osaka, Japan).
The preprint is in `paper.pdf`.

The paper proposes four gates that a cooperation-mode label (pass, broker, chain-origin, direct-help) must clear before it is used as training data — exclusion of operator and test accounts, minimum positives and negatives, leakage checks, and a model-versus-activity-baseline test — and evaluates them by injecting known defects into synthetic data and on a public request log whose post-outcome fields are documented by its creators (Random Acts of Pizza).

## Contents
| File | What it does |
|---|---|
| `gate_experiment.py` | Synthetic generator, naive and practitioner baselines, the four gates, every experiment |
| `make_tables.py` | Builds every table and every number in the text from `results/` |
| `pipeline_v2.py` | Mode labels, gates and data card from raw platform exports (member or session unit) |
| `data_card_platform.json` | Data card from the operating platform (aggregate counts only; raw exports are not released) |
| `raop_case.py` | The gates on the Random Acts of Pizza dataset (public data with documented leakage) |
| `raop_ext.py` | Operator contamination (G1) and sample-size limits (G2) on the same public dataset |
| `make_raop_fig.py` | Figure for the public-data test |
| `results/` | Result files used in the paper |

## Reproduce
```bash
pip install -r requirements.txt
python gate_experiment.py all     # Tables 1–6 and 8 (fixed seeds; about 45 minutes on 2 cores)
python gate_experiment.py extra   # adds the test-account condition (Tables 1 and 3)
python make_tables.py             # writes the LaTeX tables and macros

# public-data test: download pizza_request_dataset.tar.gz from
# https://snap.stanford.edu/data/web-RedditPizzaRequests.html and unpack it, then
python raop_case.py pizza_request_dataset/pizza_request_dataset.json   # about 30 seconds
python raop_ext.py  pizza_request_dataset/pizza_request_dataset.json   # about 2 minutes
python make_raop_fig.py
```

The Random Acts of Pizza data (Althoff, Danescu-Niculescu-Mizil and Jurafsky, ICWSM 2014) is not redistributed here; please obtain it from its creators.

Apart from the public-data test, all experimental results come from synthetic data. The platform data card reports aggregate counts only; no personal data is included.

## License
Code: MIT. See `LICENSE`.
