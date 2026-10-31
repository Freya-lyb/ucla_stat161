# Polymarket wallet ranking

## Required files

Keep these files together in the repository root:
- `wallet_features.csv`
- `pu_bagging_leave_one_insider_out.py`
- `Amount_Baseline_and_Model_Comparison.ipynb`
- `Wallet_Descriptive_Analysis_and_PCA.ipynb`
- `requirements.txt`

## Reproduce

Download the complete repository with **Code > Download ZIP**, then extract it.
The tested Python version is **3.12.14**. Use Python 3.12 with the pinned dependencies.
From the extracted repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook, select the installed Python environment, restart its kernel,
and run all cells in order. No database credentials, web downloads from inside the
notebook, or precomputed result folders are required.

- `Wallet_Descriptive_Analysis_and_PCA.ipynb`: descriptive plots and nine-feature PCA.
- `Amount_Baseline_and_Model_Comparison.ipynb`: amount ranking, 19 evaluation folds,
  and a fresh fit of both PU variants by importing the team model `.py`.
- `pu_bagging_leave_one_insider_out.py`: original team model implementation.
  It can also run separately with `python pu_bagging_leave_one_insider_out.py --data wallet_features.csv`.

The comparison notebook does not use old result caches. It generates tables and figures
under `outputs/model_comparison/`. The descriptive notebook writes its exported plots
and tables to the repository root. These generated outputs do not need to be uploaded.

## Expected comparison

Split seed = 0, PU seed = 0. Positive count = 19. No bot filter. Each fold tests one
positive against 11,186 held-out unlabeled wallets. Percentiles count ties against the positive.

| Method | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| Amount ranking | 5/19 | 12/19 | 16/19 |
| Bagging PU, tree | 9/19 | 14/19 | 16/19 |
| Bagging PU, logistic | 10/19 | 17/19 | 18/19 |

Nine-feature PCA: PC1 = 45.20%, PC2 = 16.99%, total = 62.19%.
Small numerical differences may occur across platforms or dependency versions.

Unlabeled wallets are not verified negatives. Reported recall concerns the supplied labels,
not the detection rate of all real insider trading. Related wallets and all-row
standardization limit generalization claims.
