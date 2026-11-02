# Time-series forecasting: reproducible comparison suite

This suite builds on the author's original notebooks. The **first three experiment notebooks** share the same model implementations, target representation, train/validation/test protocol, and training budgets. A **fourth, Tourism Monthly**, is a deliberately separate *global panel forecast* experiment; use its per-series MASE rather than comparing it directly with the others' RMSE.

| Notebook | Dataset | Task | Source |
|---|---|---|---|
| `01_CO2_final.ipynb` | Mauna Loa atmospheric CO₂ | 156 weekly measurements → 13 weeks | Self-contained `statsmodels` sample embedded in notebook |
| `02_ETTh1_final.ipynb` | ETTh1 transformer temperature | 168 past hours → 24 hours of oil temperature | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| `03_Jena_Weather_NEW.ipynb` | Jena Climate (NEW) | 168 past hours → 24 hours of temperature | [Keras weather example](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/md/timeseries_weather_forecasting.md) |
| `05_Tourism_Monthly_GLOBAL_NEW.ipynb` | Tourism Monthly (NEW; 366 series) | Global multi-series 18-month lookback → 24-month forecast | [Monash Tourism Monthly / Zenodo](https://zenodo.org/records/4656096) |
| `04_Compare_executed_runs.ipynb` | Three single-series runs + Tourism when available | Separate within-dataset RMSE/MASE visualizations | Reads generated CSVs; requires first three, Tourism optional |

The `reference_originals` directory retains the earlier **electricity** and **Beijing PM2.5** notebooks, which were made under other experiment settings. They are **not** pooled as controlled cross-dataset results. You may still link them separately from the Medium post as original evidence.

## Run

1. Install Python 3.10+ and packages: `pip install numpy pandas matplotlib torch catboost nbformat jupyter`
2. Open each experiment notebook (including Tourism Monthly) and choose **Kernel → Restart & Run All** (or equivalent). Set the runtime to a GPU where available; CatBoost uses CPU in this suite.
3. ETTh1, Jena and Tourism Monthly download their public datasets on the first run. If your environment blocks downloads, manually place `ETTh1.csv`, `jena_climate_2009_2016.csv.zip`, or `tourism_monthly_dataset.zip` inside `benchmark_data/`. Tourism source: https://zenodo.org/records/4656096 .
4. Execute `04_Compare_executed_runs.ipynb` afterward. It refuses to invent single-series results when any required CSV is absent; it shows Tourism only if its CSV is available.
5. Upload executed notebooks **and** `forecast_challenge_outputs/` (especially each `comparison.csv` + `run_info.json`) for article updates.

## Comparison details

- Target windows do not cross chronological split boundaries; training history always ends before each forecast origin. Validation/test histories *may* include earlier observations from a previous partition, as they would in production.
- Target is last-observation-relative for neural nets and the primary CatBoost models, with a training-only scale. Inputs include the same past target/sensor sequences and only known-in-advance future calendar covariates. No future sensor measurements are used.
- CatBoost **raw-lag** and **engineered** versions are shown as separate, explicit feature-ablation variants. Neural models are given the same raw history and future calendar; therefore a win by engineered CatBoost demonstrates a **pipeline** advantage, not an architecture-only advantage. Optional `NEURAL_ENGINEERED_FEATURES=True` tests the augmented neural ablation.
- Models: Linear, MLP, 1D CNN, GRU, LSTM, TCN, N-BEATS-style, Patch Transformer; optional full Transformer. CatBoost is configured for **300 maximum trees**, with validation early stopping and 4 CPU threads. 30 max neural epochs with validation early stopping. This is **not a matched compute-time or hyperparameter-optimization budget**.
- Large datasets use uniformly spaced forecast origins (max 8000 train / 1600 validation / 1600 test); these are identical for every model, chosen before outcomes are examined.
- Predictions overlap in time at adjacent origins. RMSE uncertainty cannot be estimated using naive independent errors. For stronger inference, use rolling-origin backtests or block bootstrapping, different seeds and sensitivity to budget.
- 90-tree and 300-tree performance should not be conflated. The article's previous CO₂ conclusion should be re-evaluated after *fresh* 300-tree outputs are saved.
- **Do not choose a winning dataset from the test results** and call that generalization. Jena was selected in advance because its measurements and patterns differ from CO₂/ETT, not because of any observed neural victory.

## Tourism Monthly: a better opportunity for global sequential models

The fourth experiment was added in response to a published Hugging Face article on the 366-series Monash Tourism Monthly data. That blog reports a Transformer MASE of 1.256 and a CatBoost benchmark MASE of 1.699 ([source](https://huggingface.co/blog/time-series-transformers)). **Those are not outcomes produced by these notebooks and not directly reproducible using our different pipeline.** The Tourism notebook uses 18 months of history to forecast a separate final 24-month test block for each series; the preceding 24 months are used for validation. Every model sees the same normalized past values and known calendar phases. Models include a compact patch-based Transformer, which is NOT a faithful PatchTST reproduction. Scores are macro-averaged per-series MASE, and no training targets overlap validation/test blocks. CatBoost has both raw lags and an engineered causal feature ablation.

## Interpret results

`04_Compare_executed_runs.ipynb` reports percentage RMSE improvement **relative to the MLP within each dataset**, and a separate CatBoost comparison. Do **not** compare RMSE absolute values across ppm and degrees Celsius, and do not treat differences across unrelated task horizons as apples-to-apples.

## Data attribution

- CO₂ sample: `statsmodels` Mauna Loa atmospheric CO₂ weekly observations, credited to C. D. Keeling / T. P. Whorf.
- ETTh1: Haoyi Zhou et al. `ETDataset` and accompanying paper (AAAI 2021).
- Jena Climate: Max Planck Institute for Biogeochemistry (Jena), sourced using the public archive linked in the Keras time-series weather forecasting example. All Jena model inputs are *past* observed weather; values at forecast times are never supplied.

- Tourism Monthly: Godahewa et al. [Zenodo/Monash](https://zenodo.org/records/4656096), 366 monthly tourism series, CC BY 4.0.
