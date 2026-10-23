# Replication package: How reliable is a single Google Trends pull?

Repeated daily Google Trends pulls for 59 tourism queries in 15 source markets of Turkey
(21 August to 15 October 2026), with a parallel series without the Travel category filter.
Accompanies the manuscript by Umut Güleryüz (Bitlis Eren University).
Licences: code under MIT (`LICENSE`), data under CC BY 4.0 (`LICENSE-DATA.md`). Cite via `CITATION.cff`. [Zenodo DOI to be added after the first release.]

## Contents

| Path | Description |
|---|---|
| `data/raw/<YYYY-MM-DD>/<GEO>.csv` | One pull per day, Travel category (category code 67). Monthly series, 2016-01 to 2026-07 (127 rows); one column per query, values 0 to 100. |
| `data/raw_nocat/<YYYY-MM-DD>/<GEO>.csv` | Same queries without category filter (category 0), from 2026-08-27. |
| `data/collection_log.csv`, `data/collection_log_nocat.csv` | One row per market and pull attempt (status OK or FAIL, number of rows). FAIL rows have no CSV. |
| `collection_gaps.txt` | Days on which no pull was taken (computer off or asleep). Not recoverable retrospectively. |
| `collect_gt.py`, `collect_gt_nocat.py` | Collection scripts (pytrends). The query lists are in `collect_gt.py` (`MARKETS`). |
| `analyze_stage1_v2.py` | Variability, ICC(2,1), pairwise correlations, peak-month agreement. Writes `data/processed/s1_*`. |
| `analyze_stage1b_category_effect.py` | Category filter versus no filter (failure rates, ICC differences). |
| `analyze_stage1c_icc_ci.py` | 95% confidence intervals for ICC (analytical for keywords, bootstrap for market means). |
| `analyze_stage1d_quantization.py` | Distinct values per pull and rounding-versus-sampling diagnostics. |
| `analyze_stage1e_averaging.py` | Split-half reliability of k-pull means (does averaging help). |
| `analyze_stage1f_robustness.py` | Robustness checks (drift, transformations, block bootstrap, matched-date category comparison, rule validation). |
| `analyze_stage1g_validity.py` | External validity: co-movement of keyword series with official monthly arrivals. |
| `validity_data/ktb_annual/` | Annual border statistics workbooks (2016-2025), Ministry of Culture and Tourism of Türkiye, General Directorate of Investment and Enterprises (https://yigm.ktb.gov.tr/TR-249709/yillik-bultenler.html). Sheet `T7-Mil-Ay Göre G.Yabancı` is used. |
| `data/external/ktb_monthly_arrivals_2016_2025.csv` | Monthly foreign visitor arrivals by nationality for the 15 markets, extracted from the workbooks. |
| `analyze_stage1h_diagnostics.py` | Screening-rule diagnostics, period sensitivity, signal and noise components, and the three figures in English and Turkish versions (`figures/`; Times New Roman 10 pt). |
| `data/processed/` | Outputs produced by the analysis scripts. |

## Reproducing the analysis

Python 3.14 with the packages in `requirements.txt`. From this folder, in order:

    python analyze_stage1_v2.py
    python analyze_stage1b_category_effect.py
    python analyze_stage1c_icc_ci.py
    python analyze_stage1d_quantization.py
    python analyze_stage1e_averaging.py
    python analyze_stage1f_robustness.py
    python analyze_stage1g_validity.py
    python analyze_stage1h_diagnostics.py

The scripts read `data/raw*` and write `data/processed/`. The bootstrap uses a fixed seed
(`SEED` in `analyze_stage1c_icc_ci.py`). On Windows consoles with a non-UTF-8 code page,
keyword text may print as `?`; the output files are UTF-8 and unaffected.

## Notes and limitations

* Collection cannot be repeated: every pull is a fresh Google sample and past pulls cannot be re-drawn.
* Data were collected with the unofficial `pytrends` interface, which may change without notice.
* The wait between requests was raised from 8 to 20 seconds on 2026-09-10 (see manuscript).
* Values are relative indices (0 to 100) per request, not search counts.
* Pulls that failed after five attempts were not replaced; Georgia has the most failures under the Travel filter.
