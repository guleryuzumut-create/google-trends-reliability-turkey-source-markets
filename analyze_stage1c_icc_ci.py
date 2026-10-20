#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1c - ICC(2,1) Guven Araliklari
=====================================
Koo ve Li (2016) ICC yorumunun nokta tahmini degil %95 guven araligina gore
yapilmasini onerir. Bu script:
  (1) her (pazar, anahtar kelime) icin ICC(2,1) ve analitik %95 guven araligi
      (Shrout-Fleiss / McGraw-Wong F-dagilimi yaklasimi),
  (2) her pazar icin kelime ICC'lerinin ortalamasinin %95 guven araligi
      (aylar uzerinden bootstrap; kelimeler ayni ay ornegiyle birlikte
      yeniden orneklenir).

Ciktilar (data/processed/):
  s1c_icc_ci_by_keyword.csv
  s1c_icc_ci_by_geo.csv
  s1c_summary.txt

Calistirma: python analyze_stage1c_icc_ci.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from analyze_stage1_v2 import load_draws, icc21

OUT_DIR = os.path.join("data", "processed")
ALPHA = 0.05
N_BOOT = 1000
SEED = 20261015


def icc21_ci(matrix, alpha=ALPHA):
    """ICC(2,1) ve analitik guven araligi. matrix: (n_ay x k_cekim)."""
    m = np.asarray(matrix, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 3 or k < 2:
        return np.nan, np.nan, np.nan

    grand = m.mean()
    ss_rows = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((m - grand) ** 2).sum() - ss_rows - ss_cols

    dfr, dfc = n - 1, k - 1
    msr, msc, mse = ss_rows / dfr, ss_cols / dfc, ss_err / (dfr * dfc)
    if mse <= 0:
        return np.nan, np.nan, np.nan

    icc = (msr - mse) / (msr + dfc * mse + k * (msc - mse) / n)
    fc = msc / mse

    vn = dfc * dfr * (k * icc * fc + n * (1 + dfc * icc) - k * icc) ** 2
    vd = dfr * k ** 2 * icc ** 2 * fc ** 2 + (n * (1 + dfc * icc) - k * icc) ** 2
    if vd <= 0 or vn <= 0:
        return icc, np.nan, np.nan
    v = vn / vd

    f_up = f_dist.ppf(1 - alpha / 2, dfr, v)
    f_lo = f_dist.ppf(1 - alpha / 2, v, dfr)
    den = k * msc + (k * n - k - n) * mse
    lb = n * (msr - f_up * mse) / (f_up * den + n * msr)
    ub = n * (f_lo * msr - mse) / (den + n * f_lo * msr)
    return icc, lb, ub


def wide_by_series(panel):
    out = {}
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        out[(geo, kw)] = sub.pivot_table(
            index="date", columns="draw_date", values="value"
        ).sort_index()
    return out


def boot_geo_mean_icc(mats, rng):
    """mats: bir pazarin kelime matrisleri (aylar ortak). Ay indeksi yeniden orneklenir."""
    months = mats[0].index
    n = len(months)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        iccs = []
        for w in mats:
            iccs.append(icc21(w.values[pick]))
        vals.append(np.nanmean(iccs) if not np.all(np.isnan(iccs)) else np.nan)
    arr = np.array(vals, dtype=float)
    return np.nanpercentile(arr, 100 * ALPHA / 2), np.nanpercentile(arr, 100 * (1 - ALPHA / 2))


def label(lb):
    if np.isnan(lb):
        return "tanimsiz"
    if lb > 0.90:
        return "mukemmel"
    if lb > 0.75:
        return "iyi"
    if lb > 0.50:
        return "orta"
    return "zayif/belirsiz"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_draws()
    series = wide_by_series(panel)

    rows = []
    for (geo, kw), w in series.items():
        icc, lb, ub = icc21_ci(w.values)
        rows.append({
            "geo": geo, "keyword": kw, "n_draws": w.shape[1], "n_months": w.shape[0],
            "icc": icc, "ci_lower": lb, "ci_upper": ub,
            "class_by_lower_bound": label(lb),
        })
    by_kw = pd.DataFrame(rows).sort_values(["geo", "keyword"])
    by_kw.to_csv(os.path.join(OUT_DIR, "s1c_icc_ci_by_keyword.csv"), index=False)

    rng = np.random.default_rng(SEED)
    grows = []
    for geo, sub in by_kw.groupby("geo"):
        mats = [series[(geo, kw)] for kw in sub["keyword"]]
        lo, hi = boot_geo_mean_icc(mats, rng)
        grows.append({
            "geo": geo, "n_keywords": len(sub), "n_draws": int(sub["n_draws"].max()),
            "mean_icc": sub["icc"].mean(), "boot_ci_lower": lo, "boot_ci_upper": hi,
        })
    by_geo = pd.DataFrame(grows).sort_values("mean_icc", ascending=False)
    by_geo.to_csv(os.path.join(OUT_DIR, "s1c_icc_ci_by_geo.csv"), index=False)

    lines = ["=" * 68, "ASAMA 1c - ICC(2,1) %95 GUVEN ARALIKLARI", "=" * 68,
             f"Cekim sayisi: {panel['draw_date'].nunique()} | bootstrap: {N_BOOT} | seed: {SEED}", "",
             "--- Pazar duzeyinde ortalama ICC (bootstrap %95 GA, aylar uzerinden) ---",
             by_geo.round(3).to_string(index=False), "",
             "--- Kelime duzeyinde sinif dagilimi (GA alt sinirina gore, Koo ve Li esikleri) ---",
             by_kw["class_by_lower_bound"].value_counts().to_string(), "",
             "NOT: Bootstrap aylar arasi zamansal bagimliligi yok sayar; GA'lar",
             "bu varsayim altinda yaklasik degerlerdir. Kelime duzeyindeki GA",
             "analitiktir (iki yonlu rastgele etkiler, mutlak uyum, tek olcum).",
             "=" * 68]
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "s1c_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("Ciktilar yazildi: s1c_icc_ci_by_keyword.csv, s1c_icc_ci_by_geo.csv, s1c_summary.txt")


if __name__ == "__main__":
    main()
