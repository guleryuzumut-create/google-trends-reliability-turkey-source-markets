#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1f - Dayaniklilik kontrolleri ("seytanin avukati")
=========================================================
Bir hakemin sorabilecegi bes soruyu olcer:
  a) Gunler arasi kayma var mi? ICC(2,1) (mutlak uyum) ile ICC(3,1) (tutarlilik)
     karsilastirilir; fark buyukse cekim gunu etkisi vardir.
  b) ICC dagilim varsayimina duyarli mi? log(1+x) donusumlu ICC ve cekimler
     arasi ortalama Spearman korelasyonu.
  c) Aylar arasi zamansal bagimlilik GA'lari daraltiyor mu? 12 aylik blok
     bootstrap ile pazar ortalamasi GA'si, iid bootstrap'le karsilastirilir.
  d) Kategorili/kategorisiz karsilastirma ayni cekim gunleriyle de gecerli mi?
     (kategorisiz seri 6 gun sonra basladi) Yalniz ortak tarihler kullanilir.
  e) 'Farkli deger sayisi' kurali nesne disi dogruluk: AUC ve pazar disarida
     birakma (leave-one-market-out) esik dogrulugu.

Ciktilar (data/processed/): s1f_*.csv ve s1f_summary.txt
Calistirma: python analyze_stage1f_robustness.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analyze_stage1_v2 import load_draws, icc21
import analyze_stage1_v2 as a1

OUT = os.path.join("data", "processed")
SEED = 20261020
rng = np.random.default_rng(SEED)


def icc31(m):
    m = np.asarray(m, float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 3 or k < 2:
        return np.nan
    g = m.mean()
    ssr = k * ((m.mean(axis=1) - g) ** 2).sum()
    ssc = n * ((m.mean(axis=0) - g) ** 2).sum()
    sse = ((m - g) ** 2).sum() - ssr - ssc
    msr, mse = ssr / (n - 1), sse / ((n - 1) * (k - 1))
    d = msr + (k - 1) * mse
    return np.nan if d == 0 else (msr - mse) / d


def mean_pair_spearman(m):
    m = np.asarray(m, float)
    m = m[~np.isnan(m).any(axis=1)]
    if m.shape[0] < 3 or m.shape[1] < 2:
        return np.nan
    r = np.apply_along_axis(rankdata, 0, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(r, rowvar=False)
    iu = np.triu_indices_from(c, 1)
    return np.nanmean(c[iu]) if np.isfinite(c[iu]).any() else np.nan


def block_idx(n, b):
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, nb)
    return np.concatenate([np.arange(s, s + b) for s in starts])[:n]


def main():
    panel = load_draws()
    wides = {}
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wides[(geo, kw)] = sub.pivot_table(index="date", columns="draw_date", values="value").sort_index()

    # a) + b)
    rows = []
    for (geo, kw), w in wides.items():
        v = w.values
        rows.append({"geo": geo, "keyword": kw, "icc2": icc21(v), "icc3": icc31(v),
                     "icc2_log": icc21(np.log1p(v)), "mean_spearman": mean_pair_spearman(v)})
    r1 = pd.DataFrame(rows)
    r1["icc3_minus_icc2"] = r1["icc3"] - r1["icc2"]
    r1.to_csv(os.path.join(OUT, "s1f_transform_and_drift_by_keyword.csv"), index=False)
    g1 = r1.groupby("geo").agg(icc2=("icc2", "mean"), icc3=("icc3", "mean"),
                               icc2_log=("icc2_log", "mean"), spearman=("mean_spearman", "mean"),
                               max_abs_drift=("icc3_minus_icc2", lambda s: s.abs().max())).sort_values("icc2", ascending=False)

    # c) blok bootstrap
    B, BLOCK = 500, 12
    rows = []
    for geo in g1.index:
        mats = [wides[(geo, kw)] for kw in r1[r1.geo == geo].keyword]
        n = len(mats[0])
        blk, iid = [], []
        for _ in range(B):
            ib = block_idx(n, BLOCK)
            ii = rng.integers(0, n, n)
            blk.append(np.nanmean([icc21(w.values[ib]) for w in mats]))
            iid.append(np.nanmean([icc21(w.values[ii]) for w in mats]))
        rows.append({"geo": geo,
                     "iid_lo": np.nanpercentile(iid, 2.5), "iid_hi": np.nanpercentile(iid, 97.5),
                     "block_lo": np.nanpercentile(blk, 2.5), "block_hi": np.nanpercentile(blk, 97.5)})
    c = pd.DataFrame(rows).set_index("geo")
    c["iid_width"] = c.iid_hi - c.iid_lo
    c["block_width"] = c.block_hi - c.block_lo
    c.to_csv(os.path.join(OUT, "s1f_block_bootstrap_by_geo.csv"))

    # d) ortak tarihli kategori karsilastirmasi
    nocat_dir = os.path.join("data", "raw_nocat")
    dates_nc = sorted(d for d in os.listdir(nocat_dir) if os.path.isdir(os.path.join(nocat_dir, d)))
    old = a1.RAW_ROOT
    a1.RAW_ROOT = nocat_dir
    pn = load_draws()
    a1.RAW_ROOT = old
    rows = []
    for (geo, kw), sub in pn.groupby(["geo", "keyword"]):
        wn = sub.pivot_table(index="date", columns="draw_date", values="value").sort_index()
        wc = wides[(geo, kw)]
        common = [d for d in wc.columns if d in wn.columns]
        rows.append({"geo": geo, "keyword": kw, "n_common": len(common),
                     "icc_cat_all": icc21(wc.values), "icc_cat_matched": icc21(wc[common].values) if len(common) > 1 else np.nan,
                     "icc_nocat_matched": icc21(wn[common].values) if len(common) > 1 else np.nan})
    d = pd.DataFrame(rows)
    d["diff_matched"] = d.icc_nocat_matched - d.icc_cat_matched
    d.to_csv(os.path.join(OUT, "s1f_category_matched_dates.csv"), index=False)
    orig = pd.read_csv(os.path.join(OUT, "s1b_category_effect_by_keyword.csv"))[["geo", "keyword", "icc_diff"]]
    dm = d.merge(orig, on=["geo", "keyword"])
    corr = dm[["diff_matched", "icc_diff"]].corr(method="spearman").iloc[0, 1]

    # e) esik kurali
    q = pd.read_csv(os.path.join(OUT, "s1d_quantization_by_keyword.csv"))
    def auc(x, y):
        pos, neg = x[y], x[~y]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        r = rankdata(np.concatenate([pos, neg]))
        return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    out_e = []
    for cut in (0.5, 0.75):
        y = (q.icc_2_1 >= cut).values
        x = q.mean_distinct.values
        a = auc(x, y)
        correct, tot = 0, 0
        for geo in q.geo.unique():
            tr, te = q[q.geo != geo], q[q.geo == geo]
            best, bt = -1, None
            for t in range(1, 60):
                acc = ((tr.mean_distinct >= t) == (tr.icc_2_1 >= cut)).mean()
                if acc > best:
                    best, bt = acc, t
            correct += ((te.mean_distinct >= bt) == (te.icc_2_1 >= cut)).sum(); tot += len(te)
        fixed = ((q.mean_distinct >= 6) == (q.icc_2_1 >= cut)).mean()
        out_e.append({"icc_cut": cut, "auc": a, "lomo_accuracy": correct / tot, "accuracy_fixed_rule_ge6": fixed})
    e = pd.DataFrame(out_e)
    e.to_csv(os.path.join(OUT, "s1f_distinct_rule_validation.csv"), index=False)

    lines = ["=" * 68, "ASAMA 1f - DAYANIKLILIK KONTROLLERI", "=" * 68,
             "a,b) ICC(2,1) mutlak uyum | ICC(3,1) tutarlilik | log(1+x) ICC | ort. Spearman",
             g1.round(3).to_string(), "",
             f"En buyuk |ICC3-ICC2| (kelime duzeyi): {r1.icc3_minus_icc2.abs().max():.3f}", "",
             f"c) Pazar ortalamasi ICC %95 GA genisligi: iid vs {BLOCK}-aylik blok bootstrap",
             c[["iid_width", "block_width"]].round(3).to_string(), "",
             "d) Kategori etkisi yalniz ORTAK cekim gunleriyle: eslesmis fark ile orijinal fark arasi Spearman = "
             f"{corr:.3f}",
             dm.sort_values("diff_matched", ascending=False).head(8)[["geo", "keyword", "n_common", "icc_cat_matched", "icc_nocat_matched", "diff_matched"]].round(3).to_string(index=False), "",
             "e) 'Farkli deger sayisi >= 6' kurali", e.round(3).to_string(index=False), "=" * 68]
    with open(os.path.join(OUT, "s1f_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Yazildi: s1f_*")


if __name__ == "__main__":
    main()
