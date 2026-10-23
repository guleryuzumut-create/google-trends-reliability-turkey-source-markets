#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1i - Hakem itirazlarina yonelik ek analizler
==================================================
a) Kategorisiz (filtresiz) serinin resmi varislarla es-hareketi (kategorili seriyle
   karsilastirmali): guvenilir olan serinin gecerli olup olmadigi.
b) Pazar duzeyinde ortalama ICC icin kelime orneklemesini de iceren iki asamali
   bootstrap (kelimeler, sonra aylar) ve bir kelimeyi disarida birakma araligi.
c) Farkli deger kuralinin AUC, duyarlilik, ozgullugu ve Spearman rho'su icin
   kume (pazar/sorgu seti) bootstrap'i: ayni sorgu setini paylasan pazarlar tek
   kume sayilir.
d) Basarisiz cekimlerin nedeni (bos veri / istek siniri) toplama gunlugunden.

Ciktilar: data/processed/s1i_*.csv, s1i_summary.txt
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analyze_stage1_v2 import icc21
from analyze_stage1b_category_effect import load_draws as load_root
from analyze_stage1g_validity import load_arrivals, deseason, corr

warnings.filterwarnings("ignore")
OUT = os.path.join("data", "processed")
SEED = 20261023
rng = np.random.default_rng(SEED)
NB = 1000


def auc(score, y):
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def validity(panel, arr_s):
    rows = []
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        gt = sub.groupby("date")["value"].mean().sort_index().astype(float)
        a = arr_s[geo]
        idx = a.index.intersection(gt.index)
        x, y = gt.loc[idx], a.loc[idx].astype(float)
        idx2 = idx[(idx.year < 2020) | (idx.year > 2021)]
        x2, y2 = gt.loc[idx2], a.loc[idx2].astype(float)
        wide = sub.pivot_table(index="date", columns="draw_date", values="value").sort_index()
        rows.append({"geo": geo, "keyword": kw, "icc": icc21(wide.values),
                     "r_level": corr(x, y), "r_deseas": corr(deseason(x), deseason(y)),
                     "r_yoy": corr(x.diff(12), y.diff(12)),
                     "r_level_nocovid": corr(x2, y2), "r_deseas_nocovid": corr(deseason(x2), deseason(y2))})
    return pd.DataFrame(rows)


def main():
    lines = ["=" * 68, "ASAMA 1i - HAKEM ITIRAZLARINA YONELIK EK ANALIZLER", "=" * 68]

    # ---------------- a) kategorisiz serinin gecerligi
    arr = load_arrivals()
    arr_s = {g: d.set_index("date")["arrivals"].sort_index() for g, d in arr.groupby("geo")}
    vc = validity(load_root(os.path.join("data", "raw")), arr_s)
    vn = validity(load_root(os.path.join("data", "raw_nocat")), arr_s)
    m = vc.merge(vn, on=["geo", "keyword"], suffixes=("_cat", "_nocat"))
    m.to_csv(os.path.join(OUT, "s1i_validity_cat_vs_nocat.csv"), index=False)

    def cls(x):
        return pd.cut(x, [-1, 0.5, 0.9, 1.01], labels=["ICC<0.50", "0.50-0.90", "ICC>=0.90"])
    rows = []
    for tag, d, icc, cols in (("Travel filtreli", vc, "icc", ["r_level", "r_deseas", "r_yoy", "r_level_nocovid", "r_deseas_nocovid"]),
                              ("Filtresiz", vn, "icc", ["r_level", "r_deseas", "r_yoy", "r_level_nocovid", "r_deseas_nocovid"])):
        d = d.copy(); d["cls"] = cls(d[icc])
        g = d.groupby("cls", observed=True)[cols].median()
        g.insert(0, "n", d.groupby("cls", observed=True).size())
        g.insert(0, "seri", tag)
        rows.append(g.reset_index())
    cmp_ = pd.concat(rows)
    cmp_.to_csv(os.path.join(OUT, "s1i_validity_by_icc_class.csv"), index=False)
    lines += ["", "a) Kelime duzeyi medyan korelasyon (ICC sinifi kendi serisinin ICC'sine gore)", cmp_.round(3).to_string(index=False)]
    low = m[m.icc_cat < 0.5]
    lines += ["", f"Kategorili ICC<0.50 olan {len(low)} kelime: kategorili/filtresiz medyan r_level = "
                  f"{low.r_level_cat.median():.3f} / {low.r_level_nocat.median():.3f}; "
                  f"r_deseas = {low.r_deseas_cat.median():.3f} / {low.r_deseas_nocat.median():.3f}; "
                  f"r_deseas (2020-21 haric) = {low.r_deseas_nocovid_cat.median():.3f} / {low.r_deseas_nocovid_nocat.median():.3f}"]
    for col in ("r_level", "r_deseas", "r_deseas_nocovid"):
        ok = m[[f"{col}_cat", f"{col}_nocat"]].dropna()
        lines.append(f"{col}: filtresiz > filtreli {int((ok[col+'_nocat'] > ok[col+'_cat']).sum())}/{len(ok)} kelimede; "
                     f"medyan fark (filtresiz - filtreli) = {(ok[col+'_nocat'] - ok[col+'_cat']).median():.3f}")
    for col in ("r_level", "r_deseas", "r_deseas_nocovid"):
        ok = vn[["icc", col]].dropna()
        lines.append(f"Filtresiz seri: Spearman(ICC, {col}) = {spearmanr(ok.icc, ok[col])[0]:.3f} (n={len(ok)})")

    # ---------------- b) pazar ICC'si icin iki asamali bootstrap
    panel = load_root(os.path.join("data", "raw"))
    wides = {}
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wides[(geo, kw)] = sub.pivot_table(index="date", columns="draw_date", values="value").sort_index()
    rows = []
    for geo in sorted({g for g, _ in wides}):
        mats = [w for (g, k), w in wides.items() if g == geo]
        kws = len(mats)
        base = np.nanmean([icc21(w.values) for w in mats])
        loo = [np.nanmean([icc21(w.values) for j, w in enumerate(mats) if j != i]) for i in range(kws)] if kws > 1 else [base]
        n = mats[0].shape[0]
        vals = []
        for _ in range(NB):
            pick_m = rng.integers(0, n, n)
            pick_k = rng.integers(0, kws, kws)
            iccs = [icc21(mats[j].values[pick_m]) for j in pick_k]
            vals.append(np.nanmean(iccs) if not np.all(np.isnan(iccs)) else np.nan)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        rows.append({"geo": geo, "n_keywords": kws, "mean_icc": base, "two_stage_lo": lo, "two_stage_hi": hi,
                     "loo_min": min(loo), "loo_max": max(loo)})
    two = pd.DataFrame(rows).sort_values("mean_icc", ascending=False)
    one = pd.read_csv(os.path.join(OUT, "s1c_icc_ci_by_geo.csv"))[["geo", "boot_ci_lower", "boot_ci_upper"]]
    two = two.merge(one, on="geo")
    two["width_ratio"] = (two.two_stage_hi - two.two_stage_lo) / (two.boot_ci_upper - two.boot_ci_lower)
    two.to_csv(os.path.join(OUT, "s1i_two_stage_bootstrap_by_geo.csv"), index=False)
    lines += ["", "b) Pazar ortalama ICC: yalniz ay bootstrap'i (s1c) ile ay + kelime iki asamali bootstrap; "
                  "'loo' bir kelimeyi disarida birakinca ortalamanin alt/ust degeri", two.round(3).to_string(index=False)]

    # ---------------- c) kume bootstrap'i
    k = pd.read_csv(os.path.join(OUT, "s1_variability_by_keyword.csv"))[["geo", "keyword", "icc_2_1", "zero_share_pct"]]
    q = pd.read_csv(os.path.join(OUT, "s1d_quantization_by_keyword.csv"))[["geo", "keyword", "mean_distinct"]]
    d = k.merge(q, on=["geo", "keyword"])
    cluster = {g: ("RU+KZ" if g in ("RU", "KZ") else "IQ+SA" if g in ("IQ", "SA") else g) for g in d.geo.unique()}
    d["cl"] = d.geo.map(cluster)
    cl_names = sorted(d.cl.unique())
    groups = {c: d[d.cl == c] for c in cl_names}
    res = {"auc50": [], "rho": [], "sens": [], "spec": []}
    for _ in range(NB):
        pick = rng.integers(0, len(cl_names), len(cl_names))
        s = pd.concat([groups[cl_names[i]] for i in pick])
        y = (s.icc_2_1 >= 0.5).values
        pred = (s.mean_distinct >= 6).values
        res["auc50"].append(auc(s.mean_distinct.values, y))
        res["rho"].append(spearmanr(s.mean_distinct, s.icc_2_1)[0])
        res["sens"].append((pred & y).sum() / max(y.sum(), 1))
        res["spec"].append((~pred & ~y).sum() / max((~y).sum(), 1))
    y0 = (d.icc_2_1 >= 0.5).values
    pred0 = (d.mean_distinct >= 6).values
    out = {"AUC (ICC>=0.50)": auc(d.mean_distinct.values, y0), "Spearman rho": spearmanr(d.mean_distinct, d.icc_2_1)[0],
           "duyarlilik": (pred0 & y0).sum() / y0.sum(), "ozgullukk"[:-1]: (~pred0 & ~y0).sum() / (~y0).sum()}
    keys = ["auc50", "rho", "sens", "spec"]
    tab = pd.DataFrame({"olcut": list(out.keys()), "tahmin": list(out.values()),
                        "kume_bootstrap_lo": [np.nanpercentile(res[k_], 2.5) for k_ in keys],
                        "kume_bootstrap_hi": [np.nanpercentile(res[k_], 97.5) for k_ in keys]})
    tab.to_csv(os.path.join(OUT, "s1i_cluster_bootstrap_rule.csv"), index=False)
    lines += ["", f"c) Farkli deger kurali: {len(cl_names)} kume (pazar / ortak sorgu seti) uzerinden kume bootstrap'i", tab.round(3).to_string(index=False)]

    # ---------------- d) basarisizlik nedenleri
    log_path = "collect_gt.log" if os.path.exists("collect_gt.log") else os.path.join("data", "collect_gt.log")
    log = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    cl = pd.read_csv(os.path.join("data", "collection_log.csv"))
    fails = {(r.geo, r.draw_date_utc) for r in cl[cl.status != "OK"].itertuples()}
    last = {}
    for i, ln in enumerate(log):
        mm = re.search(r"^(\d{4}-\d{2}-\d{2}) .*\[(\w\w)\] BASARISIZ", ln)
        if mm and (mm.group(2), mm.group(1)) in fails:
            geo = mm.group(2)
            window = log[max(0, i - 12):i]
            n_empty = sum(1 for w in window if f"[{geo}] Bos veri" in w)
            n_429 = sum(1 for w in window if f"[{geo}]" in w and "429" in w)
            last[(geo, mm.group(1))] = "bos veri" if n_empty >= n_429 else "istek siniri (429)"
    cnt = {}
    for (geo, _), kind in last.items():
        cnt[(geo, kind)] = cnt.get((geo, kind), 0) + 1
    lines += ["", f"d) Gunluge gore kalici basarisiz {len(fails)} cekimden nedeni belirlenenler (son denemelerin turu)"]
    for (geo, kind), v in sorted(cnt.items()):
        lines.append(f"{geo}: {kind} = {v}")
    lines.append(f"Gunlukte eslesen: {len(last)} / {len(fails)}")
    pd.DataFrame([{"geo": g, "neden": kd, "n": v} for (g, kd), v in sorted(cnt.items())]).to_csv(
        os.path.join(OUT, "s1i_failure_causes.csv"), index=False)

    lines.append("=" * 68)
    with open(os.path.join(OUT, "s1i_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Yazildi: s1i_*")


if __name__ == "__main__":
    main()
