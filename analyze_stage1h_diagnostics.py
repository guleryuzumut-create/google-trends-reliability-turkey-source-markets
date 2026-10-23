#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1h - Tani olcutu degerlendirmesi, donem duyarliligi, sinyal/gurultu ayrimi ve sekiller
===========================================================================================
a) 'Farkli deger sayisi' kuralinin karisiklik matrisi, duyarlilik/ozgullugu, PPV/NPV,
   cogunluk-sinifi tabani ve AUC'nin bootstrap GA'si; rakip tahmin ediciler (ort. duzey,
   sifir orani, maksimum) ile karsilastirma; seyrek olmayan serilerde kural hala ayirt
   ediyor mu (mekanik iliski suphesi).
b) Donem duyarliligi: ilk 10 pazar icin yalniz 4 Eylul sonrasi cekimlerle (yeni 5 pazarla
   ayni pencere) ICC; 10 Eylul oncesi/sonrasi (bekleme suresi degisimi) ICC.
c) ICC'nin sinyal ve gurultu bilesenleri: aylar arasi SD (sinyal) ve ay-ici cekimler arasi SD
   (gurultu), ICC sinifina gore.
d) Sekiller (figures/fig1-fig3.png ve Turkce surumleri *_tr.png).

Ciktilar: data/processed/s1h_*.csv, s1h_summary.txt, figures/*.png
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_stage1_v2 import load_draws, icc21

OUT = os.path.join("data", "processed")
FIG = "figures"
SEED = 20261022
rng = np.random.default_rng(SEED)
os.makedirs(FIG, exist_ok=True)
SUB = os.path.join(FIG, "submission")
os.makedirs(SUB, exist_ok=True)


def auc(score, y):
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    k = pd.read_csv(os.path.join(OUT, "s1_variability_by_keyword.csv"))[["geo", "keyword", "mean_level", "zero_share_pct", "icc_2_1"]]
    q = pd.read_csv(os.path.join(OUT, "s1d_quantization_by_keyword.csv"))[["geo", "keyword", "mean_distinct", "max_value"]]
    d = k.merge(q, on=["geo", "keyword"])
    lines = ["=" * 68, "ASAMA 1h - TANI OLCUTU, DONEM DUYARLILIGI, SINYAL/GURULTU", "=" * 68]

    # a) tani degerlendirmesi
    rows = []
    for cut in (0.5, 0.75):
        y = (d.icc_2_1 >= cut).values
        pred = (d.mean_distinct >= 6).values
        tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
        base = max(y.mean(), 1 - y.mean())
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), len(d))
            boots.append(auc(d.mean_distinct.values[idx], y[idx]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({"icc_cut": cut, "n_pos": int(y.sum()), "n": len(d), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "PPV": tp / (tp + fp) if tp + fp else np.nan, "NPV": tn / (tn + fn) if tn + fn else np.nan,
                     "accuracy": (tp + tn) / len(d), "majority_baseline": base,
                     "auc": auc(d.mean_distinct.values, y), "auc_ci_lo": lo, "auc_ci_hi": hi})
        for name, col in (("mean_level", "mean_level"), ("neg_zero_share", None), ("max_value", "max_value")):
            s = -d.zero_share_pct.values if col is None else d[col].values
            rows[-1][f"auc_{name}"] = auc(s, y)
    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(OUT, "s1h_distinct_rule_diagnostics.csv"), index=False)
    lines += ["", "a) Farkli deger >= 6 kurali (tek cekim tani olcutu)", diag.round(3).T.to_string(), ""]
    sub = d[d.zero_share_pct < 50]
    from scipy.stats import spearmanr
    rho_all = spearmanr(d.mean_distinct, d.icc_2_1)[0]
    rho_sub = spearmanr(sub.mean_distinct, sub.icc_2_1)[0]
    lines.append(f"Spearman(farkli deger, ICC): tum kelimeler = {rho_all:.3f} (n={len(d)}); sifir orani < %50 olanlar = {rho_sub:.3f} (n={len(sub)})")

    # b) donem duyarliligi
    panel = load_draws()
    wides = {}
    for (geo, kw), sub_ in panel.groupby(["geo", "keyword"]):
        wides[(geo, kw)] = sub_.pivot_table(index="date", columns="draw_date", values="value").sort_index()
    orig10 = {"RU", "DE", "GB", "BG", "IR", "UA", "PL", "NL", "GE", "IQ"}
    rows = []
    for (geo, kw), w in wides.items():
        cols = list(w.columns)
        late = [c for c in cols if c >= "2026-09-04"]
        pre = [c for c in cols if c < "2026-09-10"]
        post = [c for c in cols if c >= "2026-09-10"]
        rows.append({"geo": geo, "keyword": kw, "orig10": geo in orig10, "n_all": len(cols), "icc_all": icc21(w.values),
                     "n_since_0904": len(late), "icc_since_0904": icc21(w[late].values) if len(late) > 5 else np.nan,
                     "n_pre_0910": len(pre), "icc_pre_0910": icc21(w[pre].values) if len(pre) > 5 else np.nan,
                     "n_post_0910": len(post), "icc_post_0910": icc21(w[post].values) if len(post) > 5 else np.nan})
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "s1h_period_sensitivity_by_keyword.csv"), index=False)
    g = per.groupby("geo").agg(n_all=("n_all", "max"), icc_all=("icc_all", "mean"), n_since_0904=("n_since_0904", "max"),
                               icc_since_0904=("icc_since_0904", "mean"), n_pre=("n_pre_0910", "max"), icc_pre=("icc_pre_0910", "mean"),
                               n_post=("n_post_0910", "max"), icc_post=("icc_post_0910", "mean")).sort_values("icc_all", ascending=False)
    g.to_csv(os.path.join(OUT, "s1h_period_sensitivity_by_geo.csv"))
    lines += ["", "b) Donem duyarliligi (pazar ortalamasi ICC)", g.round(3).to_string()]
    for a, b in (("icc_all", "icc_since_0904"), ("icc_pre_0910", "icc_post_0910")):
        ok = per[[a, b]].dropna()
        lines.append(f"Kelime duzeyi Spearman({a}, {b}) = {ok[a].rank().corr(ok[b].rank()):.3f}; ort. mutlak fark = {(ok[a]-ok[b]).abs().mean():.3f} (n={len(ok)})")

    # c) sinyal / gurultu
    rows = []
    for (geo, kw), w in wides.items():
        m = w.values
        mean_series = np.nanmean(m, axis=1)
        signal_sd = np.nanstd(mean_series, ddof=1)
        noise_sd = np.sqrt(np.nanmean(np.nanvar(m, axis=1, ddof=1)))
        rows.append({"geo": geo, "keyword": kw, "signal_sd": signal_sd, "noise_sd": noise_sd})
    sn = pd.DataFrame(rows).merge(k, on=["geo", "keyword"])
    sn["cls"] = pd.cut(sn.icc_2_1, [-1, 0.5, 0.9, 1.01], labels=["ICC<0.50", "0.50-0.90", "ICC>=0.90"])
    snt = sn.groupby("cls", observed=True).agg(n=("keyword", "count"), med_signal_sd=("signal_sd", "median"),
                                                 med_noise_sd=("noise_sd", "median"), med_mean_level=("mean_level", "median"))
    sn.to_csv(os.path.join(OUT, "s1h_signal_noise_by_keyword.csv"), index=False)
    lines += ["", "c) Sinyal (aylar arasi SD) ve gurultu (ay-ici cekimler arasi SD), ICC sinifina gore", snt.round(3).to_string()]

    # d) sekiller (Times New Roman 10 pt, 150 mm genislik: Word'de 1:1 boyutta okunur); EN ve TR surumleri
    d2 = d.copy()
    cat = pd.read_csv(os.path.join(OUT, "s1b_category_effect_by_keyword.csv"))
    sh = pd.read_csv(os.path.join(OUT, "s1e_split_half_by_geo.csv"), index_col=0)
    sh.columns = [int(c) for c in sh.columns]
    colors = {"high": "#1b7f5f", "mid": "#b35000", "low": "#5b4b9a"}
    markers = {"high": "o", "mid": "s", "low": "^"}
    geo_icc = d.groupby("geo").icc_2_1.mean()
    grp = geo_icc.apply(lambda v: "high" if v >= 0.9 else ("mid" if v >= 0.5 else "low"))
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.titlesize": 10, "axes.linewidth": 0.8})
    W = 5.9  # inc = 150 mm
    LAB = {
        "en": {"suf": "", "dec": ".",
               "hi": "Markets with mean ICC ≥ 0.90", "mid": "Mean ICC 0.50 to 0.90", "lo": "Mean ICC below 0.50",
               "cut": "5 | 6 distinct values",
               "x1": "Mean number of distinct values per pull (out of 127 months)", "y1": "ICC(2,1) of the keyword series",
               "x2": "ICC with Travel category filter", "y2": "ICC without category filter",
               "x3": "Number of pulls averaged (k)", "y3": "Split-half ICC of the k-pull mean",
               "note3": "7 markets with mean ICC ≥ 0.90 lie above 0.94"},
        "tr": {"suf": "_tr", "dec": ",",
               "hi": "Ortalama ICC ≥ 0,90 olan pazarlar", "mid": "Ortalama ICC 0,50-0,90", "lo": "Ortalama ICC 0,50'nin altı",
               "cut": "5 | 6 farklı değer",
               "x1": "Çekim başına ortalama farklı değer sayısı (127 ay içinde)", "y1": "Anahtar kelime serisinin ICC(2,1) değeri",
               "x2": "Seyahat kategorisi filtreli ICC", "y2": "Filtresiz ICC",
               "x3": "Ortalaması alınan çekim sayısı (k)", "y3": "k çekim ortalamasının yarıya bölme ICC değeri",
               "note3": "Ortalama ICC ≥ 0,90 olan 7 pazar 0,94'ün üzerindedir"},
    }
    from matplotlib.ticker import FuncFormatter
    for lang, L in LAB.items():
        fmt = FuncFormatter(lambda v, _, dc=L["dec"]: f"{v:.1f}".replace(".", dc))
        fmt2 = FuncFormatter(lambda v, _, dc=L["dec"]: f"{v:.2f}".replace(".", dc))
        fig, ax = plt.subplots(figsize=(W, 4.6))
        for gname, lab in (("high", L["hi"]), ("mid", L["mid"]), ("low", L["lo"])):
            s_ = d2[d2.geo.map(grp) == gname]
            ax.scatter(s_.mean_distinct, s_.icc_2_1, s=30, alpha=0.8, color=colors[gname], marker=markers[gname], label=lab, linewidths=0)
        ax.axvline(5.5, color="grey", ls="--", lw=0.9)
        ax.text(6.3, -0.07, L["cut"], fontsize=10, color="dimgrey")
        ax.set_xlabel(L["x1"]); ax.set_ylabel(L["y1"])
        ax.set_ylim(-0.12, 1.04); ax.yaxis.set_major_formatter(fmt)
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig1_icc_vs_distinct_values{L['suf']}.png"), dpi=300)
        if lang == "en": fig.savefig(os.path.join(SUB, "Fig1.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ok = cat.dropna(subset=["icc_2_1_cat", "icc_2_1_nocat"])
        ax.scatter(ok.icc_2_1_cat, ok.icc_2_1_nocat, s=30, color="#2b5c8a", alpha=0.8, linewidths=0)
        ax.plot([-0.1, 1.05], [-0.1, 1.05], color="grey", lw=0.9, ls="--")
        ax.set_xlabel(L["x2"]); ax.set_ylabel(L["y2"])
        ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05); ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig2_filtered_vs_unfiltered_icc{L['suf']}.png"), dpi=300)
        if lang == "en": fig.savefig(os.path.join(SUB, "Fig2.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(W, 4.8))
        ends = []
        for geo in sh.index:
            s_ = sh.loc[geo].dropna()
            ax.plot(s_.index, s_.values, marker=markers[grp[geo]], ms=3.5, lw=1.1, color=colors[grp[geo]], alpha=0.9)
            if grp[geo] != "high":
                ends.append([geo, s_.index[-1], s_.values[-1]])
        ends.sort(key=lambda e: -e[2])
        ypos = [e[2] for e in ends]
        for i in range(1, len(ypos)):
            if ypos[i - 1] - ypos[i] < 0.05:
                ypos[i] = ypos[i - 1] - 0.05
        for (geo, x, y), yl in zip(ends, ypos):
            ax.text(x + 0.35, yl, geo, fontsize=10, va="center", color=colors[grp[geo]])
        ax.text(16.0, 1.04, L["note3"], fontsize=10, ha="right", va="bottom", color=colors["high"])
        ax.set_xlabel(L["x3"]); ax.set_ylabel(L["y3"])
        ax.set_xlim(0.5, 18.6); ax.set_ylim(-0.05, 1.1); ax.set_xticks([1, 2, 3, 5, 8, 12, 16]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(fmt2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig3_reliability_vs_pulls_averaged{L['suf']}.png"), dpi=300)
        if lang == "en": fig.savefig(os.path.join(SUB, "Fig3.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    from PIL import Image
    for n in ("Fig1", "Fig2", "Fig3"):  # Springer: RGB, 8 bit
        pth = os.path.join(SUB, n + ".tif")
        Image.open(pth).convert("RGB").save(pth, dpi=(600, 600), compression="tiff_lzw")

    with open(os.path.join(OUT, "s1h_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines + ["=" * 68]))
    print("Yazildi: s1h_*, figures/*.png")


if __name__ == "__main__":
    main()
