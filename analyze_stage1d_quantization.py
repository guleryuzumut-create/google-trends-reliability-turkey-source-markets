#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1d - Kuantizasyon (Tam Sayi Yuvarlama) Etkisi
====================================================
Google Trends degerleri 0-100 arasi TAM SAYIDIR. Bir kelimenin seviyesi
dusukse seri yalnizca birkac farkli tam sayi alir; bu durumda yuvarlama
tek basina cekimler arasi farka yol acar.

Her (pazar, kelime) icin olculur:
  mean_distinct : bir cekimdeki ortalama FARKLI DEGER sayisi (127 aydan)
  max_value     : serinin gordugu en yuksek deger (cekim ortalamasi)
  mean_sd       : ortalamasi > 0 olan aylarda cekimler arasi SD ortalamasi
  rounding_sd   : tek basina tam sayi yuvarlamanin bekleyecegi SD (1/sqrt(12))
  sd_ratio      : mean_sd / rounding_sd  (1'e yakinsa fark yuvarlamayla
                  aciklanabilir, cok buyukse ornekleme gurultusu baskin)

Ciktilar (data/processed/): s1d_quantization_by_keyword.csv, s1d_summary.txt
Calistirma: python analyze_stage1d_quantization.py
"""

import os
import numpy as np
import pandas as pd

from analyze_stage1_v2 import load_draws

OUT_DIR = os.path.join("data", "processed")
ROUNDING_SD = 1 / np.sqrt(12)


def main():
    panel = load_draws()
    icc = pd.read_csv(os.path.join(OUT_DIR, "s1_variability_by_keyword.csv"))[
        ["geo", "keyword", "mean_level", "zero_share_pct", "icc_2_1"]
    ]

    rows = []
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wide = sub.pivot_table(index="date", columns="draw_date", values="value")
        distinct = wide.nunique(axis=0)
        sd_month = wide.std(axis=1, ddof=1)
        active = wide.mean(axis=1) > 0
        rows.append({
            "geo": geo, "keyword": kw, "n_draws": wide.shape[1],
            "mean_distinct": distinct.mean(),
            "max_value": wide.max(axis=0).mean(),
            "mean_sd": sd_month[active].mean() if active.any() else np.nan,
        })
    q = pd.DataFrame(rows).merge(icc, on=["geo", "keyword"])
    q["rounding_sd"] = ROUNDING_SD
    q["sd_ratio"] = q["mean_sd"] / ROUNDING_SD
    q.sort_values(["geo", "keyword"]).to_csv(
        os.path.join(OUT_DIR, "s1d_quantization_by_keyword.csv"), index=False)

    ok = q.dropna(subset=["icc_2_1", "mean_distinct"])
    rho_d = ok["mean_distinct"].rank().corr(ok["icc_2_1"].rank())
    rho_m = ok["max_value"].rank().corr(ok["icc_2_1"].rank())

    bins = pd.cut(q["mean_distinct"], [0, 5, 10, 20, 40, 200],
                  labels=["<=5", "6-10", "11-20", "21-40", ">40"])
    tab = q.groupby(bins, observed=True).agg(
        n_kelime=("keyword", "count"),
        medyan_icc=("icc_2_1", "median"),
        ort_icc=("icc_2_1", "mean"),
        medyan_sd_orani=("sd_ratio", "median"),
    ).round(3)

    lines = ["=" * 68, "ASAMA 1d - KUANTIZASYON (TAM SAYI YUVARLAMA) ETKISI", "=" * 68,
             f"Cekim sayisi: {panel['draw_date'].nunique()} | kelime: {len(q)}", "",
             f"Spearman(ortalama farkli deger sayisi, ICC) = {rho_d:.3f}",
             f"Spearman(maksimum deger, ICC)               = {rho_m:.3f}", "",
             "--- Farkli deger sayisina gore ICC ---", tab.to_string(), "",
             "--- Yuvarlama SD'sine gore (sd_ratio = gozlenen SD / 0.289) ---",
             "sd_ratio ~ 1: cekimler arasi fark yuvarlamayla aciklanabilir.",
             "sd_ratio >> 1: ornekleme gurultusu yuvarlamadan cok buyuk.", "",
             q.groupby("geo").agg(
                 mean_level=("mean_level", "mean"),
                 mean_distinct=("mean_distinct", "mean"),
                 max_value=("max_value", "mean"),
                 sd_ratio=("sd_ratio", "mean"),
                 icc=("icc_2_1", "mean"),
             ).sort_values("icc", ascending=False).round(2).to_string(), "",
             "=" * 68]
    with open(os.path.join(OUT_DIR, "s1d_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Yazildi: s1d_quantization_by_keyword.csv, s1d_summary.txt")


if __name__ == "__main__":
    main()
