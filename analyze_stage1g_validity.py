#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1g - Dis gecerlik: Google Trends serisi ile resmi aylik varis sayilari
===========================================================================
Kultur ve Turizm Bakanligi yillik sinir istatistik dosyalarindan (T7 sayfasi:
milliyete ve aylara gore gelen yabanci ziyaretci) 15 pazarin aylik varis sayilari
cikarilir (2016-2025) ve her anahtar kelimenin cekim ortalamasi serisiyle
karsilastirilir. Amac: guvenilirlik (ICC) ile gecerlik (varisla ilisi) arasinda
iliski var mi?

Olculer (her kelime icin, 120 ay):
  r_level   : Pearson korelasyonu (duzey)
  r_deseas  : ay-ortalamasi cikarilmis (mevsimsellikten arindirilmis) serilerin korelasyonu
  r_yoy     : 12 aylik farklarin (yillik degisim) korelasyonu

Girdi : validity_data/ktb_annual/sinir_<yil>.xlsx (Bakanlik, YIGM)
Cikti : data/external/ktb_monthly_arrivals_2016_2025.csv,
        data/processed/s1g_validity_by_keyword.csv, s1g_validity_by_geo.csv, s1g_summary.txt
"""

import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import openpyxl

from analyze_stage1_v2 import load_draws

warnings.filterwarnings("ignore")

SRC = os.path.join("validity_data", "ktb_annual")
EXT = os.path.join("data", "external")
OUT = os.path.join("data", "processed")
MONTHS = ["OCAK", "ŞUBAT", "MART", "NİSAN", "MAYIS", "HAZİRAN", "TEMMUZ", "AĞUSTOS", "EYLÜL", "EKİM", "KASIM", "ARALIK"]
NAMES = {
    "RU": "rusya", "DE": "almanya", "GB": "ingiltere", "BG": "bulgaristan", "IR": "iran",
    "UA": "ukrayna", "PL": "polonya", "NL": "hollanda", "GE": "gurcistan", "IQ": "irak",
    "KZ": "kazakistan", "AZ": "azerbaycan", "SA": "suudi arabistan", "FR": "fransa",
    "US": "amerika birlesik devletleri",
}


def norm(s):
    s = str(s).replace("İ", "i").replace("I", "i").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def load_arrivals():
    rows = []
    for y in range(2016, 2026):
        ws = openpyxl.load_workbook(os.path.join(SRC, f"sinir_{y}.xlsx"), read_only=True, data_only=True)["T7-Mil-Ay Göre G.Yabancı"]
        found = {}
        for r in ws.iter_rows(values_only=True):
            if len(r) < 14 or r[1] is None:
                continue
            key = norm(r[1])
            for geo, nm in NAMES.items():
                if key.startswith(nm) and geo not in found:
                    vals = [r[2 + i] for i in range(12)]
                    found[geo] = [float(v) if isinstance(v, (int, float)) else np.nan for v in vals]
        for geo, nm in NAMES.items():
            if geo not in found:
                raise SystemExit(f"{y}: {geo} bulunamadi")
            for m, v in enumerate(found[geo], start=1):
                rows.append({"geo": geo, "date": pd.Timestamp(y, m, 1), "arrivals": v})
    df = pd.DataFrame(rows)
    os.makedirs(EXT, exist_ok=True)
    df.to_csv(os.path.join(EXT, "ktb_monthly_arrivals_2016_2025.csv"), index=False)
    return df


def deseason(s):
    return s - s.groupby(s.index.month).transform("mean")


def corr(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 24 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    arr = load_arrivals()
    arr_s = {g: d.set_index("date")["arrivals"].sort_index() for g, d in arr.groupby("geo")}
    panel = load_draws()
    k = pd.read_csv(os.path.join(OUT, "s1_variability_by_keyword.csv"))[["geo", "keyword", "icc_2_1", "mean_level", "zero_share_pct"]]
    q = pd.read_csv(os.path.join(OUT, "s1d_quantization_by_keyword.csv"))[["geo", "keyword", "mean_distinct"]]

    rows = []
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        gt = sub.groupby("date")["value"].mean().sort_index()
        a = arr_s[geo]
        idx = a.index.intersection(gt.index)
        x, y = gt.loc[idx].astype(float), a.loc[idx].astype(float)
        idx2 = idx[(idx.year < 2020) | (idx.year > 2021)]
        x2, y2 = gt.loc[idx2].astype(float), a.loc[idx2].astype(float)
        rows.append({
            "geo": geo, "keyword": kw, "n_months": len(idx),
            "r_level": corr(x, y), "r_deseas": corr(deseason(x), deseason(y)),
            "r_yoy": corr(x.diff(12), y.diff(12)),
            "r_level_nocovid": corr(x2, y2), "r_deseas_nocovid": corr(deseason(x2), deseason(y2)),
        })
    v = pd.DataFrame(rows).merge(k, on=["geo", "keyword"]).merge(q, on=["geo", "keyword"])
    v.to_csv(os.path.join(OUT, "s1g_validity_by_keyword.csv"), index=False)

    # pazar duzeyi: kelime serilerinin ortalamasi (z-skor ortalamasi)
    grows = []
    for geo, sub in panel.groupby("geo"):
        zs = []
        for kw, s in sub.groupby("keyword"):
            g = s.groupby("date")["value"].mean().sort_index().astype(float)
            if g.std() > 0:
                zs.append((g - g.mean()) / g.std())
        comp = pd.concat(zs, axis=1).mean(axis=1)
        a = arr_s[geo]
        idx = a.index.intersection(comp.index)
        x, y = comp.loc[idx], a.loc[idx].astype(float)
        grows.append({"geo": geo, "r_level": corr(x, y), "r_deseas": corr(deseason(x), deseason(y)),
                      "r_yoy": corr(x.diff(12), y.diff(12))})
    g = pd.DataFrame(grows)
    icc_geo = k.groupby("geo")["icc_2_1"].mean().rename("mean_icc")
    g = g.merge(icc_geo, left_on="geo", right_index=True).sort_values("mean_icc", ascending=False)
    g.to_csv(os.path.join(OUT, "s1g_validity_by_geo.csv"), index=False)

    def rho(x, y):
        ok = x.notna() & y.notna()
        return x[ok].rank().corr(y[ok].rank())

    v["icc_class"] = pd.cut(v["icc_2_1"], [-1, 0.5, 0.9, 1.01], labels=["ICC<0.50", "0.50-0.90", "ICC>=0.90"])
    cls = v.groupby("icc_class", observed=True).agg(n=("keyword", "count"), med_r_level=("r_level", "median"),
                                                   med_r_deseas=("r_deseas", "median"), med_r_yoy=("r_yoy", "median"),
                                                   med_level_nocovid=("r_level_nocovid", "median"),
                                                   med_deseas_nocovid=("r_deseas_nocovid", "median"))
    lines = ["=" * 68, "ASAMA 1g - DIS GECERLIK (Bakanlik aylik varis sayilari, 2016-2025)", "=" * 68,
             f"Karsilastirilan ay sayisi: {int(v.n_months.median())} | kelime: {len(v)}", "",
             "Pazar duzeyi (kelime serilerinin z-skor ortalamasi ile varis):", g.round(3).to_string(index=False), "",
             "Kelime duzeyi: ICC sinifina gore medyan korelasyon", cls.round(3).to_string(), "",
             f"Spearman(ICC, r_level)   = {rho(v.icc_2_1, v.r_level):.3f}",
             f"Spearman(ICC, r_deseas)  = {rho(v.icc_2_1, v.r_deseas):.3f}",
             f"Spearman(ICC, r_yoy)     = {rho(v.icc_2_1, v.r_yoy):.3f}",
             f"Spearman(farkli deger, r_deseas) = {rho(v.mean_distinct, v.r_deseas):.3f}",
             f"COVID (2020-2021) disarida: Spearman(ICC, r_level) = {rho(v.icc_2_1, v.r_level_nocovid):.3f}, "
             f"Spearman(ICC, r_deseas) = {rho(v.icc_2_1, v.r_deseas_nocovid):.3f}", "",
             "NOT: Bu analiz nedensellik degil, ayni donemde es-hareketi olcer. COVID-19 (2020-2021)",
             "donemi dahildir. Varis sayilari milliyete gore ulusal giris sayisidir.", "=" * 68]
    with open(os.path.join(OUT, "s1g_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Yazildi: s1g_*, data/external/ktb_monthly_arrivals_2016_2025.csv")


if __name__ == "__main__":
    main()
