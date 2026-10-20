#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1e - Kac Cekimin Ortalamasi Gerekli? (bolunmus-yari guvenilirlik)
=====================================================================
Her (pazar, kelime) icin cekimler rastgele iki ayri gruba bolunur (her biri k
cekim), her grubun aylik ortalamasi alinir ve iki ortalama serisi arasindaki
ICC(2,1) hesaplanir. 50 rastgele bolme uzerinden ortalama alinir. k=1, tek
cekimin tekrar cekilebilirligidir; k arttikca ortalamanin kararliligi gorulur.
Ek olarak tek cekim ICC'sinden Spearman-Brown ile kuramsal k hesaplanir.

Ciktilar (data/processed/): s1e_split_half_by_keyword.csv, s1e_split_half_by_geo.csv,
                            s1e_summary.txt
Calistirma: python analyze_stage1e_averaging.py
"""

import os
import numpy as np
import pandas as pd

from analyze_stage1_v2 import load_draws, icc21

OUT_DIR = os.path.join("data", "processed")
KS = [1, 2, 3, 5, 8, 12, 16]
N_SPLIT = 50
SEED = 20261019


def main():
    panel = load_draws()
    rng = np.random.default_rng(SEED)
    rows = []
    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wide = sub.pivot_table(index="date", columns="draw_date", values="value").sort_index()
        m = wide.values
        n = m.shape[1]
        for k in KS:
            if 2 * k > n:
                continue
            vals = []
            for _ in range(N_SPLIT):
                perm = rng.permutation(n)
                a = m[:, perm[:k]].mean(axis=1)
                b = m[:, perm[k:2 * k]].mean(axis=1)
                vals.append(icc21(np.column_stack([a, b])))
            rows.append({"geo": geo, "keyword": kw, "k": k, "n_pulls": n,
                         "split_half_icc": np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan})
    by_kw = pd.DataFrame(rows)
    by_kw.to_csv(os.path.join(OUT_DIR, "s1e_split_half_by_keyword.csv"), index=False)

    by_geo = by_kw.groupby(["geo", "k"])["split_half_icc"].mean().unstack("k")
    by_geo = by_geo.sort_values(1, ascending=False)
    by_geo.to_csv(os.path.join(OUT_DIR, "s1e_split_half_by_geo.csv"))

    one = by_kw[by_kw["k"] == 1].set_index(["geo", "keyword"])["split_half_icc"]
    lines = ["=" * 68, "ASAMA 1e - ORTALAMA ALINAN CEKIM SAYISINA GORE BOLUNMUS-YARI ICC", "=" * 68,
             f"Rastgele bolme sayisi: {N_SPLIT} | seed: {SEED}",
             "Hucre: pazardaki kelimelerin ortalama ICC'si (iki ayri k-cekimlik ortalama arasinda).",
             "Bos hucre: o pazarda 2k cekim yok.", "",
             by_geo.round(3).to_string(), "",
             "Kelime duzeyinde: k=1'de ICC<0.50 olan kelimelerin k=16 (veya mevcut en buyuk k) ortalamasi:"]
    low = one[one < 0.5].index
    kmax = by_kw.groupby(["geo", "keyword"])["k"].max()
    last = by_kw.merge(kmax.rename("kmax").reset_index(), on=["geo", "keyword"])
    last = last[last["k"] == last["kmax"]].set_index(["geo", "keyword"])["split_half_icc"]
    cmp = pd.DataFrame({"icc_k1": one.loc[low], "icc_kmax": last.loc[low], "kmax": kmax.loc[low]})
    lines.append(cmp.round(3).to_string())
    lines.append("")
    lines.append(f"k=1'de ICC<0.50 olan {len(cmp)} kelimeden {(cmp['icc_kmax'] >= 0.75).sum()} tanesi "
                 f"en buyuk k'de >=0.75'e, {(cmp['icc_kmax'] >= 0.50).sum()} tanesi >=0.50'ye ulasti.")
    lines.append("=" * 68)
    with open(os.path.join(OUT_DIR, "s1e_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Yazildi: s1e_*.csv, s1e_summary.txt")


if __name__ == "__main__":
    main()
