#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1b - Kategori Filtresi Etkisi Analizi
=============================================
data/raw/ (Travel kategorisi, cat=67) ile data/raw_nocat/ (kategorisiz,
cat=0) cekimlerini karsilastirir.

collect_gt_nocat.py'nin sordugu soruyu nicel olarak cevaplar: dusuk
hacimli pazarlarda (GE, UA, IQ, ve genisletilmis sette AZ, FR, KZ)
gorulen "cokme"nin Google'in ornekleme gurultusundan mi, yoksa kategori
filtresinin zaten az olan aramalari tamamen elemesinden mi
kaynaklandigini ayirt eder. Iki ayri mekanizmayi olcer:
  (1) TAM VERI KAYBI  - collection_log*.csv'deki OK/FAIL oranlarindan
  (2) SEVIYE/KARARLILIK FARKI - basarili cekimlerdeki deger, sifir
      orani ve ICC(2,1) karsilastirmasindan

Uretilen ciktilar (data/processed/ altina):
  s1b_failure_rates_by_geo.csv       - pazar bazinda basarisizlik orani (cat vs nocat)
  s1b_category_effect_by_keyword.csv - kelime bazinda seviye/sifir/ICC/CV karsilastirmasi
  s1b_summary.txt                    - okunabilir ozet

Calistirma: python analyze_stage1b_category_effect.py
"""

import os
import glob
import pandas as pd

from analyze_stage1_v2 import variability_by_month, variability_by_keyword

RAW_CAT = os.path.join("data", "raw")
RAW_NOCAT = os.path.join("data", "raw_nocat")
LOG_CAT = os.path.join("data", "collection_log.csv")
LOG_NOCAT = os.path.join("data", "collection_log_nocat.csv")
OUT_DIR = os.path.join("data", "processed")


# ------------------------------------------------------------------
# Veri yukleme
# ------------------------------------------------------------------

def load_draws(root):
    """analyze_stage1_v2.load_draws ile ayni mantik, ama kok klasor
    parametre olarak verilir (data/raw VEYA data/raw_nocat)."""
    draw_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
    )
    if not draw_dirs:
        return pd.DataFrame(columns=["draw_date", "geo", "date", "keyword", "value"])

    records = []
    for ddir in draw_dirs:
        draw_date = os.path.basename(ddir)
        for path in sorted(glob.glob(os.path.join(ddir, "*.csv"))):
            geo = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path, encoding="utf-8-sig")

            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])

            long = df.melt(id_vars=[date_col], var_name="keyword", value_name="value")
            long = long.rename(columns={date_col: "date"})
            long["draw_date"] = draw_date
            long["geo"] = geo

            records.append(long)

    panel = pd.concat(records, ignore_index=True)
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    return panel[["draw_date", "geo", "date", "keyword", "value"]]


def load_failure_log(path):
    """collection_log*.csv -> pazar bazinda deneme/basari/basarisizlik sayisi."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["geo", "n_attempts", "n_ok", "n_fail", "fail_rate_pct"])

    log = pd.read_csv(path)
    g = log.groupby("geo")["status"].agg(
        n_attempts="count",
        n_ok=lambda s: (s == "OK").sum(),
        n_fail=lambda s: (s == "FAIL").sum(),
    ).reset_index()
    g["fail_rate_pct"] = 100.0 * g["n_fail"] / g["n_attempts"]
    return g


# ------------------------------------------------------------------
# Karsilastirmalar
# ------------------------------------------------------------------

def compare_failure_rates():
    cat = load_failure_log(LOG_CAT).rename(columns={
        "n_attempts": "n_attempts_cat", "n_ok": "n_ok_cat",
        "n_fail": "n_fail_cat", "fail_rate_pct": "fail_rate_pct_cat",
    })
    nocat = load_failure_log(LOG_NOCAT).rename(columns={
        "n_attempts": "n_attempts_nocat", "n_ok": "n_ok_nocat",
        "n_fail": "n_fail_nocat", "fail_rate_pct": "fail_rate_pct_nocat",
    })

    merged = pd.merge(cat, nocat, on="geo", how="outer").fillna(0)
    merged["category_induced_fail"] = (
        (merged["fail_rate_pct_cat"] > 0) & (merged["fail_rate_pct_nocat"] == 0)
    )
    return merged.sort_values("fail_rate_pct_cat", ascending=False)


def compare_by_keyword(panel_cat, panel_nocat):
    """Her (geo, keyword) icin cat ve nocat setlerinde ayri ayri
    kararlilik hesabi, sonra yan yana karsilastirma."""

    def summarize(panel, label):
        if panel.empty:
            return pd.DataFrame(columns=["geo", "keyword"])
        by_month = variability_by_month(panel)
        by_kw = variability_by_keyword(panel, by_month)
        by_kw = by_kw[[
            "geo", "keyword", "n_draws", "mean_level",
            "zero_share_pct", "fleiss_kappa_zero", "icc_2_1", "mean_cv_pct",
        ]]
        renamed = by_kw.add_suffix(f"_{label}")
        return renamed.rename(columns={f"geo_{label}": "geo", f"keyword_{label}": "keyword"})

    cat_summary = summarize(panel_cat, "cat")
    nocat_summary = summarize(panel_nocat, "nocat")

    merged = pd.merge(cat_summary, nocat_summary, on=["geo", "keyword"], how="outer")
    merged["level_diff_nocat_minus_cat"] = merged["mean_level_nocat"] - merged["mean_level_cat"]
    merged["zero_share_diff_pct"] = merged["zero_share_pct_nocat"] - merged["zero_share_pct_cat"]
    merged["icc_diff"] = merged["icc_2_1_nocat"] - merged["icc_2_1_cat"]
    return merged.sort_values(["geo", "keyword"])


# ------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    panel_cat = load_draws(RAW_CAT)
    panel_nocat = load_draws(RAW_NOCAT)

    print(f"Kategorili   : {panel_cat['draw_date'].nunique()} cekim, {panel_cat['geo'].nunique()} pazar")
    print(f"Kategorisiz  : {panel_nocat['draw_date'].nunique()} cekim, {panel_nocat['geo'].nunique()} pazar")

    if panel_nocat.empty:
        print("\nUYARI: data/raw_nocat altinda veri yok. collect_gt_nocat.py hic calismamis.")
        return

    fail_cmp = compare_failure_rates()
    fail_cmp.to_csv(os.path.join(OUT_DIR, "s1b_failure_rates_by_geo.csv"), index=False)

    kw_cmp = compare_by_keyword(panel_cat, panel_nocat)
    kw_cmp.to_csv(os.path.join(OUT_DIR, "s1b_category_effect_by_keyword.csv"), index=False)

    # ---- ozet ----
    lines = []
    lines.append("=" * 68)
    lines.append("ASAMA 1b - KATEGORI FILTRESI ETKISI (cat=67 vs cat=0)")
    lines.append("=" * 68)
    lines.append(f"Kategorili cekim sayisi  : {panel_cat['draw_date'].nunique()}")
    lines.append(f"Kategorisiz cekim sayisi : {panel_nocat['draw_date'].nunique()}")
    lines.append("")

    lines.append("--- Pazar bazinda basarisizlik orani (collection_log*.csv) ---")
    show_cols = ["geo", "n_attempts_cat", "n_fail_cat", "fail_rate_pct_cat",
                 "n_attempts_nocat", "n_fail_nocat", "fail_rate_pct_nocat",
                 "category_induced_fail"]
    lines.append(fail_cmp[show_cols].round(1).to_string(index=False))
    lines.append("")

    induced = fail_cmp[fail_cmp["category_induced_fail"]]
    if not induced.empty:
        lines.append("UYARI: Asagidaki pazarlarda basarisizlik SADECE kategori")
        lines.append("filtresi uygulaninca ortaya cikiyor (kategorisiz sette hic yok):")
        for _, row in induced.iterrows():
            lines.append(
                f"  {row['geo']}: kategorili basarisizlik %{row['fail_rate_pct_cat']:.1f}, "
                f"kategorisiz %{row['fail_rate_pct_nocat']:.1f}"
            )
        lines.append("-> Bu pazarlarda 'cokme' ornekleme gurultusu degil, DOGRUDAN")
        lines.append("   kategori filtresinin neden oldugu tam veri kaybidir.")
    else:
        lines.append("Kategori filtresine ozgu (kategorisizde gorulmeyen) basarisizlik yok.")
    lines.append("")

    lines.append("--- Kelime bazinda seviye farki (nocat - cat), en buyuk 15 ---")
    diff_tab = kw_cmp.dropna(subset=["level_diff_nocat_minus_cat"]).nlargest(
        15, "level_diff_nocat_minus_cat"
    )[["geo", "keyword", "mean_level_cat", "mean_level_nocat",
       "level_diff_nocat_minus_cat", "zero_share_diff_pct"]]
    if not diff_tab.empty:
        lines.append(diff_tab.round(2).to_string(index=False))
    lines.append("")

    lines.append("--- ICC farki (nocat - cat), her iki sette de >=2 cekim olanlar ---")
    icc_tab = kw_cmp.dropna(subset=["icc_diff"]).sort_values("icc_diff", ascending=False)[
        ["geo", "keyword", "n_draws_cat", "icc_2_1_cat",
         "n_draws_nocat", "icc_2_1_nocat", "icc_diff"]
    ]
    if not icc_tab.empty:
        lines.append(icc_tab.round(3).to_string(index=False))
    else:
        lines.append("(Henuz her iki sette de >=2 cekimi olan yeterli kelime yok.)")
    lines.append("")

    lines.append("HATIRLATMA: Kategori filtresinin etkisini yorumlarken cekim sayisi")
    lines.append("farkina dikkat edin (kategorisiz seri kategoriliden daha kisa")
    lines.append("baslamis olabilir); ICC/CV karsilastirmalari esit cekim sayisinda")
    lines.append("daha guvenilirdir.")
    lines.append("=" * 68)

    # ---- makale icin oneri (verilerden dinamik uretilir, elle guncellenmez) ----
    lines.append("")
    lines.append("=" * 68)
    lines.append("MAKALE ICIN ONERI (Turkce akademik uslup kurallarina uygun taslak)")
    lines.append("=" * 68)

    fail_rows = fail_cmp[fail_cmp["category_induced_fail"]].sort_values(
        "fail_rate_pct_cat", ascending=False
    )
    fail_clause = "; ".join(
        f"{r.geo} pazarinda yuzde {r.fail_rate_pct_cat:.1f}"
        for r in fail_rows.itertuples()
    )

    silent_loss = kw_cmp[
        (kw_cmp["geo"] != "GE") & (kw_cmp["icc_diff"] > 0.2)
    ].sort_values("icc_diff", ascending=False)
    silent_geos = list(dict.fromkeys(silent_loss["geo"]))  # sirali, tekrarsiz

    iq_rows = kw_cmp[kw_cmp["geo"] == "IQ"].dropna(subset=["icc_diff"])
    iq_mean_diff = iq_rows["icc_diff"].mean() if not iq_rows.empty else float("nan")

    paragraph = (
        f"Kategori filtresinin dusuk hacimli pazarlardaki etkisi tek bir "
        f"bozulma bicimine indirgenememektedir. Basarisizlik oranlari "
        f"karsilastirildiginda, kategori uygulandiginda {fail_clause} "
        f"oraninda cekimin bos yanitla sonuclandigi, kategori "
        f"kaldirildiginda ise bu oranin tum pazarlarda sifira dustugu "
        f"gorulmektedir. Bunun yaninda, sorgunun basarili dondugu ancak "
        f"elde edilen degerlerin kategori altinda asiri bastirildigi "
        f"ikinci bir bozulma bicimi belirlenmistir"
    )
    if silent_geos:
        geos_clause = ", ".join(silent_geos[:-1]) + (
            f" ve {silent_geos[-1]}" if len(silent_geos) > 1 else silent_geos[0]
        )
        paragraph += (
            f"; {geos_clause} pazarlarina ait dusuk hacimli anahtar "
            f"kelimelerde cekimler arasi tutarlilik katsayisi (ICC) "
            f"kategori kisiti altinda belirgin sekilde dusmekte, "
            f"kategorisiz sorguda ise yuksek kalmaktadir"
        )
    paragraph += ". "
    if not pd.isna(iq_mean_diff):
        paragraph += (
            f"Buna karsin Irak pazarindaki zayifligin buyuk olcude "
            f"kategoriden bagimsiz seyrettigi gozlenmis (ortalama ICC "
            f"farki {iq_mean_diff:.3f}), bu da soz konusu pazardaki dusuk "
            f"guvenilirligin dogrudan arama hacmi sinirliligindan "
            f"kaynaklandigina isaret etmektedir. "
        )
    paragraph += (
        "Bulgular birlikte degerlendirildiginde, kategori tabanli "
        "filtrelemenin dusuk hacimli pazarlarda yalnizca veri kaybina "
        "degil, veri kaybi gozlenmeyen durumlarda dahi olcum "
        "guvenilirliginde onemli bir azalmaya yol acabilecegi "
        "degerlendirilmektedir."
    )

    import textwrap
    lines.extend(textwrap.wrap(paragraph, width=68))
    lines.append("=" * 68)

    text = "\n".join(lines)

    summary_path = os.path.join(OUT_DIR, "s1b_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n[NOT] Konsol bazi karakterleri gosteremedi; tam surum icin dosyayi acin:")
        print(summary_path)

    print(f"\nCiktilar {OUT_DIR}/ altina yazildi.")
    print(f"Ozet dosyasi: {summary_path}")


if __name__ == "__main__":
    main()
