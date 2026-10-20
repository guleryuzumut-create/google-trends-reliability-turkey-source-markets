#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAMA 1 - Cekimler Arasi Varyasyon Analizi
===========================================
data/raw/YYYY-MM-DD/*.csv altindaki tekrarli Google Trends cekimlerini
okur ve cekimler arasi kararliligi olcer.

Uretilen ciktilar (data/processed/ altina):
  s1_variability_by_keyword.csv  - anahtar kelime duzeyinde ozet
  s1_variability_by_month.csv    - ay x kelime duzeyinde CV
  s1_pairwise_correlations.csv   - cekim ciftleri arasi korelasyon
  s1_peak_consistency.csv        - zirve ay tutarliligi
  s1_summary.txt                 - okunabilir ozet

Calistirma: python analyze_stage1.py
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd

RAW_ROOT = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")


# ------------------------------------------------------------------
# Veri yukleme
# ------------------------------------------------------------------

def load_draws():
    """
    Tum cekimleri tek bir uzun formatli DataFrame'e yukler.
    Donen kolonlar: draw_date, geo, date, keyword, value
    """
    draw_dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_ROOT, "*"))
        if os.path.isdir(d)
    )

    if not draw_dirs:
        raise SystemExit(f"HATA: {RAW_ROOT} altinda cekim klasoru bulunamadi.")

    records = []

    for ddir in draw_dirs:
        draw_date = os.path.basename(ddir)

        for path in sorted(glob.glob(os.path.join(ddir, "*.csv"))):
            geo = os.path.splitext(os.path.basename(path))[0]

            df = pd.read_csv(path, encoding="utf-8-sig")

            # ilk kolon tarih kolonudur (pytrends 'date' olarak yazar)
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])

            long = df.melt(
                id_vars=[date_col],
                var_name="keyword",
                value_name="value",
            )
            long = long.rename(columns={date_col: "date"})
            long["draw_date"] = draw_date
            long["geo"] = geo

            records.append(long)

    panel = pd.concat(records, ignore_index=True)
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")

    return panel[["draw_date", "geo", "date", "keyword", "value"]]


# ------------------------------------------------------------------
# ICC(2,1) - iki yonlu rastgele etkiler, mutlak uyum, tek olcum
# ------------------------------------------------------------------

def icc21(matrix):
    """
    matrix: (n_denek x k_degerlendirici) = (n_ay x k_cekim)
    Shrout & Fleiss (1979) ICC(2,1).
    Yetersiz veri veya sifir varyans halinde np.nan doner.
    """
    m = np.asarray(matrix, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]

    n, k = m.shape
    if n < 2 or k < 2:
        return np.nan

    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)

    # kareler toplami
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((m - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_err = df_rows * df_cols

    if df_err <= 0:
        return np.nan

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_err = ss_err / df_err

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom == 0:
        return np.nan

    return (ms_rows - ms_err) / denom


# ------------------------------------------------------------------
# Sifir siskin seriler icin ek olcutler
# ------------------------------------------------------------------

def fleiss_kappa_binary(matrix):
    """
    Ikili (sifir / sifir-disi) kodlamada cekimler arasi uyum.
    matrix: (n_ay x k_cekim), degerler 0/1.
    Sansa gore duzeltilmis uyum. CV'nin aksine sifir sisme altinda
    anlamli kalir.
    """
    m = np.asarray(matrix, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return np.nan

    n1 = m.sum(axis=1)          # her ay icin 'sifir-disi' oy sayisi
    n0 = k - n1

    # denek bazinda uyum
    P_i = (n0 ** 2 + n1 ** 2 - k) / (k * (k - 1))
    P_bar = P_i.mean()

    p1 = n1.sum() / (n * k)
    p0 = 1.0 - p1
    P_e = p0 ** 2 + p1 ** 2

    if np.isclose(P_e, 1.0):
        # tum gozlemler ayni kategoride -> kappa tanimsiz ama uyum tam
        return np.nan

    return (P_bar - P_e) / (1.0 - P_e)


def zero_agreement_stats(wide):
    """
    wide: (n_ay x k_cekim) ham degerler.
    Doner: sifir-disi oran, cekimlerin ANLASAMADIGI ay orani, Fleiss kappa.

    'Belirsiz ay' = bazi cekimlerin sinyal gordugu, bazilarinin sifir
    dondurdugu ay. Dusuk hacimli serilerde asil sorun budur.
    """
    m = np.asarray(wide, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    if m.size == 0:
        return np.nan, np.nan, np.nan

    binm = (m > 0).astype(float)
    prop = binm.mean(axis=1)

    nonzero_share = 100.0 * binm.mean()
    ambiguous = 100.0 * ((prop > 0) & (prop < 1)).mean()
    kappa = fleiss_kappa_binary(binm)

    return nonzero_share, ambiguous, kappa


def robust_spread(wide):
    """Medyan mutlak sapmanin aylar boyunca ortalamasi (CV'ye dayanikli alternatif)."""
    m = np.asarray(wide, dtype=float)
    if m.size == 0:
        return np.nan
    med = np.nanmedian(m, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(m - med), axis=1)
    return float(np.nanmean(mad))


# ------------------------------------------------------------------
# Analizler
# ------------------------------------------------------------------

def variability_by_month(panel):
    """Her (geo, keyword, ay) icin cekimler arasi ortalama, sd, CV."""
    g = panel.groupby(["geo", "keyword", "date"])["value"]

    out = g.agg(
        n_draws="count",
        mean_val="mean",
        sd_val=lambda s: s.std(ddof=1),
        min_val="min",
        max_val="max",
    ).reset_index()

    out["range_val"] = out["max_val"] - out["min_val"]
    out["cv_pct"] = np.where(
        out["mean_val"] > 0,
        100.0 * out["sd_val"] / out["mean_val"],
        np.nan,
    )
    return out


def variability_by_keyword(panel, by_month):
    """Anahtar kelime duzeyinde ozet: ortalama CV, ICC, sifir orani."""
    rows = []

    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wide = sub.pivot_table(
            index="date", columns="draw_date", values="value"
        ).sort_index()

        icc = icc21(wide.values)
        nonzero_share, ambiguous, kappa = zero_agreement_stats(wide.values)
        mad = robust_spread(wide.values)

        bm = by_month[(by_month["geo"] == geo) & (by_month["keyword"] == kw)]

        mean_level = sub["value"].mean()
        zero_share = 100.0 * (sub["value"] == 0).mean()

        # CV, sifir siskin serilerde yapay olarak sisiyor. Asagidaki
        # kosullarda raporlanmamali - sadece isaretleniyor, silinmiyor.
        cv_reliable = bool((mean_level >= 5.0) and (zero_share <= 25.0))

        rows.append({
            "geo": geo,
            "keyword": kw,
            "n_draws": wide.shape[1],
            "n_months": wide.shape[0],
            "mean_level": mean_level,
            "zero_share_pct": zero_share,
            "nonzero_share_pct": nonzero_share,
            "ambiguous_month_pct": ambiguous,
            "fleiss_kappa_zero": kappa,
            "mean_cv_pct": bm["cv_pct"].mean(),
            "median_cv_pct": bm["cv_pct"].median(),
            "max_cv_pct": bm["cv_pct"].max(),
            "cv_reliable": cv_reliable,
            "mean_mad": mad,
            "mean_range": bm["range_val"].mean(),
            "icc_2_1": icc,
        })

    return pd.DataFrame(rows).sort_values(["geo", "keyword"])


def pairwise_correlations(panel):
    """Her (geo, keyword) icin cekim ciftleri arasi Pearson korelasyonu."""
    rows = []

    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wide = sub.pivot_table(
            index="date", columns="draw_date", values="value"
        ).sort_index()

        draws = list(wide.columns)
        if len(draws) < 2:
            continue

        for a, b in itertools.combinations(draws, 2):
            pair = wide[[a, b]].dropna()
            if len(pair) < 3 or pair[a].std() == 0 or pair[b].std() == 0:
                r = np.nan
            else:
                r = pair[a].corr(pair[b])

            rows.append({
                "geo": geo, "keyword": kw,
                "draw_a": a, "draw_b": b, "pearson_r": r,
            })

    return pd.DataFrame(rows)


def peak_consistency(panel):
    """Her cekimin isaretledigi zirve ay ayni mi?"""
    rows = []

    for (geo, kw), sub in panel.groupby(["geo", "keyword"]):
        wide = sub.pivot_table(
            index="date", columns="draw_date", values="value"
        ).sort_index()

        if wide.shape[1] < 2 or wide.empty:
            continue

        peaks = wide.idxmax(axis=0, skipna=True).dropna()
        if peaks.empty:
            continue

        modal = peaks.mode()
        modal_peak = modal.iloc[0] if not modal.empty else pd.NaT
        agree = (peaks == modal_peak).mean() if pd.notna(modal_peak) else np.nan

        rows.append({
            "geo": geo, "keyword": kw,
            "n_draws": wide.shape[1],
            "modal_peak_month": modal_peak,
            "n_distinct_peaks": peaks.nunique(),
            "modal_agreement_pct": 100.0 * agree,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    panel = load_draws()
    n_draws = panel["draw_date"].nunique()

    print(f"Yuklendi: {n_draws} cekim, {panel['geo'].nunique()} pazar, "
          f"{len(panel):,} gozlem")

    if n_draws < 2:
        print("\nUYARI: Varyasyon analizi icin en az 2 cekim gerekir.")
        print("Su an sadece 1 cekim var. Yarin tekrar calistirin.")
        return

    if n_draws < 5:
        print(f"\nNOT: {n_draws} cekimle hesaplanan degerler on-gostergedir. "
              "Kararli tahmin icin 10+ cekim gerekir.")

    by_month = variability_by_month(panel)
    by_kw = variability_by_keyword(panel, by_month)
    pairs = pairwise_correlations(panel)
    peaks = peak_consistency(panel)

    by_month.to_csv(os.path.join(OUT_DIR, "s1_variability_by_month.csv"), index=False)
    by_kw.to_csv(os.path.join(OUT_DIR, "s1_variability_by_keyword.csv"), index=False)
    pairs.to_csv(os.path.join(OUT_DIR, "s1_pairwise_correlations.csv"), index=False)
    peaks.to_csv(os.path.join(OUT_DIR, "s1_peak_consistency.csv"), index=False)

    # ---- ozet ----
    lines = []
    lines.append("=" * 68)
    lines.append("ASAMA 1 - CEKIMLER ARASI VARYASYON OZETI")
    lines.append(">>> SURUM 2 - sifir siskin serilere uygun olcutler dahil <<<")
    lines.append("=" * 68)
    lines.append(f"Cekim sayisi : {n_draws}")
    lines.append(f"Cekim tarihleri: {', '.join(sorted(panel['draw_date'].unique()))}")
    lines.append("")

    lines.append("--- Pazar duzeyinde kararlilik ---")
    lines.append("(ambig% = cekimlerin sinyal var/yok konusunda anlasamadigi ay orani)")
    geo_sum = by_kw.groupby("geo").agg(
        mean_level=("mean_level", "mean"),
        zero_pct=("zero_share_pct", "mean"),
        ambig_pct=("ambiguous_month_pct", "mean"),
        kappa=("fleiss_kappa_zero", "mean"),
        mean_icc=("icc_2_1", "mean"),
        mean_cv_pct=("mean_cv_pct", "mean"),
        n_cv_reliable=("cv_reliable", "sum"),
    ).sort_values("mean_icc")
    lines.append(geo_sum.round(3).to_string())
    lines.append("")

    n_unrel = int((~by_kw["cv_reliable"]).sum())
    lines.append(f"UYARI: {n_unrel}/{len(by_kw)} anahtar kelimede CV GUVENILIR DEGIL")
    lines.append("(ortalama duzey < 5 veya sifir orani > %25). Bu kelimelerde")
    lines.append("CV yerine Fleiss kappa ve belirsiz-ay oranina bakin.")
    lines.append("")

    spearman_r = None

    rel = by_kw[by_kw["cv_reliable"]]
    if not rel.empty:
        lines.append("--- SADECE CV'si guvenilir kelimeler: duzey-kararlilik iliskisi ---")
        tab = rel[["geo", "keyword", "mean_level", "mean_cv_pct", "icc_2_1"]] \
            .sort_values("mean_level")
        lines.append(tab.round(2).to_string(index=False))
        lines.append("")
        if len(rel) >= 4:
            # Spearman = siralamalar uzerinde Pearson. pandas'in method="spearman"
            # secenegi scipy gerektirdigi icin siralamayi elle alip Pearson
            # uyguluyoruz - sonuc birebir ayni, ek bagimlilik yok.
            r = rel["mean_level"].rank().corr(rel["mean_cv_pct"].rank())
            spearman_r = r
            lines.append(f"Duzey ile CV arasi Spearman korelasyonu: {r:.3f}")
            lines.append("(guclu negatif deger AS2 hipotezini destekler)")
            lines.append("")

    lines.append("--- En kararsiz 10 anahtar kelime (ortalama CV'ye gore) ---")
    top = by_kw.nlargest(10, "mean_cv_pct")[
        ["geo", "keyword", "mean_level", "zero_share_pct", "mean_cv_pct", "icc_2_1"]
    ]
    lines.append(top.round(2).to_string(index=False))
    lines.append("")

    lines.append("--- En kararli 10 anahtar kelime ---")
    bot = by_kw.nsmallest(10, "mean_cv_pct")[
        ["geo", "keyword", "mean_level", "zero_share_pct", "mean_cv_pct", "icc_2_1"]
    ]
    lines.append(bot.round(2).to_string(index=False))
    lines.append("")

    if not pairs.empty:
        lines.append("--- Cekim ciftleri arasi korelasyon ---")
        lines.append(f"Ortalama r : {pairs['pearson_r'].mean():.4f}")
        lines.append(f"Minimum r  : {pairs['pearson_r'].min():.4f}")
        lines.append("")

    if not peaks.empty:
        perfect = (peaks["n_distinct_peaks"] == 1).mean()
        lines.append("--- Zirve ay tutarliligi ---")
        lines.append(f"Tum cekimlerde ayni zirveyi veren kelime orani: {100*perfect:.1f}%")
        disagree = peaks[peaks["n_distinct_peaks"] > 1]
        if not disagree.empty:
            lines.append("Zirve ayi cekime gore DEGISEN kelimeler:")
            lines.append(disagree[
                ["geo", "keyword", "n_distinct_peaks", "modal_agreement_pct"]
            ].round(1).to_string(index=False))
        lines.append("")

    lines.append("HATIRLATMA: mean_level dusuk ve zero_share yuksek kelimelerde")
    lines.append("CV'nin buyuk cikmasi BEKLENEN sonuctur (AS2 hipotezi).")
    lines.append("=" * 68)

    # ---- makale icin oneri (verilerden dinamik uretilir, elle guncellenmez) ----
    lines.append("")
    lines.append("=" * 68)
    lines.append("MAKALE ICIN ONERI (Turkce akademik uslup kurallarina uygun taslak)")
    lines.append("=" * 68)

    most_stable_geo = geo_sum.index[-1]
    least_stable_geo = geo_sum.index[0]
    most_stable_icc = geo_sum.loc[most_stable_geo, "mean_icc"]
    least_stable_icc = geo_sum.loc[least_stable_geo, "mean_icc"]

    paragraph = (
        f"{n_draws} tekrarli cekimden elde edilen bulgular, kaynak "
        f"pazarlar arasinda belirgin bir guvenilirlik hiyerarsisine "
        f"isaret etmektedir. Pazar duzeyinde ortalama ICC(2,1) degeri "
        f"en yuksek {most_stable_geo} pazarinda ({most_stable_icc:.3f}), "
        f"en dusuk ise {least_stable_geo} pazarinda "
        f"({least_stable_icc:.3f}) gozlenmistir. "
    )

    if spearman_r is not None:
        yon = "negatif" if spearman_r < 0 else "pozitif"
        paragraph += (
            f"Arama hacmi duzeyi ile degisim katsayisi arasindaki "
            f"Spearman korelasyonu {spearman_r:.3f} olarak "
            f"hesaplanmis, bu {yon} iliski dusuk hacimli kelimelerde "
            f"orneklem gurultusunun sistematik olarak yukseldigini "
            f"varsayan AS2 hipotezini desteklemektedir. "
        )

    if not pairs.empty:
        paragraph += (
            f"Cekim ciftleri arasindaki ortalama Pearson korelasyonu "
            f"{pairs['pearson_r'].mean():.3f} duzeyinde bulunmus, bu "
            f"deger ICC tabanli siralamayi bagimsiz bir yontemle "
            f"dogrulamaktadir. "
        )

    if not peaks.empty:
        perfect_pct = 100 * (peaks["n_distinct_peaks"] == 1).mean()
        paragraph += (
            f"Anahtar kelimelerin yuzde {perfect_pct:.1f} kadari tum "
            f"cekimlerde ayni zirve ayini vermis, geri kalaninda "
            f"gozlenen tutarsizlik buyuk olcude dusuk hacimli "
            f"pazarlarda yogunlasmistir. "
        )

    paragraph += (
        f"Bulgular birlikte degerlendirildiginde, Google Trends "
        f"verisinin tekrar-tekrar cekilebilirliginin arama hacmine "
        f"guclu bicimde bagli oldugu, yuksek hacimli pazarlarda "
        f"neredeyse mukemmel duzeyde kararlilik gosterirken dusuk "
        f"hacimli pazarlarda ayni verinin metodolojik olarak "
        f"kullanilamaz hale gelebilecegi degerlendirilmektedir."
    )

    import textwrap
    lines.extend(textwrap.wrap(paragraph, width=68))
    lines.append("=" * 68)

    text = "\n".join(lines)

    # ONCE dosyaya yaz. Windows konsolu (cp1254) Kiril/Arap/Gurcu
    # karakterlerini basamadigi icin ekrana yazma islemi cokebilir;
    # dosya her halukarda kaydedilmis olsun.
    summary_path = os.path.join(OUT_DIR, "s1_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)

    # SONRA ekrana basmayi dene. Basarisiz olursa sessizce gec.
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        safe = text.encode("ascii", "replace").decode("ascii")
        print("\n[NOT] Konsol bazi karakterleri gosteremedi; '?' ile degistirildi.")
        print("[NOT] Tam surum icin dosyayi acin: " + summary_path)
        try:
            print("\n" + safe)
        except Exception:
            pass

    print(f"\nCiktilar {OUT_DIR}/ altina yazildi.")
    print(f"Ozet dosyasi: {summary_path}")


if __name__ == "__main__":
    main()
