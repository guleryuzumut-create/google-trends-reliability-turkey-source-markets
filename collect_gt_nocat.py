#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARALEL CEKIM - KATEGORI KISITI OLMADAN
========================================
Ana toplama ile AYNI anahtar kelimeleri, AYNI ulkeleri ve AYNI donemi
kullanir; tek fark kategori kisitinin kaldirilmis olmasidir (cat=0).

AMAC: Dusuk hacimli pazarlarda (GE, UA, IQ) serilerin cokmesinin sebebi
Google'in ornekleme gurultusu mu, yoksa Travel kategorisi filtresinin
zaten az olan aramalari tamamen elemesi mi? Bu iki aciklamayi ayirmak
icin kategorisiz bir kontrol serisi gerekiyor.

Anahtar kelimeler collect_gt.py'den ICE AKTARILIR, kopyalanmaz.
Boylece iki set arasindaki TEK farkin kategori oldugu garanti edilir.

Cikti:  data/raw_nocat/YYYY-MM-DD/<ulke>.csv
Log:    data/collection_log_nocat.csv

Calistirma: python collect_gt_nocat.py
"""

import os
import csv
import time
import random
import logging
from datetime import datetime, timezone

from pytrends.request import TrendReq

# Ana script'ten ice aktarim - kelimeler ve donem BIREBIR ayni olsun diye
from collect_gt import (MARKETS, TIMEFRAME, MAX_RETRY, BASE_SLEEP, fetch_market,
                        RETRY_ROUNDS, ROUND_WAIT)

CATEGORY_NOCAT = 0     # 0 = kategori kisiti yok (ana script'te 67 = Travel)

OUT_ROOT = os.path.join("data", "raw_nocat")
LOG_CSV = os.path.join("data", "collection_log_nocat.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | NOCAT | %(message)s",
    handlers=[
        logging.FileHandler("collect_gt_nocat.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,  # collect_gt import'u kendi basicConfig()'ini zaten calistirdi;
                 # force olmadan bu cagri sessizce yok sayilir ve NOCAT loglari
                 # yanlislikla collect_gt.log'a karisir.
)
log = logging.getLogger(__name__)


def fetch_nocat(pytrends, geo, keywords):
    """fetch_market ile ayni mantik, ancak cat=0 ile."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            pytrends.build_payload(
                kw_list=keywords,
                cat=CATEGORY_NOCAT,
                timeframe=TIMEFRAME,
                geo=geo,
                gprop="",
            )
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                log.warning("[%s] Bos veri dondu (deneme %d/%d)", geo, attempt, MAX_RETRY)
            else:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                log.info("[%s] Basarili - %d satir", geo, len(df))
                return df

        except Exception as exc:
            log.warning("[%s] Hata (deneme %d/%d): %s", geo, attempt, MAX_RETRY, exc)

        wait = (2 ** attempt) + random.uniform(0, 4)
        time.sleep(wait)

    log.error("[%s] BASARISIZ", geo)
    return None


def append_log(row):
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    new_file = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "draw_date_utc", "run_timestamp_utc", "geo",
                "status", "n_rows", "n_keywords", "timeframe", "category",
            ])
        w.writerow(row)


def main():
    now_utc = datetime.now(timezone.utc)
    draw_date = now_utc.strftime("%Y-%m-%d")
    run_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir = os.path.join(OUT_ROOT, draw_date)

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        log.warning("Bugunun kategorisiz cekimi zaten yapilmis. Cikiliyor.")
        return

    os.makedirs(out_dir, exist_ok=True)
    log.info("=" * 60)
    log.info("KATEGORISIZ CEKIM - UTC tarih: %s | cat=%d", draw_date, CATEGORY_NOCAT)
    log.info("Donem: %s (ana script ile ayni)", TIMEFRAME)
    log.info("=" * 60)

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

    ok = 0
    pending = list(MARKETS)

    # Ana script ile ayni ek tur mantigi: FAIL yalnizca tum turlar bitince yazilir.
    for rnd in range(RETRY_ROUNDS + 1):
        if rnd > 0:
            if not pending:
                break
            log.info("=" * 60)
            log.info("EK TUR %d/%d - %s icin %d sn bekleniyor, yeni oturum acilacak",
                     rnd, RETRY_ROUNDS, ", ".join(pending), ROUND_WAIT)
            time.sleep(ROUND_WAIT)
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

        still_failed = []
        for geo in pending:
            keywords = MARKETS[geo]
            log.info("--- %s isleniyor (%d kelime) ---", geo, len(keywords))

            df = fetch_nocat(pytrends, geo, keywords)

            if df is not None:
                df.to_csv(os.path.join(out_dir, f"{geo}.csv"), encoding="utf-8-sig")
                append_log([draw_date, run_ts, geo, "OK", len(df),
                            len(keywords), TIMEFRAME, CATEGORY_NOCAT])
                ok += 1
            else:
                still_failed.append(geo)

            time.sleep(BASE_SLEEP + random.uniform(0, 5))
        pending = still_failed

    for geo in pending:
        append_log([draw_date, run_ts, geo, "FAIL", 0,
                    len(MARKETS[geo]), TIMEFRAME, CATEGORY_NOCAT])
    fail = len(pending)

    log.info("=" * 60)
    log.info("BITTI - Basarili: %d | Basarisiz: %d", ok, fail)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
