#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends Tekrarli Cekim Toplayicisi
=========================================
Turkiye'nin kaynak pazarlari icin GT serilerini HER GUN BIR KEZ ceker
ve tarih damgali klasorlere kaydeder. 30 gun boyunca calistirilmalidir.

Calistirma:  python collect_gt.py
Cikti:       data/raw/YYYY-MM-DD/<ulke>.csv  +  data/collection_log.csv

ONEMLI: Google, GT verisini 24 saat onbellekte tutar ve UTC gece yarisi
yeni ornek ceker. Bu yuzden gunde BIR kez calistirmak yeterlidir; ayni
gun icinde tekrar calistirmak ayni veriyi verir (bilimsel olarak
yeni bir cekim SAYILMAZ).
"""

import os
import csv
import time
import random
import logging
from datetime import datetime, timezone

import pandas as pd
from pytrends.request import TrendReq

# ============================================================
# YAPILANDIRMA
# ============================================================

# Analiz donemi. Baslangici sabit tutun; bitisi her ay guncelleyin
# ya da tum cekimler boyunca SABIT birakin (tercih edilen: sabit).
TIMEFRAME = "2016-01-01 2026-07-31"

CATEGORY = 67          # Google Trends "Travel" kategorisi
GPROP = ""             # bos = web aramasi
MAX_RETRY = 5
BASE_SLEEP = 20        # sorgular arasi bekleme (saniye) - 429 hatasini onler
                       # 2026-09-10: 8sn'den yukseltildi - gunluk sorgu sayisi
                       # 15 pazar x 2 script (cat+nocat) = 30'a ciktiktan
                       # sonra 429 hatalari artti (IR ilk kez basarisiz oldu)
RETRY_ROUNDS = 3      # ilk turda basarisiz olan pazarlar icin ek tur sayisi
ROUND_WAIT = 180      # ek turdan once bekleme (saniye)
                       # 2026-09-24: GE ayni oturumda 5 denemede de bos dondu,
                       # saatler sonra YENI oturumla ilk denemede geldi. Hizli
                       # tekrar yetmiyor; bekleyip yeni oturumla denemek gerekiyor.

# Kaynak pazarlar ve kendi dillerindeki anahtar kelimeler.
# NOT: Tek sorguda en fazla 5 kelime olabilir. Ayni sorgudaki kelimeler
# ayni olcege normalize edilir - bu bizim icin istenen davranistir.
MARKETS = {
    "RU": ["отдых в Турции", "тур в Турцию", "отели Турция", "авиабилеты Турция"],
    "DE": ["Türkei Urlaub", "Flug Türkei", "Hotel Türkei", "Antalya Urlaub"],
    "GB": ["Turkey holiday", "flights to Turkey", "Turkey hotels", "Antalya holiday"],
    "BG": ["почивка Турция", "хотели Турция", "екскурзия Турция", "самолетни билети Турция"],
    "IR": ["تور ترکیه", "سفر به ترکیه", "هتل استانبول", "بلیط ترکیه"],
    "UA": ["відпочинок у Туреччині", "тур до Туреччини", "готелі Туреччина", "авіаквитки Туреччина"],
    "PL": ["wakacje Turcja", "loty Turcja", "hotele Turcja", "Antalya wakacje"],
    "NL": ["vakantie Turkije", "vluchten Turkije", "hotel Turkije", "Antalya vakantie"],
    "GE": ["ტურები თურქეთში", "სასტუმრო თურქეთი", "თურქეთი მოგზაურობა"],
    "IQ": ["السياحة في تركيا", "فنادق اسطنبول", "تركيا سياحة", "تذاكر طيران تركيا"],
    # --- 2026-09-05 tarihinde eklendi: genisletilmis pazar seti ---
    # Bu 5 pazarin gecmise donuk cekimi YOK; seri bu tarihten itibaren baslar.
    "KZ": ["отдых в Турции", "тур в Турцию", "отели Турция", "авиабилеты Турция"],  # Kazakistan - Rusca arama baskin
    "AZ": ["Türkiyəyə səyahət", "Türkiyə turu", "İstanbul otel", "Antalya turu"],   # Azerbaycan
    "SA": ["السياحة في تركيا", "فنادق اسطنبول", "تركيا سياحة", "تذاكر طيران تركيا"],  # Suudi Arabistan - IQ ile ayni Arapca set
    "FR": ["vacances en Turquie", "vols vers la Turquie", "hôtel Turquie", "vacances à Antalya"],  # Fransa
    "US": ["Turkey vacation", "flights to Turkey", "Turkey hotels", "Antalya vacation"],  # ABD
}

OUT_ROOT = os.path.join("data", "raw")
LOG_CSV = os.path.join("data", "collection_log.csv")

# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler("collect_gt.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def fetch_market(pytrends, geo, keywords):
    """Tek bir kaynak pazar icin GT serisini ceker. Basarisizsa None doner."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            pytrends.build_payload(
                kw_list=keywords,
                cat=CATEGORY,
                timeframe=TIMEFRAME,
                geo=geo,
                gprop=GPROP,
            )
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                log.warning("[%s] Bos veri dondu (deneme %d/%d)", geo, attempt, MAX_RETRY)
            else:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                log.info("[%s] Basarili - %d satir, %d kolon", geo, len(df), len(df.columns))
                return df

        except Exception as exc:
            log.warning("[%s] Hata (deneme %d/%d): %s", geo, attempt, MAX_RETRY, exc)

        # ustel geri cekilme + jitter
        wait = (2 ** attempt) + random.uniform(0, 4)
        log.info("[%s] %.1f sn bekleniyor...", geo, wait)
        time.sleep(wait)

    log.error("[%s] BASARISIZ - %d deneme sonunda veri alinamadi", geo, MAX_RETRY)
    return None


def append_log(row):
    """Her cekimin sonucunu merkezi log dosyasina yazar."""
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    new_file = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "draw_date_utc", "run_timestamp_utc", "geo",
                "status", "n_rows", "n_keywords", "timeframe",
            ])
        w.writerow(row)


def main():
    now_utc = datetime.now(timezone.utc)
    draw_date = now_utc.strftime("%Y-%m-%d")
    run_ts = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir = os.path.join(OUT_ROOT, draw_date)

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        log.warning(
            "UYARI: %s klasoru zaten dolu. Google onbellegi nedeniyle bugun "
            "tekrar cekmek YENI BIR ORNEK VERMEZ. Cikiliyor.", out_dir
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    log.info("=" * 60)
    log.info("CEKIM BASLIYOR - UTC tarih: %s", draw_date)
    log.info("Donem: %s | Kategori: %s", TIMEFRAME, CATEGORY)
    log.info("=" * 60)

    # NOT: retries/backoff_factor parametreleri BILEREK verilmiyor.
    # pytrends bunlari alinca urllib3'un eski 'method_whitelist' arayuzunu
    # kullaniyor; urllib3 2.x bu parametreyi tanimadigi icin hata veriyor.
    # Zaten fetch_market() icinde kendi yeniden deneme dongumuz var.
    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

    ok = 0
    pending = list(MARKETS)

    # FAIL kaydi yalnizca tum turlar bittikten sonra yazilir; boylece log'da
    # her pazar icin tek satir olur.
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

            df = fetch_market(pytrends, geo, keywords)

            if df is not None:
                path = os.path.join(out_dir, f"{geo}.csv")
                df.to_csv(path, encoding="utf-8-sig")
                append_log([draw_date, run_ts, geo, "OK", len(df), len(keywords), TIMEFRAME])
                ok += 1
            else:
                still_failed.append(geo)

            # pazarlar arasi nazik bekleme
            time.sleep(BASE_SLEEP + random.uniform(0, 5))
        pending = still_failed

    for geo in pending:
        append_log([draw_date, run_ts, geo, "FAIL", 0, len(MARKETS[geo]), TIMEFRAME])
    fail = len(pending)

    log.info("=" * 60)
    log.info("CEKIM BITTI - Basarili: %d | Basarisiz: %d", ok, fail)
    if fail:
        log.warning(
            "%d pazar alinamadi. Bunlari ELLE cekip ayni klasore ayni isimle "
            "kaydedin, YA DA collection_log.csv'deki FAIL kaydini oldugu gibi "
            "birakip makalede eksik cekim olarak raporlayin.", fail
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
