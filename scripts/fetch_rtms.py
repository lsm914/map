#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
국토부 RTMS 아파트매매 실거래가 수집/집계 파이프라인
- 입력: ref/lawd_codes.csv, ref/sigungu_centroids.csv, env MOLIT_API_KEY
- 출력: data/all_trades.parquet, data/agg_sigungu.parquet, data/meta.json
- 기간: 최근 12개월 (오늘 기준)
- 신축/구축 구분: 거래연도 - 준공연도 <= 10년 → 신축, 그 외 구축
- 타입(면적대): <60, 60~85, 85~102, >102 (전용면적 m²)
"""

import os, math, json, time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
REF_DIR  = BASE_DIR / "ref"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY   = os.environ.get("MOLIT_API_KEY")  # 인코딩된 서비스키
BASE_URL  = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
NUM_ROWS  = 999

# ---------- 1) 기간(최근 12개월) ----------
today = datetime.today()
date_list = [(today - relativedelta(months=i)).strftime("%Y%m") for i in range(12)]
date_list = sorted(set(date_list))  # 오래→최근 순

# ---------- 2) 기준 테이블 ----------
lawd_df = pd.read_csv("lawd_codes.csv")  # 컬럼: lawd_cd(5자리), sido_nm, sigungu_nm
cent_df = pd.read_csv("sigungu_centroids.csv")  # 컬럼: lawd_cd, lat, lng, region_group(수도권/특광역시/지방)

lawd_list = lawd_df["LAWD_CD"].astype(str).str.zfill(5).unique().tolist()

# ---------- 3) 수집 ----------
records = []
sess = requests.Session()
for ym in date_list:
    for lawd_cd in lawd_list:
        # 1페이지 호출
        url = f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={ym}&serviceKey={API_KEY}&pageNo=1&numOfRows={NUM_ROWS}"
        res = sess.get(url, timeout=20)
        root = ET.fromstring(res.content)
        total_count = int(root.findtext(".//totalCount", "0"))
        if total_count == 0:
            continue
        total_pages = math.ceil(total_count / NUM_ROWS)

        for page in range(1, total_pages + 1):
            url_p = f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={ym}&serviceKey={API_KEY}&pageNo={page}&numOfRows={NUM_ROWS}"
            r = sess.get(url_p, timeout=20)
            root_p = ET.fromstring(r.content)
            for it in root_p.findall(".//item"):
                records.append({
                    "LAWD_CD": lawd_cd,
                    "DEAL_YM": ym,  # YYYYMM
                    "aptNm":          (it.findtext("aptNm", "") or "").strip(),
                    "dealAmount":     (it.findtext("dealAmount", "") or "").replace(",", "").strip(), # 만원 문자열
                    "excluUseAr":     (it.findtext("excluUseAr", "") or "").strip(),                 # 전용(m²)
                    "floor":          (it.findtext("floor", "") or "").strip(),
                    "buildYear":      (it.findtext("buildYear", "") or "").strip(),
                    "dealYear":       (it.findtext("dealYear", "") or "").strip(),
                    "dealMonth":      (it.findtext("dealMonth", "") or "").strip(),
                    "dealDay":        (it.findtext("dealDay", "") or "").strip(),
                    "estateAgentSggNm": (it.findtext("estateAgentSggNm", "") or "").strip(),
                    "dong":           (it.findtext("dong", "") or "").strip(),
                    "jibun":          (it.findtext("jibun", "") or "").strip(),
                })
        # 속도/쿨다운(과도 요청 방지)
        time.sleep(0.1)

# ---------- 4) 정규화 ----------
if len(records) == 0:
    raise SystemExit("수집된 데이터가 없습니다.")

df = pd.DataFrame(records)
# 숫자형 변환
for col in ["dealAmount", "excluUseAr", "buildYear", "dealYear", "dealMonth", "dealDay", "floor"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 금액: 만원 → 원
df["price_krw"] = (df["dealAmount"] * 10000).astype("Int64")

# 거래일자
def _mk_date(r):
    try:
        return datetime(int(r["dealYear"]), int(r["dealMonth"]), int(r["dealDay"]))
    except Exception:
        # 일자 누락 대비: 해당 월 1일
        y = int(str(r["DEAL_YM"])[:4]); m = int(str(r["DEAL_YM"])[4:])
        return datetime(y, m, 1)
df["deal_date"] = df.apply(_mk_date, axis=1)

# 면적대(타입)
def _area_band(x):
    if pd.isna(x): return None
    if x < 60: return "<60"
    if x < 85: return "60~85"
    if x < 102: return "85~102"
    return "102+"
df["area_band"] = df["excluUseAr"].apply(_area_band)

# 신축/구축 (거래연도 기준 10년 이내)
df["age_years"] = df["dealYear"] - df["buildYear"]
df["new_old"] = df["age_years"].apply(lambda a: "신축(≤10년)" if pd.notna(a) and a <= 10 else "구축(>10년)")

# m²당 가격
df["price_per_m2"] = (df["price_krw"] / df["excluUseAr"]).round().astype("Int64")

# 조인(명칭, 좌표, 권역)
df = df.merge(lawd_df.rename(columns={"lawd_cd":"LAWD_CD"}), on="LAWD_CD", how="left")
df = df.merge(cent_df.rename(columns={"lawd_cd":"LAWD_CD"}), on="LAWD_CD", how="left")  # lat,lng,region_group

# ---------- 5) 기간 버킷 ----------
# 1년전~6개월전, 6개월전~3개월전, 3개월전~1개월전, 1개월 내
now = datetime.today()
b1_start = now - relativedelta(months=12)
b2_start = now - relativedelta(months=6)
b3_start = now - relativedelta(months=3)
b4_start = now - relativedelta(months=1)

def _bucket(d):
    if d < b2_start and d >= b1_start: return "1년~6개월"
    if d < b3_start and d >= b2_start: return "6개월~3개월"
    if d < b4_start and d >= b3_start: return "3개월~1개월"
    if d >= b4_start: return "최근1개월"
    return "기타"
df["period_bucket"] = df["deal_date"].apply(_bucket)

# ---------- 6) 집계: 시군구 × 기간버킷 × 신축/구축 × 면적대 ----------
grp_cols = ["LAWD_CD","sido_nm","sigungu_nm","lat","lng","region_group","period_bucket","new_old","area_band"]
agg = (df
       .dropna(subset=["price_krw","excluUseAr"])
       .groupby(grp_cols, dropna=False)
       .agg(
           avg_price_krw=("price_krw","mean"),
           avg_price_per_m2=("price_per_m2","mean"),
           n_trades=("price_krw","count")
       ).reset_index())
agg["avg_price_krw"] = agg["avg_price_krw"].round().astype("Int64")
agg["avg_price_per_m2"] = agg["avg_price_per_m2"].round().astype("Int64")

# ---------- 7) 저장 ----------
df.to_parquet(DATA_DIR / "all_trades.parquet", index=False)
agg.to_parquet(DATA_DIR / "agg_sigungu.parquet", index=False)
with open(DATA_DIR / "meta.json","w",encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "months": date_list,
        "records": len(df),
        "group_rows": len(agg)
    }, f, ensure_ascii=False, indent=2)

print("DONE",
      {"records":len(df), "agg_rows":len(agg), "months":date_list[0]+"~"+date_list[-1]})
