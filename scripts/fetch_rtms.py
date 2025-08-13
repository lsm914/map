#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
국토부 RTMS 아파트매매 실거래가 수집/집계 파이프라인 (전체 수정본)
- 입력:
  - ref/lawd_codes.csv              # 컬럼: lawd_cd(5자리), sido_nm, sigungu_nm
  - ref/sigungu_centroids.csv       # 컬럼: lawd_cd(5자리), lat, lng, region_group(수도권/특광역시/지방)
  - 환경변수 MOLIT_API_KEY          # 인코딩된 서비스키(그대로 사용)
- 출력:
  - data/all_trades.parquet         # 정규화 원천
  - data/agg_sigungu.parquet        # 시군구×기간×신축/구축×면적대 집계
  - data/meta.json                  # 생성 메타
- 수집 기간: 최근 12개월
- 신축/구축 기준: 거래연도 - 준공연도 ≤ 10년 → 신축
- 면적대(타입): <60, 60~85, 85~102, 102+ (전용 m²)
"""

import os, math, json, time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import requests

# ---------------------- 경로/상수 ----------------------
BASE_DIR = Path(__file__).resolve().parents[1]
REF_DIR  = BASE_DIR / "ref"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY   = os.environ.get("MOLIT_API_KEY")  # 인코딩된 서비스키(그대로 사용)
BASE_URL  = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
NUM_ROWS  = 999

if not API_KEY:
    raise SystemExit("환경변수 MOLIT_API_KEY 가 설정되어 있지 않습니다. GitHub Secrets 또는 로컬 환경변수를 확인하세요.")

# ---------------------- 기간(최근 12개월) ----------------------
today = datetime.today()
date_list = [(today - relativedelta(months=i)).strftime("%Y%m") for i in range(12)]
date_list = sorted(set(date_list))  # 오래→최근

# ---------------------- 기준 테이블 로딩 (문자열 5자리 통일) ----------------------
# lawd_codes.csv
lawd_df = pd.read_csv(REF_DIR / "lawd_codes.csv", dtype={"lawd_cd": str})
lawd_df["lawd_cd"] = lawd_df["lawd_cd"].str.strip().str.zfill(5)

# sigungu_centroids.csv
cent_df = pd.read_csv(REF_DIR / "sigungu_centroids.csv", dtype={"lawd_cd": str})
cent_df["lawd_cd"] = cent_df["lawd_cd"].str.strip().str.zfill(5)

# 조인 편의 컬럼명 정리
lawd_df = lawd_df.rename(columns={"lawd_cd": "LAWD_CD"})
cent_df = cent_df.rename(columns={"lawd_cd": "LAWD_CD"})

# 수집 대상 코드 리스트
lawd_list = lawd_df["LAWD_CD"].astype(str).str.zfill(5).unique().tolist()

# ---------------------- 수집 루프 ----------------------
records = []
sess = requests.Session()

def fetch_one_month_one_code(lawd_cd: str, ym: str):
    """특정 시군구(lawd_cd), 특정 연월(ym=YYYYMM) 데이터 전 페이지 수집"""
    url = f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={ym}&serviceKey={API_KEY}&pageNo=1&numOfRows={NUM_ROWS}"
    res = sess.get(url, timeout=20)
    root = ET.fromstring(res.content)
    total_count = int(root.findtext(".//totalCount", "0"))
    if total_count == 0:
        return []

    total_pages = math.ceil(total_count / NUM_ROWS)
    out = []
    for page in range(1, total_pages + 1):
        url_p = f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={ym}&serviceKey={API_KEY}&pageNo={page}&numOfRows={NUM_ROWS}"
        r = sess.get(url_p, timeout=20)
        root_p = ET.fromstring(r.content)
        for it in root_p.findall(".//item"):
            out.append({
                "LAWD_CD":           str(lawd_cd).zfill(5),
                "DEAL_YM":           ym,  # YYYYMM
                "aptNm":             (it.findtext("aptNm", "") or "").strip(),
                "dealAmount":        (it.findtext("dealAmount", "") or "").replace(",", "").strip(),  # 만원 문자열
                "excluUseAr":        (it.findtext("excluUseAr", "") or "").strip(),                  # 전용(m²)
                "floor":             (it.findtext("floor", "") or "").strip(),
                "buildYear":         (it.findtext("buildYear", "") or "").strip(),
                "dealYear":          (it.findtext("dealYear", "") or "").strip(),
                "dealMonth":         (it.findtext("dealMonth", "") or "").strip(),
                "dealDay":           (it.findtext("dealDay", "") or "").strip(),
                "estateAgentSggNm":  (it.findtext("estateAgentSggNm", "") or "").strip(),
                "dong":              (it.findtext("dong", "") or "").strip(),
                "jibun":             (it.findtext("jibun", "") or "").strip(),
            })
    return out

print(f"[INFO] 수집 시작: {date_list[0]} ~ {date_list[-1]}, codes={len(lawd_list)}")
for ym in date_list:
    for i, lawd_cd in enumerate(lawd_list, 1):
        try:
            recs = fetch_one_month_one_code(lawd_cd, ym)
            records.extend(recs)
        except Exception as e:
            print(f"[WARN] {ym} {lawd_cd} 수집 오류: {e}")
        # 과도요청 방지(필요시 조정)
        time.sleep(0.1)
    print(f"[INFO] {ym} 완료 ({i}/{len(lawd_list)})")

if len(records) == 0:
    raise SystemExit("수집된 데이터가 없습니다. API Key/권한, 기간, 코드 테이블을 확인하세요.")

# ---------------------- 정규화 ----------------------
df = pd.DataFrame(records)

# 키 컬럼 문자열 5자리 보장
df["LAWD_CD"] = df["LAWD_CD"].astype(str).str.zfill(5)

# 숫자형 변환
for col in ["dealAmount", "excluUseAr", "buildYear", "dealYear", "dealMonth", "dealDay", "floor"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 금액: 만원 → 원
df["price_krw"] = (df["dealAmount"] * 10000).round().astype("Int64")

# 거래일자 구성
def _mk_date(r):
    try:
        if pd.notna(r["dealYear"]) and pd.notna(r["dealMonth"]) and pd.notna(r["dealDay"]):
            return datetime(int(r["dealYear"]), int(r["dealMonth"]), int(r["dealDay"]))
    except Exception:
        pass
    # fallback: DEAL_YM의 1일
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

# 신축/구축 (거래연도 기준 10년 이내 → 신축)
def _new_old(row):
    if pd.isna(row["dealYear"]) or pd.isna(row["buildYear"]):
        return "구축(>10년)"  # 보수적으로 구축 처리
    age = int(row["dealYear"]) - int(row["buildYear"])
    return "신축(≤10년)" if age <= 10 else "구축(>10년)"

df["new_old"] = df.apply(_new_old, axis=1)

# m²당 가격
df["price_per_m2"] = (df["price_krw"] / df["excluUseAr"]).round().astype("Int64")

# ---------------------- 행정구역/좌표 조인 (문자열 키) ----------------------
# 두 테이블 모두 LAWD_CD는 문자열 5자리로 통일되어 있음
df = df.merge(lawd_df, on="LAWD_CD", how="left")   # sido_nm, sigungu_nm
df = df.merge(cent_df, on="LAWD_CD", how="left")   # lat, lng, region_group

# ---------------------- 기간 버킷 ----------------------
now = datetime.today()
b1_start = now - relativedelta(months=12)
b2_start = now - relativedelta(months=6)
b3_start = now - relativedelta(months=3)
b4_start = now - relativedelta(months=1)

def _bucket(d):
    if d is None or pd.isna(d): return "기타"
    if d < b2_start and d >= b1_start: return "1년~6개월"
    if d < b3_start and d >= b2_start: return "6개월~3개월"
    if d < b4_start and d >= b3_start: return "3개월~1개월"
    if d >= b4_start: return "최근1개월"
    return "기타"

df["period_bucket"] = df["deal_date"].apply(_bucket)

# ---------------------- 집계: 시군구 × 기간버킷 × 신축/구축 × 면적대 ----------------------
grp_cols = ["LAWD_CD","sido_nm","sigungu_nm","lat","lng","region_group","period_bucket","new_old","area_band"]
agg = (
    df.dropna(subset=["price_krw","excluUseAr"])
      .groupby(grp_cols, dropna=False)
      .agg(
          avg_price_krw=("price_krw","mean"),
          avg_price_per_m2=("price_per_m2","mean"),
          n_trades=("price_krw","count")
      ).reset_index()
)
agg["avg_price_krw"] = agg["avg_price_krw"].round().astype("Int64")
agg["avg_price_per_m2"] = agg["avg_price_per_m2"].round().astype("Int64")

# ---------------------- 저장 ----------------------
df.to_parquet(DATA_DIR / "all_trades.parquet", index=False)
agg.to_parquet(DATA_DIR / "agg_sigungu.parquet", index=False)

meta = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "months": f"{date_list[0]}~{date_list[-1]}",
    "records": int(len(df)),
    "group_rows": int(len(agg))
}
with open(DATA_DIR / "meta.json","w",encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("[DONE]", meta)
