#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
국토부 RTMS 아파트매매 실거래가 XML 수집(월별 x 시군구)
- 입력: lawd_codes.csv (전국 시군구 코드), 환경변수 MOLIT_API_KEY
- 출력: data/trades_YYYYMM.csv (월별 원천), data/all_trades.parquet (정규화)

컬럼 기준(주요):
- lawd_cd (시군구코드)
- deal_ym (거래연월, YYYYMM)
- deal_dt (거래일자 YYYY-MM-DD, 조합)
- apt_nm, build_year, exclu_use_ar (전용면적 m2), floor, dong, jibun
- deal_amount (만원 → 원 변환 컬럼도 제공)
- sido_cd, sigungu_nm (조인으로 부여)

집계/분석은 Streamlit 앱에서 수행합니다.
"""

import os, sys, time, argparse, math
import pandas as pd
import numpy as np
import requests, xmltodict
from pathlib import Path
from datetime import datetime, timedelta

API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"

def month_list(n_months:int, end_month:str|None=None):
    """최근 N개월의 YYYYMM 리스트를 생성(오름차순)."""
    if end_month:
        end = datetime.strptime(end_month, "%Y%m")
    else:
        today = datetime.today()
        end = datetime(today.year, today.month, 1)
    months = []
    for i in range(n_months):
        m = end - pd.DateOffset(months=(n_months-1-i))
        months.append(f"{m.year:04d}{m.month:02d}")
    return months
    
def month_list(n_months:int, end_month:str|None=None):
    """최근 N개월의 YYYYMM 리스트(오름차순)."""
    if end_month:
        end = datetime.strptime(end_month, "%Y%m")
    else:
        today = datetime.today()
        end = datetime(today.year, today.month, 1)
    months = []
    for i in range(n_months):
        m = end - relativedelta(months=(n_months-1-i))
        months.append(f"{m.year:04d}{m.month:02d}")
    return months

def fetch_one(lawd_cd:str, ym:str, api_key:str, timeout=20, retries=3, sleep=0.4):
    params = {
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": ym,
        "serviceKey": api_key,
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=timeout, headers={"accept":"*/*"})
            r.raise_for_status()
            data = xmltodict.parse(r.text)
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            if items is None:
                items = []
            if isinstance(items, dict):
                items = [items]
            return items
        except Exception as e:
            if attempt == retries - 1:
                print(f"[WARN] {lawd_cd}-{ym} failed: {e}")
                return []
            time.sleep(sleep * (attempt + 1))

def normalize_items(items, lawd_cd, ym):
    rows = []
    for it in items:
        # XML 키 이름은 대소문자 섞임. None 처리/strip
        def g(k): 
            v = it.get(k)
            if isinstance(v, str):
                v = v.strip()
            return v
        row = {
            "lawd_cd": lawd_cd,
            "deal_ym": ym,
            "apt_nm": g("aptNm"),
            "build_year": pd.to_numeric(g("buildYear"), errors="coerce"),
            "exclu_use_ar": pd.to_numeric(g("excluUseAr"), errors="coerce"),
            "floor": pd.to_numeric(g("floor"), errors="coerce"),
            "dong": g("dong"),
            "jibun": g("jibun"),
            "deal_amount_manwon": pd.to_numeric(str(g("dealAmount")).replace(",","") if g("dealAmount") else None, errors="coerce"),
            "deal_year": pd.to_numeric(g("dealYear"), errors="coerce"),
            "deal_month": pd.to_numeric(g("dealMonth"), errors="coerce"),
            "deal_day": pd.to_numeric(g("dealDay"), errors="coerce"),
            "dealing_gbn": g("dealingGbn"),
            "buyer_gbn": g("buyerGbn"),
            "cdeal_type": g("cdealType"),
            "cdeal_day": g("cdealDay"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 전처리
    df["deal_amount_won"] = df["deal_amount_manwon"] * 10000
    # 날짜 조합
    def mk_date(r):
        try:
            y = int(r["deal_year"]) if not pd.isna(r["deal_year"]) else int(str(r["deal_ym"])[:4])
            m = int(r["deal_month"]) if not pd.isna(r["deal_month"]) else int(str(r["deal_ym"])[4:6])
            d = int(r["deal_day"]) if not pd.isna(r["deal_day"]) else 1
            return f"{y:04d}-{m:02d}-{d:02d}"
        except:
            return None
    df["deal_dt"] = df.apply(mk_date, axis=1)
    return df

def fetch_records_raw(lawd_cd: str, deal_ymd: str, service_key: str, num_of_rows: int = 999, timeout: int = 20):
    """
    params 미사용, 원문 URL 문자열로 호출 + 페이지네이션.
    필요한 5개 필드만 추출해서 dict 리스트로 반환.
    """
    records = []

    # 1) 첫 페이지 호출해 totalCount 파악
    url_first = (
        f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={deal_ymd}"
        f"&serviceKey={service_key}&pageNo=1&numOfRows={num_of_rows}"
    )
    r = requests.get(url_first, timeout=timeout, headers={"accept": "*/*"})
    r.raise_for_status()
    root = ET.fromstring(r.content)

    total_count = int(root.findtext(".//totalCount", "0"))
    total_pages = max(1, math.ceil(total_count / num_of_rows))

    # 2) 모든 페이지 순회
    for page in range(1, total_pages + 1):
        url_page = (
            f"{BASE_URL}?LAWD_CD={lawd_cd}&DEAL_YMD={deal_ymd}"
            f"&serviceKey={service_key}&pageNo={page}&numOfRows={num_of_rows}"
        )
        rp = requests.get(url_page, timeout=timeout, headers={"accept": "*/*"})
        rp.raise_for_status()
        root_page = ET.fromstring(rp.content)

        for item in root_page.findall(".//item"):
            # 요청하신 5개 + 식별용 LAWD_CD/DEAL_YMD만 수집
            records.append({
                "LAWD_CD": lawd_cd,
                "DEAL_YMD": deal_ymd,
                "estateAgentSggNm": (item.findtext("estateAgentSggNm", "") or "").strip(),
                "dealAmount": (item.findtext("dealAmount", "") or "").strip(),
                "excluUseAr": (item.findtext("excluUseAr", "") or "").strip(),
                "floor": (item.findtext("floor", "") or "").strip(),
                "buildYear": (item.findtext("buildYear", "") or "").strip(),
            })

    return records


def load_lawd_codes(path="lawd_codes.csv"):
    lc = pd.read_csv(path, dtype={"LAWD_CD":str, "SIDO_CD":str})
    # 최소 요구 컬럼 확인
    needed = {"LAWD_CD"}
    assert needed.issubset(lc.columns), f"lawd_codes.csv must include: {needed}"
    if "SIDO_CD" not in lc.columns:
        # 앞 2자리 추정
        lc["SIDO_CD"] = lc["LAWD_CD"].str[:2]
    return lc

def save_month(df_m, ym, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"trades_{ym}.csv"
    df_m.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {fp} ({len(df_m):,} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12, help="최근 N개월 수집 (기본 12개월)")
    ap.add_argument("--end-month", type=str, default=None, help="끝 YYYYMM (기본: 당월)")
    ap.add_argument("--rows", type=int, default=999, help="numOfRows (기본 999)")
    args = ap.parse_args()

    service_key = os.environ.get("MOLIT_API_KEY")
    if not service_key:
        print("ERROR: MOLIT_API_KEY is not set in environment (URL-encoded key expected).")
        sys.exit(1)

    lawd_list = load_lawd_codes("lawd_codes.csv")
    months = month_list(args.months, args.end_month)
    outdir = Path("data")

    all_parts = []

    for ym in months:
        month_records = []
        for lawd_cd in lawd_list:
            try:
                recs = fetch_records_raw(lawd_cd, ym, service_key, num_of_rows=args.rows)
                if recs:
                    month_records.extend(recs)
            except Exception as e:
                print(f"[WARN] {lawd_cd}-{ym} failed: {e}")
                continue

        if month_records:
            df_m = pd.DataFrame(month_records, columns=[
                "LAWD_CD","DEAL_YMD","estateAgentSggNm","dealAmount","excluUseAr","floor","buildYear"
            ])
            save_month(df_m, ym, outdir)
            all_parts.append(df_m)

    if all_parts:
        df_all = pd.concat(all_parts, ignore_index=True)
        # 조인: 시도/시군구 명칭
        join_cols = [c for c in ["SIDO_CD","SIDO_NM","SIGUNGU_NM"] if c in lc.columns]
        df_all = df_all.merge(lc[["LAWD_CD"] + join_cols], left_on="lawd_cd", right_on="LAWD_CD", how="left")
        # 파케 저장
        outdir.mkdir(parents=True, exist_ok=True)
        pq = outdir / "all_trades.parquet"
        df_all.to_parquet(pq, index=False)
        print(f"[OK] saved {pq} ({len(df_all):,} rows)")
    else:
        print("[WARN] no data collected.")

if __name__ == "__main__":
    main()
