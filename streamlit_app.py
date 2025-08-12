# -*- coding: utf-8 -*-
import os, pandas as pd, numpy as np, pydeck as pdk, streamlit as st
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="전국 실거래가 비교 지도", layout="wide")

st.title("전국 시군구 아파트 실거래가 비교")

DATA_PATH = Path("data/all_trades.parquet")
LAWD_CODES_PATH = Path("lawd_codes.csv")
CENTROIDS_PATH = Path("sigungu_centroids.csv")

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
    else:
        st.stop()
    lc = pd.read_csv(LAWD_CODES_PATH, dtype={"LAWD_CD":str,"SIDO_CD":str})
    ct = pd.read_csv(CENTROIDS_PATH, dtype={"LAWD_CD":str})
    return df, lc, ct

def label_region(sido_cd):
    # 수도권: 서울(11), 경기(41), 인천(28)
    # 특·광역시: 부산(26), 대구(27), 인천(28), 광주(29), 대전(30), 울산(31), 세종(36)
    if pd.isna(sido_cd):
        return "기타"
    s = str(sido_cd).zfill(2)
    if s in {"11","41","28"}:
        return "수도권"
    if s in {"26","27","28","29","30","31","36"}:
        return "특·광역시"
    return "지방"

def area_type(m2):
    # 사용자 정의 타입 구간
    if pd.isna(m2):
        return None
    x = float(m2)
    if x < 60: return "<60"
    if x < 85: return "60-85"
    if x < 102: return "85-102"
    if x < 135: return "102-135"
    return ">=135"

def is_new(build_year, deal_year):
    # 10년 이내 신축
    try:
        if pd.isna(build_year) or pd.isna(deal_year): return None
        return int(deal_year) - int(build_year) <= 10
    except:
        return None

df, lc, ct = load_data()
df["REGION"] = df["SIDO_CD"].apply(label_region) if "SIDO_CD" in df.columns else df["lawd_cd"].str[:2].apply(label_region)
df["TYPE"] = df["exclu_use_ar"].apply(area_type)
df["IS_NEW"] = df.apply(lambda r: is_new(r.get("build_year"), r.get("deal_year")), axis=1)

# 기간 버튼 그룹
st.subheader("기간 선택")
# 최근 월은 데이터 내 최대 deal_ym 기준
latest_ym = df["deal_ym"].dropna().astype(int).max()
latest_y = latest_ym // 100
latest_m = latest_ym % 100
latest_dt = datetime(latest_y, latest_m, 1)

def ym_range(start_delta_months, end_delta_months):
    # latest_dt 기준 (start, end] 구간. 예: 12~6 => (t-12, t-6]
    def add_months(d, m):
        y = d.year + (d.month - 1 + m) // 12
        mm = (d.month - 1 + m) % 12 + 1
        return datetime(y, mm, 1)
    start = add_months(latest_dt, -start_delta_months)
    end = add_months(latest_dt, -end_delta_months)
    start_ym = int(start.strftime("%Y%m"))
    end_ym = int(end.strftime("%Y%m"))
    return start_ym, end_ym

btn_cols = st.columns(4)
ranges = [
    ("1년~6개월", (12,6)),
    ("6개월~3개월", (6,3)),
    ("3개월~1개월", (3,1)),
    ("최근 1개월", (1,0))
]
selected = None
for i, (label, (a,b)) in enumerate(ranges):
    if btn_cols[i].button(label):
        selected = (a,b)

if selected is None:
    selected = (1,0)  # 기본: 최근 1개월

start_ym, end_ym = ym_range(*selected)

mask = (df["deal_ym"].astype(int) > start_ym) & (df["deal_ym"].astype(int) <= end_ym)
dfp = df.loc[mask].copy()

# 필터: 권역 / 타입 / 구축/신축
st.subheader("필터")
col1, col2, col3 = st.columns(3)
region_opt = col1.multiselect("권역", options=["수도권","특·광역시","지방"], default=["수도권","특·광역시","지방"])
type_opt = col2.multiselect("타입(전용면적)", options=["<60","60-85","85-102","102-135",">=135"], default=["<60","60-85","85-102","102-135",">=135"])
new_opt = col3.multiselect("구축/신축", options=["신축(≤10년)","구축(>10년)"], default=["신축(≤10년)","구축(>10년)"])

dfp = dfp[dfp["REGION"].isin(region_opt)]
dfp = dfp[dfp["TYPE"].isin(type_opt)]
if set(new_opt) != {"신축(≤10년)","구축(>10년)"}:
    if "신축(≤10년)" in new_opt and "구축(>10년)" not in new_opt:
        dfp = dfp[dfp["IS_NEW"] == True]
    elif "구축(>10년)" in new_opt and "신축(≤10년)" not in new_opt:
        dfp = dfp[dfp["IS_NEW"] == False]

# 시군구 단위 집계
agg = dfp.groupby(["lawd_cd"], as_index=False).agg(
    avg_price=("deal_amount_won","mean"),
    cnt=("deal_amount_won","count")
)

# 좌표 조인
agg = agg.merge(ct[["LAWD_CD","lon","lat"]], left_on="lawd_cd", right_on="LAWD_CD", how="left")
agg = agg.merge(lc[["LAWD_CD","SIGUNGU_NM","SIDO_NM"]], on="LAWD_CD", how="left")

st.markdown(f"**기간:** {start_ym} ~ {end_ym}  \n표시: 시군구별 평균가격(원) / 거래건수")

# 지도
if agg[["lon","lat"]].notna().all().all():
    # 색/크기 스케일링
    pmin, pmax = float(agg["avg_price"].min()), float(agg["avg_price"].max())
    agg["radius"] = np.interp(agg["avg_price"], [pmin, pmax] if pmin<pmax else [pmin, pmin+1], [2000, 10000]).astype(int)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg.rename(columns={"lon":"lon","lat":"lat"}),
        get_position='[lon, lat]',
        get_radius="radius",
        pickable=True,
    )
    tooltip = {
        "html": "<b>{SIDO_NM} {SIGUNGU_NM}</b><br/>평균가: {avg_price}<br/>거래건수: {cnt}<br/>LAWD: {lawd_cd}",
        "style": {"backgroundColor": "white", "color":"black"}
    }
    view_state = pdk.ViewState(latitude=36.5, longitude=127.8, zoom=6.5)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r, use_container_width=True)
else:
    st.warning("일부 시군구 좌표가 없어 지도를 그릴 수 없습니다. sigungu_centroids.csv를 확인하세요.")

# 표/다운로드
st.subheader("집계 테이블")
st.dataframe(agg[["SIDO_NM","SIGUNGU_NM","lawd_cd","avg_price","cnt"]].sort_values("avg_price", ascending=False))
st.download_button("CSV 다운로드", data=agg.to_csv(index=False).encode("utf-8-sig"), file_name="agg_by_sigungu.csv", mime="text/csv")
