# app/app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from urllib.request import urlopen
from io import BytesIO

st.set_page_config(page_title="전국 아파트 실거래가 비교", layout="wide")

# =========================
# 데이터 로더 (캐시)
# =========================
@st.cache_data(ttl=3600)
def load_parquet_local(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(ttl=3600)
def load_parquet_url(url: str) -> pd.DataFrame:
    # 일부 환경에서 pandas가 https 파케를 직접 못 읽을 수 있어 BytesIO로 우회
    with urlopen(url) as f:
        buf = BytesIO(f.read())
    return pd.read_parquet(buf)

@st.cache_data(ttl=3600)
def load_json_local(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_json_url(url: str) -> dict:
    with urlopen(url) as f:
        return json.loads(f.read().decode("utf-8"))

# 깃허브 raw 베이스 경로
DATA_BASE = "https://raw.githubusercontent.com/lsm914/map/main/data"

# 집계 데이터 로드 (로컬 우선, 실패 시 깃허브 raw)
try:
    agg = load_parquet_local("data/agg_sigungu.parquet")
except Exception:
    agg = load_parquet_url(f"{DATA_BASE}/agg_sigungu.parquet")

# 메타 로드
try:
    meta = load_json_local("data/meta.json")
except Exception:
    meta = load_json_url(f"{DATA_BASE}/meta.json")

# =========================
# 사이드바 UI
# =========================
st.sidebar.markdown("### 필터")
st.sidebar.write(f"데이터 생성: {meta.get('generated_at','-')}")

# 구버전 호환을 위해 segmented_control 대신 radio 사용
period = st.sidebar.radio(
    "기간",
    ["1년~6개월","6개월~3개월","3개월~1개월","최근1개월"],
    index=3,
    horizontal=True
)

new_old = st.sidebar.radio(
    "신축/구축",
    ["전체","신축(≤10년)","구축(>10년)"],
    index=0,
    horizontal=True
)

area_choices = ["<60","60~85","85~102","102+"]
area_band = st.sidebar.multiselect("면적대(전용 m²)", area_choices, default=["60~85","85~102"])

# region_group 컬럼이 없을 수도 있음 → 옵션 자동 구성
region_options = ["전국"]
if "region_group" in agg.columns and agg["region_group"].notna().any():
    extra = sorted([x for x in agg["region_group"].dropna().unique().tolist() if x])
    region_options += extra
region_tab = st.sidebar.radio("권역", region_options, index=0)

metric = st.sidebar.selectbox("표시값", ["평균 거래가(원)","m²당 가격(원)","거래건수"], index=1)

# =========================
# 필터 적용
# =========================
df = agg.copy()

# 기간
df = df[df["period_bucket"].eq(period)]

# 신축/구축
if new_old != "전체":
    df = df[df["new_old"].eq(new_old)]

# 면적대
if area_band:
    df = df[df["area_band"].isin(area_band)]

# 권역
if region_tab != "전국" and "region_group" in df.columns:
    df = df[df["region_group"].eq(region_tab)]

# 표시값 컬럼 결정
if metric == "평균 거래가(원)":
    value_col = "avg_price_krw"
elif metric == "m²당 가격(원)":
    value_col = "avg_price_per_m2"
else:
    value_col = "n_trades"

# 시군구 대표 포인트(동일 시군구 중복 → 평균/합)
group_cols = ["LAWD_CD","sido_nm","sigungu_nm","lat","lng"]
agg_dict = {value_col: "mean", "n_trades": "sum"}
map_df = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

# 숫자형 보정
if map_df[value_col].dtype.kind not in "iu":
    map_df[value_col] = pd.to_numeric(map_df[value_col], errors="coerce")
map_df["n_trades"] = pd.to_numeric(map_df["n_trades"], errors="coerce").fillna(0).astype(int)
map_df[value_col] = map_df[value_col].round()

# =========================
# 반경(Radius) 사전 계산 (JSON 내 함수 호출 금지 이슈 해결)
# =========================
# 원래 식: min(max(val/50, 1000), 20000)
map_df["val"] = map_df[value_col].astype(float)
map_df["radius"] = (map_df["val"] / 50.0)
map_df["radius"] = map_df["radius"].clip(lower=1000, upper=20000)
map_df["radius"] = map_df["radius"].fillna(1000)

# 좌표형 보정
map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
map_df["lng"] = pd.to_numeric(map_df["lng"], errors="coerce")

# =========================
# 지도 렌더
# =========================
st.markdown(
    f"## 전국 실거래가 지도 — {period} · {new_old} · "
    f"{', '.join(area_band) if area_band else '전체'} · {metric}"
)

# 뷰 초기값
if map_df["lat"].notna().any() and map_df["lng"].notna().any():
    mid_lat = float(map_df["lat"].mean())
    mid_lng = float(map_df["lng"].mean())
else:
    mid_lat, mid_lng = 36.5, 127.8  # 한국 중심 근사치

# pydeck Layer (함수 문자열 없이 컬럼명만 사용)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[lng, lat]',
    get_radius='radius',  # ← 함수 호출 금지 이슈 회피: 사전 계산 컬럼 사용
    pickable=True,
    auto_highlight=True
)

tooltip = {
    "html": (
        "<b>{sido_nm} {sigungu_nm}</b><br/>"
        + f"{metric}: " + "{val}<br/>거래건수: {n_trades}"
    ),
    "style": {"backgroundColor": "white", "color": "black"}
}

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

# =========================
# 표
# =========================
st.markdown("### 상위 시군구")
display_df = map_df.rename(columns={value_col: metric}).copy()
display_df = display_df[["sido_nm","sigungu_nm", metric, "n_trades", "lat", "lng"]]
display_df = display_df.sort_values(metric, ascending=False).head(30)
st.dataframe(display_df, use_container_width=True)
