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
# 유틸
# =========================
def fmt_eok(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{float(x)/1e8:,.1f}억원"
    except Exception:
        return "-"

def fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

# =========================
# 로더 (캐시)
# =========================
@st.cache_data(ttl=3600)
def load_parquet_url(url: str) -> pd.DataFrame:
    with urlopen(url) as f:
        buf = BytesIO(f.read())
    return pd.read_parquet(buf)

@st.cache_data(ttl=3600)
def load_json_url(url: str) -> dict:
    with urlopen(url) as f:
        return json.loads(f.read().decode("utf-8"))

@st.cache_data(ttl=86400)
def load_geojson_url(url: str) -> dict:
    with urlopen(url) as f:
        return json.loads(f.read().decode("utf-8"))

# =========================
# 데이터 경로
# =========================
DATA_BASE = "https://raw.githubusercontent.com/lsm914/map/main/data"

agg = load_parquet_url(f"{DATA_BASE}/agg_sigungu.parquet")
meta = load_json_url(f"{DATA_BASE}/meta.json")
sgg = load_geojson_url("sgg.geojson")
sgg_type = load_geojson_url("sgg_type.geojson")

# =========================
# 사이드바
# =========================
st.sidebar.markdown("### 필터")
st.sidebar.write(f"데이터 생성: {meta.get('generated_at','-')}")

period = st.sidebar.radio(
    "기간",
    ["1년~6개월","6개월~3개월","3개월~1개월","최근1개월"],
    index=3,
    horizontal=True
)

area_choices = ["25평형","31평형","35평형"]
area_band = st.sidebar.multiselect("평형", area_choices, default=area_choices)

region_options = ["전국"]
if "region_group" in agg.columns and agg["region_group"].notna().any():
    region_options += sorted([x for x in agg["region_group"].dropna().unique().tolist() if x])
region_tab = st.sidebar.radio("권역", region_options, index=0)

cat_value_mode = st.sidebar.selectbox("지도 값 기준", ["전체","신축(≤10년)","구축(>10년)"], index=0)

# =========================
# 데이터 필터링 + 집계
# =========================
@st.cache_data
def filter_and_aggregate(agg, period, area_band, region_tab):
    d = agg[agg["period_bucket"].eq(period)].copy()
    if area_band:
        d = d[d["area_band"].isin(area_band)]
    if region_tab != "전국" and "region_group" in d.columns:
        d = d[d["region_group"].eq(region_tab)]
    return d

base = filter_and_aggregate(agg, period, area_band, region_tab)

# 시군구별 집계
def agg_by_sigungu(df_in, new_old=None):
    d = df_in.copy()
    if new_old:
        d = d[d["new_old"].eq(new_old)]
    g = (d.groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
          .agg(avg_price_krw=("avg_price_krw","mean"),
               n_trades=("n_trades","sum"))
          .reset_index())
    return g

map_all = agg_by_sigungu(base)  # 전체
map_new = agg_by_sigungu(base, "신축(≤10년)")
map_old = agg_by_sigungu(base, "구축(>10년)")

map_dict = {"전체": map_all, "신축(≤10년)": map_new, "구축(>10년)": map_old}
map_selected = map_dict[cat_value_mode]

# 색상 스케일 (앱 최초 한 번만)
if "vmin" not in st.session_state:
    st.session_state.vmin = float(agg["avg_price_krw"].min())
    st.session_state.vmax = float(agg["avg_price_krw"].max())

vmin, vmax = st.session_state.vmin, st.session_state.vmax

# =========================
# 레이아웃: 좌우 지도
# =========================
st.markdown(f"## 지도 — 기준: {cat_value_mode}")
left, right = st.columns(2)
mid_lat, mid_lng = 36.5, 127.8

# 왼쪽: 시군구 지도
with left:
    st.markdown("#### 시군구 지도")
    deck_left = pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                sgg,   # 원본 그대로
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="""[
                    255 * (properties.avg_price_krw - %f) / (%f - %f),
                    150,
                    200,
                    160
                ]""" % (vmin, vmax, vmin),
                get_line_color=[120,120,140,120],
                line_width_min_pixels=1,
            )
        ],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": "<b>{sigungu_nm}</b><br/>평균 거래가: {avg_price_krw}<br/>거래건수: {n_trades}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck_left, use_container_width=True)

# 오른쪽: 구분 지도 (sgg_type)
with right:
    st.markdown("#### 구분 지도")
    deck_right = pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                sgg_type,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="""[
                    240 * (properties.avg_price_krw - %f) / (%f - %f),
                    200,
                    180,
                    160
                ]""" % (vmin, vmax, vmin),
                get_line_color=[100,100,120,140],
                line_width_min_pixels=1.5,
            )
        ],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": "<b>{group}</b><br/>평균 거래가: {avg_price_krw}<br/>거래건수: {n_trades}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck_right, use_container_width=True)

# =========================
# 표 (간단)
# =========================
st.markdown("### 시군구별 요약")
st.dataframe(map_selected, use_container_width=True)
