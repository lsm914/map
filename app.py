import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.request import urlopen
import json
from datetime import datetime

st.set_page_config(page_title="전국 아파트 실거래가 비교", layout="wide")

# ===== 데이터 로딩(캐시) =====
@st.cache_data(ttl=3600)
def load_parquet(path_or_url: str) -> pd.DataFrame:
    return pd.read_parquet(path_or_url)

# 로컬 파일 우선, 없으면 깃허브 raw URL 사용
DATA_BASE = "https://raw.githubusercontent.com/lsm914/map/main/data"
try:
    agg = load_parquet("data/agg_sigungu.parquet")
except Exception:
    agg = load_parquet(f"{DATA_BASE}/agg_sigungu.parquet")

# 메타
try:
    with open("data/meta.json","r",encoding="utf-8") as f:
        meta = json.load(f)
except Exception:
    with urlopen(f"{DATA_BASE}/meta.json") as f:
        meta = json.loads(f.read().decode("utf-8"))

st.sidebar.markdown("### 필터")
st.sidebar.write(f"데이터 생성: {meta.get('generated_at','-')}")

# ===== 필터 UI =====
period = st.sidebar.segmented_control(
    "기간",
    ["1년~6개월","6개월~3개월","3개월~1개월","최근1개월"],
    default="최근1개월"
)
new_old = st.sidebar.radio("신축/구축", ["전체","신축(≤10년)","구축(>10년)"], index=0, horizontal=True)
area_band = st.sidebar.multiselect("면적대(전용 m²)", ["<60","60~85","85~102","102+"], default=["60~85","85~102"])

region_tab = st.sidebar.radio("권역", ["전국","수도권","특광역시","지방"], index=0, horizontal=False)

metric = st.sidebar.selectbox("표시값", ["평균 거래가(원)","m²당 가격(원)","거래건수"], index=1)

# ===== 필터 적용 =====
df = agg[agg["period_bucket"].eq(period)].copy()
if new_old != "전체":
    df = df[df["new_old"].eq(new_old)]
if area_band:
    df = df[df["area_band"].isin(area_band)]
if region_tab != "전국":
    df = df[df["region_group"].eq(region_tab)]

# 표시값 선택
if metric == "평균 거래가(원)":
    value_col = "avg_price_krw"
elif metric == "m²당 가격(원)":
    value_col = "avg_price_per_m2"
else:
    value_col = "n_trades"

# 시군구 포인트(동일 시군구 중복 → 대표 1개로 평균)
map_df = (df
          .groupby(["LAWD_CD","sido_nm","sigungu_nm","lat","lng"], dropna=False)
          .agg({value_col:"mean", "n_trades":"sum"})
          .reset_index())

map_df[value_col] = map_df[value_col].round()

# ===== 지도 =====
st.markdown(f"## 전국 실거래가 지도 — {period} · {new_old} · {', '.join(area_band) if area_band else '전체'}")
mid_lat = map_df["lat"].dropna().mean() if not map_df.empty else 36.5
mid_lng = map_df["lng"].dropna().mean() if not map_df.empty else 127.8

tooltip = {
    "html": "<b>{sido_nm} {sigungu_nm}</b><br/>" +
            f"{metric}: " + "{val}<br/>거래건수: {n_trades}",
    "style": {"backgroundColor": "white", "color": "black"}
}

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df.rename(columns={value_col:"val"}),
    get_position='[lng, lat]',
    get_radius='min(max(val/50, 1000), 20000)',   # 값에 따라 반경(임의 스케일)
    pickable=True,
    auto_highlight=True
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
    tooltip=tooltip
)
st.pydeck_chart(r, use_container_width=True)

st.markdown("### 상위 시군구")
st.dataframe(
    map_df.rename(columns={value_col: metric})
          .sort_values(metric, ascending=False)
          .head(30),
    use_container_width=True
)
