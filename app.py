# app/app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from urllib.request import urlopen
from io import BytesIO
from copy import deepcopy

st.set_page_config(page_title="전국 아파트 실거래가 비교", layout="wide")

# =========================
# 유틸
# =========================
def fmt_int(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

# =========================
# 데이터 로더 (캐시)
# =========================
@st.cache_data(ttl=3600)
def load_parquet_local(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(ttl=3600)
def load_parquet_url(url: str) -> pd.DataFrame:
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

@st.cache_data(ttl=86400)
def load_geojson_local(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=86400)
def load_geojson_url(url: str) -> dict:
    with urlopen(url) as f:
        return json.loads(f.read().decode("utf-8"))

# =========================
# 경로
# =========================
DATA_BASE = "https://raw.githubusercontent.com/lsm914/map/main/data"
REF_BASE  = "https://raw.githubusercontent.com/lsm914/map/main/ref"

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

# 시군구 경계(GeoJSON) — 로컬 → 원격
# 기대 스키마: feature.properties.SIG_CD == 5자리 시군구 코드(= LAWD_CD)
try:
    sgg = load_geojson_local("ref/sgg.geojson")
except Exception:
    sgg = load_geojson_url(f"{REF_BASE}/sgg.geojson")

# =========================
# 사이드바 UI
# =========================
st.sidebar.markdown("### 필터")
st.sidebar.write(f"데이터 생성: {meta.get('generated_at','-')}")

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
df = df[df["period_bucket"].eq(period)]
if new_old != "전체":
    df = df[df["new_old"].eq(new_old)]
if area_band:
    df = df[df["area_band"].isin(area_band)]
if region_tab != "전국" and "region_group" in df.columns:
    df = df[df["region_group"].eq(region_tab)]

# 표시값 컬럼 결정
if metric == "평균 거래가(원)":
    value_col = "avg_price_krw"
elif metric == "m²당 가격(원)":
    value_col = "avg_price_per_m2"
else:
    value_col = "n_trades"

# 시군구 단위 대표값 (동일 시군구 복수 레코드 → 평균/합)
group_cols = ["LAWD_CD","sido_nm","sigungu_nm"]
agg_dict = {value_col: "mean", "n_trades": "sum"}
map_df = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

# 숫자형 보정
map_df[value_col] = pd.to_numeric(map_df[value_col], errors="coerce")
map_df["n_trades"] = pd.to_numeric(map_df["n_trades"], errors="coerce").fillna(0).astype(int)

# 값 범위
val_min = float(map_df[value_col].min()) if map_df[value_col].notna().any() else 0.0
val_max = float(map_df[value_col].max()) if map_df[value_col].notna().any() else 1.0
if val_min == val_max:
    val_max = val_min + 1.0

# =========================
# GeoJSON에 값 주입(조인) + 색상 사전계산
# =========================
# join key: LAWD_CD (df) <-> SIG_CD (geojson)
val_dict      = {str(k).zfill(5): float(v) for k, v in zip(map_df["LAWD_CD"], map_df[value_col])}
trades_dict   = {str(k).zfill(5): int(v)   for k, v in zip(map_df["LAWD_CD"], map_df["n_trades"])}
sido_dict     = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sido_nm"])}
sigungu_dict  = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sigungu_nm"])}

def color_scale(v, vmin, vmax):
    # 0~1 정규화 후, 연한 파랑(200,220,255) → 진한 파랑(20,60,200)로 보간
    if v is None or np.isnan(v):
        return [220, 220, 220, 100]  # 회색톤(데이터 없음)
    t = (v - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    r = int(200 + (20 - 200) * t)
    g = int(220 + (60 - 220) * t)
    b = int(255 + (200 - 255) * t)
    return [r, g, b, 160]

# GeoJSON 복사본에 값과 색상 주입
sgg_joined = deepcopy(sgg)
for ft in sgg_joined.get("features", []):
    props = ft.get("properties", {})
    sig_cd = str(props.get("SIG_CD", "")).zfill(5)
    val = val_dict.get(sig_cd)
    ntr = trades_dict.get(sig_cd, 0)
    sido_nm = sido_dict.get(sig_cd)
    sigungu_nm = sigungu_dict.get(sig_cd)
    props["LAWD_CD"] = sig_cd
    props["val"] = None if val is None or np.isnan(val) else round(val)
    props["n_trades"] = int(ntr)
    props["sido_nm"] = sido_nm
    props["sigungu_nm"] = sigungu_nm
    props["fill_color"] = color_scale(val, val_min, val_max)
    # 툴팁용 문자열(콤마 포맷)
    props["val_str"] = fmt_int(val)
    props["n_trades_str"] = fmt_int(ntr)

# =========================
# 지도 렌더 (폴리곤)
# =========================
st.markdown(
    f"## 전국 실거래가 지도 — {period} · {new_old} · "
    f"{', '.join(area_band) if area_band else '전체'} · {metric}"
)

# 전국 뷰(대략)
mid_lat, mid_lng = 36.5, 127.8

poly_layer = pdk.Layer(
    "GeoJsonLayer",
    sgg_joined,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",   # JSON 함수 없이 속성 사용
    get_line_color=[120, 120, 140, 120],
    line_width_min_pixels=1,
)

tooltip = {
    "html": (
        "<b>{properties.sido_nm} {properties.sigungu_nm}</b><br/>"
        + f"{metric}: " + "{properties.val_str}<br/>거래건수: {properties.n_trades_str}"
    ),
    "style": {"backgroundColor": "white", "color": "black"}
}

deck = pdk.Deck(
    layers=[poly_layer],
    initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

# =========================
# 표 (좌표 컬럼 제외 & 콤마 포맷)
# =========================
st.markdown("### 상위 시군구")

display_df = map_df.copy()
# 숫자 포맷된 텍스트 컬럼 추가
display_df["표시값"] = display_df[value_col].map(fmt_int)
display_df["거래건수"] = display_df["n_trades"].map(fmt_int)

# 보여줄 컬럼만 (좌표 제외)
display_df = display_df[["sido_nm","sigungu_nm","표시값","거래건수"]]
display_df = display_df.sort_values("표시값", ascending=False, key=lambda s: s.str.replace(",", "").replace("-", "0").astype(int)).head(30)

st.dataframe(display_df, use_container_width=True)
