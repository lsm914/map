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

# -------------------------
# 유틸
# -------------------------
def fmt_eok(x):
    """원화 금액 -> 억원(소수 1자리) 문자열"""
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

# -------------------------
# 로더 (캐시)
# -------------------------
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

# -------------------------
# 경로
# -------------------------
DATA_BASE = "https://raw.githubusercontent.com/lsm914/map/main/data"
REF_BASE  = "https://raw.githubusercontent.com/lsm914/map/main/ref"

# 집계 데이터 (로컬 우선, 실패 시 원격)
try:
    agg = load_parquet_local("data/agg_sigungu.parquet")
except Exception:
    agg = load_parquet_url(f"{DATA_BASE}/agg_sigungu.parquet")

# 메타
try:
    meta = load_json_local("data/meta.json")
except Exception:
    meta = load_json_url(f"{DATA_BASE}/meta.json")

# 시군구 경계 GeoJSON (로컬 → 원격)
# 기대 속성: feature.properties.SIG_CD == LAWD_CD(5자리)
try:
    sgg = load_geojson_local("ref/sgg.geojson")
except Exception:
    sgg = load_geojson_url(f"{REF_BASE}/sgg.geojson")

# -------------------------
# 사이드바
# -------------------------
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

# 권역 옵션 자동 구성
region_options = ["전국"]
if "region_group" in agg.columns and agg["region_group"].notna().any():
    extra = sorted([x for x in agg["region_group"].dropna().unique().tolist() if x])
    region_options += extra
region_tab = st.sidebar.radio("권역", region_options, index=0)

# 표에 표시할 시군구 다중 선택(코드 기반으로 식별)
# 옵션 라벨: "시도 시군구 (LAWD_CD)"
df_for_opts = agg[["LAWD_CD","sido_nm","sigungu_nm"]].drop_duplicates()
df_for_opts["label"] = df_for_opts.apply(
    lambda r: f"{(r['sido_nm'] or '').strip()} {(r['sigungu_nm'] or '').strip()} ({str(r['LAWD_CD']).zfill(5)})".strip(),
    axis=1
)
options = dict(zip(df_for_opts["label"], df_for_opts["LAWD_CD"]))
selected_labels = st.sidebar.multiselect("표에 표시할 시군구 선택", list(options.keys()), default=[])

# -------------------------
# 값 설정 (평균 거래가만 사용)
# -------------------------
metric_label = "평균 거래가(억원)"   # 표시 라벨은 억원
value_col = "avg_price_krw"        # 내부 수치는 원단위

# -------------------------
# 필터 적용
# -------------------------
df = agg.copy()
df = df[df["period_bucket"].eq(period)]
if new_old != "전체":
    df = df[df["new_old"].eq(new_old)]
if area_band:
    df = df[df["area_band"].isin(area_band)]
if region_tab != "전국" and "region_group" in df.columns:
    df = df[df["region_group"].eq(region_tab)]

# 시군구 단위 대표값 (동일 시군구 다건 → 평균/합산)
group_cols = ["LAWD_CD","sido_nm","sigungu_nm"]
agg_dict = {value_col: "mean", "n_trades": "sum"}
map_df = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

# 숫자형 보정
map_df[value_col] = pd.to_numeric(map_df[value_col], errors="coerce")
map_df["n_trades"] = pd.to_numeric(map_df["n_trades"], errors="coerce").fillna(0).astype(int)

# 범례용 최소/최대 (색상 계산에는 원단위 사용)
val_min = float(map_df[value_col].min()) if map_df[value_col].notna().any() else 0.0
val_max = float(map_df[value_col].max()) if map_df[value_col].notna().any() else 1.0
if val_min == val_max:
    val_max = val_min + 1.0

# -------------------------
# GeoJSON 주입 + 색상/툴팁 평탄화
# -------------------------
val_dict      = {str(k).zfill(5): float(v) for k, v in zip(map_df["LAWD_CD"], map_df[value_col])}
trades_dict   = {str(k).zfill(5): int(v)   for k, v in zip(map_df["LAWD_CD"], map_df["n_trades"])}
sido_dict     = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sido_nm"])}
sigungu_dict  = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sigungu_nm"])}

def color_scale(v, vmin, vmax):
    if v is None or np.isnan(v):
        return [220, 220, 220, 100]  # 데이터 없음
    t = (v - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    r = int(200 + (20 - 200) * t)
    g = int(220 + (60 - 220) * t)
    b = int(255 + (200 - 255) * t)
    return [r, g, b, 160]

sgg_joined = deepcopy(sgg)
for ft in sgg_joined.get("features", []):
    props = ft.get("properties", {})
    sig_cd = str(props.get("SIG_CD", "")).zfill(5)
    val = val_dict.get(sig_cd)
    ntr = trades_dict.get(sig_cd, 0)
    sido_nm = sido_dict.get(sig_cd)
    sigungu_nm = sigungu_dict.get(sig_cd)

    props["LAWD_CD"] = sig_cd
    props["sido_nm"] = sido_nm
    props["sigungu_nm"] = sigungu_nm
    props["val"] = None if val is None or np.isnan(val) else round(val)
    props["n_trades"] = int(ntr)
    props["fill_color"] = color_scale(val, val_min, val_max)

    # 툴팁 평탄화 & 포맷
    props["name"] = f"{sido_nm or ''} {sigungu_nm or ''}".strip()
    props["metric_str"] = fmt_eok(val)     # 억원 단위
    props["trades_str"] = fmt_int(ntr)

# -------------------------
# 지도
# -------------------------
st.markdown(
    f"## 전국 실거래가 지도 — {period} · {new_old} · "
    f"{', '.join(area_band) if area_band else '전체'} · {metric_label}"
)

mid_lat, mid_lng = 36.5, 127.8  # 전국 뷰

poly_layer = pdk.Layer(
    "GeoJsonLayer",
    sgg_joined,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[120, 120, 140, 120],
    line_width_min_pixels=1,
)

tooltip = {
    "html": (
        "<b>{name}</b><br/>"
        + f"{metric_label}: " + "{metric_str}<br/>거래건수: {trades_str}"
    ),
    "style": {"backgroundColor": "white", "color": "black"}
}

deck = pdk.Deck(
    layers=[poly_layer],
    initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

# -------------------------
# 표 (선택 시군구만 표시 / 기본은 전체)
# -------------------------
st.markdown("### 시군구별 요약")

table_df = map_df.copy()
# 선택이 있으면 그 코드만 필터
if selected_labels:
    selected_codes = [str(options[label]).zfill(5) for label in selected_labels]
    table_df = table_df[table_df["LAWD_CD"].isin(selected_codes)]

# 포맷 컬럼
table_df["평균 거래가(억원)"] = table_df[value_col].map(fmt_eok)  # 억원
table_df["거래건수"] = table_df["n_trades"].map(fmt_int)

# 표시 컬럼만
table_df = table_df[["sido_nm","sigungu_nm","평균 거래가(억원)","거래건수"]]

# 수치 기준 정렬(내림차순)
_sort_val = pd.to_numeric(map_df[value_col], errors="coerce").fillna(0)
table_df = table_df.join(_sort_val.rename("_sort_val"))
table_df = table_df.sort_values("_sort_val", ascending=False).drop(columns=["_sort_val"])

st.dataframe(table_df, use_container_width=True)
