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

def get_prop(props, keys):
    for k in keys:
        if k in props and props[k] not in (None, ""):
            return props[k]
    return None

def only_digits(s):
    return "".join(ch for ch in str(s) if ch.isdigit())

def extract_sgg_code(props):
    """
    GeoJSON properties에서 시군구 코드를 최대한 유연하게 찾는다.
    - 우선순위: SIG_CD, LAWD_CD, ADM_CD, ADM_DR_CD, CODE, code, SGG_CD ...
    - 10자리 등이면 앞 5자리만 사용(시군구 레벨).
    """
    cand = get_prop(props, [
        "SIG_CD","LAWD_CD","ADM_CD","ADM_DR_CD","SGG_CD","CODE","code","sig_cd","adm_cd"
    ])
    if cand is None:
        return None
    d = only_digits(cand)
    if len(d) >= 5:
        return d[:5]
    return None

def norm_name(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    for suf in ["시","군","구"]:
        s = s.replace(suf, "")
    return s.replace(" ", "")

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
# 어떤 속성 키를 쓰든 간에 위 extract_sgg_code로 알아서 잡는다.
try:
    sgg = load_geojson_local("sgg.geojson")
except Exception:
    sgg = load_geojson_url("sgg.geojson")

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

# 값 범위 (색상 계산용)
val_min = float(map_df[value_col].min()) if map_df[value_col].notna().any() else 0.0
val_max = float(map_df[value_col].max()) if map_df[value_col].notna().any() else 1.0
if val_min == val_max:
    val_max = val_min + 1.0

# 이름 기반 백업 매칭을 위해 정규화된 이름-코드 딕셔너리 생성
map_df["_name_norm"] = (map_df["sido_nm"].fillna("") + map_df["sigungu_nm"].fillna("")).apply(norm_name)
name_to_code = dict(zip(map_df["_name_norm"], map_df["LAWD_CD"]))

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
matched = 0

for ft in sgg_joined.get("features", []):
    props = ft.get("properties", {})

    # 1) 코드 기반 매칭 시도
    code = extract_sgg_code(props)

    # 2) 이름 기반 백업 매칭 (시도/시군구 이름을 properties에서 찾아 정규화)
    if (code is None) or (code not in val_dict):
        sido_p = get_prop(props, ["CTP_KOR_NM","SIDO_NM","SIDO","CTPRVN_NM","CTPRVN_NM_KOR"])
        sgg_p  = get_prop(props, ["SIG_KOR_NM","SIG_NM","SIG_ENG_NM","SGG_NM","SIG_NAME"])
        key_norm = norm_name((sido_p or "") + (sgg_p or ""))
        maybe = name_to_code.get(key_norm)
        if maybe:
            code = str(maybe).zfill(5)

    # 값 주입
    val = val_dict.get(code)
    ntr = trades_dict.get(code, 0)
    sido_nm = sido_dict.get(code)
    sigungu_nm = sigungu_dict.get(code)

    if code in val_dict:
        matched += 1

    props["LAWD_CD"] = code or ""
    props["sido_nm"] = sido_nm
    props["sigungu_nm"] = sigungu_nm
    props["val"] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else round(val)
    props["n_trades"] = int(ntr) if ntr is not None else 0
    props["fill_color"] = color_scale(val, val_min, val_max)

    # 툴팁 평탄화 & 포맷
    name_txt = get_prop(props, ["SIG_KOR_NM","SIG_NM","name"]) or f"{sido_nm or ''} {sigungu_nm or ''}".strip()
    props["name"] = name_txt
    props["metric_str"] = fmt_eok(val)     # 억원 단위
    props["trades_str"] = fmt_int(ntr)

# -------------------------
# 지도
# -------------------------
def format_area_band(band_list):
    mapping = {
        "6085": "60~85",
        "85102": "85~102",
        "102+": "102+",
        "<60": "<60"
    }
    return [mapping.get(b, b) for b in band_list]

# 제목 생성
st.markdown(
    f"## 전국 실거래가 지도 — {period} · {new_old} · "
    f"{' · '.join(format_area_band(area_band)) if area_band else '전체'} · {metric_label}"
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

# 매칭 통계(디버그용)
with st.expander("매칭 상태(디버그)"):
    total_feat = len(sgg_joined.get("features", []))
    st.write(f"GeoJSON 피처 수: {total_feat} | 값 매칭된 피처: {matched}")

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
table_df = table_df.assign(_sort_val=pd.to_numeric(table_df["평균 거래가(억원)"].str.replace("억원","").str.replace(",",""), errors="coerce").fillna(0.0))
table_df = table_df.sort_values("_sort_val", ascending=False).drop(columns=["_sort_val"])

st.dataframe(table_df, use_container_width=True)
