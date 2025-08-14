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

# --- 구분 지도용 이름 표준화(영문/별칭 포함) ---
def normalize_category_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for tok in ["특별자치시","광역시","특별시"," ", "·", "-", "_"]:
        s = s.replace(tok, "")
    low = s.lower()
    eng_alias = {
        "seoul":"서울","busan":"부산","daegu":"대구","incheon":"인천",
        "gwangju":"광주","daejeon":"대전","ulsan":"울산","sejong":"세종",
        "capitalregion":"수도권","capitalarea":"수도권","metropolitanarea":"수도권",
        "gyeonggi":"수도권","gyeonggido":"수도권","gg":"수도권",
        "noncapital":"지방","others":"지방","regions":"지방",
    }
    if low in eng_alias:
        return eng_alias[low]
    alias = {
        "서울":"서울","서울시":"서울","서울특별시":"서울",
        "부산":"부산","부산시":"부산","부산광역시":"부산",
        "대구":"대구","대구시":"대구","대구광역시":"대구",
        "인천":"인천","인천시":"인천","인천광역시":"인천",
        "광주":"광주","광주시":"광주","광주광역시":"광주",
        "대전":"대전","대전시":"대전","대전광역시":"대전",
        "울산":"울산","울산시":"울산","울산광역시":"울산",
        "세종":"세종","세종시":"세종","세종특별자치시":"세종",
        "수도권":"수도권","경기도":"수도권","경기":"수도권",
        "지방":"지방",
    }
    return alias.get(s, s)

def extract_type_name(props: dict) -> str:
    keys = [
        "name","NAME","Name","type","TYPE","label","LABEL",
        "sgg_type","SGG_TYPE","adm_nm","ADM_NM","adm_name","ADM_NAME",
        "sigungu","SIGUNGU","sido","SIDO","region","REGION",
        "group","GROUP","cat","CAT","category","CATEGORY"
    ]
    cand = get_prop(props, keys)
    if cand:
        return normalize_category_name(cand)
    texts = [str(v) for v in (props or {}).values() if isinstance(v, str)]
    if texts:
        texts.sort(key=len, reverse=True)
        return normalize_category_name(texts[0])
    return ""

# =========================
# 로더 (캐시)
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

try:
    agg = load_parquet_local("data/agg_sigungu.parquet")
except Exception:
    agg = load_parquet_url(f"{DATA_BASE}/agg_sigungu.parquet")

try:
    meta = load_json_local("data/meta.json")
except Exception:
    meta = load_json_url(f"{DATA_BASE}/meta.json")

try:
    sgg = load_geojson_local("sgg.geojson")
except Exception:
    sgg = load_geojson_url("sgg.geojson")

try:
    sgg_type = load_geojson_local("sgg_type.geojson")
except Exception:
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

# 평형(ETL에서 25/31/35평형 생성 전제)
area_choices = ["25평형","31평형","35평형"]
area_band = st.sidebar.multiselect("평형", area_choices, default=area_choices)

# 권역
region_options = ["전국"]
if "region_group" in agg.columns and agg["region_group"].notna().any():
    region_options += sorted([x for x in agg["region_group"].dropna().unique().tolist() if x])
region_tab = st.sidebar.radio("권역", region_options, index=0)

# 구분지도 값 기준(전체/신축/구축)
cat_value_mode = st.sidebar.selectbox("구분 지도 값 기준", ["전체","신축(≤10년)","구축(>10년)"], index=0)

# 표 시군구 선택
df_for_opts = agg[["LAWD_CD","sido_nm","sigungu_nm"]].drop_duplicates()
df_for_opts["label"] = df_for_opts.apply(
    lambda r: f"{(r['sido_nm'] or '').strip()} {(r['sigungu_nm'] or '').strip()} ({str(r['LAWD_CD']).zfill(5)})".strip(),
    axis=1
)
options = dict(zip(df_for_opts["label"], df_for_opts["LAWD_CD"]))
selected_labels = st.sidebar.multiselect("표에 표시할 시군구 선택", list(options.keys()), default=[])

# =========================
# 공통 설정
# =========================
metric_label = "평균 거래가(억원)"   # 표시는 억원
value_col = "avg_price_krw"        # 내부 수치는 원

# =========================
# 헬퍼: 집계 만들기
# =========================
def filter_base(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["period_bucket"].eq(period)].copy()
    if area_band:
        d = d[d["area_band"].isin(area_band)]
    if region_tab != "전국" and "region_group" in d.columns:
        d = d[d["region_group"].eq(region_tab)]
    return d

def agg_by_sigungu(df_in: pd.DataFrame) -> pd.DataFrame:
    """시군구별 대표값: 평균(가격), 합계(건수)"""
    g = (df_in.groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
              .agg(avg_price_krw=("avg_price_krw","mean"),
                   n_trades=("n_trades","sum"))
              .reset_index())
    g["avg_price_krw"] = pd.to_numeric(g["avg_price_krw"], errors="coerce")
    g["n_trades"] = pd.to_numeric(g["n_trades"], errors="coerce").fillna(0).astype(int)
    return g

def agg_by_sigungu_weighted(df_in: pd.DataFrame) -> pd.DataFrame:
    """신축/구축을 함께 포함한 가중평균(거래건수 가중)"""
    t = df_in[["LAWD_CD","sido_nm","sigungu_nm","avg_price_krw","n_trades"]].dropna(subset=["avg_price_krw","n_trades"]).copy()
    t["w_sum"] = t["avg_price_krw"] * t["n_trades"]
    g = (t.groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
           .agg(w_sum=("w_sum","sum"), n=("n_trades","sum"))
           .reset_index())
    g["avg_price_krw"] = np.where(g["n"]>0, g["w_sum"]/g["n"], np.nan)
    g = g.rename(columns={"n":"n_trades"})
    g["n_trades"] = g["n_trades"].fillna(0).astype(int)
    return g[["LAWD_CD","sido_nm","sigungu_nm","avg_price_krw","n_trades"]]

def build_map_df_all_new_old(df_base: pd.DataFrame):
    """전체/신축/구축 3종 집계 반환"""
    # 전체(가중평균)
    map_all = agg_by_sigungu_weighted(df_base)

    # 신축/구축 별도
    map_new = agg_by_sigungu( df_base[df_base["new_old"].eq("신축(≤10년)")] )
    map_old = agg_by_sigungu( df_base[df_base["new_old"].eq("구축(>10년)")] )

    return map_all, map_new, map_old

def inject_to_sgg(geojson_obj: dict, map_df: pd.DataFrame, vmin: float, vmax: float):
    """시군구 GeoJSON에 값/색상/툴팁 주입"""
    vals = {str(k).zfill(5): float(v) for k, v in zip(map_df["LAWD_CD"], map_df["avg_price_krw"])}
    cnts = {str(k).zfill(5): int(v)   for k, v in zip(map_df["LAWD_CD"], map_df["n_trades"])}
    sidos= {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sido_nm"])}
    sggs = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sigungu_nm"])}

    def color_scale(v, vmin, vmax):
        if v is None or np.isnan(v):
            return [220,220,220,100]
        t = (v - vmin) / (vmax - vmin)
        t = np.clip(t, 0, 1)
        r = int(200 + (20 - 200) * t)
        g = int(220 + (60 - 220) * t)
        b = int(255 + (200 - 255) * t)
        return [r, g, b, 160]

    joined = deepcopy(geojson_obj)
    for ft in joined.get("features", []):
        pr = ft.get("properties", {})
        code = extract_sgg_code(pr)
        val = vals.get(code)
        ntr = cnts.get(code, 0)
        pr["sido_nm"] = sidos.get(code)
        pr["sigungu_nm"] = sggs.get(code)
        pr["val"] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        pr["metric_str"] = fmt_eok(val)
        pr["n_trades"] = int(ntr)
        pr["trades_str"] = fmt_int(ntr)
        pr["fill_color"] = color_scale(val, vmin, vmax)
        name_txt = get_prop(pr, ["SIG_KOR_NM","SIG_NM","name"]) or f"{pr['sido_nm'] or ''} {pr['sigungu_nm'] or ''}".strip()
        pr["name"] = name_txt
    return joined

def build_cat_df_from_map(map_df: pd.DataFrame) -> pd.DataFrame:
    """시군구 집계(map_df)에서 구분(서울/…/수도권/지방) 가중평균"""
    core = map_df[["LAWD_CD","avg_price_krw","n_trades"]].dropna(subset=["avg_price_krw","n_trades"]).copy()
    core["LAWD_CD"] = core["LAWD_CD"].astype(str).str.zfill(5)
    core["w_sum"] = core["avg_price_krw"] * core["n_trades"]

    def lawd_to_category(lawd_cd: str) -> str:
        p2 = str(lawd_cd)[:2]
        if p2 == "11": return "서울"
        if p2 == "26": return "부산"
        if p2 == "27": return "대구"
        if p2 == "28": return "인천"
        if p2 == "29": return "광주"
        if p2 == "30": return "대전"
        if p2 == "31": return "울산"
        if p2 == "36": return "세종"
        if p2 == "41": return "수도권"
        return "지방"

    core["category"] = core["LAWD_CD"].map(lawd_to_category)
    cat = (core.groupby("category", dropna=False)
                .agg(w_sum=("w_sum","sum"), n_trades=("n_trades","sum"))
                .reset_index())
    cat["wavg"] = np.where(cat["n_trades"]>0, cat["w_sum"]/cat["n_trades"], np.nan)
    cat["category_norm"] = cat["category"].map(normalize_category_name)
    return cat

def inject_to_sgg_type(sgg_type_geo: dict, cat_df: pd.DataFrame):
    """구분 GeoJSON에 값/색상/툴팁 주입"""
    vmap = dict(zip(cat_df["category_norm"], cat_df["wavg"]))
    cmap = dict(zip(cat_df["category_norm"], cat_df["n_trades"]))
    vmin = float(cat_df["wavg"].min()) if cat_df["wavg"].notna().any() else 0.0
    vmax = float(cat_df["wavg"].max()) if cat_df["wavg"].notna().any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    def color_scale_cat(v, vmin, vmax):
        if v is None or np.isnan(v):
            return [220,220,220,100]
        t = (v - vmin) / (vmax - vmin)
        t = np.clip(t, 0, 1)
        r = int(240 + (30 - 240) * t)
        g = int(220 + (70 - 220) * t)
        b = int(240 + (180 - 240) * t)
        return [r, g, b, 160]

    joined = deepcopy(sgg_type_geo)
    for ft in joined.get("features", []):
        pr = ft.get("properties", {}) or {}
        nm = extract_type_name(pr)  # 서울/부산/…/수도권/지방
        v = vmap.get(nm)
        n = int(cmap.get(nm, 0))
        pr["group_name"] = nm or ""
        pr["val"] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        pr["val_str"] = fmt_eok(v)
        pr["n_trades"] = n
        pr["n_trades_str"] = fmt_int(n)
        pr["fill_color"] = color_scale_cat(v, vmin, vmax)
    return joined

# =========================
# 필터 적용 및 집계 생성
# =========================
base = filter_base(agg)

# 시군구 지도용 3종 집계
map_all, map_new, map_old = build_map_df_all_new_old(base)

# 값 범위(색상)
def get_range(dfmap):
    v = dfmap["avg_price_krw"]
    vmin = float(v.min()) if v.notna().any() else 0.0
    vmax = float(v.max()) if v.notna().any() else 1.0
    if vmin == vmax: vmax = vmin + 1.0
    return vmin, vmax

vmin_all, vmax_all = get_range(map_all)
vmin_new, vmax_new = get_range(map_new)
vmin_old, vmax_old = get_range(map_old)

# GeoJSON 주입(시군구)
sgg_all = inject_to_sgg(sgg, map_all, vmin_all, vmax_all)
sgg_new = inject_to_sgg(sgg, map_new, vmin_new, vmax_new)
sgg_old = inject_to_sgg(sgg, map_old, vmin_old, vmax_old)

# 구분 지도용 집계 소스 선택(전체/신축/구축)
cat_source = {"전체": map_all, "신축(≤10년)": map_new, "구축(>10년)": map_old}[cat_value_mode]
cat_df = build_cat_df_from_map(cat_source)
sgg_type_joined = inject_to_sgg_type(sgg_type, cat_df)

# =========================
# 레이아웃: 시군구 지도(탭) + 구분 지도
# =========================
st.markdown("## 시군구 지도")
tabs = st.tabs(["전체", "신축(≤10년)", "구축(>10년)"])
mid_lat, mid_lng = 36.5, 127.8

with tabs[0]:
    deck = pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer", sgg_all, pickable=True, stroked=True, filled=True,
                          get_fill_color="properties.fill_color",
                          get_line_color=[120,120,140,120], line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": "<b>{name}</b><br/>평균 거래가: {metric_str}<br/>거래건수: {trades_str}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck, use_container_width=True)

with tabs[1]:
    deck = pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer", sgg_new, pickable=True, stroked=True, filled=True,
                          get_fill_color="properties.fill_color",
                          get_line_color=[120,120,140,120], line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": "<b>{name}</b><br/>신축 평균: {metric_str}<br/>거래건수: {trades_str}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck, use_container_width=True)

with tabs[2]:
    deck = pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer", sgg_old, pickable=True, stroked=True, filled=True,
                          get_fill_color="properties.fill_color",
                          get_line_color=[120,120,140,120], line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": "<b>{name}</b><br/>구축 평균: {metric_str}<br/>거래건수: {trades_str}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck, use_container_width=True)

# 구분 지도
st.markdown(f"## 구분 지도 — 기준: {cat_value_mode}")
type_deck = pdk.Deck(
    layers=[pdk.Layer("GeoJsonLayer", sgg_type_joined, pickable=True, stroked=True, filled=True,
                      get_fill_color="properties.fill_color",
                      get_line_color=[100,100,120,140], line_width_min_pixels=1.5)],
    initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
    tooltip={"html": "<b>{group_name}</b><br/>평균 거래가: {val_str}<br/>거래건수: {n_trades_str}",
             "style":{"backgroundColor":"white","color":"black"}}
)
st.pydeck_chart(type_deck, use_container_width=True)

# =========================
# 표: 전체·신축·구축 모두 보여주기
# =========================
st.markdown("### 시군구별 요약 (전체·신축·구축)")

# 원천(필터 반영)
df_table_src = base.copy()

# 시군구×new_old별 가중평균
tmp = df_table_src[["LAWD_CD","sido_nm","sigungu_nm","new_old","avg_price_krw","n_trades"]].dropna(subset=["avg_price_krw","n_trades"]).copy()
tmp["w_sum"] = tmp["avg_price_krw"] * tmp["n_trades"]

g = (tmp.groupby(["LAWD_CD","sido_nm","sigungu_nm","new_old"], dropna=False)
        .agg(w_sum=("w_sum","sum"), n=("n_trades","sum")).reset_index())
g["wavg"] = g["w_sum"] / g["n"]

all_g = (g.groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
           .agg(w_sum=("w_sum","sum"), n=("n","sum")).reset_index())
all_g["wavg_all"] = all_g["w_sum"] / all_g["n"]

pivot = g.pivot_table(index=["LAWD_CD","sido_nm","sigungu_nm"],
                      columns="new_old", values="wavg", aggfunc="first").reset_index()
table = pivot.merge(all_g[["LAWD_CD","wavg_all","n"]], on="LAWD_CD", how="left")

# 라벨 통일
col_new = "신축(≤10년)" if "신축(≤10년)" in table.columns else ("신축" if "신축" in table.columns else None)
col_old = "구축(>10년)" if "구축(>10년)" in table.columns else ("구축" if "구축" in table.columns else None)
rename_map = {"sido_nm":"시도","sigungu_nm":"시군구","wavg_all":"전체(억원)","n":"거래건수"}
if col_new: rename_map[col_new] = "신축(억원)"
if col_old: rename_map[col_old] = "구축(억원)"
table = table.rename(columns=rename_map)

# 선택 시군구 필터
if selected_labels:
    selected_codes = [str(options[label]).zfill(5) for label in selected_labels]
    table = table[table["LAWD_CD"].isin(selected_codes)]

# 포맷
for c in ["전체(억원)","신축(억원)","구축(억원)"]:
    if c in table.columns:
        table[c] = table[c].map(fmt_eok)
table["거래건수"] = table["거래건수"].fillna(0).astype(int).map(fmt_int)

# 정렬(전체 기준)
sort_vals = all_g.set_index("LAWD_CD")["wavg_all"]
table = (table.join(sort_vals.rename("_sort"), on="LAWD_CD")
              .sort_values("_sort", ascending=False)
              .drop(columns=["_sort","LAWD_CD"]))

st.dataframe(
    table[["시도","시군구"]
          + ([ "전체(억원)"] if "전체(억원)" in table.columns else [])
          + ([ "신축(억원)"] if "신축(억원)" in table.columns else [])
          + ([ "구축(억원)"] if "구축(억원)" in table.columns else [])
          + ["거래건수"]],
    use_container_width=True
)
