# app/app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from urllib.request import urlopen
from io import BytesIO
from copy import deepcopy
from datetime import datetime

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

def lawd_to_category(lawd_cd: str) -> str:
    """LAWD_CD → 권역명(서울/부산/…/수도권/지방)"""
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

# 집계(지도용)
try:
    agg = load_parquet_local("data/agg_sigungu.parquet")
except Exception:
    agg = load_parquet_url(f"{DATA_BASE}/agg_sigungu.parquet")

# 원천(표/툴팁용 — 연차계산)
try:
    trades = load_parquet_local("data/all_trades.parquet")
except Exception:
    trades = load_parquet_url(f"{DATA_BASE}/all_trades.parquet")

# 메타
try:
    meta = load_json_local("data/meta.json")
except Exception:
    meta = load_json_url(f"{DATA_BASE}/meta.json")

# 지오JSON
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

# 기간 멀티 선택
period_options = ["1년~6개월","6개월~3개월","3개월~1개월","최근1개월"]
periods = st.sidebar.multiselect("기간 (복수 선택 가능)", period_options, default=["최근1개월"])

# 평형 기본 35평형
area_choices = ["25평형","31평형","35평형"]
area_band = st.sidebar.multiselect("평형", area_choices, default=["35평형"])

# 권역
region_options = ["전국"]
if "region_group" in agg.columns and agg["region_group"].notna().any():
    region_options += sorted([x for x in agg["region_group"].dropna().unique().tolist() if x])
region_tab = st.sidebar.radio("권역", region_options, index=0)

# 지도 값 기준(지도는 기존 유지)
cat_value_mode = st.sidebar.selectbox("지도 값 기준", ["전체","신축(≤10년)","구축(>10년)"], index=0)

# 표 시군구 선택
df_for_opts = agg[["LAWD_CD","sido_nm","sigungu_nm"]].drop_duplicates()
df_for_opts["label"] = df_for_opts.apply(
    lambda r: f"{(r['sido_nm'] or '').strip()} {(r['sigungu_nm'] or '').strip()} ({str(r['LAWD_CD']).zfill(5)})".strip(),
    axis=1
)
options = dict(zip(df_for_opts["label"], df_for_opts["LAWD_CD"]))
selected_labels = st.sidebar.multiselect("표에 표시할 시군구 선택", list(options.keys()), default=[])

# =========================
# 공통 필터 함수
# =========================
def filter_base(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["period_bucket"].isin(periods)].copy()
    if area_band:
        d = d[d["area_band"].isin(area_band)]
    if region_tab != "전국" and "region_group" in d.columns:
        d = d[d["region_group"].eq(region_tab)]
    return d

# =========================
# 지도용 집계(agg 사용 — 기존 유지)
# =========================
base_agg = filter_base(agg)

def agg_by_sigungu(df_in: pd.DataFrame) -> pd.DataFrame:
    g = (df_in.groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
              .agg(avg_price_krw=("avg_price_krw","mean"),
                   n_trades=("n_trades","sum"))
              .reset_index())
    g["avg_price_krw"] = pd.to_numeric(g["avg_price_krw"], errors="coerce")
    g["n_trades"] = pd.to_numeric(g["n_trades"], errors="coerce").fillna(0).astype(int)
    return g

def agg_by_sigungu_weighted(df_in: pd.DataFrame) -> pd.DataFrame:
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
    map_all = agg_by_sigungu_weighted(df_base)
    map_new = agg_by_sigungu( df_base[df_base["new_old"].eq("신축(≤10년)")] )
    map_old = agg_by_sigungu( df_base[df_base["new_old"].eq("구축(>10년)")] )
    return map_all, map_new, map_old

# =========================
# 연차 구간(원천 trades 기반) — 표 & 툴팁용
# =========================
base_trades = filter_base(trades)

def ensure_age_years_from_trades(df: pd.DataFrame) -> pd.Series:
    """dealYear 없으면 deal_date.year, buildYear 없으면 NaN"""
    if "dealYear" in df.columns and df["dealYear"].notna().any():
        deal_year = pd.to_numeric(df["dealYear"], errors="coerce")
    elif "deal_date" in df.columns:
        deal_year = pd.to_datetime(df["deal_date"], errors="coerce").dt.year
    else:
        deal_year = pd.Series(np.nan, index=df.index)
    build_year = pd.to_numeric(df.get("buildYear"), errors="coerce")
    return pd.to_numeric(deal_year, errors="coerce") - build_year

def age_bucket_2_5_10(age_years: pd.Series) -> pd.Series:
    bins = [-np.inf, 2, 5, 10, np.inf]
    labels = ["≤2년","2~5년","5~10년",">10년"]
    return pd.cut(age_years, bins=bins, labels=labels, right=True, include_lowest=True)

def build_ageband_by_sigungu(df: pd.DataFrame):
    """시군구 코드별로 연차 구간 평균가(억원 문자열) dict 생성"""
    if df.empty:
        return {}
    age_years = ensure_age_years_from_trades(df)
    t = (df.assign(__age_years__=pd.to_numeric(age_years, errors="coerce"))
           .dropna(subset=["__age_years__", "price_krw", "LAWD_CD"]))
    if t.empty:
        return {}
    t["__age_bucket__"] = age_bucket_2_5_10(t["__age_years__"])
    g = (t.groupby(["LAWD_CD","__age_bucket__"], dropna=False)
           .agg(avg=("price_krw","mean"), n=("price_krw","count"))
           .reset_index())
    p = g.pivot_table(index="LAWD_CD", columns="__age_bucket__", values="avg", aggfunc="first")
    n = g.groupby("LAWD_CD")["n"].sum()
    out = {}
    for code, row in p.iterrows():
        out[str(code).zfill(5)] = {
            "age_02_str":  fmt_eok(row.get("≤2년")),
            "age_25_str":  fmt_eok(row.get("2~5년")),
            "age_510_str": fmt_eok(row.get("5~10년")),
            "age_10p_str": fmt_eok(row.get(">10년")),
            "age_total_n": int(n.get(code, 0)),
        }
    return out

def build_ageband_by_category(df: pd.DataFrame):
    """권역(정규화명)별 연차 구간 평균가(억원 문자열) dict 생성"""
    if df.empty:
        return {}
    age_years = ensure_age_years_from_trades(df)
    t = (df.assign(__age_years__=pd.to_numeric(age_years, errors="coerce"))
           .dropna(subset=["__age_years__", "price_krw", "LAWD_CD"]))
    if t.empty:
        return {}
    t["LAWD_CD"] = t["LAWD_CD"].astype(str).str.zfill(5)
    t["category"] = t["LAWD_CD"].map(lawd_to_category)
    t["category_norm"] = t["category"].map(normalize_category_name)
    t["__age_bucket__"] = age_bucket_2_5_10(t["__age_years__"])

    g = (t.groupby(["category_norm","__age_bucket__"], dropna=False)
           .agg(avg=("price_krw","mean"), n=("price_krw","count"))
           .reset_index())
    p = g.pivot_table(index="category_norm", columns="__age_bucket__", values="avg", aggfunc="first")
    n = g.groupby("category_norm")["n"].sum()

    out = {}
    for cat, row in p.iterrows():
        out[cat] = {
            "age_02_str":  fmt_eok(row.get("≤2년")),
            "age_25_str":  fmt_eok(row.get("2~5년")),
            "age_510_str": fmt_eok(row.get("5~10년")),
            "age_10p_str": fmt_eok(row.get(">10년")),
            "age_total_n": int(n.get(cat, 0)),
        }
    return out

ageband_tip_by_sgg = build_ageband_by_sigungu(base_trades)
ageband_tip_by_cat = build_ageband_by_category(base_trades)

# =========================
# GeoJSON 주입 (연차구간 툴팁 포함)
# =========================
def inject_to_sgg(geojson_obj: dict, map_df: pd.DataFrame, vmin: float, vmax: float, extra_tip: dict | None = None):
    """시군구 GeoJSON에 값/색상/툴팁 주입 (+연차구간 툴팁)"""
    vals = {str(k).zfill(5): float(v) for k, v in zip(map_df["LAWD_CD"], map_df["avg_price_krw"]) if pd.notna(v)}
    cnts = {str(k).zfill(5): int(v)   for k, v in zip(map_df["LAWD_CD"], map_df["n_trades"])}
    sidos= {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sido_nm"])}
    sggs = {str(k).zfill(5): s        for k, s in zip(map_df["LAWD_CD"], map_df["sigungu_nm"])}

    def color_scale(v, vmin, vmax):
        if v is None or (isinstance(v, float) and np.isnan(v)):
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

        # 연차구간 툴팁 값 주입
        if extra_tip and code in extra_tip:
            tip = extra_tip[code]
            pr["age_02_str"]  = tip.get("age_02_str", "-")
            pr["age_25_str"]  = tip.get("age_25_str", "-")
            pr["age_510_str"] = tip.get("age_510_str", "-")
            pr["age_10p_str"] = tip.get("age_10p_str", "-")
            pr["age_total_n"] = tip.get("age_total_n", 0)
        else:
            pr["age_02_str"] = pr["age_25_str"] = pr["age_510_str"] = pr["age_10p_str"] = "-"
            pr["age_total_n"] = 0
    return joined

def build_cat_df_from_map(map_df: pd.DataFrame) -> pd.DataFrame:
    core = map_df[["LAWD_CD","avg_price_krw","n_trades"]].dropna(subset=["avg_price_krw","n_trades"]).copy()
    core["LAWD_CD"] = core["LAWD_CD"].astype(str).str.zfill(5)
    core["w_sum"] = core["avg_price_krw"] * core["n_trades"]
    core["category"] = core["LAWD_CD"].map(lawd_to_category)
    cat = (core.groupby("category", dropna=False)
                .agg(w_sum=("w_sum","sum"), n_trades=("n_trades","sum"))
                .reset_index())
    cat["wavg"] = np.where(cat["n_trades"]>0, cat["w_sum"]/cat["n_trades"], np.nan)
    cat["category_norm"] = cat["category"].map(normalize_category_name)
    return cat

def inject_to_sgg_type(sgg_type_geo: dict, cat_df: pd.DataFrame, extra_tip: dict | None = None):
    """권역 GeoJSON에 값/색상/툴팁 주입 (+연차구간 툴팁)"""
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
        nm = extract_type_name(pr)  # 정규화된 권역명
        v = vmap.get(nm)
        n = int(cmap.get(nm, 0))
        pr["group_name"] = nm or ""
        pr["val"] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        pr["val_str"] = fmt_eok(v)
        pr["n_trades"] = n
        pr["n_trades_str"] = fmt_int(n)
        pr["fill_color"] = color_scale_cat(v, vmin, vmax)

        # 연차구간 툴팁 값 주입
        tip = extra_tip.get(nm) if extra_tip else None
        if tip:
            pr["age_02_str"]  = tip.get("age_02_str", "-")
            pr["age_25_str"]  = tip.get("age_25_str", "-")
            pr["age_510_str"] = tip.get("age_510_str", "-")
            pr["age_10p_str"] = tip.get("age_10p_str", "-")
            pr["age_total_n"] = tip.get("age_total_n", 0)
        else:
            pr["age_02_str"] = pr["age_25_str"] = pr["age_510_str"] = pr["age_10p_str"] = "-"
            pr["age_total_n"] = 0
    return joined

# =========================
# 지도 데이터 생성
# =========================
map_all, map_new, map_old = build_map_df_all_new_old(base_agg)

def get_range(dfmap):
    v = dfmap["avg_price_krw"]
    vmin = float(v.min()) if v.notna().any() else 0.0
    vmax = float(v.max()) if v.notna().any() else 1.0
    if vmin == vmax: vmax = vmin + 1.0
    return vmin, vmax

vmin_all, vmax_all = get_range(map_all)
vmin_new, vmax_new = get_range(map_new)
vmin_old, vmax_old = get_range(map_old)

# 연차구간 툴팁 주입
sgg_all = inject_to_sgg(sgg, map_all, vmin_all, vmax_all, extra_tip=ageband_tip_by_sgg)
sgg_new = inject_to_sgg(sgg, map_new, vmin_new, vmax_new, extra_tip=ageband_tip_by_sgg)
sgg_old = inject_to_sgg(sgg, map_old, vmin_old, vmax_old, extra_tip=ageband_tip_by_sgg)

# 권역 지도 데이터(색은 기존 로직, 툴팁은 연차구간)
cat_source = {"전체": map_all, "신축(≤10년)": map_new, "구축(>10년)": map_old}[cat_value_mode]
cat_df = build_cat_df_from_map(cat_source)
sgg_type_joined = inject_to_sgg_type(sgg_type, cat_df, extra_tip=ageband_tip_by_cat)

# =========================
# 레이아웃: 좌/우 지도
# =========================
st.markdown(f"## 지도 — 기준: {cat_value_mode}")
left, right = st.columns(2)
mid_lat, mid_lng = 36.5, 127.8

# 왼쪽: 시군구 지도 (툴팁 = 연차 구간)
with left:
    st.markdown("#### 시군구 지도")
    if cat_value_mode == "전체":
        geo_src = sgg_all
    elif cat_value_mode == "신축(≤10년)":
        geo_src = sgg_new
    else:
        geo_src = sgg_old

    tooltip_left_html = (
        "<b>{name}</b>"
        "<br/>≤2년 평균: {age_02_str}"
        "<br/>2~5년 평균: {age_25_str}"
        "<br/>5~10년 평균: {age_510_str}"
        "<br/>>10년 평균: {age_10p_str}"
        "<br/>거래건수(표본): {age_total_n}"
    )

    deck_left = pdk.Deck(
        layers=[pdk.Layer(
            "GeoJsonLayer",
            geo_src,
            pickable=True, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[120,120,140,120],
            line_width_min_pixels=1)],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": tooltip_left_html,
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck_left, use_container_width=True)

# 오른쪽: 권역 지도 (툴팁 = 연차 구간)
with right:
    st.markdown("#### 권역(서울·부산·…·수도권·지방) 지도")
    tooltip_right_html = (
        "<b>{group_name}</b>"
        "<br/>≤2년 평균: {age_02_str}"
        "<br/>2~5년 평균: {age_25_str}"
        "<br/>5~10년 평균: {age_510_str}"
        "<br/>>10년 평균: {age_10p_str}"
        "<br/>거래건수(표본): {age_total_n}"
        "<br/>전체 평균(색 기준): {val_str}"
    )

    deck_right = pdk.Deck(
        layers=[pdk.Layer(
            "GeoJsonLayer",
            sgg_type_joined,
            pickable=True, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[100,100,120,140],
            line_width_min_pixels=1.5)],
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=6),
        tooltip={"html": tooltip_right_html,
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck_right, use_container_width=True)

# =========================
# 표: 시군구별 요약 (연차 구간)
# =========================
st.markdown("### 시군구별 요약 (연차 구간)")

required_cols = {"buildYear","price_krw","LAWD_CD","sido_nm","sigungu_nm"}
if base_trades.empty or not required_cols.issubset(set(base_trades.columns)):
    st.warning("표 생성을 위한 필수 컬럼이 부족합니다. all_trades.parquet에 buildYear, price_krw, LAWD_CD, sido_nm, sigungu_nm 이 있어야 합니다.")
else:
    t = base_trades.dropna(subset=["buildYear","price_krw"]).copy()
    # 연차 계산
    t["__age_years__"] = pd.to_numeric(ensure_age_years_from_trades(t), errors="coerce")
    t = t.dropna(subset=["__age_years__"])
    t["__age_bucket__"] = age_bucket_2_5_10(t["__age_years__"])

    g = (t.groupby(["LAWD_CD","sido_nm","sigungu_nm","__age_bucket__"], dropna=False)
           .agg(avg_price_krw=("price_krw","mean"),
                n=("price_krw","count"))
           .reset_index())

    pvt = g.pivot_table(index=["LAWD_CD","sido_nm","sigungu_nm"],
                        columns="__age_bucket__", values="avg_price_krw", aggfunc="first").reset_index()
    n_sum = (t.groupby(["LAWD_CD"]).agg(n_total=("price_krw","count")).reset_index())

    table = pvt.merge(n_sum, on="LAWD_CD", how="left").rename(columns={
        "sido_nm":"시도",
        "sigungu_nm":"시군구",
        "≤2년":"≤2년(억원)",
        "2~5년":"2~5년(억원)",
        "5~10년":"5~10년(억원)",
        ">10년":">10년(억원)",
        "n_total":"거래건수"
    })

    if selected_labels:
        selected_codes = [str(options[label]).zfill(5) for label in selected_labels]
        table = table[table["LAWD_CD"].astype(str).str.zfill(5).isin(selected_codes)]

    for c in ["≤2년(억원)","2~5년(억원)","5~10년(억원)",">10년(억원)"]:
        if c in table.columns:
            table[c] = table[c].map(fmt_eok)
    table["거래건수"] = table["거래건수"].fillna(0).astype(int).map(fmt_int)

    sort_key = []
    for c in ["≤2년(억원)","2~5년(억원)","5~10년(억원)",">10년(억원)"]:
        if c in table.columns:
            sort_key.append(pd.to_numeric(table[c].str.replace("억원","", regex=False).str.replace(",",""), errors="coerce"))
    if sort_key:
        sk = pd.concat(sort_key, axis=1).max(axis=1)
        table = table.assign(__sort__=sk).sort_values("__sort__", ascending=False).drop(columns="__sort__")

    st.dataframe(
        table[["시도","시군구"]
              + [c for c in ["≤2년(억원)","2~5년(억원)","5~10년(억원)",">10년(억원)"] if c in table.columns]
              + ["거래건수"]],
        use_container_width=True
    )

# =========================
# 표: 권역별 요약 (연차 구간)
# =========================
st.markdown("### 권역별 요약 (연차 구간)")

if base_trades.empty or not {"buildYear","price_krw","LAWD_CD"}.issubset(set(base_trades.columns)):
    st.warning("권역 표 생성을 위한 필수 컬럼이 부족합니다. all_trades.parquet을 확인하세요.")
else:
    tt = base_trades.dropna(subset=["buildYear","price_krw"]).copy()
    tt["LAWD_CD"] = tt["LAWD_CD"].astype(str).str.zfill(5)
    tt["category"] = tt["LAWD_CD"].map(lawd_to_category)
    tt["__age_years__"] = pd.to_numeric(ensure_age_years_from_trades(tt), errors="coerce")
    tt = tt.dropna(subset=["__age_years__"])
    tt["__age_bucket__"] = age_bucket_2_5_10(tt["__age_years__"])

    g2 = (tt.groupby(["category","__age_bucket__"], dropna=False)
            .agg(avg_price_krw=("price_krw","mean"),
                 n=("price_krw","count"))
            .reset_index())

    p2 = g2.pivot_table(index=["category"], columns="__age_bucket__",
                        values="avg_price_krw", aggfunc="first").reset_index()
    n2 = (tt.groupby(["category"]).agg(n_total=("price_krw","count")).reset_index())

    out = p2.merge(n2, on="category", how="left")
    out["권역"] = out["category"].map(normalize_category_name)
    out = out.rename(columns={
        "≤2년":"≤2년(억원)",
        "2~5년":"2~5년(억원)",
        "5~10년":"5~10년(억원)",
        ">10년":">10년(억원)",
        "n_total":"거래건수"
    })

    for c in ["≤2년(억원)","2~5년(억원)","5~10년(억원)",">10년(억원)"]:
        if c in out.columns:
            out[c] = out[c].map(fmt_eok)
    out["거래건수"] = out["거래건수"].fillna(0).astype(int).map(fmt_int)

    order = ["서울","부산","대구","인천","광주","대전","울산","세종","수도권","지방"]
    out["__order__"] = out["권역"].apply(lambda x: order.index(x) if x in order else 999)
    out = out.sort_values("__order__").drop(columns=["__order__","category"])

    cols_show = ["권역"] + [c for c in ["≤2년(억원)","2~5년(억원)","5~10년(억원)",">10년(억원)"] if c in out.columns] + ["거래건수"]
    st.dataframe(out[cols_show], use_container_width=True)
