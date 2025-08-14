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

# 시군구 경계 GeoJSON (로컬 → 원격; 필요시 경로 수정)
try:
    sgg = load_geojson_local("sgg.geojson")
except Exception:
    sgg = load_geojson_url("sgg.geojson")

# 구분 폴리곤 GeoJSON (서울/부산/.../수도권/지방)
try:
    sgg_type = load_geojson_local("sgg_type.geojson")
except Exception:
    sgg_type = load_geojson_url("sgg_type.geojson")

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

# ✅ 평형 기준 선택 (ETL에서 area_band를 25/31/35평형으로 생성해야 함)
area_choices = ["25평형","31평형","35평형"]
area_band = st.sidebar.multiselect("평형", area_choices, default=area_choices)

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
# 필터 적용 (지도용과 표용 분리)
# -------------------------
df = agg.copy()
df = df[df["period_bucket"].eq(period)]

# 지도 전용 df (신축/구축 라디오 반영)
if new_old != "전체":
    df_map = df[df["new_old"].eq(new_old)].copy()
else:
    df_map = df.copy()

# 표 전용 df (✅ 신축/구축 필터 제거: 전체/신축/구축을 동시에 보여주기 위함)
df_table_src = df.copy()

# 공통 필터: 평형/권역
if area_band:
    df_map = df_map[df_map["area_band"].isin(area_band)]
    df_table_src = df_table_src[df_table_src["area_band"].isin(area_band)]

if region_tab != "전국" and "region_group" in df.columns:
    df_map = df_map[df_map["region_group"].eq(region_tab)]
    df_table_src = df_table_src[df_table_src["region_group"].eq(region_tab)]

# -------------------------
# 지도용 시군구 대표값 (동일 시군구 다건 → 평균/합산)
# -------------------------
group_cols = ["LAWD_CD","sido_nm","sigungu_nm"]
agg_dict = {value_col: "mean", "n_trades": "sum"}
map_df = df_map.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

# 숫자형 보정
map_df[value_col] = pd.to_numeric(map_df[value_col], errors="coerce")
map_df["n_trades"] = pd.to_numeric(map_df["n_trades"], errors="coerce").fillna(0).astype(int)

# 값 범위 (색상 계산용)
val_min = float(map_df[value_col].min()) if map_df[value_col].notna().any() else 0.0
val_max = float(map_df[value_col].max()) if map_df[value_col].notna().any() else 1.0
if val_min == val_max:
    val_max = val_min + 1.0

# 이름 기반 백업 매칭을 위해 정규화된 이름-코드 딕셔너리 생성 (지도 매칭 보조)
map_df["_name_norm"] = (map_df["sido_nm"].fillna("") + map_df["sigungu_nm"].fillna("")).apply(norm_name)
name_to_code = dict(zip(map_df["_name_norm"], map_df["LAWD_CD"]))

# -------------------------
# GeoJSON 주입 + 색상/툴팁 평탄화 (시군구 지도)
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
    # 2) 이름 기반 백업 매칭
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
# 지도 ①: 시군구 경계 지도
# -------------------------
st.markdown(
    f"## 전국 실거래가 지도 — {period} · {new_old} · "
    f"{' · '.join(area_band) if area_band else '전체'} · {metric_label}"
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

with st.expander("매칭 상태(디버그)"):
    total_feat = len(sgg_joined.get("features", []))
    st.write(f"GeoJSON 피처 수: {total_feat} | 값 매칭된 피처: {matched}")

# ======================================================
# ② 구분 지도(서울/부산/.../수도권/지방) 추가
# ======================================================

# 1) 카테고리별 LAWD_CD 목록 (사용자 제공)
CATEGORY_CODES = {
    "서울":  ['11110','11140','11170','11200','11215','11230','11260','11290','11305','11320','11350','11380','11410','11440','11470','11500','11530','11545','11560','11590','11620','11650','11680','11710','11740'],
    "부산":  ['26110','26140','26170','26200','26230','26260','26290','26320','26350','26380','26410','26440','26470','26500','26530','26710'],
    "대구":  ['27110','27140','27170','27200','27230','27260','27290','27710','27720'],
    "인천":  ['28110','28140','28177','28185','28200','28237','28245','28260','28710','28720'],
    "광주":  ['29110','29140','29155','29170','29200'],
    "대전":  ['30110','30140','30170','30200','30230'],
    "울산":  ['31110','31140','31170','31200','31710'],
    "세종":  ['36110'],
    "수도권":['41111','41113','41115','41117','41131','41133','41135','41150','41171','41173','41210','41220','41250','41271','41273','41281','41285','41287','41290','41310','41360','41370','41390','41410','41430','41450','41461','41463','41465','41480','41500','41550','41570','41590','41610','41630','41650','41670','41800','41820','41830','41192','41194','41196'],
    "지방":  ['51110','51130','51150','51170','51190','51210','51230','51720','51730','51750','51760','51770','51780','51790','51800','51810','51820','51830','43130','43150','43111','43112','43113','43114','43720','43730','43740','43750','43760','43770','43800','43745','44131','44133','44150','44180','44200','44210','44230','44250','44270','44710','44760','44770','44790','44800','44810','44825','46110','46130','46150','46170','46230','46710','46720','46730','46770','46780','46790','46800','46810','46820','46830','46840','46860','46870','46880','46890','46900','46910','47111','47113','47130','47150','47170','47190','47210','47230','47250','47280','47290','47730','47750','47760','47770','47820','47830','47840','47850','47900','47920','47930','47940','48170','48220','48240','48250','48270','48310','48330','48121','48123','48125','48127','48129','48720','48730','48740','48820','48840','48850','48860','48870','48880','48890','50110','50130','52111','52113','52130','52140','52180','52190','52210','52710','52720','52730','52740','52750','52770','52790','52800'],
}

# 2) 지도에서 사용 중인 값 소스(map_df) → 카테고리 가중평균/거래건수 집계
#    - avg_price_krw는 거래건수 기준 가중평균이 맞음
map_core = df_map[["LAWD_CD", value_col, "n_trades"]].dropna(subset=[value_col, "n_trades"]).copy()
map_core["LAWD_CD"] = map_core["LAWD_CD"].astype(str).str.zfill(5)
map_core["w_sum"] = map_core[value_col] * map_core["n_trades"]

cat_rows = []
for cat, codes in CATEGORY_CODES.items():
    codes = [str(c).zfill(5) for c in codes]
    sub = map_core[map_core["LAWD_CD"].isin(codes)]
    if len(sub) == 0 or sub["n_trades"].sum() == 0:
        wavg = np.nan
        nsum = 0
    else:
        wavg = sub["w_sum"].sum() / sub["n_trades"].sum()
        nsum = int(sub["n_trades"].sum())
    cat_rows.append({"category": cat, "wavg": wavg, "n_trades": nsum})

cat_df = pd.DataFrame(cat_rows)

# 3) 구분 GeoJSON에 값 주입(이름 매칭; name/type/label 등 유연 매칭)
def extract_type_name(props):
    return get_prop(props, ["name","NAME","type","TYPE","label","LABEL","sgg_type","SGG_TYPE"])

# 색상 스케일(카테고리 값 기준)
cat_vmin = float(cat_df["wavg"].min()) if cat_df["wavg"].notna().any() else 0.0
cat_vmax = float(cat_df["wavg"].max()) if cat_df["wavg"].notna().any() else 1.0
if cat_vmin == cat_vmax:
    cat_vmax = cat_vmin + 1.0

def color_scale_cat(v, vmin, vmax):
    if v is None or np.isnan(v):
        return [220, 220, 220, 100]
    t = (v - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    r = int(240 + (30 - 240) * t)   # 연핑크 → 보라 계열 예시
    g = int(220 + (70 - 220) * t)
    b = int(240 + (180 - 240) * t)
    return [r, g, b, 160]

cat_value = dict(zip(cat_df["category"], cat_df["wavg"]))
cat_trades = dict(zip(cat_df["category"], cat_df["n_trades"]))

sgg_type_joined = deepcopy(sgg_type)
matched_cat = 0
for ft in sgg_type_joined.get("features", []):
    props = ft.get("properties", {}) or {}
    nm = extract_type_name(props)
    if nm in cat_value:
        v = cat_value[nm]
        n = cat_trades.get(nm, 0)
        matched_cat += 1
    else:
        v, n = np.nan, 0

    props["group_name"] = nm or ""
    props["val"] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
    props["val_str"] = fmt_eok(v)
    props["n_trades"] = int(n)
    props["n_trades_str"] = fmt_int(n)
    props["fill_color"] = color_scale_cat(v, cat_vmin, cat_vmax)

# 4) 구분 지도 렌더
st.markdown("## 구분 지도 — 서울·부산·대구·인천·광주·대전·울산·세종·수도권·지방")

type_layer = pdk.Layer(
    "GeoJsonLayer",
    sgg_type_joined,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[100, 100, 120, 140],
    line_width_min_pixels=1.5,
)

type_tooltip = {
    "html": "<b>{group_name}</b><br/>"
            + f"{metric_label}: " + "{val_str}<br/>거래건수: {n_trades_str}",
    "style": {"backgroundColor": "white", "color": "black"}
}

type_deck = pdk.Deck(
    layers=[type_layer],
    initial_view_state=pdk.ViewState(latitude=36.5, longitude=127.8, zoom=6),
    tooltip=type_tooltip
)
st.pydeck_chart(type_deck, use_container_width=True)

with st.expander("구분 매칭 상태(디버그)"):
    total_feat2 = len(sgg_type_joined.get("features", []))
    st.write(f"구분 폴리곤 수: {total_feat2} | 값 매칭된 구분: {matched_cat}")

# -------------------------
# 표 (선택 시군구만 표시 / ✅ 전체·신축·구축 모두)
# -------------------------
st.markdown("### 시군구별 요약 (전체·신축·구축)")

# 1) 시군구×new_old별 가중평균(거래건수 기준) 계산
tmp = df_table_src.copy()
tmp = tmp[["LAWD_CD","sido_nm","sigungu_nm","new_old","avg_price_krw","n_trades"]].dropna(subset=["avg_price_krw","n_trades"])
tmp["w_sum"] = tmp["avg_price_krw"] * tmp["n_trades"]

g = (tmp
     .groupby(["LAWD_CD","sido_nm","sigungu_nm","new_old"], dropna=False)
     .agg(w_sum=("w_sum","sum"), n=("n_trades","sum"))
     .reset_index())
g["wavg"] = g["w_sum"] / g["n"]

# 2) 전체(신축+구축 합) 가중평균도 계산
all_g = (g
         .groupby(["LAWD_CD","sido_nm","sigungu_nm"], dropna=False)
         .agg(w_sum=("w_sum","sum"), n=("n","sum"))
         .reset_index())
all_g["wavg_all"] = all_g["w_sum"] / all_g["n"]

# 3) 피벗으로 신축/구축 칼럼화
pivot = g.pivot_table(index=["LAWD_CD","sido_nm","sigungu_nm"],
                      columns="new_old", values="wavg", aggfunc="first").reset_index()

# 4) 전체와 병합
table = pivot.merge(all_g[["LAWD_CD","wavg_all","n"]], on="LAWD_CD", how="left")

# 5) 컬럼 정리(라벨 표준화)
col_new = "신축(≤10년)" if "신축(≤10년)" in table.columns else ("신축" if "신축" in table.columns else None)
col_old = "구축(>10년)" if "구축(>10년)" in table.columns else ("구축" if "구축" in table.columns else None)

rename_map = {
    "sido_nm":"시도",
    "sigungu_nm":"시군구",
    "wavg_all":"전체(억원)",
    "n":"거래건수"
}
if col_new: rename_map[col_new] = "신축(억원)"
if col_old: rename_map[col_old] = "구축(억원)"
table = table.rename(columns=rename_map)

# 6) 선택 시군구가 있으면 그 코드만 필터
if selected_labels:
    selected_codes = [str(options[label]).zfill(5) for label in selected_labels]
    table = table[table["LAWD_CD"].isin(selected_codes)]

# 7) 단위 변환/포맷(억원) + 정렬
for c in ["전체(억원)","신축(억원)","구축(억원)"]:
    if c in table.columns:
        table[c] = table[c].map(fmt_eok)

table["거래건수"] = table["거래건수"].fillna(0).astype(int).map(fmt_int)

# 정렬용 보조열(전체 기준 내림차순)
sort_vals = all_g.set_index("LAWD_CD")["wavg_all"]
table = (table
         .join(sort_vals.rename("_sort"), on="LAWD_CD")
         .sort_values("_sort", ascending=False)
         .drop(columns=["_sort","LAWD_CD"]))

st.dataframe(table[["시도","시군구"] +
                  ([ "전체(억원)"] if "전체(억원)" in table.columns else []) +
                  ([ "신축(억원)"] if "신축(억원)" in table.columns else []) +
                  ([ "구축(억원)"] if "구축(억원)" in table.columns else []) +
                  ["거래건수"]],
             use_container_width=True)
