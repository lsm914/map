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

# --- 카테고리명 표준화(구분 지도용; 한글/영문/별칭 폭넓게 대응) ---
def normalize_category_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # 흔한 구분/기호 제거
    for tok in ["특별자치시","광역시","특별시"," ", "·", "-", "_"]:
        s = s.replace(tok, "")
    # 영문 별칭 매핑(전부 소문자로 비교)
    low = s.lower()
    eng_alias = {
        "seoul": "서울",
        "busan": "부산",
        "daegu": "대구",
        "incheon": "인천",
        "gwangju": "광주",
        "daejeon": "대전",
        "ulsan": "울산",
        "sejong": "세종",
        "capitalregion": "수도권",
        "capitalarea": "수도권",
        "metropolitanarea": "수도권",
        "gyeonggi": "수도권",
        "gyeonggido": "수도권",
        "gg": "수도권",
        "noncapital": "지방",
        "others": "지방",
        "regions": "지방",
    }
    if low in eng_alias:
        return eng_alias[low]
    # 한글 별칭 통일
    alias = {
        "서울":"서울","서울시":"서울","서울특별시":"서울",
        "부산":"부산","부산시":"부산","부산광역시":"부산",
        "대구":"대구","대구시":"대구","대구광역시":"대구",
        "인천":"인천","인천시":"인천","인천광역시":"인천",
        "광주":"광주","광주시":"광주","광주광역시":"광주",
        "대전":"대전","대전시":"대전","대전광역시":"대전",
        "울산":"울산","울산시":"울산","울산광역시":"울산",
        "세종":"세종","세종시":"세종","세종특별자치시":"세종",
        # 수도권/지방
        "수도권":"수도권","경기도":"수도권","경기":"수도권",
        "지방":"지방",
    }
    return alias.get(s, s)

def extract_type_name(props: dict) -> str:
    """
    sgg_type.geojson의 properties에서 카테고리 이름을 robust하게 추출
    1) 다양한 후보 키를 우선 탐색
    2) 없으면 문자열 속성 중 가장 길이가 긴 값을 선택 (폴백)
    3) 최종적으로 normalize_category_name으로 표준화
    """
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

# ✅ 평형 기준 선택 (ETL에서 area_band를 25/31/35평형으로 생성 필요)
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
# ② 구분 지도 준비(서울/부산/.../수도권/지방)
#   └ 시군구 집계(map_df) 기반으로 카테고리 가중평균(거래건수 가중)
# -------------------------
# 시군구 단위로 이미 집계된 map_df에서 다시 카테고리 가중평균을 계산
map_core = map_df[["LAWD_CD", value_col, "n_trades"]].dropna(subset=[value_col, "n_trades"]).copy()
map_core["LAWD_CD"] = map_core["LAWD_CD"].astype(str).str.zfill(5)
map_core["w_sum"] = map_core[value_col] * map_core["n_trades"]

# 앞 2자리 -> 카테고리 매핑 (광역단위/수도권/지방 자동 판정)
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
    if p2 == "41": return "수도권"   # 경기도는 '수도권'으로 묶기
    return "지방"                    # 나머지는 '지방'

map_core["category"] = map_core["LAWD_CD"].map(lawd_to_category)

# 카테고리별 가중평균(거래건수 가중)
cat_df = (
    map_core.groupby("category", dropna=False)
            .agg(w_sum=("w_sum", "sum"), n_trades=("n_trades", "sum"))
            .reset_index()
)
cat_df["wavg"] = np.where(cat_df["n_trades"] > 0, cat_df["w_sum"] / cat_df["n_trades"], np.nan)

# GeoJSON 속성명과 안전 매칭(서울특별시 → 서울 등)
cat_df["category_norm"] = cat_df["category"].map(normalize_category_name)
cat_value  = dict(zip(cat_df["category_norm"], cat_df["wavg"]))
cat_trades = dict(zip(cat_df["category_norm"], cat_df["n_trades"]))

# 색상 스케일(구분 지도용)
cat_vmin = float(cat_df["wavg"].min()) if cat_df["wavg"].notna().any() else 0.0
cat_vmax = float(cat_df["wavg"].max()) if cat_df["wavg"].notna().any() else 1.0
if cat_vmin == cat_vmax:
    cat_vmax = cat_vmin + 1.0

def color_scale_cat(v, vmin, vmax):
    if v is None or np.isnan(v):
        return [220, 220, 220, 100]
    t = (v - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    r = int(240 + (30 - 240) * t)   # 연핑크 → 보라 계열
    g = int(220 + (70 - 220) * t)
    b = int(240 + (180 - 240) * t)
    return [r, g, b, 160]

# sgg_type.geojson에 값 주입
sgg_type_joined = deepcopy(sgg_type)
matched_cat = 0
for ft in sgg_type_joined.get("features", []):
    props = ft.get("properties", {}) or {}
    nm = extract_type_name(props)  # 표준화된 카테고리명(서울/부산/…/수도권/지방)
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

# ---- 디버그: sgg_type 안의 이름이 무엇으로 읽히는지 확인 ----
detected = []
for ft in sgg_type.get("features", []):
    props = ft.get("properties", {}) or {}
    nm = extract_type_name(props)
    # 원시 속성 일부 샘플
    sample = {k: props.get(k) for k in list(props.keys())[:8]}
    detected.append({"detected_name": nm, "has_value": nm in cat_value, "raw_props_sample": sample})
debug_df = pd.DataFrame(detected).drop_duplicates("detected_name")

# -------------------------
# 두 지도를 좌우로 배치
# -------------------------
left, right = st.columns(2)

with left:
    st.markdown(
        f"## 시군구 지도 — {period} · {new_old} · "
        f"{' · '.join(area_band) if area_band else '전체'} · {metric_label}"
    )
    deck = pdk.Deck(
        layers=[pdk.Layer(
            "GeoJsonLayer",
            sgg_joined,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[120, 120, 140, 120],
            line_width_min_pixels=1,
        )],
        initial_view_state=pdk.ViewState(latitude=36.5, longitude=127.8, zoom=6),
        tooltip={
            "html": "<b>{name}</b><br/>"
                    + f"{metric_label}: " + "{metric_str}<br/>거래건수: {trades_str}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    )
    st.pydeck_chart(deck, use_container_width=True)
    with st.expander("시군구 매칭 상태(디버그)"):
        total_feat = len(sgg_joined.get("features", []))
        st.write(f"GeoJSON 피처 수: {total_feat} | 값 매칭된 피처: {matched}")

with right:
    st.markdown("## 구분 지도 — 서울·부산·대구·인천·광주·대전·울산·세종·수도권·지방")
    type_deck = pdk.Deck(
        layers=[pdk.Layer(
            "GeoJsonLayer",
            sgg_type_joined,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[100, 100, 120, 140],
            line_width_min_pixels=1.5,
        )],
        initial_view_state=pdk.ViewState(latitude=36.5, longitude=127.8, zoom=6),
        tooltip={
            "html": "<b>{group_name}</b><br/>"
                    + f"{metric_label}: " + "{val_str}<br/>거래건수: {n_trades_str}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    )
    st.pydeck_chart(type_deck, use_container_width=True)
    with st.expander("구분 매칭 상태(디버그)"):
        total_feat2 = len(sgg_type_joined.get("features", []))
        st.write(f"구분 폴리곤 수: {total_feat2} | 값 매칭된 구분: {matched_cat}")
        st.write("cat_df(집계):", cat_df.assign(억원=cat_df["wavg"].map(fmt_eok)))
        st.write("sgg_type에서 읽힌 이름:", debug_df)

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
