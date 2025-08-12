# 전국 시군구 실거래가 비교 지도 (Streamlit + GitHub Actions)

이 저장소는 **국토교통부 RTMS 아파트 매매 실거래가 API** 데이터를
정기적으로 수집(ETL)하여 CSV/Parquet로 저장하고, **Streamlit** 앱에서
읽어 지도 및 대시보드로 시각화하는 예제 프로젝트입니다.

## 구성
- `etl_fetch.py` — API 수집 스크립트(월별, 시군구별 XML → CSV/Parquet)
- `streamlit_app.py` — 지도/필터/타입별 평균가, 구축/신축 집계
- `.github/workflows/fetch.yml` — GitHub Actions: 매일/수동 실행으로 데이터 갱신
- `requirements.txt` — 의존성
- `data/` — ETL 산출물 저장 디렉토리(커밋됨): `trades_YYYYMM.csv`, `all_trades.parquet` 등

## API 키 설정
- **절대 코드에 인증키를 직접 하드코딩하지 마세요.**
- GitHub 저장소 Settings → Secrets and variables → Actions → **New repository secret**
  - 이름: `MOLIT_API_KEY`
  - 값: 본인의 서비스키(예: `d4Cwk6...`)
- 로컬 실행 시: `export MOLIT_API_KEY="여러분의키"` (Windows PowerShell: `$env:MOLIT_API_KEY="..."`)

## 입력 파일
- `lawd_codes.csv` — 전국 시군구 코드 테이블(열: `LAWD_CD`,`SIGUNGU_NM`,`SIDO_CD`,`SIDO_NM` ...). 루트에 두세요.
- `sigungu_centroids.csv` — 시군구 중심좌표 테이블(열: `LAWD_CD`,`lon`,`lat`).
  - 앱은 이 좌표를 사용해 시군구 점 지도(센트로이드)를 그립니다.

> 둘 다 사용자가 제공한 파일을 그대로 사용하도록 설계했습니다.

## 실행 방법

### 1) 로컬에서 ETL 실행
```bash
pip install -r requirements.txt
export MOLIT_API_KEY="여러분의키"   # Windows는 set / $env: 참고
python etl_fetch.py --months 12
```

### 2) Streamlit 앱 실행
```bash
streamlit run streamlit_app.py
```

### 3) GitHub Actions로 자동 수집
- 기본 설정: 매일 새벽 4시(Asia/Seoul) 기준 19시 UTC에 실행되도록 설정되어 있습니다.
- 수동 실행도 가능합니다(딱 한 번 Run workflow).

## 앱 기능
- **기간 버튼**: ① 1년~6개월, ② 6개월~3개월, ③ 3개월~1개월, ④ 최근 1개월
- **권역 필터**: 수도권 / 특·광역시 / 지방
- **구축/신축(10년 기준)**, **타입(전용면적 구간)**별 평균가 및 거래건수
- **지도**: 시군구 센트로이드에 집계값(예: 최근월 평균가)을 색/크기로 표시

## 참고
- API 엔드포인트 예시 (XML):
  `https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade?LAWD_CD=11110&DEAL_YMD=202508&serviceKey=...`
- 파라미터: `LAWD_CD`(시군구), `DEAL_YMD`(YYYYMM), `serviceKey`(인증키)

