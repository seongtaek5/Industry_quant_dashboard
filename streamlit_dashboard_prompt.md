# Streamlit Dashboard 개발 프롬프트
## 미국 산업별 Valuation & Momentum Dashboard

---

## 1. 프로젝트 개요

미국 GICS 기준 산업별 **PBR(Valuation)** 과 **ETF 모멘텀** 을 동시에 분석하는 Streamlit 대시보드를 개발한다.
두 지표 모두 **Time-Series Rolling Z-score** 와 **Cross-Sectional Min-Max Scaler(-1~1)** 로 표준화하여 시각화한다.

---

## 2. 데이터 소스 및 구조

### 2-1. PBR 데이터 (`US_IND_PBR.xlsx`)

- **시트명**: `값`
- **구조 (헤더가 여러 줄에 걸쳐 있음 — 아래 파싱 로직 필수)**:
  - Row 1: `PX_TO_BOOK_RATIO` (메타)
  - Row 4: 산업명(한글) — col index 3번째부터 (0-indexed)
  - Row 5: 블룸버그 티커 (`S5TELSX Index` 등)
  - Row 7~끝: 실제 데이터. **col index 2** = 날짜(`datetime`), **col index 3~24** = 각 산업 PBR 값
- **주기**: 일별(Daily)
- **기간**: 2001-01-01 ~ 현재
- **22개 산업 컬럼 순서** (col 3~24):

  | 한글명 | 블룸버그 티커 |
  |---|---|
  | 정보통신서비스 | S5TELSX Index |
  | 기술 하드웨어 | S5TECH Index |
  | 소프트웨어 | S5SFTW Index |
  | 유통산업 | S5RETL Index |
  | 제약생명공학 | S5PHRM Index |
  | 미디어산업 | S5MEDA Index |
  | 원자재산업 | S5MATRX Index |
  | 보험산업 | S5INSU Index |
  | 가정용품 | S5HOUS Index |
  | 소비자서비스 | S5HOTR Index |
  | 헬스케어 | S5HCES Index |
  | 식품 | S5FDSR Index |
  | 음식료담배 | S5FDBT Index |
  | 다각화금융 | S5DIVF Index |
  | 자본재산업 | S5CPGS Index |
  | 상업전문서비스 | S5COMS Index |
  | 내구소비재 | S5CODU Index |
  | 은행산업 | S5BANKX Index |
  | 자동차와부품 | S5AUCO Index |
  | 에너지산업 | S5ENRSX Index |
  | 유틸리티 | S5UTILX Index |
  | 운송 | S5TRAN Index |

**PBR 파싱 코드 예시**:
```python
import pandas as pd
from openpyxl import load_workbook
import datetime

wb = load_workbook("US_IND_PBR.xlsx", read_only=True)
ws = wb["값"]
rows = list(ws.iter_rows(values_only=True))

# 헤더 추출
industry_names = [v for v in rows[3][3:] if v is not None]   # row index 3 = 4번째 줄
bbg_tickers    = [v for v in rows[4][3:] if v is not None]   # row index 4 = 5번째 줄

# 데이터 추출 (row index 6부터)
data_rows = []
for r in rows[6:]:
    if r[2] is not None and isinstance(r[2], datetime.datetime):
        data_rows.append([r[2].date()] + list(r[3:3+len(industry_names)]))

pbr_df = pd.DataFrame(data_rows, columns=["date"] + industry_names)
pbr_df["date"] = pd.to_datetime(pbr_df["date"])
pbr_df = pbr_df.set_index("date").sort_index()
pbr_df = pbr_df.apply(pd.to_numeric, errors="coerce")
```

---

### 2-2. ETF 가격 데이터

앱 실행 시 **`yfinance`로 전체 기간 자동 다운로드** 한다 (`sector_etf_prices.csv` 파일은 데이터가 1개월치밖에 없으므로 사용 불가).
다운로드 완료 후 로컬 캐시(`etf_cache.parquet`)로 저장하여 재실행 시 재다운로드 없이 로드한다.

**ETF 매핑 테이블** (22개 GICS 산업 → 17개 ETF, 일부 ETF가 여러 산업 커버):

| 산업(한글) | ETF 티커 |
|---|---|
| 기술 하드웨어 | XLK |
| 소프트웨어 | IGV |
| 정보통신서비스 | VOX |
| 미디어산업 | XLC |
| 유통산업 | XRT |
| 제약생명공학 | IBB |
| 원자재산업 | XLB |
| 보험산업 | IAK |
| 가정용품 | XLP |
| 음식료담배 | XLP |
| 식품 | XLP |
| 소비자서비스 | XLY |
| 내구소비재 | XLY |
| 자동차와부품 | XLY |
| 헬스케어 | XLV |
| 다각화금융 | XLF |
| 은행산업 | KBE |
| 자본재산업 | XLI |
| 상업전문서비스 | XLI |
| 에너지산업 | XLE |
| 유틸리티 | XLU |
| 운송 | IYT |

**ETF 다운로드 코드 예시**:
```python
import yfinance as yf
import pandas as pd
import os

TICKERS = ["XLK","IGV","VOX","XLC","XRT","IBB","XLB","IAK",
           "XLP","XLY","XLV","XLF","KBE","XLI","XLE","XLU","IYT"]
CACHE = "etf_cache.parquet"

@st.cache_data(ttl=3600)
def load_etf_data():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    raw = yf.download(TICKERS, start="1998-01-01", auto_adjust=False, progress=False)
    adj = raw["Adj Close"].sort_index()
    adj.to_parquet(CACHE)
    return adj
```

---

## 3. 지표 계산 로직

### 3-1. 공통 전처리

- PBR과 ETF 가격 모두 **월말(Month-End) 리샘플링** 후 분석
  - PBR: `pbr_df.resample("ME").last()`
  - ETF: `adj_df.resample("ME").last()`
- 두 데이터의 **공통 날짜 구간**으로 align

---

### 3-2. Time-Series Rolling Z-score

$$Z_{t} = \frac{X_t - \mu_{[t-W, t]}}{\sigma_{[t-W, t]}}$$

- **윈도우 W**: 사용자가 슬라이더로 설정 (단위: 개월, 범위: 12~120, 기본값: 60)
- **PBR Z-score**: 각 산업별로 개별 계산 (22개 시계열 각각)
- **Momentum Z-score**: 각 ETF별 누적수익률(아래 3-3 참고)에 동일 윈도우 적용

```python
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=int(window * 0.5))
    return (series - roll.mean()) / roll.std()
```

---

### 3-3. 모멘텀 계산

- **모멘텀 기간**: 사용자가 선택 (1 / 3 / 6 / 12개월, 복수 선택 가능)
- **계산 방식**: 월말 adj_close 기준 단순 누적수익률

$$Mom_t^{(k)} = \frac{P_t}{P_{t-k}} - 1$$

- 여러 기간 선택 시 **평균 모멘텀** 사용:

$$Mom_t = \frac{1}{|K|} \sum_{k \in K} Mom_t^{(k)}$$

---

### 3-4. Cross-Sectional Min-Max Scaler (-1 ~ 1)

특정 시점 t에서 **전체 산업 횡단면(cross-section)** 기준으로 정규화:

$$CS\_Score_{i,t} = 2 \times \frac{X_{i,t} - \min_j(X_{j,t})}{\max_j(X_{j,t}) - \min_j(X_{j,t})} - 1$$

- **PBR CS Score**: 각 날짜별로 22개 산업 PBR 값에 적용
- **Momentum CS Score**: 각 날짜별로 22개 산업 모멘텀 값에 적용

```python
def cs_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """행(row) = 날짜, 열(col) = 산업. 날짜별 횡단면 정규화."""
    mn = df.min(axis=1)
    mx = df.max(axis=1)
    scaled = df.sub(mn, axis=0).div((mx - mn).replace(0, 1), axis=0)
    return scaled * 2 - 1
```

---

## 4. 대시보드 구성

### 4-1. 사이드바 (Controls)

```
[ Rolling Z-score 윈도우 ]  슬라이더: 12 ~ 120개월  기본: 60
[ 모멘텀 기간 ]             멀티셀렉트: 1M / 3M / 6M / 12M  기본: [3M, 6M, 12M]
[ 기준 날짜 ]               날짜 선택기 (기본: 최신 날짜)
```

---

### 4-2. 탭 구성

```
Tab 1: 📊 Overview (Heatmap)
Tab 2: 🎯 Snapshot Ranking
Tab 3: 🔵 Scatter Plot
Tab 4: 📈 Time-Series
```

---

### Tab 1: Overview — Heatmap

**두 개의 히트맵**을 나란히 배치:

| 왼쪽 | 오른쪽 |
|---|---|
| PBR Rolling Z-score | Momentum Rolling Z-score |
| X축: 최근 N개월 날짜 | X축: 최근 N개월 날짜 |
| Y축: 22개 산업 | Y축: 22개 산업 |

- 색상 스케일: `RdYlGn` (빨강=낮음, 초록=높음)
- 각 셀에 수치 표기 (소수점 1자리)
- 기준 날짜 기준 **최근 24개월** 기본 표시
- 표시 개월 수 슬라이더 추가 (6~60개월)

---

### Tab 2: Snapshot Ranking — 테이블

**기준 날짜** 기준 현재 스냅샷:

| 컬럼 | 내용 |
|---|---|
| 산업명 | 한글 |
| ETF | 매핑된 티커 |
| PBR | 원시 값 |
| PBR Z-score | Rolling Z-score |
| PBR CS Score | Cross-sectional (-1~1) |
| Momentum | 누적수익률 (%) |
| Mom Z-score | Rolling Z-score |
| Mom CS Score | Cross-sectional (-1~1) |
| 종합 Score | `(PBR_CS × -1 + Mom_CS) / 2` ← 저평가+고모멘텀일수록 높음 |

- **종합 Score 기준 내림차순 정렬**
- 색상 조건부 서식: 양수=초록, 음수=빨강 (`st.dataframe` styler 사용)
- CSV 다운로드 버튼 추가

---

### Tab 3: Scatter Plot

**X축**: PBR Rolling Z-score (기준 날짜 기준)
**Y축**: Momentum CS Score (기준 날짜 기준)
**버블 크기**: 고정 또는 PBR 절대값 비례
**버블 색상**: 종합 Score에 따른 컬러맵 (빨강→초록)
**레이블**: 산업명(한글) + ETF 티커

- 4사분면에 라벨 표기:
  - 1사분면 (우상): "고밸류 + 고모멘텀 (Momentum Play)"
  - 2사분면 (좌상): "저밸류 + 고모멘텀 ✅ (Sweet Spot)"
  - 3사분면 (좌하): "저밸류 + 저모멘텀 (Value Trap)"
  - 4사분면 (우하): "고밸류 + 저모멘텀 ⚠️ (Avoid)"
- X=0, Y=0 기준선 표시
- Plotly Express 사용 (`px.scatter`)

---

### Tab 4: Time-Series

- 사이드바 또는 탭 내부에서 **산업 멀티셀렉트** (최대 5개)
- 선택한 산업의 아래 4개 시계열을 **2×2 subplot**으로 표시:
  1. PBR 원시값
  2. PBR Rolling Z-score
  3. Momentum (누적수익률 %)
  4. Momentum Rolling Z-score
- Plotly `make_subplots` 사용, 호버 툴팁 포함

---

## 5. 기술 스택 및 패키지

```
streamlit
pandas
numpy
plotly
yfinance
openpyxl
pyarrow   # parquet 캐시용
```

---

## 6. 파일 구조

```
project/
├── app.py                  # 메인 Streamlit 앱
├── data_loader.py          # PBR 파싱, ETF 다운로드/캐시
├── indicators.py           # rolling_zscore, cs_minmax, momentum 계산
├── US_IND_PBR.xlsx         # PBR 원본 데이터 (사용자 제공)
└── etf_cache.parquet       # ETF 가격 캐시 (자동 생성)
```

---

## 7. 주요 구현 주의사항

1. **ETF-산업 매핑**: 여러 산업이 동일 ETF를 공유하므로 (예: XLK → 기술하드웨어 + 소프트웨어), 모멘텀 계산은 ETF 단위로 하되 산업 단위로 assign
2. **데이터 align**: PBR(22개 산업)과 ETF(17개 티커 → 22개 매핑)을 월말 기준으로 merge할 때 공통 날짜만 사용, NaN 처리 명시
3. **Rolling Z-score min_periods**: `window * 0.5`로 설정하여 초기 구간 NaN 최소화
4. **Cross-sectional 계산**: 행(row) 방향으로 min/max 적용 — `axis=1`
5. **Streamlit 캐싱**: `@st.cache_data`로 데이터 로딩 및 지표 계산 캐싱, 윈도우/모멘텀 기간 변경 시 자동 재계산
6. **종합 Score**: PBR이 낮을수록 매력적이므로 `PBR_CS × (-1)` 로 방향 반전 후 Momentum CS와 평균
