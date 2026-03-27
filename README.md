# Quant Market Dashboard

## 설치

pip install -r requirements.txt

## 로컬 .env 설정

cp .env.example .env

.env 파일에 아래 키를 입력하세요.

- ECOS_API_KEY=your_ecos_api_key_here
- FRED_API_KEY=your_fred_api_key_here

## 금리 API 키 설정

1. 한국은행 ECOS: https://ecos.bok.or.kr/api/ 에서 키 발급
2. 미국 FRED: https://fred.stlouisfed.org/docs/api/api_key.html 에서 무료 키 발급
3. .env에 값을 저장

참고: 한국/일본 채권 데이터는 아래 소스를 사용합니다.

- 한국: ECOS 시장금리(일별) API
- 일본: 일본 재무성 MOF CSV
- 미국: FRED Treasury Yield 시리즈

코드에서는 환경변수 ECOS_API_KEY, FRED_API_KEY를 우선 사용합니다.

## 최초 실행 (데이터 수집)

python scheduler.py --run-now

채권 데이터는 data/rates.parquet 에 장기 포맷으로 저장됩니다.
국가별 만기 구성이 달라도 대시보드에서 각 국가의 실제 만기축으로 금리커브를 그립니다.

## 대시보드 실행

streamlit run dashboard.py

## 주간 자동 수집 스케줄러 실행 (백그라운드)

python scheduler.py
# 매주 월요일 07:00 KST에 자동 수집

## GitHub Actions (Secrets 사용)

워크플로 파일: .github/workflows/weekly-collect.yml

GitHub Repository Secrets에 아래 키를 등록하세요.

- ECOS_API_KEY
- FRED_API_KEY

워크플로는 Secrets의 ECOS_API_KEY, FRED_API_KEY를 사용해 수집을 실행하고,
data/*.parquet 변경이 있으면 자동 커밋/푸시합니다.
