"""
Sector ETF Price Data Downloader
- Yahoo Finance (yfinance) 사용
- Pivot Table 형식 (date × ticker)
- 매일 최근 30일 데이터로 업데이트
- 초기 설정: python download_sector_etf.py --init (2001-01-01부터 모든 데이터)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse

ETF_LIST = [
    ("XLK",  "기술 하드웨어+소프트웨어",       "Technology Select Sector SPDR",          "1998-12"),
    ("IGV",  "소프트웨어",                      "iShares Expanded Tech-Software",         "2001-07"),
    ("VOX",  "정보통신서비스",                   "Vanguard Communication Services",        "2004-09"),
    ("XLC",  "미디어",                          "Communication Services Select SPDR",     "2018-06"),
    ("XRT",  "유통산업",                         "SPDR S&P Retail",                        "2006-06"),
    ("IBB",  "제약·바이오",                      "iShares Nasdaq Biotechnology",           "2001-02"),
    ("XLB",  "원자재산업",                       "Materials Select Sector SPDR",           "1998-12"),
    ("IAK",  "보험산업",                         "iShares U.S. Insurance",                 "2005-05"),
    ("XLP",  "가정용품·음식료·식품유통",          "Consumer Staples Select Sector SPDR",    "1998-12"),
    ("XLY",  "소비자서비스·내구소비재·자동차",    "Consumer Discretionary Select SPDR",     "1998-12"),
    ("XLV",  "헬스케어",                         "Health Care Select Sector SPDR",         "1998-12"),
    ("XLF",  "다각화금융",                       "Financial Select Sector SPDR",           "1998-12"),
    ("KBE",  "은행산업",                         "SPDR S&P Bank",                          "2005-11"),
    ("XLI",  "자본재·상업전문서비스",             "Industrials Select Sector SPDR",         "1998-12"),
    ("XLE",  "에너지산업",                       "Energy Select Sector SPDR",              "1998-12"),
    ("XLU",  "유틸리티",                         "Utilities Select Sector SPDR",           "1998-12"),
    ("IYT",  "운송",                             "iShares Transportation Average",         "2003-10"),
]

OUTPUT = "data/sector_etf_prices.csv"


def download_data(start_date, end_date):
    """모든 ETF 데이터 다운로드"""
    all_frames = []

    for ticker, industry, name, inception in ETF_LIST:
        print(f"  [{ticker}] {industry} 다운로드 중...", end=" ")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                print("데이터 없음 — 스킵")
                continue

            # 멀티인덱스 컬럼 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.rename(columns={"Date": "date", "Adj Close": "price"}, inplace=True)
            df = df[["date", "price"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ticker"] = ticker
            df["price"] = df["price"].round(4)

            all_frames.append(df)
            print(f"{len(df):,} rows ({df['date'].min()} ~ {df['date'].max()})")

        except Exception as e:
            print(f"오류: {e}")

    if not all_frames:
        return None

    result = pd.concat(all_frames, ignore_index=True)
    return result


def to_pivot_table(df):
    """데이터를 pivot table 형식으로 변환 (date × ticker)"""
    if df is None or df.empty:
        return None
    
    pivot = df.pivot_table(
        index="date",
        columns="ticker",
        values="price",
        aggfunc="last"  # 같은 날짜에 여러 값이 있으면 마지막 값 사용
    )
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("date").reset_index(drop=True)
    return pivot


def merge_with_existing(new_pivot, csv_path):
    """새 데이터를 기존 CSV와 병합"""
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        new_pivot["date"] = pd.to_datetime(new_pivot["date"]).dt.strftime("%Y-%m-%d")
        
        # 기존 데이터에서 새 데이터 범위 이전의 데이터만 유지
        min_new_date = new_pivot["date"].min()
        existing = existing[existing["date"] < min_new_date]
        
        # 병합
        merged = pd.concat([existing, new_pivot], ignore_index=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        
        return merged
    else:
        return new_pivot


def main(init_mode=False):
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    if init_mode:
        print("🔄 초기 모드: 2001-01-01부터 모든 데이터 수집 중...")
        start_date = "2001-01-01"
    else:
        print("🔄 일일 업데이트 모드: 최근 30일 데이터 수집 중...")
        start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # 데이터 다운로드
    df = download_data(start_date, end_date)
    if df is None:
        print("수집된 데이터가 없습니다.")
        return
    
    # Pivot table로 변환
    pivot = to_pivot_table(df)
    
    # 기존 데이터와 병합 (일일 모드에서만)
    if not init_mode:
        pivot = merge_with_existing(pivot, OUTPUT)
    
    # CSV 저장
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    pivot.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 저장 완료: {OUTPUT}")
    print(f"📊 데이터: {len(pivot):,} rows × {len(pivot.columns)-1} tickers")
    print(f"📅 기간: {pivot['date'].min()} ~ {pivot['date'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector ETF Price Downloader")
    parser.add_argument("--init", action="store_true", help="초기 설정 모드 (2001-01-01부터 모든 데이터)")
    args = parser.parse_args()
    
    main(init_mode=args.init)
