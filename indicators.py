"""
시계열 지표 계산 모듈
- Rolling Z-score
- Momentum (누적수익률)
- Cross-Sectional Min-Max Scaler
"""

import pandas as pd
import numpy as np


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Time-Series Rolling Z-score 계산
    
    Parameters:
    -----------
    series : pd.Series
        입력 시계열
    window : int
        윈도우 크기 (개월)
    
    Returns:
    --------
    pd.Series
        Z-score 결과
    """
    roll = series.rolling(window, min_periods=int(window * 0.5))
    z_score = (series - roll.mean()) / roll.std()
    return z_score


def rolling_zscore_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    DataFrame의 각 열에 rolling Z-score 적용
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 DataFrame (각 열 = 시계열)
    window : int
        윈도우 크기 (개월)
    
    Returns:
    --------
    pd.DataFrame
        Z-score 결과
    """
    return df.apply(lambda col: rolling_zscore(col, window), axis=0)


def calculate_momentum(price_df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    Momentum 계산 (누적수익률)
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        가격 데이터 (Index: date, Columns: 자산)
    periods : list
        기간 리스트 (개월) [1, 3, 6, 12]
    
    Returns:
    --------
    pd.DataFrame
        평균 모멘텀 (누적수익률 %, 컬럼=자산)
    """
    momentum_sum = None

    for period in periods:
        mom = (price_df / price_df.shift(period) - 1) * 100
        momentum_sum = mom if momentum_sum is None else momentum_sum.add(mom, fill_value=0)

    return momentum_sum / len(periods)


def cs_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-Sectional Min-Max Scaler (-1 ~ 1)
    
    각 날짜별로 전체 산업 횡단면(cross-section) 정규화
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 DataFrame (행=날짜, 열=산업/자산)
    
    Returns:
    --------
    pd.DataFrame
        정규화된 DataFrame (-1 ~ 1)
    """
    clipped = df.copy()
    q_low = clipped.quantile(0.05, axis=1)
    q_high = clipped.quantile(0.95, axis=1)
    clipped = clipped.clip(lower=q_low, upper=q_high, axis=0)

    mn = clipped.min(axis=1)
    mx = clipped.max(axis=1)
    denom = (mx - mn).replace(0, np.nan)  # 분모가 0이면 중립값(0) 처리
    scaled = clipped.sub(mn, axis=0).div(denom, axis=0)
    return (scaled * 2 - 1).fillna(0)
