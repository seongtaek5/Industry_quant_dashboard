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
        각 기간별 모멘텀 (누적수익률 %)
    """
    momentum_list = []
    
    for period in periods:
        mom = (price_df / price_df.shift(period) - 1) * 100
        # 각 행의 평균 모멘텀
        momentum_list.append(mom.mean(axis=1))
    
    # 평균 모멘텀
    momentum_df = pd.DataFrame({"Momentum": pd.concat(momentum_list, axis=1).mean(axis=1)})
    
    return momentum_df


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
    mn = df.min(axis=1)
    mx = df.max(axis=1)
    denom = (mx - mn).replace(0, 1)  # 분모가 0이면 1로 대체
    scaled = df.sub(mn, axis=0).div(denom, axis=0)
    return scaled * 2 - 1


def calculate_composite_score(pbr_cs: pd.DataFrame, mom_cs: pd.DataFrame) -> pd.DataFrame:
    """
    종합 Score 계산: (PBR_CS × -1 + Mom_CS) / 2
    
    낮은 PBR (저평가) + 높은 모멘텀 → 높은 점수
    
    Parameters:
    -----------
    pbr_cs : pd.DataFrame
        PBR Cross-Sectional Score
    mom_cs : pd.DataFrame
        Momentum Cross-Sectional Score
    
    Returns:
    --------
    pd.DataFrame
        종합 Score
    """
    return (-pbr_cs + mom_cs) / 2
