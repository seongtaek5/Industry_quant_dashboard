from __future__ import annotations

import pandas as pd


def calculate_correlation(
    df: pd.DataFrame,
    use_weekly: bool = True,
    window: int = 52,
) -> pd.DataFrame:
    """최신 롤링 상관관계 행렬을 계산한다.

    Args:
        df: 가격 wide-format 데이터프레임
        use_weekly: True면 주간 수익률 기준으로 계산
        window: 롤링 윈도우 길이(주간 기준이면 주 수)
    """
    if df is None or df.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    returns = data.pct_change(fill_method=None)
    if use_weekly:
        # 주간 종가로 변환 후 주간 수익률을 사용한다.
        weekly_prices = data.resample("W-FRI").last()
        returns = weekly_prices.pct_change(fill_method=None)

    window_returns = returns.tail(window)
    if window_returns.dropna(how="all").empty:
        raise ValueError("상관관계를 계산할 충분한 수익률 데이터가 없습니다.")

    corr = window_returns.corr()
    return corr.round(2)
