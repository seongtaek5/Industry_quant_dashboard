from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

DataInput = Union[str, Path, pd.DataFrame]


def _load_input(data: DataInput) -> pd.DataFrame:
    """입력을 DataFrame으로 변환한다."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
        df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _calc_point_return(series: pd.Series, periods: int) -> float:
    """N 영업일 기준 수익률을 계산한다."""
    clean = series.dropna()
    if len(clean) <= periods:
        return float("nan")
    current = clean.iloc[-1]
    prev = clean.iloc[-(periods + 1)]
    if prev == 0:
        return float("nan")
    return (current / prev - 1.0) * 100.0


def _calc_ytd_return(series: pd.Series) -> float:
    """연초 대비 수익률을 계산한다."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")

    last_date = clean.index[-1]
    year_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    ytd_slice = clean[clean.index >= year_start]
    if ytd_slice.empty:
        return float("nan")

    first = ytd_slice.iloc[0]
    current = ytd_slice.iloc[-1]
    if first == 0:
        return float("nan")
    return (current / first - 1.0) * 100.0


def calculate_returns(df: DataInput, asset_names: list[str]) -> pd.DataFrame:
    """1W/1M/3M/6M/YTD/1Y 수익률을 계산해 반환한다."""
    data = _load_input(df)
    periods = {
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    records: list[dict[str, float]] = []
    index_labels: list[str] = []

    for col in asset_names:
        if col not in data.columns:
            continue
        series = data[col]

        row = {label: _calc_point_return(series, lag) for label, lag in periods.items()}
        row["YTD"] = _calc_ytd_return(series)

        ordered_row = {
            "1W": row["1W"],
            "1M": row["1M"],
            "3M": row["3M"],
            "6M": row["6M"],
            "YTD": row["YTD"],
            "1Y": row["1Y"],
        }
        records.append(ordered_row)
        index_labels.append(col)

    result = pd.DataFrame(records, index=index_labels)
    return result.round(2)
