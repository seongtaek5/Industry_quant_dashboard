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
    return df.sort_index()


def _signal_from_z(z: float) -> str:
    """Z-Score에 따른 신호 문자열을 반환한다."""
    if pd.isna(z):
        return "정상"
    if z > 2.0:
        return "과매수 ⚠️"
    if z < -2.0:
        return "과매도 ⚠️"
    return "정상"


def calculate_zscore(df: DataInput, asset_names: list[str]) -> pd.DataFrame:
    """252일 롤링 Z-Score와 신호를 계산해 반환한다."""
    data = _load_input(df)

    rows: list[dict[str, object]] = []
    for col in asset_names:
        if col not in data.columns:
            continue

        series = data[col].dropna()
        if series.empty:
            rows.append({"자산명": col, "현재 Z-Score": float("nan"), "신호": "정상"})
            continue

        roll_mean = series.rolling(window=252, min_periods=252).mean()
        roll_std = series.rolling(window=252, min_periods=252).std()
        z_series = (series - roll_mean) / roll_std
        z_val = float(z_series.iloc[-1]) if not z_series.empty else float("nan")

        rows.append(
            {
                "자산명": col,
                "현재 Z-Score": round(z_val, 2) if pd.notna(z_val) else float("nan"),
                "신호": _signal_from_z(z_val),
            }
        )

    return pd.DataFrame(rows)
