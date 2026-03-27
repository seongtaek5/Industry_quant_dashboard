from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


def _load_config() -> dict[str, Any]:
    """설정 파일을 로드한다."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml 파일이 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _append_to_parquet(new_df: pd.DataFrame, parquet_path: Path) -> None:
    """기존 Parquet에 신규 데이터를 append하고 중복 인덱스를 제거한다."""
    if new_df.empty:
        logger.warning("저장할 신규 데이터가 없습니다.")
        return

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    combined = combined.sort_index()
    combined.to_parquet(parquet_path)


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """티커별 컬럼명을 통일된 형태로 변환한다."""
    rename_map = {col: f"{ticker}_{col}" for col in df.columns}
    out = df.rename(columns=rename_map)
    out.index.name = "Date"
    return out


def collect_equities() -> None:
    """주가지수 데이터를 수집해 Parquet로 저장한다."""
    try:
        config = _load_config()
    except Exception as e:
        logger.exception("설정 파일 로드 실패: %s", e)
        return

    items: list[dict[str, Any]] = config.get("equities", [])
    history_period = config.get("history_period", "2y")
    if not items:
        logger.warning("equities 설정이 비어 있습니다.")
        return

    frames: list[pd.DataFrame] = []
    for item in items:
        ticker = str(item.get("ticker", "")).strip()
        if not ticker:
            logger.warning("유효하지 않은 티커 항목을 건너뜁니다: %s", item)
            continue

        try:
            raw = yf.download(
                ticker,
                period=history_period,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning("티커 데이터가 비어 있어 건너뜁니다: %s", ticker)
                time.sleep(0.5)
                continue

            # yfinance 버전에 따라 MultiIndex가 내려올 수 있어 방어적으로 처리한다.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            frame = _normalize_columns(raw, ticker)
            frames.append(frame)
            logger.info("수집 완료: %s (%d rows)", ticker, len(frame))
        except Exception as e:
            logger.warning("티커 수집 실패, 건너뜀: %s | %s", ticker, e)
        finally:
            time.sleep(0.5)

    if not frames:
        logger.warning("수집된 주가지수 데이터가 없습니다.")
        return

    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    parquet_path = Path(__file__).parent.parent / "data" / "equities.parquet"
    try:
        _append_to_parquet(merged, parquet_path)
        logger.info("주가지수 저장 완료: %s", parquet_path)
    except Exception as e:
        logger.exception("주가지수 저장 실패: %s", e)
