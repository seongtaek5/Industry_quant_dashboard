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
        logger.warning("저장할 신규 크립토 데이터가 없습니다.")
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


def _build_yf_ticker(item: dict[str, Any]) -> str:
    """설정 항목으로부터 yfinance 티커를 생성한다."""
    explicit_ticker = str(item.get("ticker", "")).strip().upper()
    if explicit_ticker:
        return explicit_ticker

    symbol = str(item.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("symbol 값이 없습니다.")
    return f"{symbol}-USD"


def collect_crypto() -> None:
    """yfinance로 크립토 데이터를 수집해 저장한다."""
    try:
        config = _load_config()
    except Exception as e:
        logger.exception("설정 파일 로드 실패: %s", e)
        return

    items: list[dict[str, Any]] = config.get("crypto", [])
    history_period = str(config.get("history_period", "2y"))
    if not items:
        logger.warning("crypto 설정이 비어 있습니다.")
        return

    frames: list[pd.DataFrame] = []
    for item in items:
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            logger.warning("유효하지 않은 코인 설정을 건너뜁니다: %s", item)
            continue

        try:
            ticker = _build_yf_ticker(item)
            raw = yf.download(
                ticker,
                period=history_period,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning("크립토 티커 데이터가 비어 있어 건너뜁니다: %s", ticker)
                time.sleep(0.5)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.rename(columns={"Close": f"{symbol}_Close"})
            if f"{symbol}_Close" not in df.columns:
                logger.warning("Close 컬럼이 없어 건너뜁니다: %s", ticker)
                time.sleep(0.5)
                continue

            df = df[[f"{symbol}_Close"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            df = df[~df.index.duplicated(keep="last")].sort_index()
            frames.append(df)
            logger.info("크립토 수집 완료: %s (%d rows)", symbol, len(df))
        except Exception as e:
            logger.warning("크립토 수집 실패, 건너뜀: %s | %s", symbol, e)
        finally:
            time.sleep(0.5)

    if not frames:
        logger.warning("수집된 크립토 데이터가 없습니다.")
        return

    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    merged.index.name = "Date"
    merged = merged[~merged.index.duplicated(keep="last")]

    parquet_path = Path(__file__).parent.parent / "data" / "crypto.parquet"
    try:
        _append_to_parquet(merged, parquet_path)
        logger.info("크립토 저장 완료: %s", parquet_path)
    except Exception as e:
        logger.exception("크립토 저장 실패: %s", e)
