from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from fredapi import Fred

from utils.logger import get_logger

logger = get_logger(__name__)

ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
DEFAULT_MOF_CURRENT_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/"
    "reference/interest_rate/jgbcme.csv"
)
DEFAULT_MOF_HISTORY_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/"
    "reference/interest_rate/historical/jgbcme_all.csv"
)
LONG_COLUMNS = ["date", "country", "series_id", "name", "maturity_yr", "value", "source"]


def _load_config() -> dict[str, Any]:
    """설정 파일을 로드한다."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml 파일이 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_start_date(value: Any) -> pd.Timestamp:
    """수집 시작일을 Timestamp로 정규화한다."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(datetime.now().date()).normalize() - pd.DateOffset(years=2)
    return pd.Timestamp(parsed).normalize()


def _make_long_frame(
    series: pd.Series,
    country: str,
    series_id: str,
    name: str,
    maturity_yr: float,
    source: str,
) -> pd.DataFrame:
    """시계열을 장기 포맷 데이터프레임으로 변환한다."""
    frame = pd.DataFrame({"date": pd.to_datetime(series.index), "value": pd.to_numeric(series.values, errors="coerce")})
    frame["country"] = country
    frame["series_id"] = series_id
    frame["name"] = name
    frame["maturity_yr"] = float(maturity_yr)
    frame["source"] = source
    return frame[LONG_COLUMNS].dropna(subset=["date", "value"])


def _append_to_parquet(new_df: pd.DataFrame, parquet_path: Path) -> None:
    """장기 포맷 금리 데이터를 parquet에 upsert 저장한다."""
    if new_df.empty:
        logger.warning("저장할 신규 금리 데이터가 없습니다.")
        return

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined = new_df.copy()

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        if set(LONG_COLUMNS).issubset(existing.columns):
            combined = pd.concat([existing[LONG_COLUMNS], combined], ignore_index=True)
        else:
            logger.warning("기존 금리 parquet 스키마가 달라 새 포맷으로 덮어씁니다: %s", parquet_path)

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["maturity_yr"] = pd.to_numeric(combined["maturity_yr"], errors="coerce")
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined.dropna(subset=["date", "country", "series_id", "name", "maturity_yr", "value"])
    combined = combined.drop_duplicates(subset=["date", "country", "series_id"], keep="last")
    combined = combined.sort_values(["date", "country", "maturity_yr", "series_id"]).reset_index(drop=True)
    combined.to_parquet(parquet_path, index=False)


def _collect_fred_country(
    country: str,
    country_cfg: dict[str, Any],
    start_date: pd.Timestamp,
    fred: Fred,
) -> list[pd.DataFrame]:
    """FRED 기반 금리를 수집한다."""
    frames: list[pd.DataFrame] = []
    for item in country_cfg.get("series", []):
        series_id = str(item.get("id") or item.get("series") or "").strip()
        name = str(item.get("name") or series_id).strip()
        maturity_yr = item.get("maturity_yr")
        if not series_id or maturity_yr is None:
            logger.warning("유효하지 않은 FRED 금리 설정을 건너뜁니다: %s", item)
            continue

        try:
            series = fred.get_series(series_id, observation_start=start_date.strftime("%Y-%m-%d"))
            if series is None or len(series) == 0:
                logger.warning("빈 금리 시리즈: %s (%s)", series_id, country)
                continue

            clean = pd.Series(series.astype(float), index=pd.to_datetime(series.index))
            frames.append(_make_long_frame(clean, country, series_id, name, float(maturity_yr), "fred"))
            logger.info("금리 수집 완료: %s (%s, FRED)", series_id, country)
        except Exception as e:
            logger.warning("FRED 금리 수집 실패, 건너뜀: %s | %s", series_id, e)

    return frames


def _fetch_ecos_series(
    api_key: str,
    stat_code: str,
    start_date: pd.Timestamp,
    item_code: str,
) -> pd.Series:
    """ECOS 단일 시리즈를 조회한다."""
    url = (
        f"{ECOS_BASE_URL}/{api_key}/json/kr/1/100000/{stat_code}/D/"
        f"{start_date.strftime('%Y%m%d')}/{datetime.today().strftime('%Y%m%d')}/{item_code}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    body = response.json()

    if "StatisticSearch" not in body:
        message = body.get("RESULT", {}).get("MESSAGE", str(body))
        raise ValueError(message)

    rows = body["StatisticSearch"].get("row", [])
    if not rows:
        return pd.Series(dtype=float)

    frame = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
    frame["date"] = pd.to_datetime(frame["TIME"], format="%Y%m%d", errors="coerce")
    frame["value"] = pd.to_numeric(frame["DATA_VALUE"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).set_index("date")
    return frame["value"].sort_index()


def _collect_ecos_country(
    country: str,
    country_cfg: dict[str, Any],
    start_date: pd.Timestamp,
    ecos_api_key: str,
) -> list[pd.DataFrame]:
    """ECOS 기반 금리를 수집한다."""
    stat_code = str(country_cfg.get("stat_code") or "").strip()
    if not stat_code:
        logger.warning("ECOS stat_code가 없어 %s 금리 수집을 건너뜁니다.", country)
        return []

    frames: list[pd.DataFrame] = []
    for item in country_cfg.get("series", []):
        series_id = str(item.get("id") or item.get("item_code") or "").strip()
        name = str(item.get("name") or series_id).strip()
        maturity_yr = item.get("maturity_yr")
        if not series_id or maturity_yr is None:
            logger.warning("유효하지 않은 ECOS 금리 설정을 건너뜁니다: %s", item)
            continue

        try:
            series = _fetch_ecos_series(ecos_api_key, stat_code, start_date, series_id)
            if series.empty:
                logger.warning("ECOS 빈 금리 시리즈: %s (%s)", series_id, country)
                continue

            frames.append(_make_long_frame(series, country, series_id, name, float(maturity_yr), "ecos"))
            logger.info("금리 수집 완료: %s (%s, ECOS)", series_id, country)
        except Exception as e:
            logger.warning("ECOS 금리 수집 실패, 건너뜀: %s | %s", series_id, e)

    return frames


def _read_mof_csv(url: str) -> pd.DataFrame:
    """MOF CSV URL을 정제된 데이터프레임으로 읽는다."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text), skiprows=1, na_values=["-", " ", ""])
    frame.rename(columns={frame.columns[0]: "Date"}, inplace=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_index()


def _collect_mof_country(country: str, country_cfg: dict[str, Any], start_date: pd.Timestamp) -> list[pd.DataFrame]:
    """MOF CSV 기반 금리를 수집한다."""
    history_url = str(country_cfg.get("history_url") or DEFAULT_MOF_HISTORY_URL)
    current_url = str(country_cfg.get("current_url") or DEFAULT_MOF_CURRENT_URL)

    sources: list[pd.DataFrame] = []
    for label, url in (("history", history_url), ("current", current_url)):
        try:
            sources.append(_read_mof_csv(url))
            logger.info("MOF %s CSV 로드 완료: %s", label, url)
        except Exception as e:
            logger.warning("MOF %s CSV 로드 실패: %s | %s", label, url, e)

    if not sources:
        return []

    combined = pd.concat(sources)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined[combined.index >= start_date]

    frames: list[pd.DataFrame] = []
    for item in country_cfg.get("series", []):
        series_id = str(item.get("id") or item.get("column") or "").strip()
        name = str(item.get("name") or series_id).strip()
        maturity_yr = item.get("maturity_yr")
        if not series_id or maturity_yr is None:
            logger.warning("유효하지 않은 MOF 금리 설정을 건너뜁니다: %s", item)
            continue
        if series_id not in combined.columns:
            logger.warning("MOF CSV에 만기 컬럼이 없습니다: %s (%s)", series_id, country)
            continue

        series = combined[series_id].dropna()
        if series.empty:
            logger.warning("MOF 빈 금리 시리즈: %s (%s)", series_id, country)
            continue

        frames.append(_make_long_frame(series, country, series_id, name, float(maturity_yr), "mof"))
        logger.info("금리 수집 완료: %s (%s, MOF)", series_id, country)

    return frames


def collect_rates() -> None:
    """국가별 소스(FRED, ECOS, MOF) 기반으로 금리 데이터를 수집해 저장한다."""
    load_dotenv(Path(__file__).parent.parent / ".env")

    try:
        config = _load_config()
    except Exception as e:
        logger.exception("설정 파일 로드 실패: %s", e)
        return

    rates_cfg = config.get("rates", {})
    if not rates_cfg:
        logger.warning("rates 설정이 비어 있습니다.")
        return

    start_date = _parse_start_date(rates_cfg.get("start_date", "2020-01-01"))
    parquet_name = str(rates_cfg.get("parquet_name", "rates.parquet"))

    fred_client: Fred | None = None
    fred_api_key = os.getenv("FRED_API_KEY") or config.get("fred_api_key")
    if fred_api_key and fred_api_key != "YOUR_FRED_API_KEY_HERE":
        try:
            fred_client = Fred(api_key=fred_api_key)
        except Exception as e:
            logger.warning("FRED 클라이언트 초기화 실패: %s", e)

    ecos_api_key = os.getenv("ECOS_API_KEY")
    if not ecos_api_key:
        logger.warning("ECOS_API_KEY가 없어 한국 금리 수집은 건너뜁니다.")

    frames: list[pd.DataFrame] = []
    for country, country_cfg in rates_cfg.items():
        if country in {"start_date", "parquet_name"} or not isinstance(country_cfg, dict):
            continue

        source = str(country_cfg.get("source") or "").strip().lower()
        if source == "fred":
            if fred_client is None:
                logger.warning("FRED API 키가 없어 %s 금리 수집을 건너뜁니다.", country)
                continue
            frames.extend(_collect_fred_country(country, country_cfg, start_date, fred_client))
        elif source == "ecos":
            if not ecos_api_key:
                continue
            frames.extend(_collect_ecos_country(country, country_cfg, start_date, ecos_api_key))
        elif source == "mof":
            frames.extend(_collect_mof_country(country, country_cfg, start_date))
        else:
            logger.warning("알 수 없는 금리 소스(%s)로 %s 수집을 건너뜁니다.", source, country)

    if not frames:
        logger.warning("수집된 금리 데이터가 없습니다.")
        return

    merged = pd.concat(frames, ignore_index=True)
    parquet_path = Path(__file__).parent.parent / "data" / parquet_name
    try:
        _append_to_parquet(merged, parquet_path)
        logger.info("금리 데이터 저장 완료: %s", parquet_path)
    except Exception as e:
        logger.exception("금리 데이터 저장 실패: %s", e)
