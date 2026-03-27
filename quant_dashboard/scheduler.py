from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from collectors.commodities import collect_commodities
from collectors.crypto import collect_crypto
from collectors.equities import collect_equities
from collectors.fx import collect_fx
from collectors.rates import collect_rates
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_config() -> dict[str, Any]:
    """설정 파일을 로드한다."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml 파일이 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_run(task_name: str, task_fn: Callable[[], None]) -> None:
    """개별 수집 작업을 예외 안전하게 실행한다."""
    try:
        logger.info("작업 시작: %s", task_name)
        task_fn()
        logger.info("작업 완료: %s", task_name)
    except Exception as e:
        logger.exception("작업 실패(계속 진행): %s | %s", task_name, e)


def run_all_collectors() -> None:
    """지정된 순서로 전체 수집기를 실행한다."""
    pipeline: list[tuple[str, Callable[[], None]]] = [
        ("equities", collect_equities),
        ("commodities", collect_commodities),
        ("fx", collect_fx),
        ("crypto", collect_crypto),
        ("rates", collect_rates),
    ]

    for task_name, task_fn in pipeline:
        _safe_run(task_name, task_fn)


def start_scheduler() -> None:
    """크론 기반 주간 스케줄러를 시작한다."""
    try:
        config = _load_config()
    except Exception as e:
        logger.exception("설정 로드 실패로 스케줄러를 시작할 수 없습니다: %s", e)
        return

    schedule_cfg = config.get("scheduler", {})
    day_of_week = schedule_cfg.get("day_of_week", "mon")
    hour = int(schedule_cfg.get("hour", 7))
    minute = int(schedule_cfg.get("minute", 0))

    scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Seoul"))
    scheduler.add_job(
        run_all_collectors,
        trigger="cron",
        day_of_week=day_of_week,
        hour=hour,
        minute=minute,
        id="weekly_quant_collection",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "스케줄러 시작: 매주 %s %02d:%02d (KST)",
        day_of_week,
        hour,
        minute,
    )

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("스케줄러 종료 신호 수신")
    finally:
        scheduler.shutdown(wait=False)
        logger.info("스케줄러가 종료되었습니다.")


def _parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Quant Dashboard 데이터 수집 스케줄러")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="스케줄러 시작 전 즉시 1회 수집을 실행합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 로컬 실행 시 .env를 먼저 로드한다.
    load_dotenv(Path(__file__).parent / ".env")

    args = _parse_args()
    if args.run_now:
        logger.info("즉시 실행(--run-now) 시작")
        run_all_collectors()
        logger.info("즉시 실행(--run-now) 완료")
    else:
        start_scheduler()
