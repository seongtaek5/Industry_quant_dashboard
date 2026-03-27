from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from processors.corr_matrix import calculate_correlation
from processors.returns import calculate_returns

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
THEME = "plotly_white"
CHART_HEIGHT = 500
FONT_CACHE_DIR = BASE_DIR / ".cache" / "fonts"
NANUM_GOTHIC_URL = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"


def _ensure_fallback_korean_font() -> str | None:
    """시스템 한글 폰트가 없을 때 NanumGothic을 내려받아 등록한다."""
    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_path = FONT_CACHE_DIR / "NanumGothic-Regular.ttf"

    if not font_path.exists():
        try:
            resp = requests.get(NANUM_GOTHIC_URL, timeout=15)
            resp.raise_for_status()
            font_path.write_bytes(resp.content)
        except requests.RequestException:
            return None

    try:
        fm.fontManager.addfont(str(font_path))
        return fm.FontProperties(fname=str(font_path)).get_name()
    except Exception:
        return None


def setup_matplotlib_korean_font() -> None:
    """Matplotlib 한글 폰트를 설정한다."""
    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), None)
    if not selected:
        selected = _ensure_fallback_korean_font()
    if selected:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


setup_matplotlib_korean_font()

st.set_page_config(
    page_title="Quant Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(ttl=3600)
def load_config() -> dict[str, Any]:
    """설정 파일을 로드한다."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_data(ttl=3600)
def load_parquet(name: str) -> pd.DataFrame:
    """Parquet 데이터를 로드한다."""
    file_path = DATA_DIR / name
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(file_path)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        if str(df.index.name or "").lower() == "date":
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_bond_data(name: str) -> pd.DataFrame:
    """장기 포맷 채권 parquet 데이터를 로드한다."""
    file_path = DATA_DIR / name
    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path)
    except Exception:
        return pd.DataFrame()

    required = {"date", "country", "series_id", "name", "maturity_yr", "value"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["maturity_yr"] = pd.to_numeric(out["maturity_yr"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "country", "series_id", "name", "maturity_yr"])
    return out.sort_values(["country", "date", "maturity_yr"]).reset_index(drop=True)


def get_last_update_time() -> str:
    """데이터 파일의 마지막 수정 시각을 반환한다."""
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        return "데이터 없음"
    latest = max(parquet_files, key=lambda p: p.stat().st_mtime)
    ts = pd.to_datetime(latest.stat().st_mtime, unit="s")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def warn_if_missing(df: pd.DataFrame) -> bool:
    """데이터 부재 시 경고를 표시하고 True를 반환한다."""
    if df.empty:
        st.warning("데이터가 없습니다. 사이드바에서 수집을 실행하세요.")
        return True
    return False


def make_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """주말/휴일 결측을 전진채움해 시계열을 연속화한다."""
    if df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full_index).ffill()
    out.index.name = "Date"
    return out


def get_bond_meta(bond_df: pd.DataFrame, country: str) -> pd.DataFrame:
    """국가별 채권 만기 메타데이터를 반환한다."""
    if bond_df.empty:
        return pd.DataFrame(columns=["series_id", "name", "maturity_yr"])

    meta = bond_df.loc[bond_df["country"] == country, ["series_id", "name", "maturity_yr"]].drop_duplicates("series_id")
    return meta.sort_values("maturity_yr").reset_index(drop=True)


def get_curve_snapshot(bond_df: pd.DataFrame, country: str, days_ago: int) -> pd.DataFrame:
    """국가별 특정 시점의 금리커브 스냅샷을 반환한다."""
    if bond_df.empty:
        return pd.DataFrame(columns=["date", "series_id", "name", "maturity_yr", "value"])

    country_df = bond_df[bond_df["country"] == country].copy()
    if country_df.empty:
        return pd.DataFrame(columns=["date", "series_id", "name", "maturity_yr", "value"])

    pivot_date = country_df["date"].max() - pd.Timedelta(days=days_ago)
    history = country_df[country_df["date"] <= pivot_date]
    if history.empty:
        return pd.DataFrame(columns=["date", "series_id", "name", "maturity_yr", "value"])

    snapshot_date = history["date"].max()
    snapshot = history[history["date"] == snapshot_date].copy()
    snapshot = snapshot.dropna(subset=["value"]).drop_duplicates("series_id", keep="last")
    return snapshot.sort_values("maturity_yr").reset_index(drop=True)


def get_country_bond_wide(bond_df: pd.DataFrame, country: str) -> pd.DataFrame:
    """국가별 금리 시계열을 wide 포맷으로 변환한다."""
    if bond_df.empty:
        return pd.DataFrame()

    country_df = bond_df[bond_df["country"] == country]
    if country_df.empty:
        return pd.DataFrame()

    wide = country_df.pivot_table(index="date", columns="name", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "Date"
    return wide.sort_index()


def get_series_name_by_maturity(meta_df: pd.DataFrame, maturity_yr: float) -> str | None:
    """만기에 해당하는 시리즈명을 찾는다."""
    if meta_df.empty:
        return None
    matched = meta_df.loc[np.isclose(meta_df["maturity_yr"], maturity_yr), "name"]
    if matched.empty:
        return None
    return str(matched.iloc[0])


def extract_close_columns(df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """티커 목록에 해당하는 close 컬럼만 추출한다."""
    cols: list[str] = []
    names: list[str] = []
    for t in tickers:
        c = f"{t}_Close"
        if c in df.columns:
            cols.append(c)
            names.append(t)

    if not cols:
        return pd.DataFrame(index=df.index), []

    out = df[cols].copy()
    out.columns = names
    return out, names


def normalize_to_base(df: pd.DataFrame) -> pd.DataFrame:
    """첫 유효값 기준 누적 수익률(%)로 변환한다."""
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        base = s.iloc[0]
        if base == 0:
            continue
        out[c] = (df[c] / base - 1.0) * 100.0
    return out


def latest_value(df: pd.DataFrame, col: str) -> float:
    """컬럼의 최신 유효값을 반환한다."""
    if col not in df.columns:
        return float("nan")
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


def daily_delta(df: pd.DataFrame, col: str) -> float:
    """전일 대비 변화율을 계산한다."""
    if col not in df.columns:
        return float("nan")
    s = df[col].dropna()
    if len(s) < 2 or s.iloc[-2] == 0:
        return float("nan")
    return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0


@st.cache_data(ttl=3600)
def fetch_btc_dominance_now() -> float | None:
    """CoinGecko /global에서 현재 BTC 도미넌스(%)를 조회한다."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        pct = resp.json().get("data", {}).get("market_cap_percentage", {})
        return round(float(pct.get("btc", 0)), 2) if pct else None
    except requests.RequestException:
        return None


def build_asset_map(config: dict[str, Any], key: str, code_key: str = "ticker") -> dict[str, str]:
    """코드-표시명 매핑을 생성한다."""
    return {str(item.get(code_key, "")): str(item.get("name", "")) for item in config.get(key, [])}


def render_seaborn_heatmap(
    df: pd.DataFrame,
    title: str,
    cmap: str = "RdYlGn",
    scale: float = 0.7,
    annot_size: int = 7,
    tick_size: int = 8,
    outlier_sigma: float | None = None,
) -> None:
    """Seaborn 히트맵을 렌더링한다."""
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    rows = max(8, len(df.index))
    cols = max(4, len(df.columns))
    base_w = 0.95 * cols + 4
    base_h = 0.45 * rows + 2.5
    fig_w = min(12, max(5, base_w * scale))
    fig_h = min(18, max(5, base_h * scale))

    # 색상 범위: 아웃라이어(|v - μ| > outlier_sigma * σ) 제외 표본으로 vmin/vmax 설정
    # 어노테이션은 원본 df 그대로 사용 → 실제 수익률 표시
    vmin = None
    vmax = None
    if outlier_sigma is not None:
        vals = df.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            mu = float(np.mean(finite))
            sigma = float(np.std(finite))
            if sigma > 1e-9:
                inliers = finite[np.abs(finite - mu) <= outlier_sigma * sigma]
            else:
                inliers = finite
            if inliers.size > 0:
                vmin = float(inliers.min())
                vmax = float(inliers.max())

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.35,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=tick_size)
    ax.tick_params(axis="y", labelrotation=0, labelsize=tick_size)
    st.pyplot(fig)
    plt.close(fig)


def compute_monthly_zscore_heatmap(close_df: pd.DataFrame) -> pd.DataFrame:
    """최근 1년 월말 Z-Score 히트맵용 데이터프레임을 계산한다."""
    if close_df.empty:
        return pd.DataFrame()

    z_all = pd.DataFrame(index=close_df.index)
    for col in close_df.columns:
        s = close_df[col].dropna()
        if s.empty:
            continue
        mean_252 = s.rolling(window=252, min_periods=120).mean()
        std_252 = s.rolling(window=252, min_periods=120).std()
        z_all[col] = (s - mean_252) / std_252

    z_monthly = z_all.resample("ME").last().tail(12)
    if z_monthly.empty:
        return pd.DataFrame()

    z_monthly.index = z_monthly.index.strftime("%Y-%m")
    return z_monthly.T.round(2)


def main() -> None:
    """대시보드 메인 렌더링 함수."""
    config = load_config()

    with st.sidebar:
        st.subheader("Dashboard Control")
        st.write(f"마지막 수집 시각: {get_last_update_time()}")

        if st.button("지금 수집 실행", type="primary"):
            result = subprocess.run(
                ["python", "scheduler.py", "--run-now"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                st.success("데이터 수집이 완료되었습니다.")
                load_parquet.clear()
                load_bond_data.clear()
            else:
                st.error("데이터 수집 중 오류가 발생했습니다.")
                st.code((result.stderr or result.stdout)[:1500])

        _ = st.date_input("기준 날짜", value=date.today())

    st.title("Quant Market Dashboard")

    tabs = st.tabs(
        [
            "📊 Market Overview",
            "📈 Equities",
            "🏦 Rates & Curves",
            "🛢️ Commodities",
            "💱 FX",
            "₿ Crypto",
            "🔬 Signals",
        ]
    )

    equities_df = load_parquet("equities.parquet")
    commodities_df = load_parquet("commodities.parquet")
    fx_df = load_parquet("fx.parquet")
    crypto_df = load_parquet("crypto.parquet")
    bond_df = load_bond_data("rates.parquet")

    eq_map = build_asset_map(config, "equities")
    cm_map = build_asset_map(config, "commodities")
    fx_map = build_asset_map(config, "fx")
    cr_map = build_asset_map(config, "crypto", code_key="symbol")

    with tabs[0]:
        st.subheader("Market Overview")

        if all(d.empty for d in [equities_df, commodities_df, fx_df, crypto_df]):
            st.warning("데이터가 없습니다. 사이드바에서 수집을 실행하세요.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S&P 500", f"{latest_value(equities_df, '^GSPC_Close'):,.2f}", f"{daily_delta(equities_df, '^GSPC_Close'):.2f}%")
            c2.metric("VIX", f"{latest_value(equities_df, '^VIX_Close'):,.2f}", f"{daily_delta(equities_df, '^VIX_Close'):.2f}%")
            c3.metric("DXY", f"{latest_value(fx_df, 'DX-Y.NYB_Close'):,.2f}", f"{daily_delta(fx_df, 'DX-Y.NYB_Close'):.2f}%")
            c4.metric("BTC", f"{latest_value(crypto_df, 'BTC_Close'):,.2f}", f"{daily_delta(crypto_df, 'BTC_Close'):.2f}%")

            blocks: list[pd.DataFrame] = []
            for src_df, mapping in [(equities_df, eq_map), (commodities_df, cm_map), (fx_df, fx_map), (crypto_df, cr_map)]:
                if src_df.empty:
                    continue
                sub, names = extract_close_columns(src_df, list(mapping.keys()))
                if sub.empty:
                    continue
                sub.columns = [mapping.get(n, n) for n in names]
                ret = calculate_returns(sub, list(sub.columns))
                if not ret.empty:
                    blocks.append(ret[["1W", "1M", "3M", "YTD"]])

            if blocks:
                heatmap_df = pd.concat(blocks, axis=0)
                render_seaborn_heatmap(
                    heatmap_df,
                    "전 자산군 수익률 히트맵 (%)",
                    cmap="RdYlGn",
                    scale=0.54,
                    annot_size=5,
                    tick_size=6,
                    outlier_sigma=2.0,
                )
            else:
                st.info("수익률 히트맵 데이터가 없습니다.")

    with tabs[1]:
        st.subheader("Equities")
        if warn_if_missing(equities_df):
            pass
        else:
            regions = ["미국", "한국", "일본", "중국", "유럽", "신흥국"]
            eq_regions: dict[str, str] = {str(i.get("ticker")): str(i.get("region")) for i in config.get("equities", [])}
            selected_regions = st.multiselect("지역 필터", regions, default=regions)

            target = [
                t
                for t, r in eq_regions.items()
                if r in selected_regions
                or (r in ["홍콩"] and "중국" in selected_regions)
                or (r in ["독일", "영국"] and "유럽" in selected_regions)
            ]

            sub, names = extract_close_columns(equities_df, target)
            if sub.empty:
                st.info("선택된 지역의 데이터가 없습니다.")
            else:
                sub.columns = [eq_map.get(n, n) for n in names]
                sub = make_continuous(sub)
                ret_curve = normalize_to_base(sub)

                fig = px.line(ret_curve, title="지역별 누적 수익률 추이 (%)")
                fig.update_layout(template=THEME, height=CHART_HEIGHT, yaxis_title="누적 수익률(%)")
                st.plotly_chart(fig, width="stretch")

                period = st.selectbox("수익률 기간", ["1W", "1M", "3M", "YTD"], index=0)
                ret = calculate_returns(sub, list(sub.columns))
                if not ret.empty:
                    bar_df = ret[[period]].sort_values(period, ascending=False)
                    fig_bar = px.bar(bar_df, x=bar_df.index, y=period, text_auto=".2f", title=f"지수별 {period} 수익률 (%)")
                    fig_bar.update_layout(template=THEME, height=450)
                    st.plotly_chart(fig_bar, width="stretch")

    with tabs[2]:
        st.subheader("Rates & Curves")
        if warn_if_missing(bond_df):
            pass
        else:
            country_map = {"미국": "us", "한국": "kr", "일본": "jp"}
            available_countries = [label for label, key in country_map.items() if not get_bond_meta(bond_df, key).empty]
            country_label = st.selectbox("국가 선택", available_countries, index=0)
            country_key = country_map[country_label]
            meta_df = get_bond_meta(bond_df, country_key)
            curve_now = get_curve_snapshot(bond_df, country_key, 0)
            curve_1m = get_curve_snapshot(bond_df, country_key, 30)
            curve_1y = get_curve_snapshot(bond_df, country_key, 365)

            fig_curve = go.Figure()
            for trace_name, curve_df in [("현재", curve_now), ("1개월 전", curve_1m), ("1년 전", curve_1y)]:
                if curve_df.empty:
                    continue
                fig_curve.add_trace(
                    go.Scatter(
                        x=curve_df["maturity_yr"],
                        y=curve_df["value"],
                        mode="lines+markers",
                        name=trace_name,
                        text=curve_df["name"],
                        hovertemplate="%{text}<br>금리=%{y:.2f}%<extra></extra>",
                    )
                )

            tickvals = meta_df["maturity_yr"].tolist()
            ticktext = [str(name).replace(f"{country_label} ", "") for name in meta_df["name"].tolist()]
            fig_curve.update_layout(
                title=f"{country_label} Yield Curve",
                xaxis_title="만기(년)",
                yaxis_title="금리(%)",
                template=THEME,
                height=450,
                xaxis={"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext},
            )
            st.plotly_chart(fig_curve, width="stretch")

            country_wide = get_country_bond_wide(bond_df, country_key)
            if not country_wide.empty:
                country_wide = make_continuous(country_wide)
                fig_rates = px.line(country_wide, title=f"{country_label} 금리 시계열")
                fig_rates.update_layout(template=THEME, height=450)
                st.plotly_chart(fig_rates, width="stretch")

            st.markdown("#### 스프레드 (10Y-2Y, 데이터 없으면 10Y-3M 대체)")
            spread_country = st.selectbox("스프레드 국가", available_countries, index=0)
            ckey = country_map[spread_country]
            spread_meta = get_bond_meta(bond_df, ckey)
            spread_wide = get_country_bond_wide(bond_df, ckey)
            s10 = get_series_name_by_maturity(spread_meta, 10)
            s2 = get_series_name_by_maturity(spread_meta, 2)
            s3m = get_series_name_by_maturity(spread_meta, 0.25)

            if s10 and s2 and s10 in spread_wide.columns and s2 in spread_wide.columns:
                spread_name = "10Y-2Y"
                spread = spread_wide[s10] - spread_wide[s2]
            elif s10 and s3m and s10 in spread_wide.columns and s3m in spread_wide.columns:
                spread_name = "10Y-3M(대체)"
                spread = spread_wide[s10] - spread_wide[s3m]
            else:
                spread_name = "N/A"
                spread = pd.Series(dtype=float)

            if spread.empty:
                st.info("해당 국가 스프레드 계산에 필요한 만기 데이터가 부족합니다.")
            else:
                spread = make_continuous(spread.to_frame("Spread"))["Spread"]
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines", name=spread_name))
                fig_spread.add_hrect(y0=-10, y1=0, line_width=0, fillcolor="red", opacity=0.08)
                fig_spread.update_layout(
                    title=f"{spread_country} {spread_name} 스프레드",
                    template=THEME,
                    height=450,
                )
                st.plotly_chart(fig_spread, width="stretch")

    with tabs[3]:
        st.subheader("Commodities")
        if warn_if_missing(commodities_df):
            pass
        else:
            cm_close, names = extract_close_columns(commodities_df, list(cm_map.keys()))
            cm_close.columns = [cm_map.get(n, n) for n in names]
            cm_close = make_continuous(cm_close)

            st.markdown("### 에너지 (WTI, 브렌트 / 천연가스 이중축)")
            energy_cols_left = [c for c in ["WTI 원유", "브렌트 원유"] if c in cm_close.columns]
            energy_col_right = "천연가스"
            if energy_cols_left and energy_col_right in cm_close.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for c in energy_cols_left:
                    fig.add_trace(go.Scatter(x=cm_close.index, y=cm_close[c], name=c), secondary_y=False)
                fig.add_trace(go.Scatter(x=cm_close.index, y=cm_close[energy_col_right], name=energy_col_right), secondary_y=True)
                fig.update_yaxes(title_text="WTI/브렌트", secondary_y=False)
                fig.update_yaxes(title_text="천연가스", secondary_y=True)
                fig.update_layout(template=THEME, height=CHART_HEIGHT)
                st.plotly_chart(fig, width="stretch")

            st.markdown("### 금속 (금/은/구리/백금/팔라듐)")
            metal_cols = [c for c in ["금", "은", "구리", "백금", "팔라듐"] if c in cm_close.columns]
            if metal_cols:
                metal_ret = normalize_to_base(cm_close[metal_cols])
                fig_m = px.line(metal_ret, title="금속 누적 수익률 (%)")
                fig_m.update_layout(template=THEME, height=CHART_HEIGHT)
                st.plotly_chart(fig_m, width="stretch")

            st.markdown("### 농산물 (밀/쌀/옥수수)")
            agri_cols = [c for c in ["밀", "쌀", "옥수수"] if c in cm_close.columns]
            if agri_cols:
                agri_ret = normalize_to_base(cm_close[agri_cols])
                fig_a = px.line(agri_ret, title="농산물 누적 수익률 (%)")
                fig_a.update_layout(template=THEME, height=CHART_HEIGHT)
                st.plotly_chart(fig_a, width="stretch")

    with tabs[4]:
        st.subheader("FX")
        if warn_if_missing(fx_df):
            pass
        else:
            fx_close, names = extract_close_columns(fx_df, list(fx_map.keys()))
            if fx_close.empty:
                st.info("FX 데이터가 없습니다.")
            else:
                fx_close.columns = [fx_map.get(n, n) for n in names]
                fx_close = make_continuous(fx_close)

                all_pairs = list(fx_close.columns)
                selected = st.multiselect("표시할 FX 선택", all_pairs, default=all_pairs)
                if selected:
                    fx_ret = normalize_to_base(fx_close[selected])
                    fig_fx = px.line(fx_ret, title="달러인덱스/FX 페어 누적 수익률 (%)")
                    fig_fx.update_layout(template=THEME, height=CHART_HEIGHT, yaxis_title="누적 수익률(%)")
                    st.plotly_chart(fig_fx, width="stretch")

                current_tbl = pd.DataFrame({"현재값": [fx_close[c].dropna().iloc[-1] for c in all_pairs]}, index=all_pairs).round(4)
                st.markdown("#### 현재 실제 값")
                st.dataframe(current_tbl, width="stretch")

    with tabs[5]:
        st.subheader("Crypto")
        if warn_if_missing(crypto_df):
            pass
        else:
            symbols = [str(i.get("symbol", "")).upper() for i in config.get("crypto", [])]
            cols = [f"{s}_Close" for s in symbols if f"{s}_Close" in crypto_df.columns]
            if cols:
                close_df = crypto_df[cols].copy()
                close_df.columns = [c.replace("_Close", "") for c in cols]
                close_df = make_continuous(close_df)
                ret = normalize_to_base(close_df)

                fig_c = px.line(ret, title="BTC/ETH/SOL 누적 수익률 (%)")
                fig_c.update_layout(template=THEME, height=CHART_HEIGHT)
                st.plotly_chart(fig_c, width="stretch")

                ret_tbl = calculate_returns(close_df, list(close_df.columns))
                if not ret_tbl.empty:
                    fig_b = px.bar(ret_tbl[["1W", "1M", "3M", "YTD"]], x=ret_tbl.index, y=["1W", "1M", "3M", "YTD"], barmode="group", title="수익률 비교(%)")
                    fig_b.update_layout(template=THEME, height=450)
                    st.plotly_chart(fig_b, width="stretch")

            st.divider()
            btc_dom = fetch_btc_dominance_now()
            if btc_dom is not None:
                st.metric("BTC 도미넌스 (현재)", f"{btc_dom:.2f}%")
            else:
                st.info("BTC 도미넌스 조회 실패 (CoinGecko API 오류)")

    with tabs[6]:
        st.subheader("Signals")

        frames: list[pd.DataFrame] = []
        for src_df, mapping in [(equities_df, eq_map), (commodities_df, cm_map), (fx_df, fx_map)]:
            if src_df.empty:
                continue
            sub, names = extract_close_columns(src_df, list(mapping.keys()))
            if sub.empty:
                continue
            sub.columns = [mapping.get(n, n) for n in names]
            frames.append(sub)

        if not crypto_df.empty:
            ccols = [c for c in crypto_df.columns if c.endswith("_Close")]
            cr = crypto_df[ccols].copy() if ccols else pd.DataFrame()
            if not cr.empty:
                cr.columns = [c.replace("_Close", "") for c in cr.columns]
                frames.append(cr)

        if not frames:
            st.warning("신호 계산에 필요한 데이터가 없습니다.")
        else:
            all_close = pd.concat(frames, axis=1, join="outer")
            all_close = all_close.loc[:, ~all_close.columns.duplicated()]
            all_close = make_continuous(all_close)

            st.markdown("### 월말 Rolling Z-Score 히트맵 (최근 1년)")
            z_heat = compute_monthly_zscore_heatmap(all_close)
            render_seaborn_heatmap(
                z_heat,
                "월말 Z-Score 히트맵",
                cmap="RdBu_r",
                scale=0.54,
                annot_size=5,
                tick_size=6,
            )

            st.markdown("### 주간 수익률 기반 롤링 상관관계")
            candidates = list(all_close.columns)
            default_assets = [c for c in ["S&P 500", "WTI 원유", "DXY (달러인덱스)", "BTC", "USD/KRW"] if c in candidates]
            selected_assets = st.multiselect("상관관계 자산 선택", candidates, default=default_assets or candidates[:6])
            window_weeks = st.slider("롤링 윈도우(주)", min_value=26, max_value=104, value=52, step=1)

            if len(selected_assets) < 2:
                st.info("상관관계 계산을 위해 최소 2개 자산을 선택하세요.")
            else:
                corr = calculate_correlation(all_close[selected_assets], use_weekly=True, window=window_weeks)
                render_seaborn_heatmap(
                    corr,
                    f"주간 수익률 상관관계 (최근 {window_weeks}주)",
                    cmap="RdBu_r",
                    scale=0.56,
                    annot_size=6,
                    tick_size=7,
                )


if __name__ == "__main__":
    main()
