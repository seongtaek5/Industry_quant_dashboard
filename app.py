"""
미국 산업별 Valuation & Momentum Dashboard
Streamlit 메인 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from data_loader import load_pbr_data, load_etf_data, align_and_resample, create_industry_momentum_map, ETF_INDUSTRY_MAP
from indicators import rolling_zscore_df, calculate_momentum, cs_minmax

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Industry Valuation & Momentum Dashboard", layout="wide")

# ========== 데이터 로드 (캐싱) ==========
@st.cache_data
def load_all_data():
    pbr_df = load_pbr_data("data/US_IND_PBR.xlsx")
    etf_df = load_etf_data("data/sector_etf_prices.csv")
    pbr_monthly, etf_monthly = align_and_resample(pbr_df, etf_df)
    industry_etf_monthly = create_industry_momentum_map(etf_monthly)
    return pbr_monthly, industry_etf_monthly

pbr_monthly, etf_monthly = load_all_data()

# ========== 사이드바 컨트롤 ==========
st.sidebar.title("📊 대시보드 설정")

rolling_window = st.sidebar.slider(
    "Rolling Z-score 윈도우 (개월)",
    min_value=12,
    max_value=120,
    value=60,
    step=6,
)

momentum_periods = st.sidebar.multiselect(
    "모멘텀 기간 (복수 선택 가능)",
    options=["1M", "3M", "6M", "12M"],
    default=["3M", "6M", "12M"],
)

momentum_periods_int = [int(p[0]) for p in momentum_periods] if momentum_periods else [3, 6, 12]

ref_date = st.sidebar.date_input(
    "기준 날짜",
    value=pbr_monthly.index[-1].date(),
    min_value=pbr_monthly.index[0].date(),
    max_value=pbr_monthly.index[-1].date(),
)

heatmap_months = st.sidebar.slider(
    "히트맵 표시 개월 수",
    min_value=6,
    max_value=60,
    value=24,
    step=6,
)

# ========== 지표 계산 ==========
pbr_zscore = rolling_zscore_df(pbr_monthly, rolling_window)
pbr_cs_score = cs_minmax(pbr_zscore)

# 모멘텀 계산
momentum_df = calculate_momentum(etf_monthly, momentum_periods_int)
momentum_zscore = rolling_zscore_df(momentum_df, rolling_window)
momentum_cs_score = cs_minmax(momentum_zscore)

# 기준 날짜
try:
    ref_idx = pbr_monthly.index.get_loc(pd.Timestamp(ref_date))
except:
    ref_idx = -1

ref_date_ts = pbr_monthly.index[ref_idx]

# ========== 탭 구성 ==========
tab1, tab2, tab3 = st.tabs(["📊 Overview (Heatmap)", "🔵 Scatter Plot", "📈 Time-Series"])

# ========== TAB 1: Heatmap ==========
with tab1:
    st.subheader("최신기준 Cross-Sectional Min-Max Score (-1 ~ 1)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PBR CS Score**")
        heatmap_data_pbr = pbr_cs_score.iloc[-heatmap_months:, :]
        fig_pbr = go.Figure(
            data=go.Heatmap(
                z=heatmap_data_pbr.values,
                x=heatmap_data_pbr.index.strftime("%Y-%m"),
                y=heatmap_data_pbr.columns,
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(heatmap_data_pbr.values, 1),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.2f}<extra></extra>",
            )
        )
        fig_pbr.update_layout(height=600, xaxis_tickangle=-45)
        st.plotly_chart(fig_pbr, use_container_width=True)
    
    with col2:
        st.write("**Momentum CS Score**")
        heatmap_data_mom_expanded = momentum_cs_score.iloc[-heatmap_months:, :]

        fig_mom = go.Figure(
            data=go.Heatmap(
                z=heatmap_data_mom_expanded.values,
                x=heatmap_data_mom_expanded.index.strftime("%Y-%m"),
                y=heatmap_data_mom_expanded.columns,
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(heatmap_data_mom_expanded.values, 1),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.2f}<extra></extra>",
            )
        )
        fig_mom.update_layout(height=600, xaxis_tickangle=-45)
        st.plotly_chart(fig_mom, use_container_width=True)

# ========== TAB 2: Scatter Plot ==========
with tab2:
    st.subheader("🔵 Scatter Plot: PBR Z-score vs Momentum CS Score")
    
    scatter_data = []
    for industry in pbr_monthly.columns:
        pbr_z = pbr_zscore.loc[ref_date_ts, industry]
        mom_cs = momentum_cs_score.loc[ref_date_ts, industry]
        etf = ETF_INDUSTRY_MAP.get(industry, "-")
        
        scatter_data.append({
            "Industry": industry,
            "ETF": etf,
            "PBR_Z": pbr_z,
            "Mom_CS": mom_cs,
            "PBR_CS": pbr_cs_score.loc[ref_date_ts, industry],
        })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    fig_scatter = px.scatter(
        scatter_df,
        x="PBR_Z",
        y="Mom_CS",
        color="PBR_CS",
        hover_name="Industry",
        hover_data={"ETF": True, "PBR_Z": ":.2f", "Mom_CS": ":.2f", "PBR_CS": ":.2f"},
        color_continuous_scale="RdYlGn",
        title="4사분면 분석: Sweet Spot vs Avoid Zones",
        labels={"PBR_Z": "PBR Z-score", "Mom_CS": "Momentum CS Score"},
    )
    
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_scatter.update_layout(height=600, hovermode="closest")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========== TAB 3: Time-Series ==========
with tab3:
    st.subheader("📈 시계열 분석")
    
    selected_industries = st.multiselect(
        "분석할 산업 선택 (최대 5개)",
        options=pbr_monthly.columns.tolist(),
        default=pbr_monthly.columns.tolist()[:3],
        max_selections=5,
    )
    
    if selected_industries:
        fig_ts = make_subplots(
            rows=2, cols=2,
            subplot_titles=("PBR 원시값", "PBR Rolling Z-score", "Momentum (%)", "Momentum Z-score"),
        )
        
        colors = px.colors.qualitative.Plotly
        
        for i, industry in enumerate(selected_industries):
            color = colors[i % len(colors)]
            
            fig_ts.add_trace(
                go.Scatter(x=pbr_monthly.index, y=pbr_monthly[industry], name=industry,
                          line=dict(color=color)),
                row=1, col=1
            )
            
            fig_ts.add_trace(
                go.Scatter(x=pbr_zscore.index, y=pbr_zscore[industry], name=industry,
                          line=dict(color=color), showlegend=False),
                row=1, col=2
            )
        
        fig_ts.add_trace(
            go.Scatter(x=momentum_df.index, y=momentum_df[selected_industries[0]].values, name=f"{selected_industries[0]} Momentum",
                      line=dict(color="purple", width=3)),
            row=2, col=1
        )
        
        fig_ts.add_trace(
            go.Scatter(x=momentum_zscore.index, y=momentum_zscore[selected_industries[0]].values, name=f"{selected_industries[0]} Mom Z",
                      line=dict(color="purple", width=3, dash="dash")),
            row=2, col=2
        )
        
        fig_ts.update_layout(height=800, hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>📊 Industry Valuation & Momentum Dashboard</div>", unsafe_allow_html=True)
