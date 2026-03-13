import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 3rem; }

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 0.8rem 0.8rem;
    text-align: center;
    transition: transform .2s;
    height: 115px;
    min-height: 115px;
    max-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    box-sizing: border-box;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-label {
    color: #94a3b8;
    font-size: clamp(0.55rem, 0.7vw, 0.75rem);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: .2rem;
    white-space: nowrap;
}
.metric-value {
    font-size: clamp(0.95rem, 1.3vw, 1.5rem);
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.15;
    white-space: nowrap;
}
.metric-delta {
    font-size: clamp(0.6rem, 0.75vw, 0.8rem);
    margin-top: .2rem;
    white-space: nowrap;
}
.positive { color: #22c55e; }
.negative { color: #ef4444; }

.section-header {
    color: #f8fafc; font-size: 1.4rem; font-weight: 700;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #6366f1;
}
.insight-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px; padding: 1rem 1.4rem;
    margin: 0.5rem 0 1.5rem 0; color: #cbd5e1;
    font-size: 0.93rem; line-height: 1.7;
}
div[data-testid="stSidebar"] { background: #0f172a; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 20px;
    background: #1e293b;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#dc2626", "Fear": "#f97316",
    "Neutral": "#facc15", "Greed": "#4ade80", "Extreme Greed": "#22c55e",
}

# ─── Data Loader ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fg = pd.read_csv("Datasets/fear_greed_index.csv")
    fg["date"] = pd.to_datetime(fg["date"])

    tr = pd.read_csv("Datasets/historical_data.csv")
    tr["Timestamp IST"] = pd.to_datetime(tr["Timestamp IST"], format="mixed", dayfirst=True)
    tr["date"] = tr["Timestamp IST"].dt.normalize()
    tr["Closed PnL"] = pd.to_numeric(tr["Closed PnL"], errors="coerce").fillna(0)
    tr["Size USD"]   = pd.to_numeric(tr["Size USD"],   errors="coerce").fillna(0)

    merged = pd.merge(tr, fg, on="date", how="left").dropna(subset=["value"])

    def sentiment(v):
        if v < 25: return "Extreme Fear"
        if v < 45: return "Fear"
        if v < 55: return "Neutral"
        if v < 75: return "Greed"
        return "Extreme Greed"

    merged["Sentiment"] = pd.Categorical(
        merged["value"].apply(sentiment),
        categories=SENTIMENT_ORDER, ordered=True
    )
    # Standardize Side column
    merged["Side"] = merged["Side"].str.strip().str.upper()
    merged["Side"] = merged["Side"].replace({"BUY": "Long", "SELL": "Short", "B": "Long", "S": "Short"})
    merged.loc[~merged["Side"].isin(["Long", "Short"]), "Side"] = "Other"

    # Whale flag: top 5% by Size USD
    threshold = merged["Size USD"].quantile(0.95)
    merged["Is_Whale_Trade"] = merged["Size USD"] >= threshold

    return merged, fg, threshold

merged_df, fg_df, whale_threshold = load_data()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compact_num(v, prefix="$"):
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1_000_000: return f"{sign}{prefix}{abs_v/1_000_000:.2f}M"
    elif abs_v >= 1_000:   return f"{sign}{prefix}{abs_v/1_000:.1f}K"
    else:                  return f"{sign}{prefix}{abs_v:.2f}"

def metric_html(label, value, delta=None, positive=True):
    delta_html = ""
    if delta is not None:
        cls = "positive" if positive else "negative"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

def section(title): st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
def insight(text):  st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Global Filters")
    st.markdown("---")

    selected_sentiments = st.multiselect(
        "Market Sentiment", options=SENTIMENT_ORDER, default=SENTIMENT_ORDER)

    all_coins = sorted(merged_df["Coin"].dropna().unique().tolist())
    selected_coins = st.multiselect(
        "Coin / Asset", options=all_coins,
        default=all_coins[:10] if len(all_coins) > 10 else all_coins)

    side_options = ["Long", "Short", "Other"]
    selected_sides = st.multiselect(
        "Trade Direction", options=side_options, default=["Long", "Short"],
        help="Long = BUY, Short = SELL")

    date_min = merged_df["date"].min().date()
    date_max = merged_df["date"].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    st.markdown("---")
    pnl_threshold = st.slider("PnL Outlier Clip (±USD)",
                              min_value=100, max_value=20000, value=5000, step=100)

    include_whales = st.checkbox("Include Whale Trades", value=True,
                                 help=f"Whale = Top 5% size (>{compact_num(whale_threshold)} per trade)")

    st.markdown("---")
    st.caption("💡 Hover charts for tooltip detail. Click legends to toggle. Double-click to reset zoom.")

# ─── Apply Filters ────────────────────────────────────────────────────────────
df = merged_df.copy()
if len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
df = df[df["Sentiment"].isin(selected_sentiments)]
df = df[df["Coin"].isin(selected_coins)]
df = df[df["Side"].isin(selected_sides)]
if not include_whales:
    df = df[~df["Is_Whale_Trade"]]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📊 Trader Intelligence Dashboard")
st.markdown("#### *Market Sentiment × Trader Performance — Hyperliquid EDA*")
st.caption(f"Showing **{len(df):,}** trades · **{df['date'].nunique()}** unique days · **{df['Account'].nunique()}** traders · Coins: {len(selected_coins)} selected")
st.markdown("---")

# ─── KPI Row ──────────────────────────────────────────────────────────────────
total_pnl = df["Closed PnL"].sum()
avg_pnl   = df["Closed PnL"].mean()
win_rate  = (df["Closed PnL"] > 0).mean() * 100
avg_size  = df["Size USD"].mean()
total_vol = df["Size USD"].sum()
n_traders = df["Account"].nunique()

cols = st.columns(6)
with cols[0]: st.markdown(metric_html("Total PnL", compact_num(total_pnl), "Net Realized", total_pnl>=0), unsafe_allow_html=True)
with cols[1]: st.markdown(metric_html("Avg PnL / Trade", compact_num(avg_pnl), "Per Trade", avg_pnl>=0), unsafe_allow_html=True)
with cols[2]: st.markdown(metric_html("Win Rate", f"{win_rate:.1f}%", "↑ Good" if win_rate>=50 else "↓ Below 50%", win_rate>=50), unsafe_allow_html=True)
with cols[3]: st.markdown(metric_html("Avg Trade Size", compact_num(avg_size), "USD notional"), unsafe_allow_html=True)
with cols[4]: st.markdown(metric_html("Total Volume", compact_num(total_vol), "USD traded"), unsafe_allow_html=True)
with cols[5]: st.markdown(metric_html("Unique Traders", f"{n_traders:,}", "Active accounts"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sentiment Overview",
    "⚖️ Long vs Short",
    "🧠 Trader Profiling",
    "🐋 Whale Analysis",
    "🔍 Data Explorer",
])

# ════════════════════════════════════════════════════
# TAB 1 — SENTIMENT OVERVIEW (existing analysis)
# ════════════════════════════════════════════════════
with tab1:
    section("Trade Activity by Sentiment")
    insight("<b>Finding:</b> Traders are significantly more active during <b>Greed</b> and <b>Extreme Greed</b> phases — momentum-chasing is the dominant behavior. Fear-period activity is lower but trade sizes spike — a classic <i>falling knife</i> pattern.")

    act_df = df.groupby("Sentiment", observed=True).agg(
        Trade_Count=("Trade ID", "count"),
        Total_Vol=("Size USD", "sum")
    ).reset_index().sort_values("Sentiment")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(act_df, x="Sentiment", y="Trade_Count", color="Sentiment",
                     color_discrete_map=SENTIMENT_COLORS, text="Trade_Count",
                     title="Number of Trades by Sentiment", template="plotly_dark")
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                          yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(act_df, x="Sentiment", y="Total_Vol", color="Sentiment",
                      color_discrete_map=SENTIMENT_COLORS,
                      title="Total Volume Traded (USD)", template="plotly_dark")
        fig2.update_layout(showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                           yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig2, use_container_width=True)

    section("Win Rate & PnL by Sentiment")
    insight("<b>Finding:</b> <b>Extreme Greed</b> yields the highest win rate (~46.5%) — trend-following works marginally better in euphoric conditions. Yet average PnL is volatile: losses on losing trades still outweigh gains — a skewed risk/reward pattern across all sentiments.")

    pnl_df = df.groupby("Sentiment", observed=True).agg(
        Win_Rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        Avg_PnL=("Closed PnL", "mean"),
    ).reset_index().sort_values("Sentiment")

    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=pnl_df["Sentiment"], y=pnl_df["Win_Rate"],
                              marker_color=[SENTIMENT_COLORS[s] for s in pnl_df["Sentiment"]],
                              text=[f"{v:.1f}%" for v in pnl_df["Win_Rate"]], textposition="outside"))
        fig3.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                       annotation_text="50% Breakeven", annotation_position="top right")
        fig3.update_layout(title="Win Rate by Sentiment (%)", template="plotly_dark", height=400,
                           showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[0,60], gridcolor="#1e293b"), xaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        colors_avg = ["#ef4444" if v < 0 else "#22c55e" for v in pnl_df["Avg_PnL"]]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=pnl_df["Sentiment"], y=pnl_df["Avg_PnL"],
                              marker_color=colors_avg,
                              text=[f"${v:.2f}" for v in pnl_df["Avg_PnL"]], textposition="outside"))
        fig4.add_hline(y=0, line_dash="solid", line_color="#475569")
        fig4.update_layout(title="Average PnL per Trade (USD)", template="plotly_dark", height=400,
                           showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(gridcolor="#1e293b"), xaxis=dict(showgrid=False))
        st.plotly_chart(fig4, use_container_width=True)

    section("PnL Distribution (Violin + Box)")
    insight(f"<b>Finding:</b> The IQR during <b>Extreme Greed</b> is widest — huge upside AND downside swings. Fear periods have a negative skew, confirming systematic losses during market panics. <b>Adjust the sidebar slider to clip outliers.</b>")

    clip_df = df.copy()
    clip_df["PnL_Clipped"] = clip_df["Closed PnL"].clip(-pnl_threshold, pnl_threshold)
    fig5 = px.violin(clip_df, x="Sentiment", y="PnL_Clipped", color="Sentiment",
                     color_discrete_map=SENTIMENT_COLORS, box=True, points="outliers",
                     category_orders={"Sentiment": SENTIMENT_ORDER},
                     title=f"PnL Distribution per Sentiment (clipped ±${pnl_threshold:,})",
                     template="plotly_dark")
    fig5.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig5.update_layout(height=480, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                       yaxis=dict(gridcolor="#1e293b"))
    st.plotly_chart(fig5, use_container_width=True)

    section("Daily PnL vs Fear & Greed Over Time")
    insight("<b>Finding:</b> Periods of extreme sentiment precede the biggest PnL swings. Use the hover tooltip to inspect individual trading days.")

    ts_df = df.groupby("date").agg(
        Total_PnL=("Closed PnL", "sum"),
        FG_Index=("value", "first")
    ).reset_index().sort_values("date")

    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    fig7.add_trace(go.Bar(x=ts_df["date"], y=ts_df["Total_PnL"], name="Daily PnL (USD)",
                          marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in ts_df["Total_PnL"]],
                          opacity=0.7), secondary_y=False)
    fig7.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["FG_Index"],
                              name="Fear & Greed Index", line=dict(color="#6366f1", width=2.5),
                              mode="lines"), secondary_y=True)
    fig7.update_layout(title="Daily Total PnL vs Fear & Greed Index", template="plotly_dark",
                       height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       legend=dict(x=0.01, y=0.99), hovermode="x unified",
                       xaxis=dict(gridcolor="#1e293b"),
                       yaxis=dict(gridcolor="#1e293b", title="Total Daily PnL (USD)"),
                       yaxis2=dict(title="Fear & Greed Index", range=[0, 100]))
    st.plotly_chart(fig7, use_container_width=True)

    section("Sizing vs Sentiment Index (Scatter)")
    insight("<b>Finding:</b> Large trade sizes are concentrated at <b>low Fear & Greed index values</b> — traders over-size when markets are fearful. Bubble size = daily trade count.")

    daily_agg = df.groupby("date").agg(
        Avg_Index=("value", "first"), Total_Trades=("Trade ID", "count"),
        Avg_Size=("Size USD", "mean"), Sentiment=("Sentiment", "first"),
        Total_PnL=("Closed PnL", "sum"),
    ).reset_index().dropna()

    fig6 = px.scatter(daily_agg, x="Avg_Index", y="Avg_Size",
                      size="Total_Trades", color="Sentiment",
                      color_discrete_map=SENTIMENT_COLORS,
                      hover_data={"date": True, "Total_Trades": True, "Total_PnL": ":.2f", "Avg_Index": ":.1f"},
                      title="Fear & Greed Index vs Avg Trade Size (bubble = trade count)",
                      template="plotly_dark", size_max=50,
                      category_orders={"Sentiment": SENTIMENT_ORDER})
    fig6.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis=dict(title="Fear & Greed Index", gridcolor="#1e293b"),
                       yaxis=dict(title="Avg Trade Size (USD)", gridcolor="#1e293b"))
    st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 2 — LONG vs SHORT (PHASE 2)
# ════════════════════════════════════════════════════
with tab2:
    section("⚖️ Long vs Short: Directional Performance by Sentiment")
    insight("""Does going <b>Long during Greed</b> or <b>Short during Fear</b> pay off systematically?
    Here we break down Win Rate, Average PnL, and Trade Volume separately for Long and Short positions across each sentiment phase.
    Counter-trend traders (e.g. Shorting during Extreme Greed) often face squeeze risk — this chart reveals if the data supports that.""")

    ls_df = df[df["Side"].isin(["Long", "Short"])].copy()

    # Win Rate by Side × Sentiment
    ls_summary = ls_df.groupby(["Sentiment", "Side"], observed=True).agg(
        Trade_Count=("Trade ID", "count"),
        Win_Rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        Avg_PnL=("Closed PnL", "mean"),
        Total_PnL=("Closed PnL", "sum"),
    ).reset_index()
    ls_summary["Sentiment"] = pd.Categorical(ls_summary["Sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    ls_summary = ls_summary.sort_values("Sentiment")

    fig_wr = px.bar(ls_summary, x="Sentiment", y="Win_Rate", color="Side",
                    barmode="group", text_auto=".1f",
                    color_discrete_map={"Long": "#22c55e", "Short": "#ef4444"},
                    category_orders={"Sentiment": SENTIMENT_ORDER, "Side": ["Long", "Short"]},
                    title="Win Rate: Long vs Short by Market Sentiment (%)",
                    template="plotly_dark")
    fig_wr.add_hline(y=50, line_dash="dash", line_color="#94a3b8",
                     annotation_text="50% Breakeven", annotation_position="top right")
    fig_wr.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b", range=[0, 60]))
    st.plotly_chart(fig_wr, use_container_width=True)

    col_ls1, col_ls2 = st.columns(2)
    with col_ls1:
        fig_avgpnl = px.bar(ls_summary, x="Sentiment", y="Avg_PnL", color="Side",
                            barmode="group",
                            color_discrete_map={"Long": "#22c55e", "Short": "#f97316"},
                            category_orders={"Sentiment": SENTIMENT_ORDER, "Side": ["Long", "Short"]},
                            title="Avg PnL per Trade: Long vs Short (USD)",
                            template="plotly_dark")
        fig_avgpnl.add_hline(y=0, line_dash="solid", line_color="#475569")
        fig_avgpnl.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_avgpnl, use_container_width=True)

    with col_ls2:
        fig_cnt = px.bar(ls_summary, x="Sentiment", y="Trade_Count", color="Side",
                         barmode="group",
                         color_discrete_map={"Long": "#6366f1", "Short": "#facc15"},
                         category_orders={"Sentiment": SENTIMENT_ORDER, "Side": ["Long", "Short"]},
                         title="Trade Volume: Long vs Short Count",
                         template="plotly_dark")
        fig_cnt.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_cnt, use_container_width=True)

    section("Total PnL Flow: Long vs Short")
    insight("""<b>Finding:</b> This stacked comparison shows the cumulative net effect of Long vs Short positions
    across each sentiment regime. If Shorts bleed money even during Fear, it suggests the broader market
    was not yet in a full risk-off mode during those days.""")

    fig_tot = px.bar(ls_summary, x="Sentiment", y="Total_PnL", color="Side",
                     barmode="relative",
                     color_discrete_map={"Long": "#22c55e", "Short": "#ef4444"},
                     category_orders={"Sentiment": SENTIMENT_ORDER},
                     title="Total PnL (Stacked): Long vs Short by Sentiment",
                     template="plotly_dark")
    fig_tot.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
    fig_tot.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b",
                           title="Total Realized PnL (USD)"))
    st.plotly_chart(fig_tot, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 3 — TRADER PROFILING (PHASE 2)
# ════════════════════════════════════════════════════
with tab3:
    section("🧠 Smart Money vs Retail: Trader Profiling")
    insight("""Using the <b>Account</b> hash, we profile each unique trader's net performance.
    <b>Smart Money</b> = accounts with positive total PnL (net profitable over the dataset).
    <b>Retail / Net Losers</b> = accounts with negative total PnL.
    The key question: <i>Do smart money traders use market sentiment differently than retail?</i>""")

    # Classify traders
    trader_pnl = df.groupby("Account").agg(
        Total_PnL=("Closed PnL", "sum"),
        Trade_Count=("Trade ID", "count"),
        Win_Rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        Avg_Size=("Size USD", "mean"),
    ).reset_index()
    trader_pnl["Trader_Type"] = trader_pnl["Total_PnL"].apply(
        lambda x: "Smart Money" if x > 0 else "Retail / Net Loser"
    )

    n_smart  = (trader_pnl["Trader_Type"] == "Smart Money").sum()
    n_retail = (trader_pnl["Trader_Type"] == "Retail / Net Loser").sum()

    col_tp1, col_tp2 = st.columns([1, 2])
    with col_tp1:
        st.markdown(metric_html("Smart Money Traders", f"{n_smart:,}", f"{n_smart/(n_smart+n_retail)*100:.1f}% of all traders", True), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(metric_html("Retail / Net Losers", f"{n_retail:,}", f"{n_retail/(n_smart+n_retail)*100:.1f}% of all traders", False), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig_pie = px.pie(trader_pnl, names="Trader_Type",
                         color="Trader_Type",
                         color_discrete_map={"Smart Money": "#22c55e", "Retail / Net Loser": "#ef4444"},
                         title="Trader Breakdown",
                         template="plotly_dark")
        fig_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                               legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_tp2:
        # Win Rate distribution per type
        fig_wr_dist = px.histogram(
            trader_pnl, x="Win_Rate", color="Trader_Type", nbins=40, barmode="overlay",
            color_discrete_map={"Smart Money": "#22c55e", "Retail / Net Loser": "#ef4444"},
            opacity=0.7,
            title="Win Rate Distribution: Smart Money vs Retail",
            template="plotly_dark"
        )
        fig_wr_dist.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                              annotation_text="50%", annotation_position="top right")
        fig_wr_dist.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis=dict(title="Win Rate (%)", gridcolor="#1e293b"),
                                   yaxis=dict(title="# Traders", gridcolor="#1e293b"))
        st.plotly_chart(fig_wr_dist, use_container_width=True)

    section("Do Smart Money Traders Exploit Sentiment?")
    insight("""<b>Finding:</b> Merging trade-level data with each trader's profile reveals whether profitable accounts
    concentrate their activity in specific sentiment regimes — e.g., accumulating during <b>Extreme Fear</b>
    and reducing during <b>Extreme Greed</b>, the hallmark of a disciplined contrarian strategy.""")

    df_typed = df.merge(trader_pnl[["Account", "Trader_Type"]], on="Account", how="left")

    sm_vs_ret = df_typed.groupby(["Sentiment", "Trader_Type"], observed=True).agg(
        Win_Rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        Avg_PnL=("Closed PnL", "mean"),
        Trade_Count=("Trade ID", "count"),
    ).reset_index()
    sm_vs_ret["Sentiment"] = pd.Categorical(sm_vs_ret["Sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    sm_vs_ret = sm_vs_ret.sort_values("Sentiment")

    col_tp3, col_tp4 = st.columns(2)
    with col_tp3:
        fig_sm_wr = px.bar(sm_vs_ret, x="Sentiment", y="Win_Rate", color="Trader_Type",
                           barmode="group", text_auto=".1f",
                           color_discrete_map={"Smart Money": "#22c55e", "Retail / Net Loser": "#ef4444"},
                           category_orders={"Sentiment": SENTIMENT_ORDER},
                           title="Win Rate by Sentiment: Smart Money vs Retail",
                           template="plotly_dark")
        fig_sm_wr.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
        fig_sm_wr.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b", range=[0,60]))
        st.plotly_chart(fig_sm_wr, use_container_width=True)

    with col_tp4:
        fig_sm_cnt = px.bar(sm_vs_ret, x="Sentiment", y="Trade_Count", color="Trader_Type",
                            barmode="group",
                            color_discrete_map={"Smart Money": "#6366f1", "Retail / Net Loser": "#f97316"},
                            category_orders={"Sentiment": SENTIMENT_ORDER},
                            title="Trade Activity by Sentiment: Smart Money vs Retail",
                            template="plotly_dark")
        fig_sm_cnt.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_sm_cnt, use_container_width=True)

    section("Top 20 Most Profitable Traders")
    top20 = trader_pnl.nlargest(20, "Total_PnL")[["Account", "Total_PnL", "Trade_Count", "Win_Rate", "Avg_Size"]]
    top20["Account"] = top20["Account"].str[:10] + "..."
    top20.columns = ["Account (truncated)", "Total PnL ($)", "# Trades", "Win Rate (%)", "Avg Size ($)"]
    top20 = top20.reset_index(drop=True)
    st.dataframe(top20.style.format({
        "Total PnL ($)": "{:,.2f}", "Win Rate (%)": "{:.1f}", "Avg Size ($)": "{:,.0f}"}),
        use_container_width=True
    )


# ════════════════════════════════════════════════════
# TAB 4 — WHALE ANALYSIS (PHASE 2)
# ════════════════════════════════════════════════════
with tab4:
    section(f"🐋 Whale vs Retail: Trade Size Analysis (Whale = >{compact_num(whale_threshold)})")
    insight(f"""We define <b>Whale Trades</b> as the top 5% of trades by USD size
    (threshold: <b>{compact_num(whale_threshold)}</b> per trade). The hypothesis is that institutional or whale positions
    might exploit sentiment differently — e.g. absorbing fear-driven sell-offs or distributing during euphoria.
    Use the sidebar toggle to exclude whale trades from other charts to see retail-only behavior.""")

    all_df = merged_df.copy()
    if len(date_range) == 2:
        all_df = all_df[(all_df["date"].dt.date >= date_range[0]) & (all_df["date"].dt.date <= date_range[1])]
    all_df = all_df[all_df["Sentiment"].isin(selected_sentiments)]
    all_df = all_df[all_df["Coin"].isin(selected_coins)]
    all_df["Trade_Tier"] = all_df["Is_Whale_Trade"].map({True: "Whale", False: "Retail"})

    whale_summary = all_df.groupby(["Sentiment", "Trade_Tier"], observed=True).agg(
        Trade_Count=("Trade ID", "count"),
        Win_Rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        Avg_PnL=("Closed PnL", "mean"),
        Total_PnL=("Closed PnL", "sum"),
        Avg_Size=("Size USD", "mean"),
    ).reset_index()
    whale_summary["Sentiment"] = pd.Categorical(whale_summary["Sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    whale_summary = whale_summary.sort_values("Sentiment")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        fig_w_wr = px.bar(whale_summary, x="Sentiment", y="Win_Rate", color="Trade_Tier",
                          barmode="group", text_auto=".1f",
                          color_discrete_map={"Whale": "#6366f1", "Retail": "#94a3b8"},
                          category_orders={"Sentiment": SENTIMENT_ORDER},
                          title="Win Rate: Whale vs Retail by Sentiment (%)",
                          template="plotly_dark")
        fig_w_wr.add_hline(y=50, line_dash="dash", line_color="#ef4444")
        fig_w_wr.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b", range=[0,60]))
        st.plotly_chart(fig_w_wr, use_container_width=True)

    with col_w2:
        fig_w_pnl = px.bar(whale_summary, x="Sentiment", y="Avg_PnL", color="Trade_Tier",
                           barmode="group",
                           color_discrete_map={"Whale": "#6366f1", "Retail": "#94a3b8"},
                           category_orders={"Sentiment": SENTIMENT_ORDER},
                           title="Avg PnL per Trade: Whale vs Retail (USD)",
                           template="plotly_dark")
        fig_w_pnl.add_hline(y=0, line_dash="solid", line_color="#475569")
        fig_w_pnl.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_w_pnl, use_container_width=True)

    section("Size Distribution: Where Do Whales Trade the Most?")
    insight("""<b>Finding:</b> This scatter shows individual whale trade entry points on the sentiment index.
    Clusters near extreme sentiment values reveal whether big money is contrarian or momentum-driven.""")

    whale_trades = all_df[all_df["Is_Whale_Trade"]].copy()
    fig_w_scatter = px.scatter(
        whale_trades, x="value", y="Size USD",
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        opacity=0.5, size_max=10,
        hover_data={"Coin": True, "Side": True, "Closed PnL": ":.2f"},
        title="Whale Trade Sizing vs Fear & Greed Index",
        template="plotly_dark",
        category_orders={"Sentiment": SENTIMENT_ORDER}
    )
    fig_w_scatter.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(title="Fear & Greed Index", gridcolor="#1e293b"),
                                 yaxis=dict(title="Trade Size (USD)", gridcolor="#1e293b"))
    st.plotly_chart(fig_w_scatter, use_container_width=True)

    col_w3, col_w4 = st.columns(2)
    with col_w3:
        whale_by_sent = all_df.groupby(["Sentiment", "Is_Whale_Trade"], observed=True).size().reset_index(name="Count")
        whale_by_sent["Type"] = whale_by_sent["Is_Whale_Trade"].map({True: "Whale", False: "Retail"})
        whale_by_sent["Sentiment"] = pd.Categorical(whale_by_sent["Sentiment"], categories=SENTIMENT_ORDER, ordered=True)
        fig_wcount = px.bar(whale_by_sent.sort_values("Sentiment"), x="Sentiment", y="Count",
                            color="Type", barmode="stack",
                            color_discrete_map={"Whale": "#6366f1", "Retail": "#1e293b"},
                            title="Trade Count Distribution: Whale vs Retail",
                            template="plotly_dark")
        fig_wcount.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_wcount, use_container_width=True)

    with col_w4:
        fig_box_whale = px.box(
            all_df[all_df["Is_Whale_Trade"]].assign(
                PnL_Clip=lambda x: x["Closed PnL"].clip(-pnl_threshold, pnl_threshold)
            ),
            x="Sentiment", y="PnL_Clip", color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"Sentiment": SENTIMENT_ORDER},
            title="Whale Trade PnL Distribution by Sentiment",
            template="plotly_dark"
        )
        fig_box_whale.update_layout(height=380, showlegend=False,
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig_box_whale, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ════════════════════════════════════════════════════
with tab5:
    section("🔍 Interactive Raw Data Explorer")
    insight("Browse the full merged dataset with your current sidebar filters applied. You can sort any column by clicking the header, and download filtered results as CSV.")

    cols_to_show = ["date", "Account", "Coin", "Side", "Size USD", "Closed PnL",
                    "value", "classification", "Sentiment", "Is_Whale_Trade"]
    available = [c for c in cols_to_show if c in df.columns]

    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        search_coin = st.text_input("Filter by Coin (contains)", "")
    with col_exp2:
        min_pnl = st.number_input("Min PnL ($)", value=float(df["Closed PnL"].min()), step=100.0)
    with col_exp3:
        max_pnl = st.number_input("Max PnL ($)", value=float(df["Closed PnL"].max()), step=100.0)

    view_df = df[available].copy()
    if search_coin:
        view_df = view_df[view_df["Coin"].str.contains(search_coin, case=False, na=False)]
    view_df = view_df[(view_df["Closed PnL"] >= min_pnl) & (view_df["Closed PnL"] <= max_pnl)]
    view_df = view_df.sort_values("date", ascending=False).reset_index(drop=True)

    st.caption(f"Showing {len(view_df):,} rows")
    st.dataframe(view_df, use_container_width=True, height=420)

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_trades.csv",
        mime="text/csv"
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📌 Data: Hyperliquid Historical Trades · Crypto Fear & Greed Index | Built with Streamlit & Plotly | Phase 2 EDA")
