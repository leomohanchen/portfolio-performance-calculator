"""
Portfolio Performance Calculator - Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from portfolio_calculator import run_full_analysis, calculate_var_cvar
import warnings
import tempfile
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Performance Calculator",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Portfolio Performance Calculator")
st.markdown("""
Calculate your portfolio's Time-Weighted Rate of Return (TWRR) based on transaction history.
Upload your Commsec transaction CSV file and select an end date to analyze your performance.
""")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction CSV File",
    type=['csv'],
    help="Upload your Commsec transaction CSV file"
)

# Date selector
end_date = st.sidebar.date_input(
    "Select End Date",
    value=date.today(),
    help="Select the end date for performance calculation"
)

# ── NEW ──────────────────────────────────────────────────────────────────────
# Benchmark and risk-free rate inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Benchmark Settings")

benchmark_ticker = st.sidebar.text_input(
    "Benchmark Ticker",
    value="VGS.AX",
    help="Yahoo Finance ticker for benchmark (e.g. VGS.AX, ^AXJO, STW.AX)"
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    value=5.29,
    min_value=0.0,
    max_value=20.0,
    step=0.01,
    help="Annualised risk-free rate — use your local central bank cash rate (e.g. RBA cash rate)"
) / 100
# ── END NEW ──────────────────────────────────────────────────────────────────

# ── VaR / CVaR Settings ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("VaR / CVaR Settings")

var_confidence_levels = st.sidebar.multiselect(
    "Confidence Levels",
    options=[0.90, 0.95, 0.99],
    default=[0.95, 0.99],
    format_func=lambda x: f"{x:.0%}",
    help="Confidence levels for VaR and CVaR calculation"
)
if not var_confidence_levels:
    var_confidence_levels = [0.95, 0.99]

var_custom_horizon = st.sidebar.number_input(
    "Custom Horizon (days)",
    min_value=1,
    max_value=252,
    value=30,
    step=1,
    help="Custom horizon in addition to the standard 1-day and 10-day (Basel III). Max 252 = 1 trading year."
)
var_horizons = sorted(set([1, 10, int(var_custom_horizon)]))

# Run analysis button
run_analysis = st.sidebar.button("Calculate Performance", type="primary")

# Main content
if uploaded_file is not None and run_analysis:
    try:
        with st.spinner("Analyzing portfolio performance..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name

            # ── CHANGED — now returns 7 values ───────────────────────────────
            output_df, metrics, bmark_filled, regression, ratios, episodes_df, global_dd, var_results, var_diagnostics = run_full_analysis(
                temp_file_path,
                end_date,
                benchmark_ticker=benchmark_ticker,
                risk_free_rate=risk_free_rate
            )
            # ── END CHANGED ──────────────────────────────────────────────────

            st.success("✅ Analysis Complete!")

            # ── TOP-LEVEL PAGE NAVIGATION ─────────────────────────────────────
            tab_main, tab_drawdown, tab_var = st.tabs(["📊 Performance Overview", "📉 Drawdown Analysis", "⚠️ Value at Risk"])

            # =================================================================
            # TAB 1: PERFORMANCE OVERVIEW  (all existing sections)
            # =================================================================
            with tab_main:

                # ─────────────────────────────────────────────────────────────
                # SECTION 1: KEY METRICS
                # ─────────────────────────────────────────────────────────────
                st.header("📈 Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Total Return",
                        value=f"{metrics['twrr_total']:.2%}",
                        help="Time-Weighted Rate of Return (TWRR) - Total"
                    )
                with col2:
                    st.metric(
                        label="Annualized Return",
                        value=f"{metrics['twrr_annualized']:.2%}",
                        help="Annualized TWRR"
                    )
                with col3:
                    st.metric(
                        label="Current Value",
                        value=f"${metrics['final_market_value']:,.2f}",
                        help="Current market value of portfolio"
                    )
                with col4:
                    st.metric(
                        label="Investment Period",
                        value=f"{metrics['days']} days",
                        help="Number of days from first transaction to end date"
                    )

                st.markdown(f"""
                **Analysis Period:** {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}
                """)

                # ─────────────────────────────────────────────────────────────
                # SECTION 2: PORTFOLIO VALUE CHART
                # ─────────────────────────────────────────────────────────────
                st.header("📊 Portfolio Value Over Time")

                fig_value = go.Figure()

                fig_value.add_trace(go.Scatter(
                    x=output_df['Date'],
                    y=output_df['Market_Value_End'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                fig_value.add_trace(go.Scatter(
                    x=output_df['Date'],
                    y=output_df['cumulative_cost'],
                    mode='lines',
                    name='Cost Basis (no time value)',
                    line=dict(color='#d62728', width=1.5, dash='dash'),
                ))
                fig_value.add_trace(go.Scatter(
                    x=output_df['Date'],
                    y=output_df['tv_adjusted_cost'],
                    mode='lines',
                    name=f'Cost Basis (@ {risk_free_rate * 100:.2f}% p.a. risk-free)',
                    line=dict(color='#ff7f0e', width=1.5, dash='dot'),
                ))
                fig_value.update_layout(
                    title='Portfolio Market Value vs Cost Basis',
                    xaxis_title='Date',
                    yaxis_title='Market Value ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig_value, use_container_width=True)

                with st.expander("ℹ️ How are the cost basis lines calculated?"):
                    st.markdown(f"""
                    **Cost Basis (no time value)** *(red dashed)*  
                    Running cumulative sum of all net cash flows — every buy adds to this line,
                    every sale reduces it. Shows net capital deployed at any point in time.

                    **Cost Basis (@ {risk_free_rate * 100:.2f}% p.a. risk-free)** *(orange dotted)*  
                    Each cash flow compounded forward at the risk-free rate
                    ({risk_free_rate * 100:.2f}% p.a., daily compounding) from the day it was
                    invested to each subsequent date. This is the opportunity cost — what your
                    deployed capital would have grown to in a risk-free instrument.  
                    **Your portfolio needs to stay above this line to have beaten the risk-free rate.**
                    """)

                # ─────────────────────────────────────────────────────────────
                # SECTION 3: PORTFOLIO VS BENCHMARK
                # ─────────────────────────────────────────────────────────────
                st.header("📈 Portfolio vs Benchmark — Cumulative Returns")

                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(
                    x=output_df['Date'],
                    y=output_df['cumulative_return'] * 100,
                    mode='lines',
                    name='My Portfolio',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ))
                fig_compare.add_trace(go.Scatter(
                    x=bmark_filled['Date'],
                    y=bmark_filled['cumulative_return'] * 100,
                    mode='lines',
                    name=f'Benchmark ({benchmark_ticker})',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 127, 14, 0.1)'
                ))
                fig_compare.update_layout(
                    title='Cumulative Return: Portfolio vs Benchmark (same cash flow timing)',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                st.caption(
                    f"ℹ️ Benchmark ({benchmark_ticker}) simulates investing the same cash amounts "
                    "on the same dates as your actual transactions, making the comparison fair."
                )

                # ─────────────────────────────────────────────────────────────
                # SECTION 4: REGRESSION RESULTS — ALPHA & BETA
                # ─────────────────────────────────────────────────────────────
                st.header("📐 Risk-Adjusted Performance — Alpha & Beta")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="Alpha (Annualised)",
                        value=f"{regression['alpha_annualized']:.2%}",
                        help="Excess return above what beta alone would predict."
                    )
                with col2:
                    st.metric(
                        label="Beta",
                        value=f"{regression['beta']:.4f}",
                        help="Sensitivity to benchmark moves."
                    )
                with col3:
                    st.metric(
                        label="R-Squared",
                        value=f"{regression['r_squared']:.4f}",
                        help="How much of your portfolio's movement is explained by the benchmark."
                    )
                with col4:
                    alpha_sig = "✅ Significant" if regression['p_value_alpha'] < 0.05 else "⚠️ Not significant"
                    st.metric(
                        label="Alpha p-value",
                        value=f"{regression['p_value_alpha']:.4f}",
                        help=f"Statistical significance of alpha. {alpha_sig} at 5% level."
                    )
                    st.caption(alpha_sig)

                with st.expander("📋 View Full Regression Summary"):
                    st.text(str(regression['summary']))

                # ─────────────────────────────────────────────────────────────
                # SECTION 5: PERFORMANCE RATIOS
                # ─────────────────────────────────────────────────────────────
                st.header("📊 Performance Ratios")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Sharpe Ratio", value=f"{ratios['sharpe_ratio']:.4f}",
                              help="Excess return per unit of total volatility.")
                    st.metric(label="Sortino Ratio", value=f"{ratios['sortino_ratio']:.4f}",
                              help="Like Sharpe but only penalises downside volatility.")
                with col2:
                    st.metric(label="Treynor Ratio", value=f"{ratios['treynor_ratio']:.4f}",
                              help="Excess return per unit of systematic risk (beta).")
                    st.metric(label="Information Ratio", value=f"{ratios['information_ratio']:.4f}",
                              help="Active return per unit of tracking error.")
                with col3:
                    st.metric(label="Appraisal Ratio", value=f"{ratios['appraisal_ratio']:.4f}",
                              help="Alpha per unit of unsystematic risk.")
                    st.metric(label="Annualised Tracking Error", value=f"{ratios['tracking_error']:.2%}",
                              help="Volatility of active returns vs benchmark.")

                st.subheader("📡 Capture Ratios")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Up Capture", value=f"{ratios['up_capture']:.2%}",
                              help="% of benchmark's positive days captured by portfolio.")
                with col2:
                    st.metric(label="Down Capture", value=f"{ratios['down_capture']:.2%}",
                              help="% of benchmark's negative days suffered by portfolio.")
                with col3:
                    capture_label = "✅ Favourable" if ratios['capture_ratio'] > 1 else "⚠️ Unfavourable"
                    st.metric(label="Capture Ratio (Up/Down)", value=f"{ratios['capture_ratio']:.4f}",
                              help="Up capture / Down capture. >1 means you capture more upside than downside.")
                    st.caption(capture_label)

                # ─────────────────────────────────────────────────────────────
                # SECTION 6: CURRENT HOLDINGS
                # ─────────────────────────────────────────────────────────────
                st.header("💼 Current Holdings")

                holdings_cols = [col for col in output_df.columns if col.endswith('_Holdings')]

                if holdings_cols:
                    last_row = output_df.iloc[-1]
                    holdings_data = []

                    for col in holdings_cols:
                        ticker = col.replace('_Holdings', '')
                        quantity = last_row[col]
                        price_col = f'{ticker}_Price'

                        if quantity > 0 and price_col in output_df.columns:
                            price = last_row[price_col]
                            value = quantity * price
                            holdings_data.append({
                                'Ticker': ticker,
                                'Quantity': int(quantity),
                                'Price': f'${price:.2f}',
                                'Market Value': f'${value:,.2f}',
                                'Weight': value
                            })

                    if holdings_data:
                        holdings_df = pd.DataFrame(holdings_data)
                        total_value = holdings_df['Weight'].sum()
                        holdings_df['Weight %'] = (holdings_df['Weight'] / total_value * 100).round(2)
                        holdings_df['Weight %'] = holdings_df['Weight %'].astype(str) + '%'
                        holdings_df = holdings_df.drop('Weight', axis=1)

                        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

                        fig_pie = px.pie(
                            holdings_df,
                            values='Market Value',
                            names='Ticker',
                            title='Portfolio Allocation'
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                # ─────────────────────────────────────────────────────────────
                # SECTION 7: DETAILED DATA TABLE
                # ─────────────────────────────────────────────────────────────
                with st.expander("📋 View Detailed Portfolio Data"):
                    display_df = output_df.copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    key_columns = ['Date', 'net_cash_flow', 'Market_Value_Beginning',
                                   'Market_Value_End', 'Subperiod_Return', 'cumulative_return']
                    available_columns = [col for col in key_columns if col in display_df.columns]
                    st.dataframe(display_df[available_columns], use_container_width=True, hide_index=True)

                # ─────────────────────────────────────────────────────────────
                # SECTION 8: BENCHMARK DETAILED DATA TABLE
                # ─────────────────────────────────────────────────────────────
                with st.expander("📋 View Detailed Benchmark Data"):
                    bmark_display = bmark_filled.copy()
                    bmark_display['Date'] = bmark_display['Date'].dt.strftime('%Y-%m-%d')
                    bmark_key_cols = ['Date', 'net_cash_flow', 'benchmark_market_value_beginning',
                                      'benchmark_market_value', 'benchmark_returns', 'cumulative_return']
                    bmark_available = [col for col in bmark_key_cols if col in bmark_display.columns]
                    st.dataframe(bmark_display[bmark_available], use_container_width=True, hide_index=True)

                # ─────────────────────────────────────────────────────────────
                # SECTION 9: DOWNLOAD
                # ─────────────────────────────────────────────────────────────
                st.header("💾 Download Results")

                col1, col2 = st.columns(2)
                with col1:
                    csv = output_df.to_csv(index=False)
                    st.download_button(
                        label="Download Portfolio Data (CSV)",
                        data=csv,
                        file_name=f"portfolio_analysis_{end_date}.csv",
                        mime="text/csv"
                    )
                with col2:
                    bmark_csv = bmark_filled.to_csv(index=False)
                    st.download_button(
                        label="Download Benchmark Data (CSV)",
                        data=bmark_csv,
                        file_name=f"benchmark_analysis_{end_date}.csv",
                        mime="text/csv"
                    )

            # =================================================================
            # TAB 2: DRAWDOWN ANALYSIS
            # =================================================================
            with tab_drawdown:

                st.header("📉 Drawdown Analysis")
                st.markdown(
                    "Drawdown is measured in **return space** — the difference in cumulative "
                    "return between a peak and subsequent trough, not a percentage decline of an index level."
                )

                # ── PART A: GLOBAL PEAK-TO-TROUGH ────────────────────────────
                st.subheader("🌍 Global Peak-to-Trough")
                st.markdown(
                    "Single worst drawdown based on the **one global peak** across the entire history "
                    "and the minimum cumulative return strictly after that peak date."
                )

                rec_status = str(global_dd['Recovery Date']) if global_dd['Recovery Date'] else "⚠️ Not yet recovered"
                rec_dur    = f"{global_dd['Recovery Duration (days)']} days" if global_dd['Recovery Duration (days)'] is not None else "—"
                total_dur  = f"{global_dd['Total Duration (days)']} days"    if global_dd['Total Duration (days)'] is not None else "—"

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Max Drawdown",
                              value=f"{global_dd['Max Drawdown']:.4f}",
                              help="Peak cumulative return minus trough cumulative return.")
                with col2:
                    st.metric(label="Drawdown Length",
                              value=f"{global_dd['Drawdown Length (days)']} days",
                              help="Calendar days from global peak to trough.")
                with col3:
                    st.metric(label="Recovery Duration",
                              value=rec_dur,
                              help="Calendar days from trough back to a new peak.")
                with col4:
                    st.metric(label="Total Duration",
                              value=total_dur,
                              help="Calendar days from peak to full recovery.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Peak Date:** {global_dd['Peak Date']}  \n"
                                f"**Peak Return:** {global_dd['Peak Cumulative Return']:.4f}")
                with col2:
                    st.markdown(f"**Trough Date:** {global_dd['Trough Date']}  \n"
                                f"**Trough Return:** {global_dd['Trough Cumulative Return']:.4f}")
                with col3:
                    st.markdown(f"**Recovery Date:** {rec_status}")

                st.markdown("---")

                # ── PART B: CUMULATIVE RETURN CHART WITH DRAWDOWN SHADING ────
                st.subheader("📈 Cumulative Return with Drawdown Periods")

                fig_dd = go.Figure()

                # Cumulative return line
                fig_dd.add_trace(go.Scatter(
                    x=output_df['Date'],
                    y=output_df['cumulative_return'],
                    mode='lines',
                    name='Cumulative Return',
                    line=dict(color='#1f77b4', width=2),
                ))

                # Shade every drawdown episode as a red rectangle
                for _, ep in episodes_df.iterrows():
                    peak_dt = str(ep['Peak Date'])
                    end_dt  = str(ep['Recovery Date']) if ep['Recovery Date'] is not None \
                              else output_df['Date'].max().strftime('%Y-%m-%d')
                    fig_dd.add_vrect(
                        x0=peak_dt, x1=end_dt,
                        fillcolor='rgba(214, 39, 40, 0.12)',
                        layer='below',
                        line_width=0,
                    )

                # Mark the global peak and trough with vertical lines + annotations
                # Using add_shape + add_annotation instead of add_vline to avoid
                # a Plotly bug where annotation positioning arithmetic fails on date axes
                for date_val, label, color in [
                    (str(global_dd['Peak Date']),   'Global Peak',   'green'),
                    (str(global_dd['Trough Date']), 'Global Trough', 'red'),
                ]:
                    fig_dd.add_shape(
                        type='line',
                        x0=date_val, x1=date_val,
                        y0=0, y1=1,
                        xref='x', yref='paper',
                        line=dict(color=color, width=1.5, dash='dash'),
                    )
                    fig_dd.add_annotation(
                        x=date_val,
                        y=1,
                        xref='x', yref='paper',
                        text=label,
                        showarrow=False,
                        font=dict(color=color, size=11),
                        xanchor='left',
                        yanchor='bottom',
                    )

                fig_dd.update_layout(
                    title='Cumulative Return — shaded areas are drawdown periods',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    hovermode='x unified',
                    template='plotly_white',
                    height=450,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    yaxis=dict(tickformat='.2%'),
                )

                st.plotly_chart(fig_dd, use_container_width=True)

                st.markdown("---")

                # ── PART C: ALL DRAWDOWN EPISODES ────────────────────────────
                st.subheader("📋 All Drawdown Episodes")
                st.markdown(
                    "Every distinct drawdown episode from peak to trough to recovery. "
                    "The last row may show **no recovery date** if the portfolio is still underwater."
                )

                if not episodes_df.empty:
                    # Format for display
                    display_ep = episodes_df.copy()
                    display_ep['Max Drawdown'] = display_ep['Max Drawdown'].map('{:.4f}'.format)
                    display_ep['Recovery Date'] = display_ep['Recovery Date'].apply(
                        lambda x: str(x) if pd.notna(x) else '⚠️ Unrecovered'
                    )
                    display_ep['Recovery Duration (days)'] = display_ep['Recovery Duration (days)'].apply(
                        lambda x: int(x) if pd.notna(x) else '—'
                    )
                    display_ep['Total Duration (days)'] = display_ep['Total Duration (days)'].apply(
                        lambda x: int(x) if pd.notna(x) else '—'
                    )

                    st.dataframe(display_ep, use_container_width=True, hide_index=True)

                    # Summary stats across all completed episodes
                    completed = episodes_df[episodes_df['Recovery Date'].notna()]
                    st.markdown("**Summary across all completed episodes:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Total Episodes", value=len(episodes_df))
                    with col2:
                        st.metric(label="Unrecovered Episodes",
                                  value=int(episodes_df['Recovery Date'].isna().sum()))
                    with col3:
                        if not completed.empty:
                            st.metric(label="Avg Drawdown (completed)",
                                      value=f"{completed['Max Drawdown'].mean():.4f}")
                    with col4:
                        if not completed.empty:
                            st.metric(label="Avg Total Duration (completed)",
                                      value=f"{completed['Total Duration (days)'].mean():.0f} days")

                else:
                    st.info("No drawdown episodes found — the portfolio has been in continuous growth.")

            # =================================================================
            # TAB 3: VALUE AT RISK
            # =================================================================
            with tab_var:


                st.header("⚠️ Value at Risk (VaR) & Conditional Value at Risk (CVaR)")
                st.markdown(
                    "Risk estimates based on your portfolio's **daily time-weighted returns**. "
                    "Two methods are shown: Historical Simulation (no assumptions) and "
                    "Parametric Student-t (MLE-fitted fat-tail distribution)."
                )

                # ── Recalculate with user-selected settings ───────────────────
                var_results_user, var_diag = calculate_var_cvar(
                    output_df['Subperiod_Return'],
                    confidence_levels=var_confidence_levels,
                    horizons=var_horizons,
                )
                var_df = pd.DataFrame(var_results_user)

                # ── PART A: DISTRIBUTION DIAGNOSTICS ─────────────────────────
                st.subheader("📐 Return Distribution Diagnostics")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Observations",     var_diag['n_observations'])
                col2.metric("Daily Mean",        f"{var_diag['mean']:.4%}")
                col3.metric("Daily Std Dev",     f"{var_diag['std']:.4%}")
                col4.metric("Skewness",          f"{var_diag['skewness']:.4f}")
                col5.metric("Excess Kurtosis",   f"{var_diag['excess_kurtosis']:.4f}")

                with st.expander("ℹ️ What do these diagnostics tell us?"):
                    st.markdown(f"""
                    **Skewness** measures the asymmetry of the return distribution.
                    - Zero = symmetric (like a normal distribution)
                    - Negative = left-skewed — losses tend to be more extreme than gains
                    - Your portfolio skewness: **{var_diag['skewness']:.4f}**

                    **Excess Kurtosis** measures how fat the tails are relative to a normal distribution.
                    - Zero = normal distribution tails
                    - Positive = fat tails — extreme events (both gains and losses) happen more often than normal predicts
                    - Your portfolio excess kurtosis: **{var_diag['excess_kurtosis']:.4f}**
                    - Values above 1 strongly suggest the normal distribution would *underestimate* tail risk

                    **Fitted Student-t Parameters (MLE)**
                    - Degrees of freedom: **{var_diag['t_dof']:.4f}** — lower = fatter tails (normal ≈ dof > 30)
                    - Location: **{var_diag['t_loc']:.6f}** — centre of the fitted distribution
                    - Scale: **{var_diag['t_scale']:.6f}** — spread of the fitted distribution
                    """)

                st.markdown("---")

                # ── PART B: HISTOGRAM WITH VaR / CVaR MARKERS ────────────────
                st.subheader("📊 Return Distribution with VaR & CVaR")

                returns_clean = output_df['Subperiod_Return'].dropna() * 100  # convert to %

                # Build histogram
                fig_hist = go.Figure()

                fig_hist.add_trace(go.Histogram(
                    x=returns_clean,
                    nbinsx=60,
                    name='Daily Returns',
                    marker_color='rgba(31, 119, 180, 0.6)',
                    marker_line=dict(color='rgba(31, 119, 180, 1.0)', width=0.5),
                ))

                # Overlay fitted Student-t PDF
                x_range = np.linspace(returns_clean.min(), returns_clean.max(), 300)
                dof_fit  = var_diag['t_dof']
                loc_fit  = var_diag['t_loc'] * 100
                scale_fit= var_diag['t_scale'] * 100

                # Scale PDF to match histogram height
                bin_width  = (returns_clean.max() - returns_clean.min()) / 60
                n_obs      = len(returns_clean)
                pdf_scaled = scipy_stats.t.pdf(x_range, dof_fit, loc_fit, scale_fit) * n_obs * bin_width

                fig_hist.add_trace(go.Scatter(
                    x=x_range,
                    y=pdf_scaled,
                    mode='lines',
                    name=f'Fitted Student-t (dof={dof_fit:.2f})',
                    line=dict(color='#ff7f0e', width=2),
                ))

                # Colour palette for confidence levels
                cl_colours = {0.90: ('#9467bd', '#c5b0d5'), 0.95: ('#d62728', '#ff9896'), 0.99: ('#8c564b', '#c49c94')}

                # Add VaR and CVaR vertical lines for each confidence level — Historical only on histogram
                for cl in var_confidence_levels:
                    cl_pct   = f'{cl:.0%}'
                    col_dark, col_light = cl_colours.get(cl, ('#333333', '#aaaaaa'))

                    hist_row = var_df[(var_df['Confidence Level'] == cl) &
                                     (var_df['Method'] == 'Historical Simulation') &
                                     (var_df['Horizon (days)'] == 1)]
                    t_row    = var_df[(var_df['Confidence Level'] == cl) &
                                     (var_df['Method'] == 'Parametric (Student-t)') &
                                     (var_df['Horizon (days)'] == 1)]

                    if not hist_row.empty:
                        var_h  = hist_row.iloc[0]['VaR (%)']
                        cvar_h = hist_row.iloc[0]['CVaR (%)']
                        # Historical VaR — solid line
                        fig_hist.add_shape(type='line', x0=var_h, x1=var_h, y0=0, y1=1,
                                           xref='x', yref='paper',
                                           line=dict(color=col_dark, width=2, dash='solid'))
                        fig_hist.add_annotation(x=var_h, y=0.97, xref='x', yref='paper',
                                                text=f'Hist VaR {cl_pct}', showarrow=False,
                                                font=dict(color=col_dark, size=10),
                                                xanchor='right', yanchor='top')
                        # Historical CVaR — dashed line
                        fig_hist.add_shape(type='line', x0=cvar_h, x1=cvar_h, y0=0, y1=1,
                                           xref='x', yref='paper',
                                           line=dict(color=col_dark, width=2, dash='dash'))
                        fig_hist.add_annotation(x=cvar_h, y=0.88, xref='x', yref='paper',
                                                text=f'Hist CVaR {cl_pct}', showarrow=False,
                                                font=dict(color=col_dark, size=10),
                                                xanchor='right', yanchor='top')

                    if not t_row.empty:
                        var_t  = t_row.iloc[0]['VaR (%)']
                        cvar_t = t_row.iloc[0]['CVaR (%)']
                        # Parametric VaR — solid line, lighter shade
                        fig_hist.add_shape(type='line', x0=var_t, x1=var_t, y0=0, y1=1,
                                           xref='x', yref='paper',
                                           line=dict(color=col_light, width=2, dash='solid'))
                        fig_hist.add_annotation(x=var_t, y=0.79, xref='x', yref='paper',
                                                text=f't-VaR {cl_pct}', showarrow=False,
                                                font=dict(color=col_dark, size=10),
                                                xanchor='right', yanchor='top')
                        # Parametric CVaR — dashed line, lighter shade
                        fig_hist.add_shape(type='line', x0=cvar_t, x1=cvar_t, y0=0, y1=1,
                                           xref='x', yref='paper',
                                           line=dict(color=col_light, width=2, dash='dash'))
                        fig_hist.add_annotation(x=cvar_t, y=0.70, xref='x', yref='paper',
                                                text=f't-CVaR {cl_pct}', showarrow=False,
                                                font=dict(color=col_dark, size=10),
                                                xanchor='right', yanchor='top')

                fig_hist.update_layout(
                    title='Daily Return Distribution — Historical vs Fitted Student-t',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    bargap=0.05,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                with st.expander("ℹ️ How to read this chart"):
                    st.markdown("""
                    **Blue bars** — the actual distribution of your daily portfolio returns.

                    **Orange curve** — the Student-t distribution fitted to your returns via Maximum
                    Likelihood Estimation (MLE). Where it rises above the bars, the model predicts
                    more events than actually occurred; where it falls below, it predicts fewer.

                    **Vertical lines** — VaR and CVaR thresholds for each confidence level.
                    Solid lines = VaR. Dashed lines = CVaR (always further left = larger loss).
                    Dark shades = Historical Simulation. Light shades = Parametric Student-t.

                    Returns to the **left** of a VaR line represent the worst outcomes —
                    the proportion of days beyond VaR equals (1 − confidence level).
                    """)

                st.markdown("---")

                # ── PART C: RESULTS TABLES PER HORIZON ───────────────────────
                st.subheader("📋 VaR & CVaR Results")
                st.markdown(
                    "Results shown for each selected horizon. "
                    "Negative values represent losses as a percentage of portfolio value."
                )

                for h in var_horizons:
                    st.markdown(f"**{h}-day Horizon**")
                    h_df = (
                        var_df[var_df['Horizon (days)'] == h]
                        [['Confidence Level Pct', 'Method', 'VaR (%)', 'CVaR (%)']]
                        .assign(**{
                            'VaR (%)':  lambda d: d['VaR (%)'].map('{:.2f}%'.format),
                            'CVaR (%)': lambda d: d['CVaR (%)'].map('{:.2f}%'.format),
                        })
                        .rename(columns={'Confidence Level Pct': 'Confidence'})
                        .reset_index(drop=True)
                    )
                    st.dataframe(h_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # ── PART D: METHODOLOGY FOOTNOTES ────────────────────────────
                with st.expander("📖 Methodology & Interpretation Guide"):
                    st.markdown(f"""
                    ### What is VaR?
                    **Value at Risk (VaR)** answers: *"What is the maximum loss I would not expect to
                    exceed on a given day, at a chosen confidence level?"*

                    For example, a **95% 1-day VaR of −1.5%** means: on 95 out of 100 trading days,
                    your portfolio is not expected to lose more than 1.5%. The remaining 5% of days
                    may produce losses worse than this.

                    ### What is CVaR?
                    **Conditional VaR (CVaR)**, also called Expected Shortfall, answers: *"Given that
                    losses exceed VaR, what is the average loss?"*

                    CVaR is always larger in magnitude than VaR at the same confidence level. It is
                    considered the superior metric because VaR says nothing about how bad losses can
                    get beyond the threshold — CVaR fills that gap.

                    ### Method 1 — Historical Simulation
                    - Takes your actual daily returns and reads off the empirical percentile directly
                    - **No distributional assumptions** — if your returns had fat tails or skewness, this captures it automatically
                    - Weakness: limited to scenarios that have already occurred in your sample

                    ### Method 2 — Parametric Student-t
                    - Fits a Student-t distribution to your returns using **Maximum Likelihood Estimation (MLE)**
                    - The Student-t has heavier tails than the normal distribution, controlled by degrees of freedom
                    - Fitted degrees of freedom: **{var_diag['t_dof']:.2f}** (lower = fatter tails; normal ≈ dof > 30)
                    - Allows VaR/CVaR to be computed analytically, including scenarios beyond what has been observed
                    - Weakness: still assumes a parametric shape that may not perfectly match your data

                    ### Horizon Scaling
                    Both methods compute 1-day VaR/CVaR first, then scale to longer horizons using the
                    **square root of time rule**: VaR(h days) = VaR(1 day) × √h.
                    This is the Basel III standard for regulatory capital calculation.
                    It assumes returns are independent and identically distributed across days — an approximation
                    that works well for moderate horizons but may underestimate risk over very long periods
                    when volatility clustering (GARCH effects) is present.

                    ### Confidence Levels
                    - **95%** — standard for internal daily risk monitoring
                    - **99%** — Basel III regulatory capital standard for market risk
                    - **90%** — sometimes used for less conservative internal limits

                    ### Ex-Ante Nature
                    VaR and CVaR are **ex-ante** (forward-looking) metrics. Even though they are calculated
                    from historical data, their purpose is to estimate the risk of future losses. They should
                    be recalculated regularly as market conditions evolve.
                    """)

            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass

elif uploaded_file is None:
    st.info("""
    ### 📋 Instructions:

    1. **Upload your transaction CSV file** in the sidebar
       - File should contain columns: Date, Reference, Details
       - Date format: DD/MM/YYYY
       - Details format: "B/S [quantity] [ticker] @ [price]"

    2. **Select an end date** for your analysis (defaults to today)

    3. **Set your benchmark ticker** (default: VGS.AX — MSCI World ETF)

    4. **Set your risk-free rate** (default: 4.29% — approximate RBA cash rate)

    5. **Click "Calculate Performance"** to run the analysis

    ### 📊 What you'll get:

    - **Total Return (TWRR)**: Your overall investment return
    - **Annualized Return**: Yearly equivalent return rate
    - **Portfolio vs Benchmark Chart**: Cumulative returns on the same axis with matched cash flow timing
    - **Alpha & Beta**: Jensen's alpha and market sensitivity from OLS regression
    - **Performance Ratios**: Sharpe, Sortino, Treynor, Information, Appraisal and Capture ratios
    - **Current Holdings**: Breakdown of your positions
    - **Detailed Data**: Complete analysis with downloadable CSV for both portfolio and benchmark

    ### 💡 Example CSV format:
    ```
    Date,Reference,Details
    01/01/2024,REF001,B 100 CBA @ 150.50
    15/01/2024,REF002,S 50 CBA @ 155.25
    ```
    """)
else:
    st.info("👆 Click 'Calculate Performance' in the sidebar to start the analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Portfolio Performance Calculator | TWRR Methodology</p>
</div>
""", unsafe_allow_html=True)