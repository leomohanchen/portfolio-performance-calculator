"""
Portfolio Performance Calculator - Streamlit Web App
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from portfolio_calculator import run_full_analysis
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

            # ── CHANGED ──────────────────────────────────────────────────────
            # run_full_analysis now returns 5 values instead of 2
            output_df, metrics, bmark_filled, regression, ratios = run_full_analysis(
                temp_file_path,
                end_date,
                benchmark_ticker=benchmark_ticker,
                risk_free_rate=risk_free_rate
            )
            # ── END CHANGED ──────────────────────────────────────────────────

            st.success("✅ Analysis Complete!")

            # ─────────────────────────────────────────────────────────────────
            # SECTION 1: KEY METRICS  (unchanged from original)
            # ─────────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────────
            # SECTION 2: PORTFOLIO VALUE CHART  (unchanged from original)
            # ─────────────────────────────────────────────────────────────────
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
            fig_value.update_layout(
                title='Portfolio Market Value',
                xaxis_title='Date',
                yaxis_title='Market Value ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_value, use_container_width=True)

            # ── NEW ──────────────────────────────────────────────────────────
            # SECTION 3: PORTFOLIO VS BENCHMARK CUMULATIVE RETURN CHART
            # Replaces the old standalone cumulative return chart so both
            # portfolio and benchmark are shown on the same axis for comparison
            # ─────────────────────────────────────────────────────────────────
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
            # ── END NEW ──────────────────────────────────────────────────────

            # ── NEW ──────────────────────────────────────────────────────────
            # SECTION 4: REGRESSION RESULTS — ALPHA & BETA
            # ─────────────────────────────────────────────────────────────────
            st.header("📐 Risk-Adjusted Performance — Alpha & Beta")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Alpha (Annualised)",
                    value=f"{regression['alpha_annualized']:.2%}",
                    help="Excess return above what beta alone would predict. Positive = outperforming on a risk-adjusted basis."
                )
            with col2:
                st.metric(
                    label="Beta",
                    value=f"{regression['beta']:.4f}",
                    help="Sensitivity to benchmark moves. >1 = more volatile than benchmark, <1 = less volatile."
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
            # ── END NEW ──────────────────────────────────────────────────────

            # ── NEW ──────────────────────────────────────────────────────────
            # SECTION 5: PERFORMANCE RATIOS
            # ─────────────────────────────────────────────────────────────────
            st.header("📊 Performance Ratios")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{ratios['sharpe_ratio']:.4f}",
                    help="Excess return per unit of total volatility. >1 is good, >2 is excellent."
                )
                st.metric(
                    label="Sortino Ratio",
                    value=f"{ratios['sortino_ratio']:.4f}",
                    help="Like Sharpe but only penalises downside volatility. Higher than Sharpe = good volatility skew."
                )
            with col2:
                st.metric(
                    label="Treynor Ratio",
                    value=f"{ratios['treynor_ratio']:.4f}",
                    help="Excess return per unit of systematic risk (beta). Useful if portfolio is part of a larger diversified portfolio."
                )
                st.metric(
                    label="Information Ratio",
                    value=f"{ratios['information_ratio']:.4f}",
                    help="Active return per unit of tracking error. >0.5 is good, >1.0 is excellent."
                )
            with col3:
                st.metric(
                    label="Appraisal Ratio",
                    value=f"{ratios['appraisal_ratio']:.4f}",
                    help="Alpha per unit of unsystematic (stock-specific) risk. Measures quality of stock picking."
                )
                st.metric(
                    label="Annualised Tracking Error",
                    value=f"{ratios['tracking_error']:.2%}",
                    help="Volatility of active returns vs benchmark. Lower = portfolio moves more like benchmark."
                )

            # Capture ratios in their own row with a visual indicator
            st.subheader("📡 Capture Ratios")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Up Capture",
                    value=f"{ratios['up_capture']:.2%}",
                    help="% of benchmark's positive days captured by portfolio. >100% = outperforms on up days."
                )
            with col2:
                st.metric(
                    label="Down Capture",
                    value=f"{ratios['down_capture']:.2%}",
                    help="% of benchmark's negative days suffered by portfolio. <100% = loses less on down days."
                )
            with col3:
                capture_label = "✅ Favourable" if ratios['capture_ratio'] > 1 else "⚠️ Unfavourable"
                st.metric(
                    label="Capture Ratio (Up/Down)",
                    value=f"{ratios['capture_ratio']:.4f}",
                    help="Up capture / Down capture. >1 means you capture more upside than downside — ideal."
                )
                st.caption(capture_label)
            # ── END NEW ──────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────
            # SECTION 6: CURRENT HOLDINGS  (unchanged from original)
            # ─────────────────────────────────────────────────────────────────
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

            # ─────────────────────────────────────────────────────────────────
            # SECTION 7: DETAILED DATA TABLE  (unchanged from original)
            # ─────────────────────────────────────────────────────────────────
            with st.expander("📋 View Detailed Portfolio Data"):
                display_df = output_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                key_columns = ['Date', 'net_cash_flow', 'Market_Value_Beginning',
                               'Market_Value_End', 'Subperiod_Return', 'cumulative_return']
                available_columns = [col for col in key_columns if col in display_df.columns]
                st.dataframe(display_df[available_columns], use_container_width=True, hide_index=True)

            # ── NEW ──────────────────────────────────────────────────────────
            # SECTION 8: BENCHMARK DETAILED DATA TABLE
            # ─────────────────────────────────────────────────────────────────
            with st.expander("📋 View Detailed Benchmark Data"):
                bmark_display = bmark_filled.copy()
                bmark_display['Date'] = bmark_display['Date'].dt.strftime('%Y-%m-%d')
                bmark_key_cols = ['Date', 'net_cash_flow', 'benchmark_market_value_beginning',
                                  'benchmark_market_value', 'benchmark_returns', 'cumulative_return']
                bmark_available = [col for col in bmark_key_cols if col in bmark_display.columns]
                st.dataframe(bmark_display[bmark_available], use_container_width=True, hide_index=True)
            # ── END NEW ──────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────
            # SECTION 9: DOWNLOAD  (unchanged from original)
            # ─────────────────────────────────────────────────────────────────
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
            # ── NEW ──────────────────────────────────────────────────────────
            with col2:
                bmark_csv = bmark_filled.to_csv(index=False)
                st.download_button(
                    label="Download Benchmark Data (CSV)",
                    data=bmark_csv,
                    file_name=f"benchmark_analysis_{end_date}.csv",
                    mime="text/csv"
                )
            # ── END NEW ──────────────────────────────────────────────────────

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