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

# Run analysis button
run_analysis = st.sidebar.button("Calculate Performance", type="primary")

# Main content
if uploaded_file is not None and run_analysis:
    try:
        with st.spinner("Analyzing portfolio performance..."):
            # Save uploaded file temporarily (works on Windows, Mac, Linux)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            
            # Run analysis
            output_df, metrics = run_full_analysis(temp_file_path, end_date)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Key Metrics Section
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
            
            # Period Information
            st.markdown(f"""
            **Analysis Period:** {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}
            """)
            
            # Portfolio Value Chart
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
            
            # Cumulative Return Chart
            st.header("📈 Cumulative Returns")
            
            fig_return = go.Figure()
            fig_return.add_trace(go.Scatter(
                x=output_df['Date'],
                y=output_df['cumulative_return'] * 100,
                mode='lines',
                name='Cumulative Return (%)',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.2)'
            ))
            
            fig_return.update_layout(
                title='Cumulative Returns Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_return, use_container_width=True)
            
            # Holdings breakdown (if available)
            st.header("💼 Current Holdings")
            
            # Get holdings columns
            holdings_cols = [col for col in output_df.columns if col.endswith('_Holdings')]
            
            if holdings_cols:
                # Get the last row
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
                    
                    # Display holdings table
                    st.dataframe(
                        holdings_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Pie chart of holdings
                    fig_pie = px.pie(
                        holdings_df,
                        values='Market Value',
                        names='Ticker',
                        title='Portfolio Allocation'
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed Data Table
            with st.expander("📋 View Detailed Data"):
                # Format the dataframe for display
                display_df = output_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Select key columns
                key_columns = ['Date', 'net_cash_flow', 'Market_Value_Beginning', 
                              'Market_Value_End', 'Subperiod_Return', 'cumulative_return']
                available_columns = [col for col in key_columns if col in display_df.columns]
                
                st.dataframe(
                    display_df[available_columns],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Download results
            st.header("💾 Download Results")
            
            csv = output_df.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset (CSV)",
                data=csv,
                file_name=f"portfolio_analysis_{end_date}.csv",
                mime="text/csv"
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # If cleanup fails, it's okay
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)
        # Try to clean up temp file even if there was an error
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass

elif uploaded_file is None:
    # Instructions
    st.info("""
    ### 📋 Instructions:
    
    1. **Upload your transaction CSV file** in the sidebar
       - File should contain columns: Date, Reference, Details
       - Date format: DD/MM/YYYY
       - Details format: "B/S [quantity] [ticker] @ [price]"
    
    2. **Select an end date** for your analysis (defaults to today)
    
    3. **Click "Calculate Performance"** to run the analysis
    
    ### 📊 What you'll get:
    
    - **Total Return (TWRR)**: Your overall investment return
    - **Annualized Return**: Yearly equivalent return rate
    - **Portfolio Value Chart**: Visual timeline of your portfolio growth
    - **Cumulative Returns**: Performance over time
    - **Current Holdings**: Breakdown of your positions
    - **Detailed Data**: Complete analysis with downloadable CSV
    
    ### 💡 Example CSV format:
    ```
    Date,Reference,Details
    01/01/2024,REF001,B 100 AAPL @ 150.50
    15/01/2024,REF002,S 50 AAPL @ 155.25
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
