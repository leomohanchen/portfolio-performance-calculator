"""
DEBUG SCRIPT - Portfolio Calculator
This will help you see what's happening at each step
"""

import pandas as pd
from portfolio_calculator import (
    clean_transactions_data,
    daily_net_cash_flow,
    accumulated_holdings_data,
    fill_portfolio_dates,
    price_data,
    calculate_portfolio_value,
    calculate_performance_metrics
)

# =============================================================================
# CONFIGURATION - CHANGE THIS TO YOUR FILE
# =============================================================================
csv_file = "C:/Users/Leo Chen/Desktop/Personal Finance/Trading&Performance/Transactions_5341720_01012024_22122025.csv"  # Change this to your actual CSV file
end_date = "2025-12-31"  # Change this to your desired end date

print("=" * 80)
print("PORTFOLIO CALCULATOR - DEBUG MODE")
print("=" * 80)
print(f"\nAnalyzing: {csv_file}")
print(f"End Date: {end_date}\n")

try:
    # =============================================================================
    # STEP 1: Load and Clean Transactions
    # =============================================================================
    print("-" * 80)
    print("STEP 1: Loading and cleaning transaction data...")
    print("-" * 80)
    
    filtered_transactions = clean_transactions_data(csv_file)
    
    print(f"✓ Total transactions loaded: {len(filtered_transactions)}")
    print(f"\nFirst few transactions:")
    print(filtered_transactions[['Date', 'buy_sell', 'quantity', 'ticker', 'price', 'cash_flow']].head(10))
    print(f"\nDate range: {filtered_transactions['Date'].min()} to {filtered_transactions['Date'].max()}")
    print(f"\nUnique tickers: {filtered_transactions['ticker'].unique()}")
    
    input("\n>>> Press Enter to continue to Step 2...")
    
    # =============================================================================
    # STEP 2: Calculate Daily Cash Flows
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Calculating daily net cash flows...")
    print("-" * 80)
    
    daily_cash_flow = daily_net_cash_flow(filtered_transactions)
    
    print(f"✓ Days with transactions: {len(daily_cash_flow)}")
    print(f"\nDaily cash flows:")
    print(daily_cash_flow.head(10))
    print(f"\nTotal cash invested (positive = money in): ${daily_cash_flow['net_cash_flow'].sum():,.2f}")
    
    input("\n>>> Press Enter to continue to Step 3...")
    
    # =============================================================================
    # STEP 3: Calculate Accumulated Holdings
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Calculating accumulated holdings...")
    print("-" * 80)
    
    accumulated_holding = accumulated_holdings_data(filtered_transactions)
    
    print(f"✓ Days with holding changes: {len(accumulated_holding)}")
    print(f"\nAccumulated holdings over time:")
    print(accumulated_holding.head(10))
    print(f"\nFinal holdings (last transaction date):")
    print(accumulated_holding.iloc[-1])
    
    input("\n>>> Press Enter to continue to Step 4...")
    
    # =============================================================================
    # STEP 4: Fill Missing Dates
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Filling missing dates...")
    print("-" * 80)
    
    filled_holdings = fill_portfolio_dates(accumulated_holding, end_date)
    
    print(f"✓ Total days (with filled dates): {len(filled_holdings)}")
    print(f"\nDate range after filling: {filled_holdings['Date'].min()} to {filled_holdings['Date'].max()}")
    print(f"\nLast 5 days of holdings:")
    print(filled_holdings.tail())
    
    input("\n>>> Press Enter to continue to Step 5...")
    
    # =============================================================================
    # STEP 5: Download Price Data
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Downloading historical price data...")
    print("-" * 80)
    print("(This may take a moment...)")
    
    filled_close_prices = price_data(filled_holdings, end_date)
    
    print(f"✓ Price data downloaded for {len(filled_close_prices)} days")
    print(f"\nLast 5 days of prices:")
    print(filled_close_prices.tail())
    
    # Check for missing prices
    price_cols = [col for col in filled_close_prices.columns if col != 'Date']
    for col in price_cols:
        missing = filled_close_prices[col].isna().sum()
        if missing > 0:
            print(f"⚠️  WARNING: {col} has {missing} days with missing prices")
    
    input("\n>>> Press Enter to continue to Step 6...")
    
    # =============================================================================
    # STEP 6: Calculate Portfolio Value
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Calculating portfolio value and returns...")
    print("-" * 80)
    
    output = calculate_portfolio_value(daily_cash_flow, filled_holdings, filled_close_prices)
    
    print(f"✓ Portfolio calculated for {len(output)} days")
    print(f"\nKey columns in output:")
    print(output.columns.tolist())
    
    print(f"\nLast 10 days of portfolio data:")
    key_cols = ['Date', 'net_cash_flow', 'Market_Value_Beginning', 'Market_Value_End', 
                'Subperiod_Return', 'cumulative_return']
    available_cols = [col for col in key_cols if col in output.columns]
    print(output[available_cols].tail(10))
    
    input("\n>>> Press Enter to continue to Step 7...")
    
    # =============================================================================
    # STEP 7: Calculate Performance Metrics
    # =============================================================================
    print("\n" + "-" * 80)
    print("STEP 7: Calculating final performance metrics...")
    print("-" * 80)
    
    metrics = calculate_performance_metrics(output)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nPeriod: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}")
    print(f"Duration: {metrics['days']} days")
    print(f"\nTotal Return (TWRR): {metrics['twrr_total']:.4%}")
    print(f"Annualized Return: {metrics['twrr_annualized']:.4%}")
    print(f"Final Market Value: ${metrics['final_market_value']:,.2f}")
    
    # =============================================================================
    # ADDITIONAL DIAGNOSTICS
    # =============================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTICS - Things to Check")
    print("=" * 80)
    
    # Check 1: Are there any NaN values in returns?
    nan_returns = output['Subperiod_Return'].isna().sum()
    if nan_returns > 0:
        print(f"\n⚠️  WARNING: {nan_returns} days have NaN (undefined) returns")
        print("This usually happens when there's division by zero.")
        print("\nRows with NaN returns:")
        print(output[output['Subperiod_Return'].isna()][['Date', 'Market_Value_Beginning', 
                                                           'Market_Value_End', 'net_cash_flow']])
    
    # Check 2: Are there any infinite values?
    inf_returns = output['Subperiod_Return'].isin([float('inf'), float('-inf')]).sum()
    if inf_returns > 0:
        print(f"\n⚠️  WARNING: {inf_returns} days have infinite returns")
    
    # Check 3: Check for unusually large returns (>100% in a day)
    large_returns = output[output['Subperiod_Return'].abs() > 1.0]
    if len(large_returns) > 0:
        print(f"\n⚠️  WARNING: {len(large_returns)} days have returns > 100%")
        print("Days with large returns:")
        print(large_returns[['Date', 'Market_Value_Beginning', 'Market_Value_End', 
                            'net_cash_flow', 'Subperiod_Return']])
    
    # Check 4: Summary statistics
    print(f"\n📊 Return Statistics:")
    print(f"Mean daily return: {output['Subperiod_Return'].mean():.4%}")
    print(f"Median daily return: {output['Subperiod_Return'].median():.4%}")
    print(f"Max daily return: {output['Subperiod_Return'].max():.4%}")
    print(f"Min daily return: {output['Subperiod_Return'].min():.4%}")
    
    # Check 5: Cash flow summary
    print(f"\n💰 Cash Flow Summary:")
    total_invested = daily_cash_flow[daily_cash_flow['net_cash_flow'] > 0]['net_cash_flow'].sum()
    total_withdrawn = daily_cash_flow[daily_cash_flow['net_cash_flow'] < 0]['net_cash_flow'].sum()
    print(f"Total invested (buys): ${total_invested:,.2f}")
    print(f"Total withdrawn (sells): ${-total_withdrawn:,.2f}")
    print(f"Net cash flow: ${daily_cash_flow['net_cash_flow'].sum():,.2f}")
    
    # Save detailed output
    output_file = "debug_output.csv"
    output.to_csv(output_file, index=False)
    print(f"\n💾 Full output saved to: {output_file}")
    print("You can open this in Excel to examine the data in detail.")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print("\nWhat do the results look like?")
    print("- If returns seem too high/low, check the diagnostics above")
    print("- If there are NaN or infinite values, there may be data issues")
    print("- Check debug_output.csv for detailed day-by-day analysis")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ ERROR OCCURRED")
    print("=" * 80)
    print(f"\nError message: {str(e)}")
    print(f"\nError type: {type(e).__name__}")
    
    import traceback
    print("\nFull error trace:")
    traceback.print_exc()
    
    print("\nThis error occurred during the calculation.")
    print("Check the error message above to see which step failed.")

print("\n")
