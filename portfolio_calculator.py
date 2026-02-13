"""
Portfolio Performance Calculator
Core calculation functions for portfolio performance analysis
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
import yfinance as yf


def clean_transactions_data(file_path):
    """
    Clean and standardize transactions data from Commsec
    parse and extract details from a string
    
    Parameters:
    -----------
    file_path: csv file name or path
    
    Returns:
    --------
    pandas.DataFrame
        filtered_transactions with only buy and sell transactions
    """
    transactions = pd.read_csv(file_path)
    transactions["Date"] = pd.to_datetime(transactions["Date"], format="%d/%m/%Y")
    
    # Keep only buy and sell transactions
    filtered_transactions = transactions[
        transactions["Details"].str.match(r"^[BS]", na=False)
    ]
    
    # Extract transaction details
    filtered_transactions[["buy_sell", "quantity", "ticker", "price"]] = (
        filtered_transactions["Details"]
        .str.extract(r"^([BS])\s+(\d+)\s+([A-Z0-9]+)\s+@\s+(\d+\.?\d*)")
    )
    
    filtered_transactions = filtered_transactions.sort_values("Date", ascending=True)
    filtered_transactions["quantity"] = filtered_transactions["quantity"].astype(int)
    filtered_transactions["price"] = filtered_transactions["price"].astype(float)
    
    # Calculate cash flow
    filtered_transactions['cash_flow'] = filtered_transactions.apply(
        lambda row: row['price'] * row['quantity'] if row['buy_sell'] == 'B' 
        else -row['price'] * row['quantity'], 
        axis=1
    )
    
    return filtered_transactions


def daily_net_cash_flow(filtered_transactions):
    """
    Convert transactions data to net cash flows data per day
    
    Parameters:
    -----------
    filtered_transactions: transactions data
    
    Returns:
    --------
    pandas.DataFrame
        daily_cash_flow with one record per date
    """
    # Group by date and sum the cash flows
    daily_cash_flow = filtered_transactions.groupby('Date')['cash_flow'].sum().reset_index()
    daily_cash_flow.columns = ['Date', 'net_cash_flow']
    return daily_cash_flow


def accumulated_holdings_data(filtered_transactions):
    """
    Convert transactions data to holdings data
    
    Parameters:
    -----------
    filtered_transactions: clean transactions data
    
    Returns:
    --------
    pandas.DataFrame
        accumulated holdings over time
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = filtered_transactions.copy()
    
    # Create signed quantity (positive for buy, negative for sell)
    df["signed_qty"] = df["quantity"].where(
        df["buy_sell"] == "B",
        -df["quantity"]
    )
    
    # Create pivot table
    txn_matrix = df.pivot_table(
        index="Date",
        columns="ticker",
        values="signed_qty",
        aggfunc="sum",
        fill_value=0
    )
    
    # Sort by date
    txn_matrix = txn_matrix.sort_values("Date", ascending=True)
    
    # Accumulate holdings
    accumulated_holding = (
        txn_matrix
        .sort_index()   # ensure chronological order
        .cumsum()       # accumulate holdings
    )
    
    # Reset index and clean up
    accumulated_holding = accumulated_holding.reset_index()
    accumulated_holding.columns.name = None
    
    return accumulated_holding


def fill_portfolio_dates(df, end_date):
    """
    Fill in missing dates in portfolio data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Date' column and holding or price columns
    end_date : str or datetime
        End date to fill up to
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all dates filled
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Convert Date column to datetime if not already
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True)
    
    # Convert end_date to datetime if it's a string
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date, dayfirst=True)
    
    # Set Date as index
    df_copy = df_copy.set_index('Date')
    
    # Create complete date range
    start_date = df_copy.index.min()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex with the complete date range
    df_filled = df_copy.reindex(date_range)
    
    # Forward fill missing values (modern pandas syntax)
    df_filled = df_filled.ffill()
    
    # Reset index to make Date a column again
    df_filled = df_filled.reset_index()
    df_filled = df_filled.rename(columns={'index': 'Date'})
    
    # To add .AX suffix to holdings data so the column names match exactly between holdings and market prices
    df_filled = df_filled.rename(
    columns=lambda x: f'{x}.AX' if (x != 'Date' and not x.endswith('.AX')) else x)
    return df_filled
    


def price_data(filled_holdings, end_date):
    """
    Download historical price data for all tickers
    
    Parameters:
    -----------
    filled_holdings : pandas.DataFrame
        DataFrame with holdings data
    end_date : str or datetime
        End date for price data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with close prices for all tickers
    """
    # Get the list of tickers
    tickers = [f"{col}" for col in filled_holdings.columns if col != 'Date']
    
    # Get the date range
    start_date = filled_holdings['Date'].min()
    end_date = filled_holdings['Date'].max()
    
    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract just the 'Close' prices
    if len(tickers) > 1:
        close_prices = data['Close']
    else:
        close_prices = data[['Close']].rename(columns={'Close': tickers[0]})
    
    # Reset index
    close_prices = close_prices.reset_index()
    
    # Rename the index column to 'Date' if needed
    if close_prices.columns[0] != 'Date':
        close_prices = close_prices.rename(columns={close_prices.columns[0]: 'Date'})
    
    close_prices.columns.name = None
    
    filled_close_prices = fill_portfolio_dates(close_prices, end_date)
    return filled_close_prices


def calculate_portfolio_value(daily_cash_flow, filled_holdings, filled_close_prices):
    """
    Calculate portfolio value and returns
    
    Parameters:
    -----------
    daily_cash_flow : pandas.DataFrame
        Daily net cash flows
    filled_holdings : pandas.DataFrame
        Holdings data with filled dates
    filled_close_prices : pandas.DataFrame
        Price data with filled dates
    
    Returns:
    --------
    pandas.DataFrame
        Complete portfolio analysis with returns
    """
    # Ensure Date columns are aligned
    filled_holdings = filled_holdings.set_index('Date')
    filled_close_prices = filled_close_prices.set_index('Date')
    
    # Get common columns
    common_cols = list(set(filled_holdings.columns) & set(filled_close_prices.columns))
    
    # Multiply holdings by prices for all common columns
    values = filled_holdings[common_cols] * filled_close_prices[common_cols]
    
    # Sum across all columns to get total value
    total_value = values.sum(axis=1)
    
    # Build result with organized columns
    result = pd.DataFrame(index=filled_holdings.index)
    
    # Add holdings columns
    for col in common_cols:
        result[f'{col}_Holdings'] = filled_holdings[col]
    
    # Add price columns
    for col in common_cols:
        result[f'{col}_Price'] = filled_close_prices[col]
    
    # Add Market_Value_End
    result['Market_Value_End'] = total_value
    
    result = result.reset_index()
    
    merged_dataset = pd.merge(
        daily_cash_flow, 
        result, 
        on='Date', 
        how='outer',
        suffixes=('_Cash_Flow', '_Positions')
    ).sort_values('Date').reset_index(drop=True)
    
    merged_dataset['Market_Value_Beginning'] = merged_dataset['Market_Value_End'].shift(1).fillna(0)
    merged_dataset['net_cash_flow'] = merged_dataset['net_cash_flow'].fillna(0)
    
    # Calculate subperiod return
    merged_dataset['Subperiod_Return'] = (
        merged_dataset['Market_Value_End'] - 
        merged_dataset['Market_Value_Beginning'] - 
        merged_dataset['net_cash_flow']
    ) / (merged_dataset['Market_Value_Beginning'] + merged_dataset['net_cash_flow'])
    
    # Add cumulative return column
    merged_dataset['cumulative_return'] = (1 + merged_dataset['Subperiod_Return']).cumprod() - 1
    
    return merged_dataset


def calculate_performance_metrics(output_df):
    """
    Calculate TWRR metrics from the output dataframe
    
    Parameters:
    -----------
    output_df : pandas.DataFrame
        Output from calculate_portfolio_value
    
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Get the last date
    last_date = output_df['Date'].max()
    first_date = output_df['Date'].min()
    
    # Total return
    twrr_total = output_df[output_df['Date'] == last_date]['cumulative_return'].values[0]
    
    # Annualized return
    days = (last_date - first_date).days
    twrr_annualized = (1 + twrr_total) ** (365 / days) - 1
    
    # Get final market value
    final_market_value = output_df[output_df['Date'] == last_date]['Market_Value_End'].values[0]
    
    return {
        'twrr_total': twrr_total,
        'twrr_annualized': twrr_annualized,
        'final_market_value': final_market_value,
        'start_date': first_date,
        'end_date': last_date,
        'days': days
    }


def run_full_analysis(file_path, end_date=None):
    """
    Run the complete portfolio analysis pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to the transactions CSV file
    end_date : str or datetime, optional
        End date for analysis (default: today)
    
    Returns:
    --------
    tuple
        (output_dataframe, metrics_dict)
    """
    if end_date is None:
        end_date = datetime.today()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date, dayfirst=True)
    
    # Run analysis pipeline
    filtered_transactions = clean_transactions_data(file_path)
    daily_cash_flow = daily_net_cash_flow(filtered_transactions)
    accumulated_holding = accumulated_holdings_data(filtered_transactions)
    filled_holdings = fill_portfolio_dates(accumulated_holding, end_date)
    filled_close_prices = price_data(filled_holdings, end_date)
    output = calculate_portfolio_value(daily_cash_flow, filled_holdings, filled_close_prices)
    metrics = calculate_performance_metrics(output)
    
    return output, metrics
