"""
Portfolio Performance Calculator
Core calculation functions for portfolio performance analysis
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
import yfinance as yf
import statsmodels.api as sm

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


def fill_portfolio_dates(df, end_date, add_ax_suffix=True):
    """
    Fill in missing dates in portfolio data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Date' column and holding or price columns
    end_date : str or datetime
        End date to fill up to
    add_ax_suffix: Boolean
        to add .AX to the ticker if it is ASX listed, by default it is true
    
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
    if add_ax_suffix:
        df_filled = df_filled.rename(
            columns=lambda x: f'{x}.AX' if (x != 'Date' and not x.endswith('.AX')) else x
        )

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

def benchmark_data(filled_close_prices, filtered_transactions, benchmark_ticker='VGS.AX'):
    """
    Grab the close market prices for selected benchmark and simulate the same
    transaction timing as the actual portfolio to make performance comparison fair.
    
    Instead of comparing raw returns, we simulate investing the same cash flows
    (buys and sells) into the benchmark on the same dates, then calculate the
    benchmark's TWRR using the same methodology as the actual portfolio.
    
    Parameters:
    -----------
    filled_close_prices : pandas.DataFrame
        DataFrame with all dates filled from first transaction to end_date
    filtered_transactions : pandas.DataFrame
        Output from clean_transactions_data, contains Date and cash_flow columns
    benchmark_ticker : str
        Ticker symbol for benchmark. Default is MSCI World index (VGS.AX)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns [Date, benchmark_price, benchmark_units,
        benchmark_market_value, benchmark_returns] with all dates filled
        from first transaction to end_date
    """
    target_start = filled_close_prices['Date'].min()
    target_end = filled_close_prices['Date'].max()
    
    # Add 5 padding days in case the target_start and target_end falls on a non-trading day
    download_start = target_start - pd.Timedelta(days=5)
    download_end = target_end + pd.Timedelta(days=5)

    # Download benchmark price data
    ticker_obj = yf.Ticker(benchmark_ticker)
    data = ticker_obj.history(start=download_start, end=download_end)
    data = data.reset_index()[['Date', 'Close']]
    data = data.rename(columns={'Close': benchmark_ticker})
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.columns.name = None

    # -------------------------------------------------------------------------
    # Simulate transactions: on each transaction date, instead of buying the
    # actual stocks, we buy/sell benchmark units with the same cash flow amount
    # -------------------------------------------------------------------------

    # Get daily net cash flows from actual transactions (same as your portfolio)
    daily_cf = daily_net_cash_flow(filtered_transactions)
    daily_cf['Date'] = pd.to_datetime(daily_cf['Date'])

    # Merge cash flows with benchmark prices on transaction dates
    # Use merge_asof to match each transaction date to the nearest available
    # trading day price in case transaction falls on a non-trading day
    daily_cf = daily_cf.sort_values('Date')
    data_sorted = data.sort_values('Date')

    cf_with_prices = pd.merge_asof(
        daily_cf,
        data_sorted[['Date', benchmark_ticker]],
        on='Date',
        direction='nearest'  # match to nearest trading day price
    )

    # Calculate units bought/sold on each transaction date
    # Positive cash_flow = buy (units in), negative cash_flow = sell (units out)
    cf_with_prices['units_transacted'] = cf_with_prices['net_cash_flow'] / cf_with_prices[benchmark_ticker]

    # Accumulate benchmark units held over time
    cf_with_prices['benchmark_units'] = cf_with_prices['units_transacted'].cumsum()

    # -------------------------------------------------------------------------
    # Build a full daily series by forward-filling units and merging with prices
    # -------------------------------------------------------------------------

    # Forward-fill benchmark prices across all dates
    bmark_filled = fill_portfolio_dates(data, target_end, add_ax_suffix=False)
    bmark_filled = bmark_filled[
        (bmark_filled['Date'] >= target_start) &
        (bmark_filled['Date'] <= target_end)
    ].reset_index(drop=True)

    # Forward-fill units held across all dates
    units_df = cf_with_prices[['Date', 'benchmark_units']].copy()
    units_filled = fill_portfolio_dates(units_df, target_end, add_ax_suffix=False)
    units_filled = units_filled[
        (units_filled['Date'] >= target_start) &
        (units_filled['Date'] <= target_end)
    ].reset_index(drop=True)

    # Merge prices and units
    bmark_filled = pd.merge(bmark_filled, units_filled, on='Date', how='left')
    bmark_filled['benchmark_units'] = bmark_filled['benchmark_units'].ffill().fillna(0)

    # Calculate benchmark market value each day
    bmark_filled['benchmark_market_value'] = (
        bmark_filled['benchmark_units'] * bmark_filled[benchmark_ticker]
    )

    # -------------------------------------------------------------------------
    # Calculate benchmark TWRR using same methodology as actual portfolio
    # -------------------------------------------------------------------------
    bmark_filled = pd.merge(
        bmark_filled,
        daily_cf[['Date', 'net_cash_flow']],
        on='Date',
        how='left'
    )
    bmark_filled['net_cash_flow'] = bmark_filled['net_cash_flow'].fillna(0)

    bmark_filled['benchmark_market_value_beginning'] = (
        bmark_filled['benchmark_market_value'].shift(1).fillna(0)
    )

    bmark_filled['benchmark_returns'] = (
        (bmark_filled['benchmark_market_value'] -
         bmark_filled['benchmark_market_value_beginning'] -
         bmark_filled['net_cash_flow']) /
        bmark_filled['benchmark_market_value_beginning']
    )

    # Add cumulative return column
    bmark_filled['cumulative_return'] = (1 + bmark_filled['benchmark_returns']).cumprod() - 1
    
    return bmark_filled

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

def calculate_alpha_beta(output, bmark_filled):
    """
    Calculate Jensen's Alpha and Beta of the portfolio relative to the benchmark
    using OLS regression of portfolio returns on benchmark returns
    
    Parameters:
    -----------
    output : pandas.DataFrame
        Output from calculate_portfolio_value, contains 'Date' and 'Subperiod_Return'
    bmark_filled : pandas.DataFrame
        Output from benchmark_data, contains 'Date' and 'returns'
    
    Returns:
    --------
    dict
        Dictionary containing alpha, beta, r-squared, and the full regression summary
    """
    # Merge portfolio and benchmark returns on Date
    merged = pd.merge(
        output[['Date', 'Subperiod_Return']],
        bmark_filled[['Date', 'benchmark_returns']],
        on='Date',
        how='inner'
    ).dropna()

    y = merged['Subperiod_Return']           # dependent variable: portfolio returns
    X = merged['benchmark_returns']                     # independent variable: benchmark returns
    X = sm.add_constant(X)                   # adds the intercept term

    # Fit OLS regression
    model = sm.OLS(y, X).fit()

    return {
        'alpha': model.params['const'],       # daily alpha
        'alpha_annualized': (1 + model.params['const']) ** 252 - 1,  # annualized
        'beta': model.params['benchmark_returns'],
        'r_squared': model.rsquared,
        'p_value_alpha': model.pvalues['const'],
        'p_value_beta': model.pvalues['benchmark_returns'],
        'summary': model.summary()
    }

def calculate_ratios(output, bmark_filled, alpha, beta, risk_free_rate=0.0529):
    """
    Calculate portfolio performance ratios
    
    Parameters:
    -----------
    output : pandas.DataFrame
        Output from calculate_portfolio_value, contains 'Date' and 'Subperiod_Return'
    bmark_filled : pandas.DataFrame
        Output from benchmark_data, contains 'Date' and 'returns'
    alpha : float
        Annualized alpha from calculate_alpha_beta
    beta : float
        Beta from calculate_alpha_beta
    risk_free_rate : float
        Annualized risk-free rate (default: 0.043 = 4.3%, approximate RBA cash rate)
    
    Returns:
    --------
    dict
        Dictionary containing all performance ratios
    """
    # Merge portfolio and benchmark returns on Date
    merged = pd.merge(
        output[['Date', 'Subperiod_Return']],
        bmark_filled[['Date', 'benchmark_returns']],
        on='Date',
        how='inner'
    ).dropna()

    # Convert annualized risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/365) - 1

    portfolio_returns = merged['Subperiod_Return']
    benchmark_returns = merged['benchmark_returns']
    excess_returns = portfolio_returns - daily_rf
    active_returns = portfolio_returns - benchmark_returns  # used for IR and Appraisal

    # Annualize returns and volatilities (252 trading days)
    ann_portfolio_return = (1 + portfolio_returns.mean()) ** 252 - 1
    ann_benchmark_return = (1 + benchmark_returns.mean()) ** 252 - 1
    ann_portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    ann_active_vol = active_returns.std() * np.sqrt(252)         # tracking error

    # Downside deviation (only negative excess returns, for Sortino)
    # Downside deviation using full sample size in denominator
    # Only negative excess returns contribute to the sum of squares,
    # but we divide by the full sample size, not just the count of negative days
    downside_squared = excess_returns.apply(lambda x: x**2 if x < 0 else 0)
    ann_downside_vol = np.sqrt(downside_squared.sum() / len(excess_returns)) * np.sqrt(252)

    # -------------------------------------------------------------------------
    # 1. Sharpe Ratio
    # Excess return per unit of total risk
    # -------------------------------------------------------------------------
    sharpe = (ann_portfolio_return - risk_free_rate) / ann_portfolio_vol

    # -------------------------------------------------------------------------
    # 2. Treynor Ratio
    # Excess return per unit of systematic risk (beta)
    # -------------------------------------------------------------------------
    treynor = (ann_portfolio_return - risk_free_rate) / beta

    # -------------------------------------------------------------------------
    # 3. Information Ratio
    # Active return per unit of active risk (tracking error)
    # -------------------------------------------------------------------------
    information_ratio = (ann_portfolio_return - ann_benchmark_return) / ann_active_vol

    # -------------------------------------------------------------------------
    # 4. Appraisal Ratio
    # Alpha per unit of residual/unsystematic risk
    # -------------------------------------------------------------------------
    residual_returns = portfolio_returns - (daily_rf + beta * (benchmark_returns - daily_rf))
    ann_residual_vol = residual_returns.std() * np.sqrt(252)
    appraisal_ratio = alpha / ann_residual_vol

    # -------------------------------------------------------------------------
    # 5. Sortino Ratio
    # Excess return per unit of downside risk only
    # -------------------------------------------------------------------------
    sortino = (ann_portfolio_return - risk_free_rate) / ann_downside_vol

    # -------------------------------------------------------------------------
    # 6. Capture Ratios
    # Up capture: how much of benchmark's up days does portfolio capture?
    # Down capture: how much of benchmark's down days does portfolio capture?
    # -------------------------------------------------------------------------
    up_days = merged[benchmark_returns > 0]
    down_days = merged[benchmark_returns < 0]

    up_capture = (
        ((1 + up_days['Subperiod_Return']).prod() ** (252/len(up_days)) - 1) /
        ((1 + up_days['benchmark_returns']).prod() ** (252/len(up_days)) - 1)
    )
    down_capture = (
        ((1 + down_days['Subperiod_Return']).prod() ** (252/len(down_days)) - 1) /
        ((1 + down_days['benchmark_returns']).prod() ** (252/len(down_days)) - 1)
    )
    # Capture ratio > 1 is ideal: capturing more upside than downside
    capture_ratio = up_capture / down_capture

    return {
        'sharpe_ratio':       sharpe,
        'treynor_ratio':      treynor,
        'information_ratio':  information_ratio,
        'appraisal_ratio':    appraisal_ratio,
        'sortino_ratio':      sortino,
        'up_capture':         up_capture,
        'down_capture':       down_capture,
        'capture_ratio':      capture_ratio,
        # Intermediates useful for interpretation
        'ann_portfolio_return': ann_portfolio_return,
        'ann_benchmark_return': ann_benchmark_return,
        'ann_portfolio_vol':    ann_portfolio_vol,
        'tracking_error':       ann_active_vol,
        'ann_downside_vol':     ann_downside_vol,
        'ann_residual_vol':     ann_residual_vol,
    }

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


def run_full_analysis(file_path, end_date=None, benchmark_ticker='VGS.AX', risk_free_rate=0.0429):
    """
    Run the complete portfolio analysis pipeline including benchmark comparison,
    regression analysis and performance ratios.
    
    Parameters:
    -----------
    file_path : str
        Path to the Commsec transaction CSV file.
        File must contain columns: Date (DD/MM/YYYY), Reference, Details
        Details column must follow format: "B/S [quantity] [ticker] @ [price]"
        Example: "B 100 CBA @ 150.50"
    
    end_date : str or datetime, optional
        End date for analysis in 'YYYY-MM-DD' or 'DD/MM/YYYY' format.
        Defaults to today's date if not provided.
    
    benchmark_ticker : str, optional
        Yahoo Finance ticker symbol for the benchmark index.
        Defaults to 'VGS.AX' (Vanguard MSCI Index International Shares ETF).
        Other examples: '^AXJO' (ASX 200), 'STW.AX' (ASX 200 ETF)
    
    risk_free_rate : float, optional
        Annualised risk-free rate as a decimal.
        Defaults to 0.0429 (4.29%, approximate RBA cash rate as of early 2025).
        Used in Sharpe, Sortino, Treynor and Appraisal ratio calculations.
    
    Returns:
    --------
    output : pandas.DataFrame
        Day-by-day portfolio analysis with the following key columns:
        - Date : calendar date
        - net_cash_flow : cash invested or withdrawn on that day
        - Market_Value_Beginning : portfolio value at start of day
        - Market_Value_End : portfolio value at end of day
        - Subperiod_Return : daily TWRR return
        - cumulative_return : compounded return from inception
        - [TICKER]_Holdings : units held for each security
        - [TICKER]_Price : closing price for each security

    metrics : dict
        Overall portfolio performance metrics:
        - twrr_total : total return from inception to end_date
        - twrr_annualized : annualised equivalent of total return
        - final_market_value : portfolio market value on end_date
        - start_date : date of first transaction
        - end_date : end date of analysis
        - days : total number of calendar days in analysis period

    bmark_filled : pandas.DataFrame
        Day-by-day benchmark analysis simulating the same cash flow timing
        as the actual portfolio, with the following key columns:
        - Date : calendar date
        - [benchmark_ticker] : benchmark closing price
        - benchmark_units : simulated units held in benchmark
        - benchmark_market_value : simulated benchmark portfolio value
        - net_cash_flow : cash flow on that day (mirrors actual portfolio)
        - benchmark_market_value_beginning : benchmark value at start of day
        - benchmark_returns : daily benchmark TWRR return
        - cumulative_return : compounded benchmark return from inception

    regression : dict
        OLS regression results of portfolio returns on benchmark returns:
        - alpha : daily Jensen's alpha (intercept)
        - alpha_annualized : annualised alpha = (1 + daily_alpha)^252 - 1
        - beta : portfolio beta (systematic risk relative to benchmark)
        - r_squared : proportion of portfolio variance explained by benchmark
        - p_value_alpha : statistical significance of alpha (< 0.05 = significant)
        - p_value_beta : statistical significance of beta
        - summary : full statsmodels OLS summary object

    ratios : dict
        Risk-adjusted performance ratios:
        - sharpe_ratio : excess return per unit of total volatility
        - treynor_ratio : excess return per unit of systematic risk (beta)
        - information_ratio : active return per unit of tracking error
        - appraisal_ratio : alpha per unit of residual/unsystematic risk
        - sortino_ratio : excess return per unit of downside volatility
        - up_capture : % of benchmark upside captured by portfolio
        - down_capture : % of benchmark downside suffered by portfolio
        - capture_ratio : up_capture / down_capture (> 1 is favourable)
        - ann_portfolio_return : annualised portfolio return
        - ann_benchmark_return : annualised benchmark return
        - ann_portfolio_vol : annualised portfolio volatility
        - tracking_error : annualised volatility of active returns
        - ann_downside_vol : annualised downside deviation (full sample)
        - ann_residual_vol : annualised volatility of residual returns

    Example:
    --------
    >>> output, metrics, bmark_filled, regression, ratios = run_full_analysis(
    ...     file_path='transactions.csv',
    ...     end_date='2025-12-31',
    ...     benchmark_ticker='VGS.AX',
    ...     risk_free_rate=0.0429
    ... )
    >>> print(f"Total Return:     {metrics['twrr_total']:.2%}")
    >>> print(f"Annualised Return:{metrics['twrr_annualized']:.2%}")
    >>> print(f"Alpha:            {regression['alpha_annualized']:.2%}")
    >>> print(f"Sharpe Ratio:     {ratios['sharpe_ratio']:.4f}")
    """
    if end_date is None:
        end_date = datetime.today()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date, dayfirst=True)

    filtered_transactions = clean_transactions_data(file_path)
    daily_cash_flow = daily_net_cash_flow(filtered_transactions)
    accumulated_holding = accumulated_holdings_data(filtered_transactions)
    filled_holdings = fill_portfolio_dates(accumulated_holding, end_date)
    filled_close_prices = price_data(filled_holdings, end_date)
    output = calculate_portfolio_value(daily_cash_flow, filled_holdings, filled_close_prices)
    metrics = calculate_performance_metrics(output)
    bmark_filled = benchmark_data(filled_close_prices, filtered_transactions, benchmark_ticker)
    regression = calculate_alpha_beta(output, bmark_filled)
    ratios = calculate_ratios(
        output, bmark_filled,
        alpha=regression['alpha_annualized'],
        beta=regression['beta'],
        risk_free_rate=risk_free_rate
    )

    return output, metrics, bmark_filled, regression, ratios
