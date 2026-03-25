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

    #Fix the first row as we assume the benchmark is purchased at market close price, so the return on day 1 is 0, without this step, might have -inf as return
    bmark_filled.loc[bmark_filled['benchmark_market_value_beginning'] == 0, 'benchmark_returns'] = 0
    
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

# =============================================================================
# NEW FUNCTION — add anywhere before run_full_analysis
# =============================================================================
def calculate_cost_basis(output_df, annual_rfr):
    """
    Append two cost-basis columns to the portfolio output dataframe.
 
    1. cumulative_cost  — simple running sum of net cash flows.
                          Positive flow = money deployed (buy),
                          negative flow = proceeds returned (sell).
                          Represents net capital at risk at any point in time.
 
    2. tv_adjusted_cost — each cash flow compounded forward at the risk-free rate
                          from the day it occurred to each valuation date.
                          Formula mirrors Excel: cash_flow * (1 + annual_rfr/365)^days
                          where days = valuation_date - cash_flow_date.
                          Represents the opportunity cost: what the deployed capital
                          would have grown to in a risk-free instrument.
                          Portfolio must stay above this line to have beaten the
                          risk-free rate.
 
    Parameters
    ----------
    output_df : pd.DataFrame
        Output from calculate_portfolio_value.
        Must contain 'Date' and 'net_cash_flow' columns.
    annual_rfr : float
        Annual risk-free rate as a decimal (e.g. 0.0435 for 4.35 %).
 
    Returns
    -------
    pd.DataFrame
        Same dataframe with two extra columns appended:
        'cumulative_cost' and 'tv_adjusted_cost'.
    """
    df = output_df.copy().sort_values('Date').reset_index(drop=True)
 
    cash_flows = df['net_cash_flow'].to_numpy()
    dates = df['Date'].to_numpy()
    n = len(df)
 
    # --- 1. Simple cumulative cost ---
    df['cumulative_cost'] = np.cumsum(cash_flows)
 
    # --- 2. Time-value adjusted cost (vectorised) ---
    # For each valuation date i, compound every prior cash flow j forward using:
    #   cash_flow[j] * (1 + annual_rfr)^(days/365)
    # where days = date[i] - date[j].
    # This uses true compound interest: the growth factor scales continuously
    # with the fraction of a year elapsed, rather than simple r/365 scaling.
    ts = pd.to_datetime(dates)
    day_matrix = np.array(
        [[(ts[i] - ts[j]).days if i >= j else 0 for j in range(n)] for i in range(n)],
        dtype=float
    )
    # (1 + r)^(days/365) — naturally gives 1.0 on the diagonal where days == 0
    compound_matrix = np.tril((1 + annual_rfr) ** (day_matrix / 365))
 
    df['tv_adjusted_cost'] = compound_matrix @ cash_flows
 
    return df

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

def calculate_drawdown(output_df):
    """
    Calculate drawdown metrics from the portfolio cumulative return series.
    Two outputs are returned:
    1. episodes_df — every distinct drawdown episode with:
       - Peak Date, Trough Date, Recovery Date (None if still underwater)
       - Max Drawdown (simple subtraction in return space: peak_return - trough_return)
       - Drawdown Length (days, peak to trough)
       - Recovery Duration (days, trough to recovery; None if unrecovered)
       - Total Duration (days, peak to recovery; None if unrecovered)
    2. global_dd — a single dict for the global peak-to-trough drawdown:
       - Same fields as above, derived from the single highest cumulative return
         and the minimum cumulative return strictly after that peak date.
    Parameters
    ----------
    output_df : pd.DataFrame
        Output from calculate_portfolio_value / run_full_analysis.
        Must contain 'Date' and 'cumulative_return' columns.
    Returns
    -------
    episodes_df : pd.DataFrame
        All drawdown episodes (rows), sorted chronologically.
    global_dd : dict
        Single global peak-to-trough summary.
    """
    df = output_df[['Date', 'cumulative_return']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Drop any rows where cumulative_return is NaN (e.g. non-trading days or
    # data gaps). NaN CRI values cause idxmax/idxmin on slices to return NaT
    # instead of a valid integer index, which breaks the '<' comparison in the
    # episode loop with a TypeError.
    df = df.dropna(subset=['cumulative_return']).reset_index(drop=True)
    # Work in index form internally so running-peak logic is clean,
    # but report drawdown as simple return-space subtraction (your convention).
    df['CRI'] = 1 + df['cumulative_return']
    # -------------------------------------------------------------------------
    # 1. Global peak-to-trough
    # -------------------------------------------------------------------------
    peak_idx  = df['CRI'].idxmax()
    peak_date = df.loc[peak_idx, 'Date']
    peak_ret  = df.loc[peak_idx, 'cumulative_return']
    post_peak   = df[df['Date'] > peak_date]
    trough_idx  = post_peak['CRI'].idxmin()
    trough_date = df.loc[trough_idx, 'Date']
    trough_ret  = df.loc[trough_idx, 'cumulative_return']
    mdd_global     = peak_ret - trough_ret          # simple subtraction
    dd_len_global  = (trough_date - peak_date).days
    post_trough   = df[df['Date'] > trough_date]
    recovered_g   = post_trough[post_trough['CRI'] >= df.loc[peak_idx, 'CRI']]
    if not recovered_g.empty:
        rec_date_g       = recovered_g.iloc[0]['Date']
        rec_dur_global   = (rec_date_g - trough_date).days
        total_dur_global = (rec_date_g - peak_date).days
    else:
        rec_date_g       = None
        rec_dur_global   = None
        total_dur_global = None
    global_dd = {
        'Peak Date':               peak_date.date(),
        'Peak Cumulative Return':  round(peak_ret, 6),
        'Trough Date':             trough_date.date(),
        'Trough Cumulative Return':round(trough_ret, 6),
        'Max Drawdown':            round(mdd_global, 6),
        'Drawdown Length (days)':  dd_len_global,
        'Recovery Date':           rec_date_g.date() if rec_date_g else None,
        'Recovery Duration (days)':rec_dur_global,
        'Total Duration (days)':   total_dur_global,
    }
    # -------------------------------------------------------------------------
    # 2. All distinct drawdown episodes
    # -------------------------------------------------------------------------
    episodes = []
    in_dd = False
    ep_peak_idx = ep_trough_idx = None
    ep_peak_val = ep_trough_val = None
    for i, row in df.iterrows():
        cri = row['CRI']
        running_peak = df['CRI'][:i+1].max()
        if not in_dd and cri < running_peak:
            in_dd = True
            ep_peak_idx = df['CRI'][:i+1].idxmax()
            ep_peak_val = df.loc[ep_peak_idx, 'CRI']
            ep_trough_idx = i
            ep_trough_val = cri
        elif in_dd:
            if cri < ep_trough_val:
                ep_trough_idx = i
                ep_trough_val = cri
            if cri >= ep_peak_val:          # fully recovered to a new peak
                ep_peak_date   = df.loc[ep_peak_idx,   'Date']
                ep_trough_date = df.loc[ep_trough_idx, 'Date']
                ep_rec_date    = row['Date']
                ep_peak_ret    = df.loc[ep_peak_idx,   'cumulative_return']
                ep_trough_ret  = df.loc[ep_trough_idx, 'cumulative_return']
                episodes.append({
                    'Peak Date':               ep_peak_date.date(),
                    'Trough Date':             ep_trough_date.date(),
                    'Recovery Date':           ep_rec_date.date(),
                    'Max Drawdown':            round(ep_peak_ret - ep_trough_ret, 6),
                    'Drawdown Length (days)':  (ep_trough_date - ep_peak_date).days,
                    'Recovery Duration (days)':(ep_rec_date - ep_trough_date).days,
                    'Total Duration (days)':   (ep_rec_date - ep_peak_date).days,
                })
                in_dd = False
    # Handle episode that has not yet recovered
    if in_dd:
        ep_peak_date   = df.loc[ep_peak_idx,   'Date']
        ep_trough_date = df.loc[ep_trough_idx, 'Date']
        ep_peak_ret    = df.loc[ep_peak_idx,   'cumulative_return']
        ep_trough_ret  = df.loc[ep_trough_idx, 'cumulative_return']
        episodes.append({
            'Peak Date':               ep_peak_date.date(),
            'Trough Date':             ep_trough_date.date(),
            'Recovery Date':           None,
            'Max Drawdown':            round(ep_peak_ret - ep_trough_ret, 6),
            'Drawdown Length (days)':  (ep_trough_date - ep_peak_date).days,
            'Recovery Duration (days)':None,
            'Total Duration (days)':   None,
        })
    episodes_df = pd.DataFrame(episodes)
    return episodes_df, global_dd

def fit_t_robust(returns):
    """
    Fit Student-t distribution via MLE with constrained degrees of freedom.
    Constraining dof to [2.01, 30] prevents degenerate solutions when
    kurtosis is extreme:
      - dof <= 2 implies infinite variance (CVaR formula breaks down)
      - dof > 30 is statistically indistinguishable from normal
    Uses method-of-moments estimate as initial guess for dof.
    """
    from scipy.optimize import minimize
    from scipy import stats
 
    def neg_loglik(params):
        dof, loc, scale = params
        if scale <= 0 or dof <= 2:
            return np.inf
        return -stats.t.logpdf(returns, dof, loc, scale).sum()
 
    kurt     = returns.kurt()
    dof_init = max(2.5, min(30, 4 + 6 / kurt)) if kurt > 0 else 5.0
 
    result = minimize(
        neg_loglik,
        x0=[dof_init, returns.mean(), returns.std()],
        bounds=[(2.01, 30), (None, None), (1e-6, None)],
        method='L-BFGS-B'
    )
    return result.x  # dof, loc, scale
 
 
def calculate_var_cvar(returns, confidence_levels=[0.95, 0.99], horizons=[1, 10]):
    """
    Calculate VaR and CVaR using two industry-standard methods:
      1. Historical Simulation  — no distributional assumption, uses empirical data directly
      2. Parametric (Student-t) — fits a t-distribution via MLE with constrained dof
 
    Results scaled to each requested horizon via square root of time rule (Basel III).
    VaR and CVaR expressed as percentages (e.g. -1.23 means a loss of 1.23%).
 
    Parameters
    ----------
    returns           : pd.Series of daily portfolio returns (e.g. Subperiod_Return)
    confidence_levels : list of floats, e.g. [0.95, 0.99]
    horizons          : list of ints, horizon in days, e.g. [1, 10, 30]
 
    Returns
    -------
    results     : list of dicts — one row per (confidence level, method, horizon)
    diagnostics : dict — fitted t-distribution parameters and return series statistics
    """
    from scipy import stats
 
    returns = returns.dropna()
 
    diagnostics = {
        'n_observations':  len(returns),
        'mean':            returns.mean(),
        'std':             returns.std(),
        'skewness':        returns.skew(),
        'excess_kurtosis': returns.kurt(),
        'min':             returns.min(),
        'max':             returns.max(),
    }
 
    dof, loc, scale = fit_t_robust(returns)
    diagnostics['t_dof']   = dof
    diagnostics['t_loc']   = loc
    diagnostics['t_scale'] = scale
 
    results = []
 
    for cl in confidence_levels:
        alpha = 1 - cl
 
        # Historical Simulation
        var_hist_1d  = np.percentile(returns, alpha * 100)
        cvar_hist_1d = returns[returns <= var_hist_1d].mean()
 
        # Parametric Student-t
        z         = stats.t.ppf(alpha, dof)
        var_t_1d  = stats.t.ppf(alpha, dof, loc, scale)
        pdf_z     = stats.t.pdf(z, dof)
        cvar_t_1d = loc - scale * (pdf_z / alpha) * ((dof + z**2) / (dof - 1))
 
        for h in horizons:
            scale_factor = np.sqrt(h)
 
            results.append({
                'Confidence Level':     cl,
                'Confidence Level Pct': f'{cl:.0%}',
                'Method':               'Historical Simulation',
                'Horizon (days)':       h,
                'Horizon Label':        f'{h}-day',
                'VaR (%)':              var_hist_1d  * scale_factor * 100,
                'CVaR (%)':             cvar_hist_1d * scale_factor * 100,
            })
            results.append({
                'Confidence Level':     cl,
                'Confidence Level Pct': f'{cl:.0%}',
                'Method':               'Parametric (Student-t)',
                'Horizon (days)':       h,
                'Horizon Label':        f'{h}-day',
                'VaR (%)':              var_t_1d  * scale_factor * 100,
                'CVaR (%)':             cvar_t_1d * scale_factor * 100,
            })
 
    return results, diagnostics

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
 
    output = calculate_cost_basis(output, risk_free_rate)
    # ── drawdown analysis ───────────────────────────────────────────────────
    episodes_df, global_dd = calculate_drawdown(output)
    # ── VaR / CVaR — defaults: 95%/99% confidence, 1-day and 10-day horizons
    var_results, var_diagnostics = calculate_var_cvar(
        output['Subperiod_Return'],
        confidence_levels=[0.95, 0.99],
        horizons=[1, 10]
    )
    return output, metrics, bmark_filled, regression, ratios, episodes_df, global_dd, var_results, var_diagnostics