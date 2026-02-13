# Portfolio Performance Calculator

A web-based application to calculate Time-Weighted Rate of Return (TWRR) for your investment portfolio based on transaction history.

## Features

- 📊 Calculate Total and Annualized TWRR
- 📈 Interactive portfolio value and returns charts
- 💼 Current holdings breakdown with allocation
- 📅 Flexible date range selection
- 💾 Download detailed analysis as CSV
- 🎨 Clean, intuitive web interface

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this project**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Start the Streamlit app**

```bash
streamlit run app.py
```

2. **Open your browser** - The app will automatically open at `http://localhost:8501`

3. **Upload your CSV file** - Use the file uploader in the sidebar

4. **Select end date** - Choose the end date for analysis (defaults to today)

5. **Calculate** - Click "Calculate Performance" to run the analysis

### CSV File Format

Your transaction CSV file should have the following columns:

- `Date`: Transaction date in DD/MM/YYYY format
- `Reference`: Transaction reference number
- `Details`: Transaction details in format: "B/S [quantity] [ticker] @ [price]"

**Example:**
```csv
Date,Reference,Details
01/01/2024,REF001,B 100 AAPL @ 150.50
15/01/2024,REF002,S 50 AAPL @ 155.25
01/02/2024,REF003,B 200 MSFT @ 380.75
```

Where:
- `B` = Buy, `S` = Sell
- `100` = Quantity
- `AAPL` = Ticker symbol
- `150.50` = Price per share

## Project Structure

```
portfolio-performance-calculator/
│
├── app.py                      # Streamlit web application
├── portfolio_calculator.py     # Core calculation functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Core Functions

The `portfolio_calculator.py` module contains:

- `clean_transactions_data()`: Parse and clean transaction CSV
- `daily_net_cash_flow()`: Aggregate transactions by date
- `accumulated_holdings_data()`: Track holdings over time
- `fill_portfolio_dates()`: Fill missing dates with forward-fill
- `price_data()`: Download historical prices from Yahoo Finance
- `calculate_portfolio_value()`: Calculate portfolio value and returns
- `calculate_performance_metrics()`: Compute TWRR metrics
- `run_full_analysis()`: Complete end-to-end analysis pipeline

## Understanding the Metrics

### Time-Weighted Rate of Return (TWRR)

TWRR measures the compound rate of growth in your portfolio, eliminating the impact of cash flows timing. This makes it ideal for comparing your performance against benchmarks.

**Total Return**: Overall return from start to end date

**Annualized Return**: Equivalent yearly return rate

Formula:
```
Subperiod Return = (Ending Value - Beginning Value - Cash Flow) / (Beginning Value + Cash Flow)
Cumulative Return = Product of (1 + Subperiod Returns) - 1
Annualized Return = (1 + Total Return) ^ (365 / Days) - 1
```

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
pip install -r requirements.txt --upgrade
```

**2. CSV parsing errors**
- Ensure your CSV uses the correct date format (DD/MM/YYYY)
- Check that Details column follows the format: "B/S [qty] [ticker] @ [price]"

**3. Ticker symbol not found**
- Verify ticker symbols are valid Yahoo Finance symbols
- Add `.AX` suffix for Australian stocks (e.g., `CBA.AX`)

**4. Port already in use**
```bash
streamlit run app.py --server.port 8502
```

## Advanced Usage

### Using the Core Module Directly

You can import and use the calculation functions in your own scripts:

```python
from portfolio_calculator import run_full_analysis

# Run analysis
output_df, metrics = run_full_analysis(
    file_path='transactions.csv',
    end_date='2025-12-31'
)

# Access metrics
print(f"Total Return: {metrics['twrr_total']:.2%}")
print(f"Annualized Return: {metrics['twrr_annualized']:.2%}")

# Work with the detailed dataframe
print(output_df.head())
```

## Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Run on Server

```bash
# Run on all interfaces
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Run in headless mode
streamlit run app.py --server.headless true
```

## License

This project is provided as-is for portfolio analysis purposes.

## Support

For issues or questions, please review the troubleshooting section or check the inline code documentation.

---

**Note**: This calculator uses Yahoo Finance for historical price data. Ensure you have an internet connection when running the analysis.
