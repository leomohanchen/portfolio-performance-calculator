# Quick Start Guide

## Option 1: Web Application (Recommended)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
streamlit run app.py
```

### Step 3: Use the Application
1. The app will open in your browser at `http://localhost:8501`
2. Upload your CSV file using the sidebar
3. Select an end date (or use today's date)
4. Click "Calculate Performance"
5. View your results and download the detailed CSV

---

## Option 2: Command Line Script

### Edit run_analysis.py
Update the configuration in `run_analysis.py`:
```python
csv_file = "your_transactions.csv"  # Your CSV file path
end_date = "31/12/2025"  # Your desired end date
```

### Run the Script
```bash
python run_analysis.py
```

---

## Option 3: Use as a Python Module

```python
from portfolio_calculator import run_full_analysis

# Run analysis
output_df, metrics = run_full_analysis(
    file_path='transactions.csv',
    end_date='31/12/2025'
)

# Print results
print(f"Total Return: {metrics['twrr_total']:.2%}")
print(f"Annualized Return: {metrics['twrr_annualized']:.2%}")
```

---

## Testing with Sample Data

A sample CSV file is included (`sample_transactions.csv`). Test the app with:

```bash
streamlit run app.py
# Then upload sample_transactions.csv
```

---

## Troubleshooting

**Can't install packages?**
```bash
pip install --upgrade pip
pip install -r requirements.txt --user
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Module import errors?**
Make sure you're in the correct directory:
```bash
cd /path/to/portfolio-performance-calculator
python -c "import portfolio_calculator"  # Should run without errors
```

---

## What's Next?

- Read the full README.md for detailed documentation
- Customize the app in `app.py` for your needs
- Explore the calculation functions in `portfolio_calculator.py`
- Deploy to Streamlit Cloud for online access
