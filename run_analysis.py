"""
Example script showing how to use the portfolio calculator module directly
"""

from portfolio_calculator import run_full_analysis
from datetime import datetime

def main():
    # Configuration
    csv_file = "transactions.csv"  # Replace with your CSV file path
    end_date = "31/12/2025"  # Or use datetime.today()
    
    print("=" * 60)
    print("Portfolio Performance Calculator - Command Line")
    print("=" * 60)
    print(f"\nAnalyzing: {csv_file}")
    print(f"End Date: {end_date}")
    print("\nRunning analysis...\n")
    
    try:
        # Run the analysis
        output_df, metrics = run_full_analysis(csv_file, end_date)
        
        # Display results
        print("=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"\nAnalysis Period:")
        print(f"  Start Date: {metrics['start_date'].strftime('%Y-%m-%d')}")
        print(f"  End Date:   {metrics['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Duration:   {metrics['days']} days")
        
        print(f"\nReturns:")
        print(f"  Total Return:      {metrics['twrr_total']:.2%}")
        print(f"  Annualized Return: {metrics['twrr_annualized']:.2%}")
        
        print(f"\nPortfolio Value:")
        print(f"  Final Market Value: ${metrics['final_market_value']:,.2f}")
        
        # Show recent data
        print("\n" + "=" * 60)
        print("RECENT PORTFOLIO DATA (Last 5 days)")
        print("=" * 60)
        print(output_df[['Date', 'Market_Value_End', 'cumulative_return']].tail().to_string(index=False))
        
        # Save to file
        output_file = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
        output_df.to_csv(output_file, index=False)
        print(f"\n✓ Full analysis saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_file}' not found.")
        print("Please update the csv_file variable with the correct path.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
