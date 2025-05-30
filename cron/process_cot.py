#!/usr/bin/env python3
"""
COT (Commitment of Traders) Currency Index Processor
Downloads Legacy Futures-only COT reports for currencies and calculates custom index
Runs weekly on Wednesday when new COT data is released
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cot_currency_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import cot_reports as cot
    logger.info("Successfully imported cot_reports package")
except ImportError as e:
    logger.error(f"Failed to import cot_reports: {e}")
    logger.error("Please install cot_reports package: pip install cot_reports")
    sys.exit(1)

# Currency markets we're interested in
CURRENCY_MARKETS = [
    'EURO FX - CHICAGO MERCANTILE EXCHANGE',
    'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE', 
    'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE',
    'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
    'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE',
    'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
    'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE',
    'MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE'
]

def download_cot_data(year=None):
    """
    Download Legacy Futures-only COT report for specified year
    
    Args:
        year (int): Year to download. If None, downloads current year
        
    Returns:
        pd.DataFrame: COT data
    """
    if year is None:
        year = datetime.now().year
        
    try:
        logger.info(f"Downloading Legacy Futures-only COT report for {year}...")
        
        df = cot.cot_year(
            year=year, 
            cot_report_type="legacy_fut",
            store_txt=False,  # Don't keep the txt file
            verbose=False
        )
        
        logger.info(f"Downloaded COT data. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading COT data for {year}: {e}")
        return None

def process_currency_cot_data(df_current, df_previous=None):
    """
    Process COT data to extract currency metrics and calculate index
    
    Args:
        df_current (pd.DataFrame): Current year COT data
        df_previous (pd.DataFrame): Previous year COT data (for 1-year lookback)
        
    Returns:
        pd.DataFrame: Processed currency data with index
    """
    if df_current is None or df_current.empty:
        logger.error("No current year data to process")
        return None
        
    try:
        logger.info("Processing currency COT data...")
        
        # Combine current and previous year data for 1-year lookback
        if df_previous is not None and not df_previous.empty:
            df_combined = pd.concat([df_previous, df_current], ignore_index=True)
        else:
            df_combined = df_current.copy()
            
        # Convert report date to datetime - using correct column names
        if 'As of Date in Form YYYY-MM-DD' in df_combined.columns:
            df_combined['Report_Date'] = pd.to_datetime(df_combined['As of Date in Form YYYY-MM-DD'])
        elif 'As of Date in Form YYMMDD' in df_combined.columns:
            df_combined['Report_Date'] = pd.to_datetime(df_combined['As of Date in Form YYMMDD'], format='%y%m%d')
        
        # Filter for currency markets only - using correct column name
        if 'Market and Exchange Names' not in df_combined.columns:
            logger.error("Column 'Market and Exchange Names' not found in data")
            return None
            
        currency_df = df_combined[df_combined['Market and Exchange Names'].isin(CURRENCY_MARKETS)].copy()
        
        if currency_df.empty:
            logger.error("No currency data found")
            return None
            
        logger.info(f"Filtered to {len(currency_df)} currency records")
        
        # Sort by market and date
        currency_df = currency_df.sort_values(['Market and Exchange Names', 'Report_Date'])
        
        # Calculate required metrics for each currency
        results = []
        
        for market in CURRENCY_MARKETS:
            market_data = currency_df[currency_df['Market and Exchange Names'] == market].copy()
            
            if market_data.empty:
                logger.warning(f"No data found for {market}")
                continue
                
            # Extract currency name (e.g., "EURO FX" from "EURO FX - CHICAGO MERCANTILE EXCHANGE")
            currency_name = market.split(' - ')[0]
            
            # Get required columns and calculate metrics - using correct column names
            for _, row in market_data.iterrows():
                # Extract the specific metrics requested
                noncomm_longs = row.get('Noncommercial Positions-Long (All)', 0)
                noncomm_shorts = row.get('Noncommercial Positions-Short (All)', 0)
                comm_longs = row.get('Commercial Positions-Long (All)', 0)
                comm_shorts = row.get('Commercial Positions-Short (All)', 0)
                
                # Calculate differences
                diff_a = noncomm_longs - noncomm_shorts  # Non-commercial net
                diff_b = comm_longs - comm_shorts        # Commercial net
                
                result_row = {
                    'Currency': currency_name,
                    'Date': row['Report_Date'],
                    'NonComm_Longs': noncomm_longs,
                    'NonComm_Shorts': noncomm_shorts,
                    'DIFF_A_NonComm_Net': diff_a,
                    'Comm_Longs': comm_longs,
                    'Comm_Shorts': comm_shorts,
                    'DIFF_B_Comm_Net': diff_b,
                    'Diff_A_minus_B': diff_a - diff_b
                }
                
                results.append(result_row)
        
        if not results:
            logger.error("No currency data processed")
            return None
            
        # Create DataFrame
        processed_df = pd.DataFrame(results)
        processed_df = processed_df.sort_values(['Currency', 'Date'])
        
        # Calculate the index for each currency
        # Index = current week (DIFF B - DIFF A) / maximum of (DIFF B - DIFF A) in the latest 1 year
        one_year_ago = processed_df['Date'].max() - timedelta(days=365)
        
        for currency in processed_df['Currency'].unique():
            currency_data = processed_df[processed_df['Currency'] == currency].copy()
            
            # Get 1-year lookback data
            currency_data = currency_data[currency_data['Date'] >= one_year_ago]
            
            if len(currency_data) > 0:
                # Find maximum of (DIFF B - DIFF A) in the latest 1 year
                max_diff_a_minus_b = currency_data['Diff_A_minus_B'].max()
                min_diff_a_minus_b = currency_data['Diff_A_minus_B'].min()
                
                # Avoid division by zero
                if max_diff_a_minus_b != 0:
                    # Calculate index for all records of this currency
                    currency_indices = (currency_data['Diff_A_minus_B'] - min_diff_a_minus_b) / (max_diff_a_minus_b - min_diff_a_minus_b) * 100
                else:
                    currency_indices = 0
                    
                # Update the main dataframe
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index'] = currency_indices
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index_Min'] = min_diff_a_minus_b
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index_Max'] = max_diff_a_minus_b
            else:
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index'] = 0
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index_Min'] = min_diff_a_minus_b
                processed_df.loc[processed_df['Currency'] == currency, 'COT_Index_Max'] = max_diff_a_minus_b
        
        for col in ['COT_Index', 'COT_Index_Min', 'COT_Index_Max']:
            processed_df[col] = processed_df[col].astype(float)
            
        logger.info(f"Processed {len(processed_df)} records for {processed_df['Currency'].nunique()} currencies")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing currency COT data: {e}")
        return None

def save_currency_data(df, filename_prefix="cot_currency_index"):
    """
    Save processed currency COT data to CSV files
    
    Args:
        df (pd.DataFrame): Processed currency data
        filename_prefix (str): Prefix for output files
    """
    if df is None or df.empty:
        logger.error("No data to save")
        return
        
    try:
        # Create output directory if it doesn't exist
        output_dir = "cot_data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete historical data
        complete_filename = f"{output_dir}/{filename_prefix}_complete_{timestamp}.csv"
        df.to_csv(complete_filename, index=False)
        logger.info(f"Saved complete data to {complete_filename}")
        
        # Save latest week summary for each currency
        latest_data = []
        for currency in df['Currency'].unique():
            currency_data = df[df['Currency'] == currency]
            if not currency_data.empty:
                latest_record = currency_data.iloc[-1]  # Most recent record
                
                summary_row = {
                    'Currency': currency,
                    'Latest_Date': latest_record['Date'],
                    'NonComm_Longs': latest_record['NonComm_Longs'],
                    'NonComm_Shorts': latest_record['NonComm_Shorts'],
                    'DIFF_A_NonComm_Net': latest_record['DIFF_A_NonComm_Net'],
                    'Comm_Longs': latest_record['Comm_Longs'],
                    'Comm_Shorts': latest_record['Comm_Shorts'],
                    'DIFF_B_Comm_Net': latest_record['DIFF_B_Comm_Net'],
                    'Diff_A_minus_B': latest_record['Diff_A_minus_B'],
                    'COT_Index': latest_record.get('COT_Index', 0),
                    'Records_Count': len(currency_data)
                }
                latest_data.append(summary_row)
        
        if latest_data:
            latest_df = pd.DataFrame(latest_data)
            latest_df = latest_df.sort_values('COT_Index', ascending=False)  # Sort by index
            
            latest_filename = f"{output_dir}/{filename_prefix}_latest_{timestamp}.csv"
            latest_df.to_csv(latest_filename, index=False)
            logger.info(f"Saved latest summary to {latest_filename}")
            
            # Print latest summary to console
            print("\n" + "="*80)
            print("LATEST COT CURRENCY INDEX SUMMARY")
            print("="*80)
            for _, row in latest_df.iterrows():
                print(f"{row['Currency']:<20} | Index: {row['COT_Index']:>8.3f} | "
                      f"NonComm Net: {row['DIFF_A_NonComm_Net']:>10,} | "
                      f"Comm Net: {row['DIFF_B_Comm_Net']:>10,}")
            print("="*80)
        
        logger.info("Data saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def setup_cron_job():
    """
    Setup scheduled job to run every Wednesday at 4:00 PM (after COT data release)
    """
    import platform
    
    try:
        system = platform.system().lower()
        
        if system == 'windows':
            setup_windows_task()
        else:
            setup_unix_cron()
            
    except Exception as e:
        logger.error(f"❌ Failed to setup scheduled job: {e}")
        return False

def setup_windows_task():
    """
    Setup Windows Task Scheduler task for COT processing
    """
    try:
        import subprocess
        
        # Get the absolute path to this script
        script_path = os.path.abspath(__file__)
        
        # Task name
        task_name = "COT_Currency_Index_Processor"
        
        # Delete existing task if it exists
        try:
            subprocess.run(['schtasks', '/Delete', '/TN', task_name, '/F'], 
                         capture_output=True, check=False)
        except:
            pass
        
        # Create command
        python_exe = sys.executable
        command = f'"{python_exe}" "{script_path}"'
        
        # Create task to run every Wednesday at 4:00 PM
        create_cmd = [
            'schtasks', '/Create', '/TN', task_name,
            '/TR', command,
            '/SC', 'WEEKLY',
            '/D', 'WED',  # Wednesday
            '/ST', '16:00',  # 4:00 PM
            '/F'  # Force create (overwrite if exists)
        ]
        
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        
        logger.info(f"✅ Windows Task Scheduler job created successfully for COT Currency Index processing")
        logger.info(f"📋 Task Name: {task_name}")
        logger.info(f"🎯 Command: {command}")
        logger.info(f"📅 Schedule: Every Wednesday at 4:00 PM")
        
        # Show the created task
        list_result = subprocess.run(['schtasks', '/Query', '/TN', task_name, '/FO', 'LIST'], 
                                   capture_output=True, text=True)
        if list_result.returncode == 0:
            logger.info("📋 Task Details:")
            for line in list_result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"   {line.strip()}")
        
        logger.info(f"\n💡 To manually manage this task:")
        logger.info(f"   View: schtasks /Query /TN {task_name}")
        logger.info(f"   Run:  schtasks /Run /TN {task_name}")
        logger.info(f"   Stop: schtasks /End /TN {task_name}")
        logger.info(f"   Delete: schtasks /Delete /TN {task_name} /F")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create Windows task: {e}")
        logger.error(f"   Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Error setting up Windows task: {e}")
        return False

def setup_unix_cron():
    """
    Setup Unix/Linux cron job for COT processing
    """
    try:
        from crontab import CronTab
        
        # Get current user's cron
        cron = CronTab(user=True)
        
        # Remove any existing COT jobs
        cron.remove_all(comment='COT Currency Index Processor')
        
        # Add new job for every Wednesday at 4:00 PM
        script_path = os.path.abspath(__file__)
        job = cron.new(command=f'cd {os.path.dirname(script_path)} && python {script_path}', 
                      comment='COT Currency Index Processor')
        job.setall('0 16 * * 3')  # Every Wednesday at 4:00 PM
        
        # Write the cron job
        cron.write()
        logger.info("✅ Cron job setup completed - will run every Wednesday at 4:00 PM")
        
        return True
        
    except ImportError:
        logger.error("❌ python-crontab not installed. Install with: pip install python-crontab")
        return False
    except Exception as e:
        logger.error(f"❌ Error setting up cron job: {e}")
        return False

def main():
    """
    Main function to execute COT currency processing pipeline
    """
    parser = argparse.ArgumentParser(description='COT Currency Index Processor')
    parser.add_argument('--setup-cron', action='store_true', help='Setup weekly cron job')
    parser.add_argument('--year', type=int, help='Specific year to process (default: current year)')
    args = parser.parse_args()
    
    if args.setup_cron:
        setup_cron_job()
        return True
    
    logger.info("Starting COT Currency Index processing...")
    
    current_year = args.year or datetime.now().year
    previous_year = current_year - 1
    
    # Download current year data
    current_df = download_cot_data(current_year)
    if current_df is None:
        logger.error("Failed to download current year data. Exiting.")
        return False
    
    # Download previous year data for 1-year lookback
    previous_df = download_cot_data(previous_year)
    if previous_df is None:
        logger.warning("Failed to download previous year data. Using current year only.")
    
    # Process data
    processed_df = process_currency_cot_data(current_df, previous_df)
    if processed_df is None:
        logger.error("Failed to process data. Exiting.")
        return False
    
    # Save processed data
    save_currency_data(processed_df)
    
    logger.info("COT Currency Index processing completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
