import pandas as pd
from metatrader_mcp.server import get_pending_orders_by_symbol, get_account_info, get_symbol_price
from metatrader_mcp.server import Context
import logging
import traceback
import subprocess
import sys
import os
import getpass

# Get logger for this module
logger = logging.getLogger(__name__)

def setup_windows_task_with_logon_options(script_path, task_name, timeframe, logger):
    """
    Set up Windows Task Scheduler task that runs regardless of user logon status
    
    Args:
        script_path: Path to the script file
        task_name: Name for the task
        timeframe: Timeframe for scheduling (H1, M15, M30, H4)
        logger: Logger instance for output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Delete existing task if it exists
        try:
            subprocess.run(['schtasks', '/Delete', '/TN', task_name, '/F'], 
                         capture_output=True, check=False)
            logger.info(f"🔄 Deleted existing task: {task_name}")
        except:
            pass
        
        # Get current user
        current_user = os.environ.get('USERNAME', os.environ.get('USER', ''))
        if not current_user:
            logger.error("❌ Could not determine current user")
            return False
        
        # Create command
        python_exe = sys.executable
        command = f'"{python_exe}" "{script_path}" --run-once'
        
        # Determine schedule based on timeframe
        schedule_params = []
        
        if timeframe == 'H1':
            schedule_params = ['/SC', 'HOURLY', '/MO', '1', '/ST', '00:05']
            logger.info("📅 Windows Task: Every hour at minute 5")
            
        elif timeframe == 'M15':
            schedule_params = ['/SC', 'MINUTE', '/MO', '15']
            logger.info("📅 Windows Task: Every 15 minutes")
            
        elif timeframe == 'M30':
            schedule_params = ['/SC', 'MINUTE', '/MO', '30']
            logger.info("📅 Windows Task: Every 30 minutes")
            
        elif timeframe == 'H4':
            schedule_params = ['/SC', 'HOURLY', '/MO', '4', '/ST', '00:05']
            logger.info("📅 Windows Task: Every 4 hours at minute 5")
            
        else:
            logger.error(f"❌ Unsupported timeframe: {timeframe}")
            return False
        
        # Prompt user for choice
        logger.info("\\n" + "="*60)
        logger.info("🔐 WINDOWS TASK SECURITY OPTIONS")
        logger.info("="*60)
        logger.info("Choose how you want the task to run:")
        logger.info("1. Run only when user is logged on (simpler, no password needed)")
        logger.info("2. Run whether user is logged on or not (requires password)")
        logger.info("="*60)
        
        try:
            choice = input("Enter choice (1 or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("\\n⚠️ User cancelled. Using option 1 (logged on only)")
            choice = "1"
        
        if choice == "2":
            # Option 2: Run whether user is logged on or not
            logger.info("\\n🔑 For the task to run when you're not logged on, we need your Windows password.")
            logger.info("⚠️  Your password will be used to configure the task but won't be stored in code.")
            logger.info("💡 Alternatively, you can configure this manually in Task Scheduler later.")
            
            try:
                password = getpass.getpass(f"Enter password for user '{current_user}': ")
                if not password.strip():
                    logger.warning("⚠️ No password entered. Falling back to 'logged on only' mode.")
                    choice = "1"
            except (EOFError, KeyboardInterrupt):
                logger.info("\\n⚠️ Password entry cancelled. Using 'logged on only' mode.")
                choice = "1"
        
        # Build command based on choice
        if choice == "2" and 'password' in locals() and password.strip():
            # Create task that runs whether user is logged on or not
            create_cmd = [
                'schtasks', '/Create', '/TN', task_name,
                '/TR', command,
                '/RU', current_user,  # Run as current user
                '/RP', password,      # User password
                '/RL', 'HIGHEST',     # Run with highest privileges
                '/F'                  # Force overwrite
            ] + schedule_params
            
            logger.info(f"✅ Creating task to run whether user is logged on or not")
            
        else:
            # Create task that runs only when user is logged on (default)
            create_cmd = [
                'schtasks', '/Create', '/TN', task_name,
                '/TR', command,
                '/F'  # Force overwrite
            ] + schedule_params
            
            logger.info(f"✅ Creating task to run only when user is logged on")
        
        # Execute the command
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        
        logger.info(f"✅ Windows Task Scheduler job created successfully")
        logger.info(f"📋 Task Name: {task_name}")
        logger.info(f"🎯 Command: {command}")
        logger.info(f"👤 User: {current_user}")
        
        # Show task details
        list_result = subprocess.run(['schtasks', '/Query', '/TN', task_name, '/FO', 'LIST'], 
                                   capture_output=True, text=True)
        if list_result.returncode == 0:
            logger.info("\\n📋 Task Details:")
            for line in list_result.stdout.split('\\n'):
                if line.strip() and any(keyword in line for keyword in ['Task Name:', 'Status:', 'Logon Mode:', 'Run As User:', 'Schedule Type:']):
                    logger.info(f"   {line.strip()}")
        
        # Provide management commands
        logger.info(f"\\n💡 To manually manage this task:")
        logger.info(f"   View: schtasks /Query /TN {task_name}")
        logger.info(f"   Run:  schtasks /Run /TN {task_name}")
        logger.info(f"   Stop: schtasks /End /TN {task_name}")
        logger.info(f"   Delete: schtasks /Delete /TN {task_name} /F")
        
        # Additional instructions for manual configuration
        if choice == "1":
            logger.info(f"\\n📝 To change to 'run whether user is logged on or not' later:")
            logger.info(f"   1. Open Task Scheduler (taskschd.msc)")
            logger.info(f"   2. Find task: {task_name}")
            logger.info(f"   3. Right-click -> Properties -> General tab")
            logger.info(f"   4. Select 'Run whether user is logged on or not'")
            logger.info(f"   5. Check 'Run with highest privileges' if needed")
            logger.info(f"   6. Enter your password when prompted")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create Windows task: {e}")
        if e.stderr:
            logger.error(f"   Error output: {e.stderr}")
        if e.stdout:
            logger.error(f"   Command output: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"❌ Error setting up Windows task: {e}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        return False

def check_no_pending_orders(ctx: Context, symbol: str) -> bool:
    """
    Check if there are no pending orders for a given symbol
    """
    pending_orders_df = get_pending_orders_by_symbol(ctx, symbol)
    if pending_orders_df.empty:
        return True
    else:
        return False

def calculate_position_size(ctx, symbol, stop_loss_distance, risk_per_trade):
    """Calculate position size based on risk management with logging"""
    try:
        logger.info(f"💰 Calculating position size...")
        logger.info(f"   Stop loss distance: {stop_loss_distance:.5f}")
        logger.info(f"   Risk per trade: {risk_per_trade * 100}%")
        
        # Get current account information
        logger.info(f"🔍 Getting account information...")
        account_info = get_account_info(ctx=ctx)
        
        if not account_info:
            logger.error(f"❌ Failed to get account information")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        account_balance = account_info.get('balance', 0)
        account_currency = account_info.get('currency', 'USD')
        logger.info(f"💵 Account balance: {account_balance:,.2f} {account_currency}")
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_per_trade
        logger.info(f"💸 Risk amount: {risk_amount:,.2f} {account_currency}")
        
        # Get current symbol price to calculate pip value
        symbol_price_info = get_symbol_price(ctx=ctx, symbol_name=symbol)
        
        if not symbol_price_info:
            logger.error(f"❌ Failed to get symbol price information for {symbol}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        # Use bid price for calculation
        current_price = symbol_price_info.get('bid', 0)
        if current_price <= 0:
            logger.error(f"❌ Invalid current price: {current_price}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        logger.info(f"📊 Current {symbol} price: {current_price:.5f}")
        
        # Calculate pip value based on symbol
        # For standard lots (100,000 units):
        # - For pairs like EUR/USD: 1 pip = 0.0001, pip value = 100,000 * 0.0001 = $10
        # - For JPY pairs: 1 pip = 0.01, pip value = 100,000 * 0.01 / current_price
        
        if 'JPY' in symbol:
            # JPY pairs: pip = 0.01
            pip_size = 0.01
            pip_value_per_lot = 100000 * pip_size / current_price  # In account currency per lot
        else:
            # Standard pairs: pip = 0.0001  
            pip_size = 0.0001
            if symbol.endswith('USD') and account_currency == 'USD':
                # For USD base pairs like EUR/USD when account is USD
                pip_value_per_lot = 100000 * pip_size  # $10 per lot
            else:
                # For other pairs, approximate calculation
                pip_value_per_lot = 100000 * pip_size
        
        # Convert stop loss distance to pips
        stop_loss_pips = stop_loss_distance / pip_size
        logger.info(f"📏 Stop loss distance: {stop_loss_pips:.1f} pips")
        
        # Calculate position size
        # Risk = Position_Size * Pip_Value_Per_Lot * Stop_Loss_Pips
        # Position_Size = Risk / (Pip_Value_Per_Lot * Stop_Loss_Pips)
        
        if stop_loss_pips <= 0:
            logger.error(f"❌ Invalid stop loss pips: {stop_loss_pips}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        calculated_lot_size = risk_amount / (pip_value_per_lot * stop_loss_pips)
        
        # Apply position size constraints
        min_lot_size = 0.01
        max_lot_size = 10.0  # Conservative maximum
        
        # Round to valid lot size increments (0.01)
        lot_size = round(calculated_lot_size, 2)
        
        # Apply constraints
        lot_size = max(min_lot_size, min(lot_size, max_lot_size))
        
        # Calculate actual risk with final lot size
        actual_risk = lot_size * pip_value_per_lot * stop_loss_pips
        actual_risk_percentage = (actual_risk / account_balance) * 100
        
        logger.info(f"💰 POSITION SIZE CALCULATION:")
        logger.info(f"   Pip value per lot: {pip_value_per_lot:.2f} {account_currency}")
        logger.info(f"   Calculated lot size: {calculated_lot_size:.4f}")
        logger.info(f"   Final lot size: {lot_size:.2f} lots")
        logger.info(f"   Actual risk: {actual_risk:.2f} {account_currency} ({actual_risk_percentage:.2f}%)")
        
        if lot_size == min_lot_size:
            logger.warning(f"⚠️ Using minimum lot size - calculated size was too small")
        elif lot_size == max_lot_size:
            logger.warning(f"⚠️ Using maximum lot size - calculated size was too large")
        
        return lot_size
        
    except Exception as e:
        logger.error(f"❌ Error calculating position size: {str(e)}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        logger.warning(f"⚠️ Using minimum lot size as fallback")
        return 0.01  # Minimum lot size as fallback

def check_if_range_is_channel_local(candles_df):
    """Local channel detection function with detailed logging"""
    if candles_df.empty or len(candles_df) < 2:
        logger.warning(f"⚠️ Channel check: insufficient data ({len(candles_df)} candles)")
        return False
    
    try:
        channel_range = candles_df["high"].max() - candles_df["low"].min()
        
        two_top_peaks = candles_df.nlargest(2, "high")
        two_bottom_peaks = candles_df.nsmallest(2, "low")
        
        logger.debug(f"🔍 Channel analysis:")
        logger.debug(f"   Channel range: {channel_range:.5f}")
        logger.debug(f"   Top peaks: {len(two_top_peaks)}")
        logger.debug(f"   Bottom peaks: {len(two_bottom_peaks)}")
        
        if len(two_top_peaks) < 2 or len(two_bottom_peaks) < 2:
            logger.debug(f"⚠️ Channel check: insufficient peaks")
            return False
        
        # Check if top peaks are from adjacent candles
        top_peak_indices = sorted(two_top_peaks.index.tolist())
        if len(top_peak_indices) >= 2:
            for i in range(len(top_peak_indices) - 1):
                if abs(top_peak_indices[i+1] - top_peak_indices[i]) == 1:
                    logger.debug(f"⚠️ Channel check: top peaks from adjacent candles (indices {top_peak_indices[i]} and {top_peak_indices[i+1]})")
                    return False
        
        # Check if bottom peaks are from adjacent candles
        bottom_peak_indices = sorted(two_bottom_peaks.index.tolist())
        if len(bottom_peak_indices) >= 2:
            for i in range(len(bottom_peak_indices) - 1):
                if abs(bottom_peak_indices[i+1] - bottom_peak_indices[i]) == 1:
                    logger.debug(f"⚠️ Channel check: bottom peaks from adjacent candles (indices {bottom_peak_indices[i]} and {bottom_peak_indices[i+1]})")
                    return False
        
        top_peaks_diff = two_top_peaks["high"].diff()
        bottom_peaks_diff = two_bottom_peaks["low"].diff()
        
        top_diff = abs(top_peaks_diff.iloc[1]) if not pd.isna(top_peaks_diff.iloc[1]) else 0
        bottom_diff = abs(bottom_peaks_diff.iloc[1]) if not pd.isna(bottom_peaks_diff.iloc[1]) else 0
        
        # top_threshold = channel_range * 0.2
        # bottom_threshold = channel_range * 0.2
        top_threshold = channel_range * 0.15
        bottom_threshold = channel_range * 0.15
        
        is_channel = (top_diff <= top_threshold and bottom_diff <= bottom_threshold)
        
        logger.debug(f"   Top diff: {top_diff:.5f} (threshold: {top_threshold:.5f})")
        logger.debug(f"   Bottom diff: {bottom_diff:.5f} (threshold: {bottom_threshold:.5f})")
        logger.debug(f"   Top peak indices: {top_peak_indices}")
        logger.debug(f"   Bottom peak indices: {bottom_peak_indices}")
        logger.debug(f"   Is channel: {is_channel}")
        
        return is_channel

    except Exception as e:
        logger.error(f"Error in channel detection: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
