#!/usr/bin/env python3
"""
USDJPY Position Cleanup Task
Runs every 4 hours to clean up conflicting pending orders based on open positions
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime, timedelta
from crontab import CronTab
from logging.handlers import RotatingFileHandler
import traceback
import json
import argparse

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from metatrader_mcp.server import *
from dataclasses import dataclass

# Configuration
SYMBOL = "USDJPY"
TASK_NAME = "Position_Cleanup_USDJPY"

@dataclass
class AppContext:
    client: str

# Enhanced logging configuration with rotation
def setup_logging():
    """Setup enhanced logging with rotation and proper encoding"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'cleanup_usdjpy.log')
    
    # Create logger
    logger = logging.getLogger('USJDPYCleanup')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create rotating file handler (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Check if we can use emojis (Windows console compatibility)
def supports_emojis():
    """Check if the current environment supports emoji display"""
    try:
        # Try to encode an emoji
        "🔧".encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False

# Global emoji support flag
EMOJI_SUPPORT = supports_emojis()

def emoji_or_text(emoji, text):
    """Return emoji if supported, otherwise return text"""
    return emoji if EMOJI_SUPPORT else text

@dataclass
class CleanupResult:
    """Result of cleanup operation"""
    success: bool
    open_positions: int
    position_type: str = None
    pending_orders_found: int = 0
    orders_cancelled: int = 0
    error_message: str = None

class USJDPYCleanupTask:
    """USDJPY Position Cleanup Task"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.ctx = None
        self.execution_id = None
        
    def generate_execution_id(self):
        """Generate unique execution ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_id = f"cleanup_{timestamp}"
        return self.execution_id
    
    def initialize(self):
        """Initialize MT5 connection with detailed logging"""
        try:
            self.logger.info(f"{emoji_or_text('🔌', '[INIT]')} Initializing MT5 connection...")
            
            client = initialize_client()
            self.ctx = AppContext(client=client)
            
            self.logger.info(f"{emoji_or_text('✅', '[OK]')} MT5 client initialized successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} MT5 initialization failed: {str(e)}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
            return False
    
    def test_mt5_connection(self):
        """Test MT5 connection and log account info"""
        try:
            self.logger.info(f"{emoji_or_text('🔌', '[CONN]')} Testing MT5 connection...")
            
            # Test connection by getting account info
            account_info = get_account_info(ctx=self.ctx)
            
            if account_info:
                self.logger.info(f"{emoji_or_text('✅', '[OK]')} MT5 connection successful")
                self.logger.info(f"{emoji_or_text('💰', '[ACC]')} Account: {account_info.get('login', 'N/A')}")
                self.logger.info(f"{emoji_or_text('💵', '[BAL]')} Balance: ${account_info.get('balance', 0):,.2f}")
                self.logger.info(f"{emoji_or_text('📊', '[EQU]')} Equity: ${account_info.get('equity', 0):,.2f}")
                return True
            else:
                self.logger.error(f"{emoji_or_text('❌', '[ERR]')} Failed to get account info")
                return False
                
        except Exception as e:
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} Connection test failed: {str(e)}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
            return False
    
    def get_usdjpy_positions(self):
        """Get open USDJPY positions"""
        try:
            self.logger.info(f"{emoji_or_text('🔍', '[POS]')} Checking {SYMBOL} open positions...")
            
            positions = get_positions_by_symbol(ctx=self.ctx, symbol=SYMBOL)
            
            if positions is not None and not positions.empty:
                self.logger.info(f"{emoji_or_text('📊', '[POS]')} Found {len(positions)} open {SYMBOL} position(s)")
                
                # Log position details
                for idx, pos in positions.iterrows():
                    pos_type = "BUY" if pos['type'] == 0 else "SELL"
                    self.logger.info(f"{emoji_or_text('💼', '[POS]')} Position {pos['ticket']}: {pos_type} {pos['volume']} lots at {pos['price_open']}")
                
                return positions
            else:
                self.logger.info(f"{emoji_or_text('📭', '[POS]')} No open {SYMBOL} positions found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} Error getting {SYMBOL} positions: {str(e)}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
            return None
    
    def get_usdjpy_pending_orders(self):
        """Get pending USDJPY orders"""
        try:
            self.logger.info(f"{emoji_or_text('📋', '[ORD]')} Checking {SYMBOL} pending orders...")
            
            pending_orders = get_pending_orders_by_symbol(ctx=self.ctx, symbol=SYMBOL)
            
            if pending_orders is not None and not pending_orders.empty:
                self.logger.info(f"{emoji_or_text('📊', '[ORD]')} Found {len(pending_orders)} pending {SYMBOL} order(s)")
                
                # Log order details
                for idx, order in pending_orders.iterrows():
                    order_type_map = {
                        2: "BUY_LIMIT", 3: "SELL_LIMIT", 
                        4: "BUY_STOP", 5: "SELL_STOP"
                    }
                    order_type = order_type_map.get(order['type'], f"TYPE_{order['type']}")
                    self.logger.info(f"{emoji_or_text('📝', '[ORD]')} Order {order['ticket']}: {order_type} {order['volume']} lots at {order['price_open']}")
                
                return pending_orders
            else:
                self.logger.info(f"{emoji_or_text('📭', '[ORD]')} No pending {SYMBOL} orders found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} Error getting {SYMBOL} pending orders: {str(e)}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
            return None
    
    def cancel_conflicting_orders(self, position_type, pending_orders):
        """Cancel orders that conflict with the current position"""
        cancelled_count = 0
        
        try:
            for idx, order in pending_orders.iterrows():
                order_type = order['type']
                order_ticket = order['ticket']
                
                # Determine if this order conflicts with the position
                should_cancel = False
                
                if position_type == "BUY":
                    # If we have a BUY position, cancel SELL orders (SELL_LIMIT=3, SELL_STOP=5)
                    if order_type in [3, 5]:
                        should_cancel = True
                        order_desc = "SELL_LIMIT" if order_type == 3 else "SELL_STOP"
                elif position_type == "SELL":
                    # If we have a SELL position, cancel BUY orders (BUY_LIMIT=2, BUY_STOP=4)
                    if order_type in [2, 4]:
                        should_cancel = True
                        order_desc = "BUY_LIMIT" if order_type == 2 else "BUY_STOP"
                
                if should_cancel:
                    self.logger.info(f"{emoji_or_text('🗑️', '[DEL]')} Cancelling conflicting {order_desc} order {order_ticket}")
                    
                    try:
                        result = cancel_pending_order(ctx=self.ctx, id=order_ticket)
                        if result:
                            self.logger.info(f"{emoji_or_text('✅', '[OK]')} Successfully cancelled order {order_ticket}")
                            cancelled_count += 1
                        else:
                            self.logger.warning(f"{emoji_or_text('⚠️', '[WARN]')} Failed to cancel order {order_ticket}")
                    except Exception as e:
                        self.logger.error(f"{emoji_or_text('💥', '[ERR]')} Error cancelling order {order_ticket}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} Error in cancel_conflicting_orders: {str(e)}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
        
        return cancelled_count
    
    def run_cleanup(self):
        """Main cleanup logic"""
        execution_id = self.generate_execution_id()
        
        self.logger.info("=" * 80)
        self.logger.info(f"{emoji_or_text('🔧', '[START]')} Starting {SYMBOL} cleanup task - ID: {execution_id}")
        self.logger.info("=" * 80)
        
        try:
            # Initialize MT5 client
            if not self.initialize():
                return CleanupResult(
                    success=False,
                    open_positions=0,
                    error_message="MT5 initialization failed"
                )
            
            # Test MT5 connection
            if not self.test_mt5_connection():
                return CleanupResult(
                    success=False,
                    open_positions=0,
                    error_message="MT5 connection failed"
                )
            
            # Get open positions
            positions = self.get_usdjpy_positions()
            if positions is None:
                return CleanupResult(
                    success=False,
                    open_positions=0,
                    error_message="Failed to retrieve positions"
                )
            
            # If no positions, nothing to do
            if positions.empty:
                self.logger.info(f"{emoji_or_text('✅', '[OK]')} No open {SYMBOL} positions - cleanup not needed")
                return CleanupResult(
                    success=True,
                    open_positions=0
                )
            
            # Determine position type (assume all positions are same direction)
            # In MT5: type 0 = BUY, type 1 = SELL
            first_position_type = "BUY" if positions.iloc[0]['type'] == 0 else "SELL"
            self.logger.info(f"{emoji_or_text('📊', '[POS]')} Current {SYMBOL} position type: {first_position_type}")
            
            # Get pending orders
            pending_orders = self.get_usdjpy_pending_orders()
            if pending_orders is None:
                return CleanupResult(
                    success=False,
                    open_positions=len(positions),
                    position_type=first_position_type,
                    error_message="Failed to retrieve pending orders"
                )
            
            # If no pending orders, nothing to cancel
            if pending_orders.empty:
                self.logger.info(f"{emoji_or_text('✅', '[OK]')} No pending {SYMBOL} orders - cleanup not needed")
                return CleanupResult(
                    success=True,
                    open_positions=len(positions),
                    position_type=first_position_type,
                    pending_orders_found=0,
                    orders_cancelled=0
                )
            
            # Cancel conflicting orders
            cancelled_count = self.cancel_conflicting_orders(first_position_type, pending_orders)
            
            result = CleanupResult(
                success=True,
                open_positions=len(positions),
                position_type=first_position_type,
                pending_orders_found=len(pending_orders),
                orders_cancelled=cancelled_count
            )
            
            self.logger.info(f"{emoji_or_text('📊', '[SUM]')} Cleanup summary:")
            self.logger.info(f"{emoji_or_text('💼', '[SUM]')} Open positions: {result.open_positions}")
            self.logger.info(f"{emoji_or_text('📋', '[SUM]')} Pending orders found: {result.pending_orders_found}")
            self.logger.info(f"{emoji_or_text('🗑️', '[SUM]')} Orders cancelled: {result.orders_cancelled}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error in cleanup: {str(e)}"
            self.logger.error(f"{emoji_or_text('💥', '[ERR]')} {error_msg}")
            self.logger.error(f"{emoji_or_text('📋', '[TRC]')} {traceback.format_exc()}")
            
            return CleanupResult(
                success=False,
                open_positions=0,
                error_message=error_msg
            )
        
        finally:
            self.logger.info("=" * 80)
            self.logger.info(f"{emoji_or_text('🏁', '[END]')} Cleanup task completed - ID: {execution_id}")
            self.logger.info("=" * 80)

def setup_windows_task():
    """Set up Windows Task Scheduler task"""
    try:
        import subprocess
        
        # Get the absolute path to this script
        script_path = os.path.abspath(__file__)
        
        # Task name
        task_name = f"{TASK_NAME}_4H"
        
        # Delete existing task if it exists
        try:
            subprocess.run(['schtasks', '/Delete', '/TN', task_name, '/F'], 
                         capture_output=True, check=False)
        except:
            pass
        
        # Create command - run every 4 hours
        python_exe = sys.executable
        
        # Create the task to run every 4 hours
        cmd = [
            'schtasks', '/Create',
            '/TN', task_name,
            '/TR', f'"{python_exe}" "{script_path}" --run',
            '/SC', 'HOURLY',
            '/MO', '4',  # Every 4 hours
            '/ST', '00:00',  # Start at midnight
            '/F'  # Force create (overwrite if exists)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Windows task '{task_name}' created successfully!")
            print(f"📅 Schedule: Every 4 hours starting at 00:00")
            print(f"📁 Script: {script_path}")
        else:
            print(f"❌ Failed to create Windows task: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error setting up Windows task: {e}")

def setup_unix_cron():
    """Set up Unix cron job"""
    try:
        script_path = os.path.abspath(__file__)
        python_exe = sys.executable
        
        # Cron command (every 4 hours at minute 0)
        cron_command = f"0 */4 * * * {python_exe} {script_path} --run"
        
        cron = CronTab(user=True)
        
        # Remove existing job
        cron.remove_all(comment=f'{TASK_NAME}_4H')
        
        # Add new job
        job = cron.new(command=f'{python_exe} {script_path} --run')
        job.setall('0 */4 * * *')  # Every 4 hours at minute 0
        job.set_comment(f'{TASK_NAME}_4H')
        
        cron.write()
        
        print(f"✅ Unix cron job created successfully!")
        print(f"📅 Schedule: Every 4 hours (0 */4 * * *)")
        print(f"📁 Script: {script_path}")
        
    except Exception as e:
        print(f"❌ Error setting up Unix cron: {e}")

def clean_USDJPY():
    """Main function to run the cleanup task"""
    cleanup_task = USJDPYCleanupTask()
    result = cleanup_task.run_cleanup()
    return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description=f'{SYMBOL} Position Cleanup Task')
    parser.add_argument('--run', action='store_true', help='Run the cleanup task')
    parser.add_argument('--setup-cron', action='store_true', help='Set up cron job')
    parser.add_argument('--run-once', action='store_true', help='Run once for testing')
    
    args = parser.parse_args()
    
    if args.setup_cron:
        print(f"🔧 Setting up cron job for {SYMBOL} cleanup task...")
        if os.name == 'nt':  # Windows
            setup_windows_task()
        else:  # Unix/Linux/Mac
            setup_unix_cron()
    elif args.run or args.run_once:
        result = clean_USDJPY()
        if not result.success:
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()