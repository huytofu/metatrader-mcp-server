#!/usr/bin/env python3
"""
Automated Range Straddle Trading Strategy
Runs as a cron job to identify and execute range breakout trades
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime, timedelta
from crontab import CronTab

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from metatrader_mcp.server import *
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('range_straddle_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# STRATEGY CONFIGURATION
# These settings should be updated based on your historical test results
STRATEGY_CONFIG = {
    'symbol': 'AUDUSD',  # Best performing pair from historical tests
    'timeframe': 'M15',   # Best performing timeframe
    'percentile_num': 0.1,  # Best performing percentile threshold (10% or 20%)
    'num_candles': 8,    # Best performing range candles (8 or 12)
    'safety_factor': 0.2,  # Entry buffer
    'tp_sl_ratio': 2.0,  # Best performing TP:SL ratio (1.0, 1.5, or 2.0)
    'analysis_period': 2400,  # Analysis window for percentile calculation
    'risk_per_trade': 0.01,  # 1% risk per trade
    'max_spread_pips': 1,  # Maximum spread to allow trades
    'min_range_pips': 10,  # Minimum range width in pips
    'max_daily_trades': 6,  # Maximum trades per day
}

@dataclass
class AppContext:
    client: str

class RangeStraddleStrategy:
    """
    Automated Range Straddle Strategy Implementation
    """
    
    def __init__(self, config):
        self.config = config
        self.ctx = None
        self.daily_trades_count = 0
        self.last_trade_date = None
        
    def initialize(self):
        """Initialize MT5 connection"""
        try:
            client = initialize_client()
            self.ctx = AppContext(client=client)
            logger.info("✅ MT5 connection initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize MT5 connection: {e}")
            return False
    
    def check_market_conditions(self):
        """Check if market is open and conditions are suitable for trading"""
        try:
            # Check if market is open
            current_time = datetime.now()
            
            # Reset daily trade count if new day
            if self.last_trade_date != current_time.date():
                self.daily_trades_count = 0
                self.last_trade_date = current_time.date()
            
            # Check daily trade limit
            if self.daily_trades_count >= self.config['max_daily_trades']:
                logger.info(f"⏸️ Daily trade limit reached ({self.daily_trades_count}/{self.config['max_daily_trades']})")
                return False
            
            # Add market hours check here if needed
            # For forex, market is generally open 24/5
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking market conditions: {e}")
            return False
    
    def get_market_data(self):
        """Retrieve latest market data"""
        try:
            total_candles_needed = self.config['analysis_period'] + self.config['num_candles'] + 10
            
            candles_df = get_candles_latest(
                ctx=self.ctx,
                symbol_name=self.config['symbol'],
                timeframe=self.config['timeframe'],
                count=total_candles_needed
            )
            
            if candles_df.empty or len(candles_df) < self.config['analysis_period']:
                logger.error(f"❌ Insufficient data for {self.config['symbol']}")
                return None
            
            # Prepare data
            candles_df['time'] = pd.to_datetime(candles_df['time'])
            candles_df = candles_df.sort_values('time').reset_index(drop=True)
            candles_df['candle_range'] = candles_df['high'] - candles_df['low']
            
            logger.info(f"📊 Retrieved {len(candles_df)} candles for analysis")
            return candles_df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving market data: {e}")
            return None
    
    def analyze_range_opportunity(self, candles_df):
        """Analyze current range for trading opportunity"""
        try:
            # Get latest complete candle (not current forming candle)
            latest_idx = len(candles_df) - 2  # -2 to get last complete candle
            range_start_idx = latest_idx - self.config['num_candles'] + 1
            range_candles = candles_df.iloc[range_start_idx:latest_idx+1]
            
            # Calculate range metrics
            range_high = range_candles['high'].max()
            range_low = range_candles['low'].min()
            range_width = range_high - range_low
            current_avg_candle_height = range_candles['candle_range'].mean()
            
            # Calculate percentile threshold
            analysis_start_idx = max(0, latest_idx - self.config['analysis_period'])
            analysis_candles = candles_df.iloc[analysis_start_idx:latest_idx+1]
            percentile_threshold = np.percentile(
                analysis_candles['candle_range'], 
                self.config['percentile_num'] * 100
            )
            
            # Check conditions
            is_narrow = current_avg_candle_height <= percentile_threshold
            is_channel = self.check_if_range_is_channel_local(range_candles)
            
            # Convert range to pips (for USDJPY, 1 pip = 0.01)
            pip_multiplier = 100 if 'JPY' in self.config['symbol'] else 10000
            range_pips = range_width * pip_multiplier
            
            current_time = candles_df.iloc[latest_idx]['time']
            
            logger.info(f"🕐 {current_time} | {self.config['symbol']} {self.config['timeframe']}")
            logger.info(f"📊 Range: {range_width:.5f} ({range_pips:.1f} pips)")
            logger.info(f"📏 Avg Height: {current_avg_candle_height:.5f} | Threshold: {percentile_threshold:.5f}")
            logger.info(f"✅ Narrow: {is_narrow} | Channel: {is_channel} | Min Range: {range_pips >= self.config['min_range_pips']}")
            
            if is_narrow and is_channel and range_pips >= self.config['min_range_pips']:
                return {
                    'signal': True,
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_width': range_width,
                    'range_pips': range_pips,
                    'current_time': current_time
                }
            else:
                logger.info(f"⏸️ No trade opportunity")
                return {'signal': False}
                
        except Exception as e:
            logger.error(f"❌ Error analyzing range opportunity: {e}")
            return {'signal': False}
    
    def check_if_range_is_channel_local(self, candles_df):
        """Local channel detection function"""
        if candles_df.empty or len(candles_df) < 2:
            return False
        
        try:
            channel_range = candles_df["high"].max() - candles_df["low"].min()
            
            two_top_peaks = candles_df.nlargest(2, "high")
            two_bottom_peaks = candles_df.nsmallest(2, "low")
            
            if len(two_top_peaks) < 2 or len(two_bottom_peaks) < 2:
                return False
            
            top_peaks_diff = two_top_peaks["high"].diff()
            bottom_peaks_diff = two_bottom_peaks["low"].diff()
            
            if len(top_peaks_diff) < 2 or len(bottom_peaks_diff) < 2:
                return False
                
            top_diff = abs(top_peaks_diff.iloc[1]) if not pd.isna(top_peaks_diff.iloc[1]) else 0
            bottom_diff = abs(bottom_peaks_diff.iloc[1]) if not pd.isna(bottom_peaks_diff.iloc[1]) else 0
            
            return (top_diff <= channel_range * 0.2 and bottom_diff <= channel_range * 0.2)
            
        except Exception as e:
            logger.warning(f"⚠️ Channel detection error: {e}, defaulting to False")
            return False
    
    def calculate_position_size(self, stop_loss_distance):
        """Calculate position size based on risk management"""
        try:
            # This is a simplified calculation
            # In practice, you'd get account balance and implement proper risk management
            
            # For now, using a fixed lot size
            # TODO: Implement proper position sizing based on account balance and risk
            lot_size = 0.1
            
            logger.info(f"💰 Calculated position size: {lot_size} lots")
            return lot_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.01  # Minimum lot size as fallback
    
    def place_oco_trades(self, opportunity_data):
        """Place OCO (One Cancels Other) trades"""
        try:
            # First check if there are any existing open positions for this symbol
            logger.info(f"🔍 Checking existing positions for {self.config['symbol']}...")
            
            try:
                existing_positions = get_positions_by_symbol(ctx=self.ctx, symbol=self.config['symbol'])
                
                if existing_positions is not None and len(existing_positions) > 0:
                    logger.warning(f"⚠️ Found {len(existing_positions)} existing open position(s) for {self.config['symbol']}")
                    logger.warning(f"🛑 Skipping trade placement to avoid overexposure")
                    logger.info("📊 Current positions:")
                    for idx, pos in existing_positions.iterrows():
                        pos_type = "BUY" if pos.get('type', 0) == 0 else "SELL"
                        logger.info(f"   Position {pos.get('ticket', 'N/A')}: {pos_type} {pos.get('volume', 0)} lots at {pos.get('price_open', 0)}")
                    return False
                else:
                    logger.info(f"✅ No existing positions found for {self.config['symbol']} - proceeding with trade setup")
                    
            except Exception as e:
                logger.error(f"❌ Error checking existing positions: {e}")
                logger.warning(f"🛑 Skipping trade placement due to position check failure")
                return False
            
            range_high = opportunity_data['range_high']
            range_low = opportunity_data['range_low']
            range_width = opportunity_data['range_width']
            
            # Calculate trade levels
            buy_stop_price = range_high + (self.config['safety_factor'] * range_width)
            sell_stop_price = range_low - (self.config['safety_factor'] * range_width)
            
            # Calculate TP/SL levels
            stop_loss_distance = range_width
            take_profit_distance = self.config['tp_sl_ratio'] * range_width
            
            buy_tp = buy_stop_price + take_profit_distance
            buy_sl = buy_stop_price - stop_loss_distance
            sell_tp = sell_stop_price - take_profit_distance
            sell_sl = sell_stop_price + stop_loss_distance
            
            # Calculate position size
            lot_size = self.calculate_position_size(stop_loss_distance)
            
            logger.info(f"🚀 TRADE OPPORTUNITY DETECTED!")
            logger.info(f"📈 BUY STOP: {buy_stop_price:.5f} | TP: {buy_tp:.5f} | SL: {buy_sl:.5f}")
            logger.info(f"📉 SELL STOP: {sell_stop_price:.5f} | TP: {sell_tp:.5f} | SL: {sell_sl:.5f}")
            logger.info(f"💱 Lot Size: {lot_size}")
            
            # Place BUY STOP order using MCP tool
            logger.info("📤 Placing BUY STOP order...")
            buy_order_result = place_pending_order(
                ctx=self.ctx,
                symbol=self.config['symbol'],
                volume=lot_size,
                type='BUY',
                price=buy_stop_price,
                stop_loss=buy_sl,
                take_profit=buy_tp
            )
            
            if buy_order_result.get('error', False):
                logger.error(f"❌ Failed to place BUY order: {buy_order_result.get('message', 'Unknown error')}")
                return False
            else:
                logger.info(f"✅ BUY order placed successfully: {buy_order_result.get('message', 'Success')}")
            
            # Place SELL STOP order using MCP tool
            logger.info("📤 Placing SELL STOP order...")
            sell_order_result = place_pending_order(
                ctx=self.ctx,
                symbol=self.config['symbol'],
                volume=lot_size,
                type='SELL',
                price=sell_stop_price,
                stop_loss=sell_sl,
                take_profit=sell_tp
            )
            
            if sell_order_result.get('error', False):
                logger.error(f"❌ Failed to place SELL order: {sell_order_result.get('message', 'Unknown error')}")
                # Cancel the BUY order if SELL order fails
                if 'data' in buy_order_result and hasattr(buy_order_result['data'], 'order'):
                    try:
                        cancel_result = cancel_pending_order(ctx=self.ctx, id=buy_order_result['data'].order)
                        logger.info(f"🔄 Cancelled BUY order due to SELL order failure")
                    except Exception as e:
                        logger.error(f"❌ Failed to cancel BUY order: {e}")
                return False
            else:
                logger.info(f"✅ SELL order placed successfully: {sell_order_result.get('message', 'Success')}")
            
            # Log trade summary
            logger.info(f"🎯 OCO TRADE SETUP COMPLETED!")
            logger.info(f"   📈 BUY Order ID: {buy_order_result.get('data', {}).order if 'data' in buy_order_result else 'N/A'}")
            logger.info(f"   📉 SELL Order ID: {sell_order_result.get('data', {}).order if 'data' in sell_order_result else 'N/A'}")
            logger.info(f"   💼 Risk: {range_width:.5f} | Reward: {take_profit_distance:.5f} | R:R = 1:{self.config['tp_sl_ratio']}")
            
            self.daily_trades_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing OCO trades: {e}")
            return False
    
    def run_strategy(self):
        """Main strategy execution function"""
        logger.info(f"🚀 Starting Range Straddle Strategy - {self.config['symbol']} {self.config['timeframe']}")
        
        try:
            # Initialize connection
            if not self.initialize():
                return False
            
            # Check market conditions
            if not self.check_market_conditions():
                return False
            
            # Get market data
            candles_df = self.get_market_data()
            if candles_df is None:
                return False
            
            # Analyze for opportunities
            opportunity = self.analyze_range_opportunity(candles_df)
            
            if opportunity['signal']:
                # Place trades
                success = self.place_oco_trades(opportunity)
                if success:
                    logger.info(f"✅ Strategy execution completed successfully")
                else:
                    logger.error(f"❌ Failed to place trades")
                return success
            else:
                logger.info(f"⏸️ No trading opportunity found")
                return True
                
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            return False

def strat_one():
    """Main entry point for the strategy"""
    strategy = RangeStraddleStrategy(STRATEGY_CONFIG)
    return strategy.run_strategy()

def setup_cron_job():
    """Set up scheduled job based on operating system"""
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
    """Set up Windows Task Scheduler task"""
    try:
        import subprocess
        
        # Get the absolute path to this script
        script_path = os.path.abspath(__file__)
        project_path = os.path.dirname(os.path.dirname(script_path))
        
        # Task name - make it unique for strat_two
        task_name = f"RangeStraddleStrategy_{STRATEGY_CONFIG['symbol']}_{STRATEGY_CONFIG['timeframe']}"
        
        # Delete existing task if it exists
        try:
            subprocess.run(['schtasks', '/Delete', '/TN', task_name, '/F'], 
                         capture_output=True, check=False)
        except:
            pass
        
        # Create command based on timeframe
        python_exe = sys.executable
        command = f'"{python_exe}" "{script_path}" --run-once'
        
        # Set schedule based on timeframe
        if STRATEGY_CONFIG['timeframe'] == 'H1':
            # Run every hour at minute 5
            schedule = '/SC HOURLY /MO 1 /ST 00:05'
            logger.info("📅 Windows Task: Every hour at minute 5")
            
        elif STRATEGY_CONFIG['timeframe'] == 'M15':
            # Run every 15 minutes (Windows Task Scheduler limitation - will run every minute and check internally)
            schedule = '/SC MINUTE /MO 1'
            logger.info("📅 Windows Task: Every minute (will check internally for 15-min intervals)")
            
        elif STRATEGY_CONFIG['timeframe'] == 'M30':
            # Run every 30 minutes
            schedule = '/SC MINUTE /MO 30'
            logger.info("📅 Windows Task: Every 30 minutes")
            
        elif STRATEGY_CONFIG['timeframe'] == 'H4':
            # Run every 4 hours
            schedule = '/SC HOURLY /MO 4 /ST 00:05'
            logger.info("📅 Windows Task: Every 4 hours at minute 5")
            
        else:
            logger.error(f"❌ Unsupported timeframe: {STRATEGY_CONFIG['timeframe']}")
            return False
        
        # Create the task
        create_cmd = [
            'schtasks', '/Create', '/TN', task_name,
            '/TR', command,
            '/SC', 'MINUTE', '/MO', '1' if STRATEGY_CONFIG['timeframe'] == 'M15' else '60',
            '/F'  # Force create (overwrite if exists)
        ]
        
        # For H1, H4 - use hourly schedule
        if STRATEGY_CONFIG['timeframe'] in ['H1', 'H4']:
            mo = '1' if STRATEGY_CONFIG['timeframe'] == 'H1' else '4'
            create_cmd = [
                'schtasks', '/Create', '/TN', task_name,
                '/TR', command,
                '/SC', 'HOURLY', '/MO', mo,
                '/ST', '00:05',  # Start at 5 minutes past the hour
                '/F'
            ]
        elif STRATEGY_CONFIG['timeframe'] == 'M30':
            create_cmd = [
                'schtasks', '/Create', '/TN', task_name,
                '/TR', command,
                '/SC', 'MINUTE', '/MO', '30',
                '/F'
            ]
        elif STRATEGY_CONFIG['timeframe'] == 'M15':
            # For M15, create a task that runs every 15 minutes
            create_cmd = [
                'schtasks', '/Create', '/TN', task_name,
                '/TR', command,
                '/SC', 'MINUTE', '/MO', '15',
                '/F'
            ]
        
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        
        logger.info(f"✅ Windows Task Scheduler job created successfully for {STRATEGY_CONFIG['timeframe']} strategy")
        logger.info(f"📋 Task Name: {task_name}")
        logger.info(f"🎯 Command: {command}")
        
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
    """Set up Unix/Linux cron job"""
    try:
        from crontab import CronTab
        
        cron = CronTab(user=True)  # Use current user instead of root
        
        # Remove existing jobs for this strategy
        cron.remove_all(comment='range_straddle_strategy_two')
        
        # Get the absolute path to this script
        script_path = os.path.abspath(__file__)
        project_path = os.path.dirname(os.path.dirname(script_path))
        
        # Create command
        command = f"cd {project_path} && python {script_path} --run-once"
        
        # Create new job based on timeframe
        job = cron.new(command=command, comment='range_straddle_strategy_two')
        
        if STRATEGY_CONFIG['timeframe'] == 'H1':
            # Run every hour at minute 5 (5 minutes after candle close)
            job.setall('5 * * * *')
            logger.info("📅 Scheduled for H1: Every hour at minute 5")
            
        elif STRATEGY_CONFIG['timeframe'] == 'M15':
            # Run every 15 minutes at 1 minute past (1 minute after candle close)
            job.setall('1,16,31,46 * * * *')
            logger.info("📅 Scheduled for M15: Every 15 minutes at minute 1")
            
        elif STRATEGY_CONFIG['timeframe'] == 'M30':
            # Run every 30 minutes at 2 minutes past
            job.setall('2,32 * * * *')
            logger.info("📅 Scheduled for M30: Every 30 minutes at minute 2")
            
        elif STRATEGY_CONFIG['timeframe'] == 'H4':
            # Run every 4 hours at minute 5
            job.setall('5 */4 * * *')
            logger.info("📅 Scheduled for H4: Every 4 hours at minute 5")
            
        else:
            logger.error(f"❌ Unsupported timeframe: {STRATEGY_CONFIG['timeframe']}")
            return False
        
        # Write the cron job
        cron.write()
        logger.info(f"✅ Cron job scheduled successfully for {STRATEGY_CONFIG['timeframe']} strategy")
        
        # List current jobs
        logger.info("📋 Current cron jobs:")
        for job in cron:
            if job.comment == 'range_straddle_strategy_two':
                logger.info(f"   {job}")
        
        return True
        
    except ImportError:
        logger.error("❌ python-crontab not installed. Install with: pip install python-crontab")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to setup cron job: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Range Straddle Trading Strategy')
    parser.add_argument('--setup-cron', action='store_true', help='Setup cron job')
    parser.add_argument('--run-once', action='store_true', help='Run strategy once')
    
    args = parser.parse_args()
    
    if args.setup_cron:
        setup_cron_job()
    elif args.run_once:
        strat_one()
    else:
        # Default behavior - run the strategy
        strat_one()
