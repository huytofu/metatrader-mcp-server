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
from logging.handlers import RotatingFileHandler
import traceback
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils import check_no_pending_orders, calculate_position_size, setup_windows_task_with_logon_options, check_if_range_is_channel_local
from metatrader_mcp.server import *
from dataclasses import dataclass

# Enhanced logging configuration with rotation
def setup_logging():
    """Setup enhanced logging with rotation and detailed formatting"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Main strategy log with rotation - separate file for strat_three
    log_file = os.path.join(log_dir, 'range_straddle_strategy_three.log')
    
    # Create formatter with more details
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup rotating file handler (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Ensure UTF-8 encoding for file
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler with fallback encoding
    console_handler = logging.StreamHandler()
    
    # Create a simpler formatter for console (no emojis on Windows)
    import platform
    if platform.system().lower() == 'windows':
        # Windows console formatter without emojis
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Filter out emoji characters for console on Windows
        class NoEmojiFilter(logging.Filter):
            def filter(self, record):
                # Replace common emojis with text equivalents for console
                if hasattr(record, 'msg'):
                    msg = str(record.msg)
                    msg = msg.replace('🚀', '[START]')
                    msg = msg.replace('🏁', '[END]')
                    msg = msg.replace('✅', '[OK]')
                    msg = msg.replace('❌', '[ERROR]')
                    msg = msg.replace('⚠️', '[WARN]')
                    msg = msg.replace('📊', '[DATA]')
                    msg = msg.replace('📅', '[TIME]')
                    msg = msg.replace('💎', '[SYMBOL]')
                    msg = msg.replace('⏰', '[TIMEFRAME]')
                    msg = msg.replace('📋', '[STATUS]')
                    msg = msg.replace('💰', '[TRADE]')
                    msg = msg.replace('⏸️', '[NO_TRADE]')
                    msg = msg.replace('📝', '[REASON]')
                    msg = msg.replace('🧪', '[TEST]')
                    msg = msg.replace('🔌', '[CONNECT]')
                    msg = msg.replace('🔗', '[STEP]')
                    msg = msg.replace('🏪', '[MARKET]')
                    msg = msg.replace('📈', '[BUY]')
                    msg = msg.replace('📉', '[SELL]')
                    msg = msg.replace('🔍', '[ANALYZE]')
                    msg = msg.replace('📏', '[RANGE]')
                    msg = msg.replace('🎯', '[TARGET]')
                    msg = msg.replace('💼', '[RISK]')
                    msg = msg.replace('🎁', '[REWARD]')
                    msg = msg.replace('⚖️', '[RATIO]')
                    msg = msg.replace('🕐', '[HOUR]')
                    msg = msg.replace('🔄', '[CANCEL]')
                    msg = msg.replace('📤', '[PLACE]')
                    msg = msg.replace('🛑', '[SKIP]')
                    record.msg = msg
                return True
        
        console_handler.addFilter(NoEmojiFilter())
    else:
        console_formatter = formatter
    
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup logging
logger = setup_logging()

# STRATEGY CONFIGURATION
# These settings should be updated based on your historical test results
STRATEGY_CONFIG = {
    'symbol': 'EURUSD',  # Best performing pair from historical tests
    'timeframe': 'M15',   # Best performing timeframe
    'percentile_num': 0.1,  # Best performing percentile threshold (10% or 20%)
    'percentile_larger_num': 0.2,  # Best performing percentile larger threshold (20% or 30%)
    'num_candles': 12,    # Best performing range candles (8 or 12)
    'safety_factor': 0.2,  # Entry buffer
    'tp_sl_ratio': 2.0,  # Best performing TP:SL ratio (1.0, 1.5, or 2.0)
    'analysis_period': 2400,  # Analysis window for percentile calculation
    'risk_per_trade': 0.01,  # 1% risk per trade
    'max_spread_pips': 1,  # Maximum spread to allow trades
    'min_range_pips': 5,  # Minimum range width in pips
    'max_daily_trades': 4,  # Maximum trades per day
    'range_width_threshold_factor': 4,  # Factor for range width threshold comparison
    'range_width_larger_threshold_factor': 3,  # Factor for larger range width threshold comparison
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
        self.execution_id = None
        
    def log_execution_start(self):
        """Log the start of strategy execution with detailed context"""
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("=" * 80)
        logger.info(f"🚀 STRATEGY EXECUTION START - ID: {self.execution_id}")
        logger.info("=" * 80)
        logger.info(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🎯 Symbol: {self.config['symbol']}")
        logger.info(f"⏰ Timeframe: {self.config['timeframe']}")
        logger.info(f"📊 Configuration:")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")
        logger.info("-" * 80)
        
    def log_execution_end(self, success, reason=""):
        """Log the end of strategy execution"""
        status = "SUCCESS" if success else "FAILED"
        logger.info("-" * 80)
        logger.info(f"🏁 STRATEGY EXECUTION END - ID: {self.execution_id} - {status}")
        if reason:
            logger.info(f"📝 Reason: {reason}")
        logger.info("=" * 80)
        logger.info("")  # Empty line for readability
        
    def initialize(self):
        """Initialize MT5 connection with detailed logging"""
        try:
            logger.info("🔌 Initializing MT5 connection...")
            
            client = initialize_client()
            self.ctx = AppContext(client=client)
            
            # Test connection by getting account info
            logger.info("🧪 Testing MT5 connection...")
            account_info = get_account_info(ctx=self.ctx)
            
            if account_info:
                logger.info("✅ MT5 connection successful")
                logger.info(f"💰 Account Info:")
                logger.info(f"   Balance: {account_info.get('balance', 'N/A')}")
                logger.info(f"   Equity: {account_info.get('equity', 'N/A')}")
                logger.info(f"   Currency: {account_info.get('currency', 'N/A')}")
                logger.info(f"   Leverage: {account_info.get('leverage', 'N/A')}")
                return True
            else:
                logger.error("❌ Failed to get account info - connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 connection failed: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    def check_market_conditions(self):
        """Check if market is open and conditions are suitable for trading"""
        try:
            logger.info("🏪 Checking market conditions...")
            
            # Check if market is open
            current_time = datetime.now()
            logger.info(f"📅 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %A')}")
            
            # Reset daily trade count if new day
            if self.last_trade_date != current_time.date():
                old_count = self.daily_trades_count
                self.daily_trades_count = 0
                self.last_trade_date = current_time.date()
                logger.info(f"📅 New day detected - resetting trade count from {old_count} to 0")
            
            # Check daily trade limit
            logger.info(f"📊 Daily trades: {self.daily_trades_count}/{self.config['max_daily_trades']}")
            
            if self.daily_trades_count >= self.config['max_daily_trades']:
                logger.warning(f"⏸️ Daily trade limit reached ({self.daily_trades_count}/{self.config['max_daily_trades']})")
                logger.info("📋 Market conditions check: FAILED - Daily limit reached")
                return False
            
            # Check market hours (forex is generally 24/5)
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            if weekday >= 5:  # Saturday or Sunday
                logger.warning(f"⏸️ Weekend detected (weekday: {weekday}) - market likely closed")
                logger.info("📋 Market conditions check: FAILED - Weekend")
                return False
            
            # Check if it's within reasonable trading hours (optional)
            hour = current_time.hour
            logger.info(f"🕐 Current hour: {hour}")
            
            logger.info("✅ Market conditions check: PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking market conditions: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    def get_market_data(self):
        """Retrieve latest market data with detailed logging"""
        try:
            total_candles_needed = self.config['analysis_period'] + self.config['num_candles'] + 10
            
            logger.info(f"📊 Requesting market data...")
            logger.info(f"   Symbol: {self.config['symbol']}")
            logger.info(f"   Timeframe: {self.config['timeframe']}")
            logger.info(f"   Candles needed: {total_candles_needed}")
            
            candles_df = get_candles_latest(
                ctx=self.ctx,
                symbol_name=self.config['symbol'],
                timeframe=self.config['timeframe'],
                count=total_candles_needed
            )
            
            if candles_df is None or candles_df.empty:
                logger.error(f"❌ No data returned for {self.config['symbol']}")
                return None
                
            logger.info(f"📈 Retrieved {len(candles_df)} candles")
            
            if len(candles_df) < self.config['analysis_period']:
                logger.error(f"❌ Insufficient data: got {len(candles_df)}, need {self.config['analysis_period']}")
                return None
            
            # Prepare data
            candles_df['time'] = pd.to_datetime(candles_df['time'])
            candles_df = candles_df.sort_values('time').reset_index(drop=True)
            candles_df['candle_range'] = candles_df['high'] - candles_df['low']
            
            # Log latest candle info
            latest_candle = candles_df.iloc[-1]
            logger.info(f"📊 Latest candle ({latest_candle['time']}):")
            logger.info(f"   OHLC: {latest_candle['open']:.5f} | {latest_candle['high']:.5f} | {latest_candle['low']:.5f} | {latest_candle['close']:.5f}")
            logger.info(f"   Range: {latest_candle['candle_range']:.5f}")
            
            logger.info(f"✅ Market data retrieved successfully")
            return candles_df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving market data: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return None
    
    def analyze_range_opportunity(self, candles_df):
        """Analyze current range for trading opportunity with detailed logging"""
        try:
            logger.info("🔍 Starting range analysis...")
            
            # Get latest complete candle (not current forming candle)
            latest_idx = len(candles_df) - 2  # -2 to get last complete candle
            range_start_idx = latest_idx - self.config['num_candles'] + 1
            range_candles = candles_df.iloc[range_start_idx:latest_idx+1]
            
            logger.info(f"📊 Analyzing range from index {range_start_idx} to {latest_idx}")
            logger.info(f"📊 Range candles: {len(range_candles)} candles")
            
            # Calculate range metrics
            range_high = range_candles['high'].max()
            range_low = range_candles['low'].min()
            range_width = range_high - range_low
            current_avg_candle_height = range_candles['candle_range'].mean()
            
            logger.info(f"📏 Range metrics:")
            logger.info(f"   High: {range_high:.5f}")
            logger.info(f"   Low: {range_low:.5f}")
            logger.info(f"   Width: {range_width:.5f}")
            logger.info(f"   Avg candle height: {current_avg_candle_height:.5f}")
            
            # Calculate percentile threshold
            analysis_start_idx = max(0, latest_idx - self.config['analysis_period'])
            analysis_candles = candles_df.iloc[analysis_start_idx:latest_idx+1]
            percentile_threshold = np.percentile(
                analysis_candles['candle_range'], 
                self.config['percentile_num'] * 100
            )
            percentile_larger_threshold = np.percentile(
                analysis_candles['candle_range'], 
                self.config['percentile_larger_num'] * 100
            )
            
            logger.info(f"📊 Percentile analysis:")
            logger.info(f"   Analysis period: {len(analysis_candles)} candles")
            logger.info(f"   Percentile: {self.config['percentile_num'] * 100}%")
            logger.info(f"   Percentile larger: {self.config['percentile_larger_num'] * 100}%")
            logger.info(f"   Threshold: {percentile_threshold:.5f}")
            logger.info(f"   Threshold larger: {percentile_larger_threshold:.5f}")
            
            # Check conditions
            is_narrow = (current_avg_candle_height <= percentile_threshold) or \
                        (range_width <= percentile_threshold * self.config['range_width_threshold_factor']) or \
                        (range_width <= percentile_larger_threshold * self.config['range_width_larger_threshold_factor'])
            # if range width is less than or equal to range_width_threshold_factor times the percentile threshold, or range_width_larger_threshold_factor times the percentile larger threshold, then it is a narrow range
            is_channel = check_if_range_is_channel_local(range_candles)
            
            # Convert range to pips (for EURUSD, 1 pip = 0.0001)
            pip_multiplier = 100 if 'JPY' in self.config['symbol'] else 10000
            range_pips = range_width * pip_multiplier
            min_range_met = range_pips >= self.config['min_range_pips']
            
            current_time = candles_df.iloc[latest_idx]['time']
            
            logger.info(f"🧪 Condition checks:")
            logger.info(f"   Narrow condition: {is_narrow} (avg height {current_avg_candle_height:.5f} <= threshold {percentile_threshold:.5f}) or (range width {range_width:.5f} <= threshold*{self.config['range_width_threshold_factor']} {percentile_threshold * self.config['range_width_threshold_factor']:.5f}) or (range width {range_width:.5f} <= larger threshold*{self.config['range_width_larger_threshold_factor']} {percentile_larger_threshold * self.config['range_width_larger_threshold_factor']:.5f})")
            logger.info(f"   Channel condition: {is_channel}")
            logger.info(f"   Min range condition: {min_range_met} (range {range_pips:.1f} pips >= min {self.config['min_range_pips']} pips)")
            
            # Log detailed analysis results
            logger.info(f"📋 ANALYSIS SUMMARY:")
            logger.info(f"   Time: {current_time}")
            logger.info(f"   Symbol: {self.config['symbol']} {self.config['timeframe']}")
            logger.info(f"   Range: {range_width:.5f} ({range_pips:.1f} pips)")
            logger.info(f"   Conditions: Narrow={is_narrow} | Channel={is_channel} | MinRange={min_range_met}")
            
            all_conditions_met = is_narrow and is_channel and min_range_met
            
            if all_conditions_met:
                logger.info(f"✅ TRADE OPPORTUNITY DETECTED!")
                logger.info(f"   All conditions satisfied for trade setup")
                
                return {
                    'signal': True,
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_width': range_width,
                    'range_pips': range_pips,
                    'current_time': current_time,
                    'conditions': {
                        'is_narrow': is_narrow,
                        'is_channel': is_channel,
                        'min_range_met': min_range_met,
                        'avg_candle_height': current_avg_candle_height,
                        'percentile_threshold': percentile_threshold
                    }
                }
            else:
                failed_conditions = []
                if not is_narrow:
                    failed_conditions.append(f"Range is not Narrow (avg height {current_avg_candle_height:.5f} > threshold {percentile_threshold:.5f})")
                    failed_conditions.append(f"Range is not Narrow (range width {range_width:.5f} > threshold*{self.config['range_width_threshold_factor']} {percentile_threshold * self.config['range_width_threshold_factor']:.5f})")
                    failed_conditions.append(f"Range is not Narrow (range width {range_width:.5f} > larger threshold*{self.config['range_width_larger_threshold_factor']} {percentile_larger_threshold * self.config['range_width_larger_threshold_factor']:.5f})")
                if not is_channel:
                    failed_conditions.append("Range is not a Channel (range not detected as channel)")
                if not min_range_met:
                    failed_conditions.append(f"Range is too small (range {range_pips:.1f} pips < min {self.config['min_range_pips']} pips)")
                
                logger.info(f"⏸️ NO TRADE OPPORTUNITY")
                logger.info(f"   Failed conditions: {', '.join(failed_conditions)}")
                
                return {
                    'signal': False,
                    'failed_conditions': failed_conditions,
                    'conditions': {
                        'is_narrow': is_narrow,
                        'is_channel': is_channel,
                        'min_range_met': min_range_met,
                        'avg_candle_height': current_avg_candle_height,
                        'percentile_threshold': percentile_threshold
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error analyzing range opportunity: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return {'signal': False, 'error': str(e)}
    
    def place_oco_trades(self, opportunity_data):
        """Place OCO (One Cancels Other) trades with comprehensive logging"""
        try:
            logger.info(f"🚀 INITIATING TRADE PLACEMENT")
            logger.info(f"=" * 60)
            
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
                logger.error(f"❌ Error checking existing positions: {str(e)}")
                logger.error(f"🔍 Error details: {traceback.format_exc()}")
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
            lot_size = calculate_position_size(self.ctx, self.config['symbol'], stop_loss_distance, self.config['risk_per_trade'])
            
            # Convert to pips for logging
            pip_multiplier = 100 if 'JPY' in self.config['symbol'] else 10000
            
            logger.info(f"📊 TRADE CALCULATION SUMMARY:")
            logger.info(f"   Range: {range_low:.5f} - {range_high:.5f} (width: {range_width:.5f})")
            logger.info(f"   Safety factor: {self.config['safety_factor']}")
            logger.info(f"   TP:SL ratio: {self.config['tp_sl_ratio']}")
            logger.info(f"   Position size: {lot_size} lots")
            logger.info(f"")
            logger.info(f"📈 BUY STOP ORDER:")
            logger.info(f"   Entry: {buy_stop_price:.5f}")
            logger.info(f"   Take Profit: {buy_tp:.5f} (+{take_profit_distance * pip_multiplier:.1f} pips)")
            logger.info(f"   Stop Loss: {buy_sl:.5f} (-{stop_loss_distance * pip_multiplier:.1f} pips)")
            logger.info(f"")
            logger.info(f"📉 SELL STOP ORDER:")
            logger.info(f"   Entry: {sell_stop_price:.5f}")
            logger.info(f"   Take Profit: {sell_tp:.5f} (+{take_profit_distance * pip_multiplier:.1f} pips)")
            logger.info(f"   Stop Loss: {sell_sl:.5f} (-{stop_loss_distance * pip_multiplier:.1f} pips)")
            logger.info(f"")
            
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
            
            logger.info(f"📋 BUY order result: {buy_order_result}")
            
            if buy_order_result.get('error', False):
                logger.error(f"❌ Failed to place BUY order: {buy_order_result.get('message', 'Unknown error')}")
                return False
            else:
                buy_order_id = buy_order_result.get('data', {}).order if 'data' in buy_order_result else 'N/A'
                logger.info(f"✅ BUY order placed successfully - Order ID: {buy_order_id}")
            
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
            
            logger.info(f"📋 SELL order result: {sell_order_result}")
            
            if sell_order_result.get('error', False):
                logger.error(f"❌ Failed to place SELL order: {sell_order_result.get('message', 'Unknown error')}")
                # Cancel the BUY order if SELL order fails
                if 'data' in buy_order_result and hasattr(buy_order_result['data'], 'order'):
                    try:
                        logger.info(f"🔄 Attempting to cancel BUY order due to SELL order failure...")
                        cancel_result = cancel_pending_order(ctx=self.ctx, id=buy_order_result['data'].order)
                        logger.info(f"🔄 BUY order cancelled: {cancel_result}")
                    except Exception as e:
                        logger.error(f"❌ Failed to cancel BUY order: {str(e)}")
                return False
            else:
                sell_order_id = sell_order_result.get('data', {}).order if 'data' in sell_order_result else 'N/A'
                logger.info(f"✅ SELL order placed successfully - Order ID: {sell_order_id}")
            
            # Log comprehensive trade summary
            logger.info(f"")
            logger.info(f"🎯 OCO TRADE SETUP COMPLETED SUCCESSFULLY!")
            logger.info(f"=" * 60)
            logger.info(f"📅 Time: {opportunity_data['current_time']}")
            logger.info(f"💎 Symbol: {self.config['symbol']}")
            logger.info(f"⏰ Timeframe: {self.config['timeframe']}")
            logger.info(f"📈 BUY Order ID: {buy_order_id}")
            logger.info(f"📉 SELL Order ID: {sell_order_id}")
            logger.info(f"💼 Risk: {range_width * pip_multiplier:.1f} pips")
            logger.info(f"🎁 Reward: {take_profit_distance * pip_multiplier:.1f} pips")
            logger.info(f"⚖️ Risk:Reward = 1:{self.config['tp_sl_ratio']}")
            logger.info(f"💰 Position Size: {lot_size} lots")
            logger.info(f"📊 Trade #{self.daily_trades_count + 1} of max {self.config['max_daily_trades']} today")
            logger.info(f"=" * 60)
            
            self.daily_trades_count += 1
            logger.info(f"📈 Updated daily trade count: {self.daily_trades_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing OCO trades: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    def run_strategy(self):
        """Main strategy execution function with comprehensive logging"""
        self.log_execution_start()
        
        try:
            # Initialize connection
            logger.info("🔗 Step 1: Initializing MT5 connection...")
            if not self.initialize():
                self.log_execution_end(False, "MT5 connection failed")
                return False
            
            # Check market conditions
            logger.info("🏪 Step 2: Checking market conditions...")
            if not self.check_market_conditions():
                self.log_execution_end(False, "Market conditions not suitable")
                return False
            
            # Get market data
            logger.info("📊 Step 3: Retrieving market data...")
            candles_df = self.get_market_data()
            if candles_df is None:
                self.log_execution_end(False, "Failed to retrieve market data")
                return False
            
            # Analyze for opportunities
            logger.info("🔍 Step 4: Analyzing range opportunities...")
            opportunity = self.analyze_range_opportunity(candles_df)
            
            if opportunity['signal']:
                # Place trades
                logger.info("🚀 Step 5: Placing trades...")
                if check_no_pending_orders(self.ctx, self.config['symbol']):
                    success = self.place_oco_trades(opportunity)
                    if success:
                        self.log_execution_end(True, "Trade placed successfully")
                    else:
                        self.log_execution_end(False, "Failed to place trades")
                    return success
                else:
                    self.log_execution_end(False, "Failed to place trades - Pending orders found")
                    return False
            else:
                reason = f"No trading opportunity - {', '.join(opportunity.get('failed_conditions', ['Unknown reasons']))}"
                self.log_execution_end(True, reason)
                return True
                
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {str(e)}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            self.log_execution_end(False, f"Exception: {str(e)}")
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
    
    # Get the absolute path to this script
    script_path = os.path.abspath(__file__)
    
    # Task name - make it unique for strat_three
    task_name = f"RangeStraddleStrategy_{STRATEGY_CONFIG['symbol']}_{STRATEGY_CONFIG['timeframe']}_Three"
    
    return setup_windows_task_with_logon_options(
        script_path=script_path,
        task_name=task_name, 
        timeframe=STRATEGY_CONFIG['timeframe'],
        logger=logger
    )

def setup_unix_cron():
    """Set up Unix/Linux cron job"""
    try:
        cron = CronTab(user=True)  # Use current user instead of root
        
        # Remove existing jobs for this strategy
        cron.remove_all(comment='range_straddle_strategy_three')
        
        # Get the absolute path to this script
        script_path = os.path.abspath(__file__)
        project_path = os.path.dirname(os.path.dirname(script_path))
        
        # Create command
        command = f"cd {project_path} && python {script_path} --run-once"
        
        # Create new job based on timeframe
        job = cron.new(command=command, comment='range_straddle_strategy_three')
        
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
            if job.comment == 'range_straddle_strategy_three':
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
