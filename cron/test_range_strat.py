import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from metatrader_mcp.server import * 

def run_comprehensive_strategy_tests(ctx: AppContext):
    """
    Run comprehensive Range Straddle strategy tests using REAL historical data via MCP tools
    - 1H timeframe: 2400 analysis period, 2 years of data
    - 15min timeframe: 4800 analysis period, 1 year of data
    """
    print("🚀 COMPREHENSIVE RANGE STRADDLE STRATEGY ANALYSIS - REAL DATA")
    print("=" * 70)
    print("📊 1H Timeframe: 2400 candle analysis period, 2 years historical data")
    print("📊 15min Timeframe: 4800 candle analysis period, 1 year historical data")
    print("=" * 70)
    
    # Test parameters - these are the actual strategy parameters, not made-up numbers
    symbols = ["USDJPY", "EURUSD", "AUDUSD"]
    timeframes = ["H1", "M15"]
    tp_sl_ratios = [1.0, 1.5, 2.0]  # These determine actual TP distance calculations
    
    # Strategy parameter combinations to test
    core_combinations = [
        {"percentile_num": 0.2, "num_candles": 8, "safety_factor": 0.2},
        {"percentile_num": 0.1, "num_candles": 8, "safety_factor": 0.2},
        {"percentile_num": 0.2, "num_candles": 12, "safety_factor": 0.2},
        {"percentile_num": 0.1, "num_candles": 12, "safety_factor": 0.2},
    ]
    
    all_results = []
    
    # Run tests for each combination
    for timeframe in timeframes:
        print(f"\n{'='*60}")
        print(f"⏰ TESTING TIMEFRAME: {timeframe}")
        print(f"{'='*60}")
        
        # Set analysis period and historical data requirements based on timeframe
        if timeframe == "H1":
            analysis_period = 2400  # 2400 candles analysis window
            total_candles_needed = 20000  # ~800 days of 1H data
            print(f"📈 Analysis Period: {analysis_period} candles")
            print(f"📊 Historical Data: {total_candles_needed} candles (~800 days)")
        else:  # M15
            analysis_period = 4800  # 4800 candles analysis window  
            total_candles_needed = 25000  # ~400 days of 15min data
            print(f"📈 Analysis Period: {analysis_period} candles")
            print(f"📊 Historical Data: {total_candles_needed} candles (~400 days)")
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"📈 TESTING {symbol} on {timeframe}")
            print(f"{'='*50}")
            
            for tp_sl_ratio in tp_sl_ratios:
                print(f"\n💰 TP:SL Ratio 1:{tp_sl_ratio}")
                print("-" * 30)
                
                for i, params in enumerate(core_combinations, 1):
                    print(f"\n🧪 Test {i}/4: {params}")
                    
                    result = real_historical_strategy_test(
                        symbol=symbol,
                        timeframe=timeframe,
                        tp_sl_ratio=tp_sl_ratio,
                        analysis_period=analysis_period,
                        total_candles_needed=total_candles_needed,
                        ctx=ctx,
                        **params
                    )
                    
                    if "error" not in result:
                        all_results.append(result)
                        
                        # Print immediate results for this combination
                        summary = result['summary']
                        print(f"    ✅ Opportunities: {summary['total_opportunities']}")
                        print(f"    ⚡ Triggered: {summary['trades_triggered']}")  
                        print(f"    🏆 Wins: {summary['winning_trades']}")
                        print(f"    📉 Losses: {summary['losing_trades']}")
                        print(f"    🎯 Win Rate: {summary['win_rate_percent']:.1f}%")
                        print(f"    📉 Loss Rate: {summary['loss_rate_percent']:.1f}%")
                        print(f"    ⚖️ Win/Loss Ratio: {summary['win_to_loss_ratio']:.2f}")
                        print(f"    💰 Expected Return: {summary['expected_return_per_trade']:.3f}")
                    else:
                        print(f"    ❌ Error: {result['error']}")
    
    # Analyze results
    analyze_comprehensive_results(all_results)
    analyze_timeframe_comparison(all_results)
    
    return all_results

def real_historical_strategy_test(symbol: str, timeframe: str, percentile_num: float, 
                                num_candles: int, safety_factor: float, tp_sl_ratio: float,
                                analysis_period: int, total_candles_needed: int, ctx=None) -> Dict:
    """
    Test Range Straddle strategy using REAL historical data via MCP tools
    No hardcoded numbers - purely deterministic based on actual price data
    """
    try:
        print(f"  Getting real historical data for {symbol} {timeframe}...")
        
        # Use actual MCP tool to get real historical data
        candles_df = get_candles_latest(
            ctx=ctx,
            symbol_name=symbol,
            timeframe=timeframe,
            count=total_candles_needed
        )
        # len(candles_df) here will be equal to total_candles_needed which is 20000 for H1 and 25000 for M15
        
        if candles_df.empty or len(candles_df) < analysis_period:
            return {"error": f"Insufficient historical data for {symbol} {timeframe}"}
        
        print(f"    Retrieved {len(candles_df)} real candles for analysis")
        
        # Convert time column and sort
        candles_df['time'] = pd.to_datetime(candles_df['time'])
        candles_df = candles_df.sort_values('time').reset_index(drop=True)
        
        # Calculate actual range for each candle
        candles_df['candle_range'] = candles_df['high'] - candles_df['low']
        print("candles_df handling complete!")
        
        # Strategy execution tracking
        total_opportunities = 0
        trades_triggered = 0
        winning_trades = 0
        losing_trades = 0
        pending_trades = 0
        ambiguous_trades = 0
        trade_details = []
        
        # Test strategy on each potential entry point
        # Calculate buffer needed for trade execution and outcome determination
        # Max lookforward: 7 days (trigger) + 7 days (outcome) = 14 days
        if timeframe == "H1":
            buffer_candles = 14 * 24  # 14 days * 24 hours (conservative estimate)
        elif timeframe == "M15":
            buffer_candles = 14 * 24 * 4  # 14 days * 24 hours * 4 (15min periods per hour)
        else:
            buffer_candles = 500  # Conservative default for other timeframes
            
        max_i = len(candles_df) - buffer_candles
        # print(f"max_i: {max_i}")
        triggered_trades = []
        print("Starting strategy execution for all opportunities...")
        for i in range(num_candles, max_i):  # Leave sufficient buffer for trade execution
            current_time = candles_df.iloc[i]['time']
            
            # Step 1: Calculate range of latest num_candles
            range_start_idx = i - num_candles + 1
            range_candles = candles_df.iloc[range_start_idx:i+1]
            # print(f"range_candles: {range_candles.head()}")
            range_high = range_candles['high'].max()
            range_low = range_candles['low'].min()
            range_width = range_high - range_low
            
            # Step 2: Calculate average candle height of current range
            current_avg_candle_height = range_candles['candle_range'].mean()
            
            # Step 3: Calculate percentile threshold using INDIVIDUAL candle heights
            # Get analysis window for percentile calculation
            analysis_start_idx = max(0, i - analysis_period)
            analysis_candles = candles_df.iloc[analysis_start_idx:i+1]
            # print(f"analysis_candles: {analysis_candles.head()}")
            # Only proceed if we have sufficient data points for reliable percentile
            if len(analysis_candles) >= 50:
                percentile_threshold = np.percentile(analysis_candles['candle_range'], percentile_num * 100)
            else:
                continue  # Skip if insufficient data for reliable percentile
            
            # Step 4: Check if average candle height qualifies (narrow + channel)
            if current_avg_candle_height <= percentile_threshold:
                # Use local function to check if range is a channel (avoids MCP serialization issues)
                is_channel = check_if_range_is_channel_local(range_candles)
                
                if is_channel:
                    total_opportunities += 1
                    
                    # Calculate trade levels based on real range data
                    buy_stop_price = range_high + (safety_factor * range_width)
                    sell_stop_price = range_low - (safety_factor * range_width)
                    
                    # Step 5: Check if trade would be triggered using real historical data
                    trigger_result = historical_test_get_position_trigger_date(
                        ctx=ctx,
                        symbol_name=symbol,
                        timeframe=timeframe,
                        order_placement_date=current_time.strftime('%Y-%m-%d %H:%M'),
                        position_type="OCO_STOP",
                        entry_price_buy=buy_stop_price,
                        entry_price_sell=sell_stop_price
                    )
                    
                    buy_triggered = trigger_result.get('buy') is not None
                    sell_triggered = trigger_result.get('sell') is not None
                    
                    if buy_triggered or sell_triggered:
                        # Step 5: Calculate TP/SL based on actual range and ratio
                        # Stop loss = 1 * range_width, Take profit = tp_sl_ratio * range_width
                        stop_loss_distance = range_width
                        take_profit_distance = tp_sl_ratio * range_width
                        
                        # Determine which side triggered first
                        if buy_triggered and sell_triggered:
                            trades_triggered += 2
                            buy_time = pd.to_datetime(trigger_result['buy'])
                            sell_time = pd.to_datetime(trigger_result['sell'])

                            triggered_trades.append({
                                "direction": "BUY",
                                "entry_price": buy_stop_price,
                                "trigger_time": buy_time,
                                "opportunity_time": current_time,
                                "range_width": range_width,
                                "percentile_threshold": percentile_threshold,
                                "take_profit": buy_stop_price + take_profit_distance,
                                "stop_loss": buy_stop_price - stop_loss_distance
                            })

                            triggered_trades.append({
                                "direction": "SELL",
                                "entry_price": sell_stop_price,
                                "trigger_time": sell_time,
                                "opportunity_time": current_time,
                                "range_width": range_width,
                                "percentile_threshold": percentile_threshold,
                                "take_profit": sell_stop_price - take_profit_distance,
                                "stop_loss": sell_stop_price + stop_loss_distance
                            })
                        elif buy_triggered:
                            trades_triggered += 1
                            buy_time = pd.to_datetime(trigger_result['buy'])
                            triggered_trades.append({
                                "direction": "BUY",
                                "entry_price": buy_stop_price,
                                "trigger_time": buy_time,
                                "opportunity_time": current_time,
                                "range_width": range_width,
                                "percentile_threshold": percentile_threshold,
                                "take_profit": buy_stop_price + take_profit_distance,
                                "stop_loss": buy_stop_price - stop_loss_distance
                            })
                        else:
                            trades_triggered += 1
                            sell_time = pd.to_datetime(trigger_result['sell'])
                            triggered_trades.append({
                                "direction": "SELL",
                                "entry_price": sell_stop_price,
                                "trigger_time": sell_time,
                                "opportunity_time": current_time,
                                "range_width": range_width,
                                "percentile_threshold": percentile_threshold,
                                "take_profit": sell_stop_price - take_profit_distance,
                                "stop_loss": sell_stop_price + stop_loss_distance
                            })                     

        print("Filtering triggered trades for overlaps...")              
        # Step 6: Filter triggered trades to remove overlapping positions
        # Remove trades that are too close to each other (adjacent candles)
        if triggered_trades:
            print(f"    🔄 Filtering {len(triggered_trades)} triggered trades for overlaps...")
            
            # Sort trades by trigger time to process chronologically
            triggered_trades.sort(key=lambda x: x['trigger_time'])
            
            # Define minimum time gap based on timeframe
            if timeframe == "H1":
                min_gap = timedelta(hours=1)  # 1 hour gap for H1
            elif timeframe == "M15":
                min_gap = timedelta(minutes=15)  # 15 minute gap for M15
            else:
                min_gap = timedelta(hours=1)  # Default 1 hour gap
            
            # Filter out trades that are too close to previous trades
            filtered_trades = []
            last_buy_trigger_time = None
            last_sell_trigger_time = None

            for trade in triggered_trades:
                current_trigger_time = trade['trigger_time']
                
                # Convert to pandas Timestamp if it's a string
                if isinstance(current_trigger_time, str):
                    current_trigger_time = pd.to_datetime(current_trigger_time)
                
                # Check if this trade is far enough from the last trade
                if trade['direction'] == "BUY":
                    if last_buy_trigger_time is None or (current_trigger_time - last_buy_trigger_time) >= min_gap:
                        filtered_trades.append(trade)
                        last_buy_trigger_time = current_trigger_time
                        print(f"    ✅ Keeping trade at {current_trigger_time} ({trade['direction']})")
                    else:
                        print(f"    ❌ Filtering out trade at {current_trigger_time} ({trade['direction']}) - too close to previous buy")
                elif trade['direction'] == "SELL":
                    if last_sell_trigger_time is None or (current_trigger_time - last_sell_trigger_time) >= min_gap:
                        filtered_trades.append(trade)
                        last_sell_trigger_time = current_trigger_time
                        print(f"    ✅ Keeping trade at {current_trigger_time} ({trade['direction']})")
                    else:
                        print(f"    ❌ Filtering out trade at {current_trigger_time} ({trade['direction']}) - too close to previous sell")
            
            print(f"    📊 Filtered trades: {len(triggered_trades)} → {len(filtered_trades)} (removed {len(triggered_trades) - len(filtered_trades)} overlapping)")
            triggered_trades = filtered_trades
        
        # Update trades_triggered count to reflect filtered trades
        trades_triggered = len(triggered_trades)
        
        # Step 7: Check actual trade outcome using real historical data in PARALLEL
        print("Checking actual trade outcome...")
        if triggered_trades:
            print(f"    🚀 Processing {len(triggered_trades)} trade outcomes in parallel...")
            
            # Prepare data for parallel processing
            trade_data_list = [(trade, symbol, timeframe, ctx) for trade in triggered_trades]
            
            # Use ThreadPoolExecutor for parallel I/O operations
            max_workers = min(8, len(triggered_trades))  # Limit to 8 concurrent threads
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all trade outcome requests
                future_to_trade = {
                    executor.submit(process_trade_outcome, trade_data): i 
                    for i, trade_data in enumerate(trade_data_list)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_trade):
                    trade_index = future_to_trade[future]
                    try:
                        result = future.result()
                        outcome = result['outcome']
                        outcome_date = result['outcome_date']
                        trade_details_data = result['trade_details']
                        
                        # Update counters
                        if outcome == "Win":
                            winning_trades += 1
                        elif outcome == "Lose":
                            losing_trades += 1
                        elif outcome == "Pending":
                            pending_trades += 1
                        elif outcome == "Ambiguous":
                            ambiguous_trades += 1

                        # Add outcome data to trade details
                        trade_details_data.update({
                            'outcome': outcome,
                            'outcome_date': outcome_date,
                            'tp_sl_ratio': tp_sl_ratio
                        })
                        trade_details.append(trade_details_data)
                        
                        print(f"    ✅ Trade {trade_index + 1}: {trade_details_data['direction']} @ {trade_details_data['entry_price']:.5f} -> {outcome}")
                        
                    except Exception as e:
                        print(f"    ❌ Error processing trade {trade_index + 1}: {e}")
            
            print(f"    🎯 Parallel processing completed for {len(triggered_trades)} trades")
        
        # Calculate final statistics
        win_rate = (winning_trades / (trades_triggered - pending_trades - ambiguous_trades) * 100) if (trades_triggered - pending_trades - ambiguous_trades) > 0 else 0
        loss_rate = (losing_trades / (trades_triggered - pending_trades - ambiguous_trades) * 100) if (trades_triggered - pending_trades - ambiguous_trades) > 0 else 0
        trigger_rate = (trades_triggered / total_opportunities * 100) if total_opportunities > 0 else 0
        
        # Calculate expected return per trade (based on actual TP:SL ratio)
        avg_win_amount = tp_sl_ratio  # Win amount = TP:SL ratio
        avg_loss_amount = 1.0         # Loss amount = 1 (stop loss distance)
        expected_return_per_trade = (win_rate/100 * avg_win_amount) - (loss_rate/100 * avg_loss_amount)
        win_to_loss_ratio = win_rate / loss_rate if loss_rate > 0 else 0
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'parameters': {
                'num_candles': num_candles,
                'analysis_period': analysis_period,
                'percentile_num': percentile_num,
                'safety_factor': safety_factor,
                'tp_sl_ratio': tp_sl_ratio
            },
            'summary': {
                'total_opportunities': total_opportunities,
                'trades_triggered': trades_triggered,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'pending_trades': pending_trades,
                'ambiguous_trades': ambiguous_trades,
                'win_rate_percent': round(win_rate, 2),
                'loss_rate_percent': round(loss_rate, 2),
                'trigger_rate_percent': round(trigger_rate, 2),
                'expected_return_per_trade': round(expected_return_per_trade, 3),
                'win_to_loss_ratio': round(win_to_loss_ratio, 2)
            },
            'trade_details': trade_details
        }
        
        print(f"    📊 Results: {total_opportunities} opportunities, {trades_triggered} triggered, "
              f"{winning_trades} wins, {losing_trades} losses, Win Rate: {win_rate:.1f}%, Loss Rate: {loss_rate:.1f}%, Win/Loss Ratio: {win_to_loss_ratio:.2f}")
        
        return results
        
    except Exception as e:
        print(f"    ❌ Error in real strategy test: {e}")
        return {"error": str(e)}

def analyze_comprehensive_results(all_results):
    """
    Analyze and compare results across all tests with detailed statistics for each combination
    """
    if not all_results:
        print("No results to analyze")
        return
    
    print("\n" + "=" * 100)
    print("📊 COMPREHENSIVE STRATEGY ANALYSIS - DETAILED STATISTICS")
    print("=" * 100)
    
    # Create detailed comparison DataFrame
    comparison_data = []
    for result in all_results:
        params = result['parameters']
        summary = result['summary']
        comparison_data.append({
            'Symbol': result['symbol'],
            'Timeframe': result['timeframe'],
            'TP_SL_Ratio': f"1:{params['tp_sl_ratio']}",
            'Percentile': params['percentile_num'],
            'Candles': params['num_candles'],
            'Safety': params['safety_factor'],
            'Analysis_Period': params['analysis_period'],
            'Opportunities': summary['total_opportunities'],
            'Triggered': summary['trades_triggered'],
            'Wins': summary['winning_trades'],
            'Losses': summary['losing_trades'],
            'Pending': summary['pending_trades'],
            'Ambiguous': summary['ambiguous_trades'],
            'Win_Rate_%': summary['win_rate_percent'],
            'Loss_Rate_%': summary['loss_rate_percent'],
            'Trigger_Rate_%': summary['trigger_rate_percent'],
            'Expected_Return': summary['expected_return_per_trade'],
            'Win_to_Loss_Ratio': summary['win_to_loss_ratio']
        })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv('comprehensive_results.csv', index=False)
    
    # Display results by currency pair and timeframe
    for symbol in sorted(df['Symbol'].unique()):
        for timeframe in sorted(df['Timeframe'].unique()):
            symbol_tf_data = df[(df['Symbol'] == symbol) & (df['Timeframe'] == timeframe)]
            if len(symbol_tf_data) > 0:
                print(f"\n📈 {symbol} {timeframe} RESULTS:")
                print("=" * 80)
                
                # Display detailed table for this symbol-timeframe combination
                display_cols = ['TP_SL_Ratio', 'Percentile', 'Candles', 'Safety', 
                              'Opportunities', 'Triggered', 'Wins', 'Losses', 'Pending', 'Ambiguous',
                              'Win_Rate_%', 'Loss_Rate_%', 'Trigger_Rate_%', 'Expected_Return', 'Win_to_Loss_Ratio']
                print(symbol_tf_data[display_cols].to_string(index=False))
                
                # Summary statistics for this symbol-timeframe
                print(f"\n📊 {symbol} {timeframe} SUMMARY:")
                print(f"   Average Opportunities: {symbol_tf_data['Opportunities'].mean():.1f}")
                print(f"   Average Win Rate: {symbol_tf_data['Win_Rate_%'].mean():.1f}%")
                print(f"   Average Loss Rate: {symbol_tf_data['Loss_Rate_%'].mean():.1f}%")
                print(f"   Average Win to Loss Ratio: {symbol_tf_data['Win_to_Loss_Ratio'].mean():.2f}")
                print(f"   Average Expected Return: {symbol_tf_data['Expected_Return'].mean():.3f}")
                print(f"   Best Expected Return: {symbol_tf_data['Expected_Return'].max():.3f}")
                print(f"   Total Trades Analyzed: {symbol_tf_data['Triggered'].sum()}")
    
    # Find best strategies overall
    print(f"\n🏆 TOP PERFORMING STRATEGIES OVERALL:")
    print("=" * 60)
    
    # Best by expected return
    if len(df) > 0:
        best_return = df.loc[df['Expected_Return'].idxmax()]
        print(f"🥇 BEST EXPECTED RETURN: {best_return['Expected_Return']:.3f}")
        print(f"   {best_return['Symbol']} {best_return['Timeframe']} | TP:SL {best_return['TP_SL_Ratio']} | "
              f"P={best_return['Percentile']} | C={best_return['Candles']} | S={best_return['Safety']}")
        print(f"   Opportunities: {best_return['Opportunities']} | Triggered: {best_return['Triggered']} | "
              f"Win Rate: {best_return['Win_Rate_%']:.1f}% | Loss Rate: {best_return['Loss_Rate_%']:.1f}% | Win to Loss Ratio: {best_return['Win_to_Loss_Ratio']:.2f}")
        
        # Best by win rate
        best_win_rate = df.loc[df['Win_Rate_%'].idxmax()]
        print(f"\n🥈 BEST WIN RATE: {best_win_rate['Win_Rate_%']:.1f}%")
        print(f"   {best_win_rate['Symbol']} {best_win_rate['Timeframe']} | TP:SL {best_win_rate['TP_SL_Ratio']} | "
              f"P={best_win_rate['Percentile']} | C={best_win_rate['Candles']} | S={best_win_rate['Safety']}")
        print(f"   Loss Rate: {best_win_rate['Loss_Rate_%']:.1f}% | Win/Loss Ratio: {best_win_rate['Win_to_Loss_Ratio']:.2f} | "
              f"Expected Return: {best_win_rate['Expected_Return']:.3f} | Triggered: {best_win_rate['Triggered']}")
        
        # Most active strategy
        most_active = df.loc[df['Triggered'].idxmax()]
        print(f"\n🥉 MOST ACTIVE STRATEGY: {most_active['Triggered']} trades triggered")
        print(f"   {most_active['Symbol']} {most_active['Timeframe']} | TP:SL {most_active['TP_SL_Ratio']} | "
              f"P={most_active['Percentile']} | C={most_active['Candles']} | S={most_active['Safety']}")
        print(f"   Win Rate: {most_active['Win_Rate_%']:.1f}% | Loss Rate: {most_active['Loss_Rate_%']:.1f}% | Win to Loss Ratio: {most_active['Win_to_Loss_Ratio']:.2f} | Expected Return: {most_active['Expected_Return']:.3f}")
    
    # Analysis by TP:SL ratio
    print(f"\n📊 PERFORMANCE BY TP:SL RATIO:")
    print("=" * 50)
    for ratio in sorted(df['TP_SL_Ratio'].unique()):
        ratio_data = df[df['TP_SL_Ratio'] == ratio]
        avg_win_rate = ratio_data['Win_Rate_%'].mean()
        avg_loss_rate = ratio_data['Loss_Rate_%'].mean()
        win_to_loss_ratio = avg_win_rate / avg_loss_rate if avg_loss_rate > 0 else 0
        avg_return = ratio_data['Expected_Return'].mean()
        total_trades = ratio_data['Triggered'].sum()
        print(f"{ratio}: Avg Win Rate = {avg_win_rate:.1f}%, Avg Loss Rate = {avg_loss_rate:.1f}%, Win to Loss Ratio = {win_to_loss_ratio:.2f}, Avg Expected Return = {avg_return:.3f}, Total Trades = {total_trades}")
    
    # Analysis by percentile
    print(f"\n📊 PERFORMANCE BY PERCENTILE THRESHOLD:")
    print("=" * 50)
    for percentile in sorted(df['Percentile'].unique()):
        percentile_data = df[df['Percentile'] == percentile]
        avg_opportunities = percentile_data['Opportunities'].mean()
        avg_win_rate = percentile_data['Win_Rate_%'].mean()
        avg_loss_rate = percentile_data['Loss_Rate_%'].mean()
        win_to_loss_ratio = avg_win_rate / avg_loss_rate if avg_loss_rate > 0 else 0
        avg_return = percentile_data['Expected_Return'].mean()
        print(f"P={percentile}: Avg Opportunities = {avg_opportunities:.1f}, Avg Win Rate = {avg_win_rate:.1f}%, Avg Loss Rate = {avg_loss_rate:.1f}%, Win to Loss Ratio = {win_to_loss_ratio:.2f}, Avg Expected Return = {avg_return:.3f}")
    
    return df

def analyze_timeframe_comparison(all_results):
    """
    Analyze performance differences between timeframes
    """
    if not all_results:
        return
    
    print("\n" + "=" * 80)
    print("⏰ TIMEFRAME COMPARISON ANALYSIS")
    print("=" * 80)
    
    df = pd.DataFrame([{
        'Symbol': result['symbol'],
        'Timeframe': result['timeframe'],
        'TP_SL_Ratio': f"1:{result['parameters']['tp_sl_ratio']}",
        'Percentile': result['parameters']['percentile_num'],
        'Candles': result['parameters']['num_candles'],
        'Opportunities': result['summary']['total_opportunities'],
        'Win_Rate_%': result['summary']['win_rate_percent'],
        'Loss_Rate_%': result['summary']['loss_rate_percent'],
        'Win_to_Loss_Ratio': result['summary']['win_to_loss_ratio'],
        'Expected_Return': result['summary']['expected_return_per_trade']
    } for result in all_results])
    
    # Compare timeframes
    for timeframe in df['Timeframe'].unique():
        tf_data = df[df['Timeframe'] == timeframe]
        avg_opportunities = tf_data['Opportunities'].mean()
        avg_win_rate = tf_data['Win_Rate_%'].mean()
        avg_loss_rate = tf_data['Loss_Rate_%'].mean()
        avg_win_to_loss_ratio = tf_data['Win_to_Loss_Ratio'].mean()
        avg_expected_return = tf_data['Expected_Return'].mean()
        
        print(f"\n📊 {timeframe} TIMEFRAME SUMMARY:")
        print(f"   Average Opportunities: {avg_opportunities:.1f}")
        print(f"   Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   Average Loss Rate: {avg_loss_rate:.1f}%")
        print(f"   Average Win to Loss Ratio: {avg_win_to_loss_ratio:.2f}")
        print(f"   Average Expected Return: {avg_expected_return:.3f}")
    
    # Best strategies per timeframe
    print(f"\n🏆 BEST STRATEGIES BY TIMEFRAME:")
    print("-" * 50)
    
    for timeframe in sorted(df['Timeframe'].unique()):
        tf_data = df[df['Timeframe'] == timeframe]
        if len(tf_data) > 0:
            best_return = tf_data.loc[tf_data['Expected_Return'].idxmax()]
            print(f"\n⏰ {timeframe} - BEST EXPECTED RETURN: {best_return['Expected_Return']:.3f}")
            print(f"   {best_return['Symbol']} | TP:SL {best_return['TP_SL_Ratio']} | "
                  f"P={best_return['Percentile']} | C={best_return['Candles']}")
            print(f"   Win Rate: {best_return['Win_Rate_%']:.1f}% | Loss Rate: {best_return['Loss_Rate_%']:.1f}% | "
                  f"Win/Loss Ratio: {best_return['Win_to_Loss_Ratio']:.2f}")
    
    # Direct timeframe comparison for same parameters
    print(f"\n🔄 HEAD-TO-HEAD TIMEFRAME COMPARISON:")
    print("-" * 60)
    
    # Group by symbol, ratio, and parameters to compare timeframes directly
    comparison_groups = df.groupby(['Symbol', 'TP_SL_Ratio', 'Percentile', 'Candles'])
    
    better_h1 = 0
    better_m15 = 0
    
    for group_key, group_data in comparison_groups:
        if len(group_data) == 2:  # Has both timeframes
            h1_data = group_data[group_data['Timeframe'] == 'H1']
            m15_data = group_data[group_data['Timeframe'] == 'M15']
            
            if len(h1_data) > 0 and len(m15_data) > 0:
                h1_return = h1_data['Expected_Return'].iloc[0]
                m15_return = m15_data['Expected_Return'].iloc[0]
                
                symbol, ratio, percentile, candles = group_key
                
                if h1_return > m15_return:
                    better_h1 += 1
                    winner = "H1"
                    difference = h1_return - m15_return
                else:
                    better_m15 += 1
                    winner = "M15"
                    difference = m15_return - h1_return
                
                print(f"{symbol} {ratio} P={percentile} C={candles}: {winner} wins by {difference:.3f}")
    
    print(f"\n📈 TIMEFRAME PERFORMANCE SUMMARY:")
    print(f"   H1 outperformed M15: {better_h1} times")
    print(f"   M15 outperformed H1: {better_m15} times")

def process_trade_outcome(trade_data):
    """
    Helper function to process a single trade outcome - designed for parallel execution
    """
    triggered_trade, symbol, timeframe, ctx = trade_data
    
    trigger_time = triggered_trade['trigger_time']
    direction = triggered_trade['direction']
    take_profit = triggered_trade['take_profit']
    stop_loss = triggered_trade['stop_loss']
    entry_price = triggered_trade['entry_price']
    opportunity_time = triggered_trade['opportunity_time']
    range_width = triggered_trade['range_width']
    percentile_threshold = triggered_trade['percentile_threshold']
    
    # Convert trigger_time to the format expected by MCP functions: 'YYYY-MM-DD HH:MM'
    if isinstance(trigger_time, pd.Timestamp):
        trigger_time_str = trigger_time.strftime('%Y-%m-%d %H:%M')
    elif isinstance(trigger_time, str):
        # Handle ISO format strings, convert to expected format
        try:
            # Parse the ISO string and convert to expected format
            parsed_time = pd.to_datetime(trigger_time)
            trigger_time_str = parsed_time.strftime('%Y-%m-%d %H:%M')
        except:
            trigger_time_str = str(trigger_time)
    else:
        trigger_time_str = str(trigger_time)
    
    # Get trade outcome via MCP call
    results = historical_test_get_position_outcome(
        ctx=ctx,
        symbol_name=symbol,
        timeframe=timeframe,
        order_trigger_date=trigger_time_str,
        position_type=direction,
        take_profit=take_profit,
        stop_loss=stop_loss
    )
    
    return {
        'outcome': results['outcome'],
        'outcome_date': results['outcome_date'],
        'trade_details': {
            'opportunity_time': opportunity_time.isoformat(),
            'trigger_time': trigger_time_str,
            'direction': direction,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'range_width': range_width,
            'percentile_threshold': percentile_threshold,
            'tp_sl_ratio': 0  # Will be updated in main function
        }
    }

def check_if_range_is_channel_local(candles_df: pd.DataFrame) -> bool:
    """
    Local function to check if the range is a channel using DataFrame directly.
    This avoids MCP serialization issues with pandas DataFrames.
    
    Args:
        candles_df (pd.DataFrame): DataFrame containing the candles for the range
        
    Returns:
        bool: True if the range is a channel, False otherwise
    """
    if candles_df.empty or len(candles_df) < 2:
        return False
    
    try:
        channel_range = candles_df["high"].max() - candles_df["low"].min()
        
        # Get top 2 peaks, but handle case where there might be fewer unique values
        two_top_peaks = candles_df.nlargest(2, "high")
        two_bottom_peaks = candles_df.nsmallest(2, "low")
        
        # Ensure we have at least 2 rows
        if len(two_top_peaks) < 2 or len(two_bottom_peaks) < 2:
            return False
        
        top_peaks_diff = two_top_peaks["high"].diff()
        bottom_peaks_diff = two_bottom_peaks["low"].diff()
        
        # Check bounds before accessing iloc[1]
        if len(top_peaks_diff) < 2 or len(bottom_peaks_diff) < 2:
            return False
            
        # Check if the differences are within the channel range threshold
        # print(f"top_peaks_diff: {top_peaks_diff}")
        # print(f"bottom_peaks_diff: {bottom_peaks_diff}")
        top_diff = abs(top_peaks_diff.iloc[1]) if not pd.isna(top_peaks_diff.iloc[1]) else 0
        bottom_diff = abs(bottom_peaks_diff.iloc[1]) if not pd.isna(bottom_peaks_diff.iloc[1]) else 0
        
        if top_diff <= channel_range * 0.2 and bottom_diff <= channel_range * 0.2:
            return True
        else:
            return False
            
    except (IndexError, KeyError, Exception) as e:
        # If any error occurs, default to False (not a channel)
        print(f"    ⚠️ Channel detection error: {e}, defaulting to False")
        return False

@dataclass
class AppContext:
	client: str

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Range Straddle Strategy Testing...")
    print("🔍 Using actual MCP tools and large historical datasets\n")
    
    # Run comprehensive strategy tests with large datasets
    print("📊 Running comprehensive strategy analysis...")
    client = initialize_client()
    ctx = AppContext(client=client)
    all_results = run_comprehensive_strategy_tests(ctx=ctx)
    
    print(f"\n\n🎯 COMPREHENSIVE HISTORICAL STRATEGY TESTING COMPLETED!")
    print("=" * 70)
    print("✅ All strategy combinations tested with real historical data")
    print("📊 Detailed statistics available above")
    print("🚀 No more simulation - 100% real market data analysis!")
