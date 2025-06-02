#!/usr/bin/env python3
"""
Daily Summary Report for Range Straddle Strategy
Generates a comprehensive daily report of strategy performance
"""

import os
import sys
from datetime import datetime, timedelta

# Import both log analyzers
sys.path.append(os.path.dirname(__file__))
from log_monitor_one import StrategyLogAnalyzer as StrategyOneAnalyzer
from log_monitor_two import StrategyLogAnalyzer as StrategyTwoAnalyzer

def generate_daily_summary():
    """Generate a comprehensive daily summary report for both strategies"""
    
    # Initialize analyzers for both strategies
    analyzer_one = StrategyOneAnalyzer()
    analyzer_two = StrategyTwoAnalyzer()
    
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 100)
    print(f"📊 DAILY STRATEGY SUMMARY REPORT - {today}")
    print("=" * 100)
    print("🎯 Analyzing both Strategy One (USDJPY H1) and Strategy Two (AUDUSD M15)")
    print("=" * 100)
    
    # Analyze Strategy One
    print(f"\n🔵 STRATEGY ONE ANALYSIS (USDJPY H1)")
    print("-" * 50)
    
    daily_executions_one = analyzer_one.get_latest_executions(24)
    
    if not daily_executions_one:
        print("❌ No Strategy One executions found")
        today_executions_one = []
    else:
        # Filter for today's executions
        today_executions_one = []
        for execution in daily_executions_one:
            try:
                exec_date = execution['start_time'].split(' ')[0]
                if exec_date == today:
                    today_executions_one.append(execution)
            except:
                continue
        
        print(f"📅 Strategy One Today: {len(today_executions_one)} executions")
        
        if today_executions_one:
            successful_today_one = sum(1 for e in today_executions_one if e['status'] == 'SUCCESS')
            trades_today_one = sum(1 for e in today_executions_one if e['trade_placed'])
            errors_today_one = sum(1 for e in today_executions_one if e['status'] == 'FAILED')
            
            print(f"✅ Successful: {successful_today_one}/{len(today_executions_one)}")
            print(f"💰 Trades placed: {trades_today_one}")
            print(f"❌ Failed: {errors_today_one}")
        else:
            print("📭 No executions found for today")
    
    # Analyze Strategy Two
    print(f"\n🟡 STRATEGY TWO ANALYSIS (AUDUSD M15)")
    print("-" * 50)
    
    daily_executions_two = analyzer_two.get_latest_executions(96)  # 96 = 24 hours * 4 (15-min intervals)
    
    if not daily_executions_two:
        print("❌ No Strategy Two executions found")
        today_executions_two = []
    else:
        # Filter for today's executions
        today_executions_two = []
        for execution in daily_executions_two:
            try:
                exec_date = execution['start_time'].split(' ')[0]
                if exec_date == today:
                    today_executions_two.append(execution)
            except:
                continue
        
        print(f"📅 Strategy Two Today: {len(today_executions_two)} executions")
        
        if today_executions_two:
            successful_today_two = sum(1 for e in today_executions_two if e['status'] == 'SUCCESS')
            trades_today_two = sum(1 for e in today_executions_two if e['trade_placed'])
            errors_today_two = sum(1 for e in today_executions_two if e['status'] == 'FAILED')
            
            print(f"✅ Successful: {successful_today_two}/{len(today_executions_two)}")
            print(f"💰 Trades placed: {trades_today_two}")
            print(f"❌ Failed: {errors_today_two}")
        else:
            print("📭 No executions found for today")
    
    # Combined Summary
    print(f"\n📊 COMBINED DAILY SUMMARY")
    print("-" * 30)
    
    total_today = len(today_executions_one) + len(today_executions_two)
    total_trades = (trades_today_one if 'trades_today_one' in locals() else 0) + (trades_today_two if 'trades_today_two' in locals() else 0)
    total_successful = (successful_today_one if 'successful_today_one' in locals() else 0) + (successful_today_two if 'successful_today_two' in locals() else 0)
    total_errors = (errors_today_one if 'errors_today_one' in locals() else 0) + (errors_today_two if 'errors_today_two' in locals() else 0)
    
    print(f"🎯 Total executions today: {total_today}")
    print(f"💰 Total trades placed: {total_trades}")
    print(f"✅ Total successful: {total_successful}")
    print(f"❌ Total errors: {total_errors}")
    
    if total_trades > 0:
        print(f"\n🎯 ALL TRADES PLACED TODAY:")
        print("-" * 25)
        trade_count = 1
        
        # List Strategy One trades
        if 'today_executions_one' in locals():
            for execution in today_executions_one:
                if execution['trade_placed']:
                    print(f"{trade_count}. {execution['start_time']} - Strategy One: {execution['symbol']} {execution.get('timeframe', 'N/A')}")
                    trade_count += 1
        
        # List Strategy Two trades
        if 'today_executions_two' in locals():
            for execution in today_executions_two:
                if execution['trade_placed']:
                    print(f"{trade_count}. {execution['start_time']} - Strategy Two: {execution['symbol']} {execution.get('timeframe', 'N/A')}")
                    trade_count += 1
    
    # Detailed condition analysis for both strategies
    print(f"\n🔍 CONDITION FAILURE ANALYSIS")
    print("-" * 40)
    
    if daily_executions_one:
        print(f"\n🔵 Strategy One (USDJPY H1) - Last 7 days:")
        weekly_executions_one = analyzer_one.get_latest_executions(168)  # 7 * 24 hours
        analyzer_one.analyze_condition_failures(len(weekly_executions_one))
    
    if daily_executions_two:
        print(f"\n🟡 Strategy Two (AUDUSD M15) - Last 7 days:")
        weekly_executions_two = analyzer_two.get_latest_executions(672)  # 7 * 24 * 4 (15-min intervals)
        analyzer_two.analyze_condition_failures(len(weekly_executions_two))
    
    # Performance trends
    print(f"\n📈 PERFORMANCE TRENDS")
    print("-" * 25)
    
    # Strategy One trends
    if len(daily_executions_one) >= 7:
        last_week_one = daily_executions_one[-168:] if len(daily_executions_one) >= 168 else daily_executions_one
        weekly_trades_one = sum(1 for e in last_week_one if e['trade_placed'])
        weekly_success_one = sum(1 for e in last_week_one if e['status'] == 'SUCCESS')
        
        print(f"🔵 Strategy One (Last 7 days):")
        print(f"   Total executions: {len(last_week_one)}")
        print(f"   Success rate: {weekly_success_one/len(last_week_one)*100:.1f}%")
        print(f"   Trade frequency: {weekly_trades_one/len(last_week_one)*100:.1f}%")
        print(f"   Avg trades per day: {weekly_trades_one/7:.1f}")
    
    # Strategy Two trends
    if len(daily_executions_two) >= 7:
        last_week_two = daily_executions_two[-672:] if len(daily_executions_two) >= 672 else daily_executions_two
        weekly_trades_two = sum(1 for e in last_week_two if e['trade_placed'])
        weekly_success_two = sum(1 for e in last_week_two if e['status'] == 'SUCCESS')
        
        print(f"🟡 Strategy Two (Last 7 days):")
        print(f"   Total executions: {len(last_week_two)}")
        print(f"   Success rate: {weekly_success_two/len(last_week_two)*100:.1f}%")
        print(f"   Trade frequency: {weekly_trades_two/len(last_week_two)*100:.1f}%")
        print(f"   Avg trades per day: {weekly_trades_two/7:.1f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    # Strategy One recommendations
    if 'today_executions_one' in locals() and len(today_executions_one) > 0:
        narrow_failures_one = sum(1 for e in today_executions_one 
                                if not e['trade_placed'] and not e['conditions'].get('narrow', True))
        channel_failures_one = sum(1 for e in today_executions_one 
                                 if not e['trade_placed'] and not e['conditions'].get('channel', True))
        
        print("🔵 Strategy One (USDJPY H1):")
        if narrow_failures_one > len(today_executions_one) * 0.8:
            print("   📋 Consider adjusting percentile_num - narrow condition failing frequently")
        if channel_failures_one > len(today_executions_one) * 0.5:
            print("   📋 Consider reviewing channel detection logic")
        if 'trades_today_one' in locals() and trades_today_one == 0:
            print("   📋 No trades today - review market conditions and strategy parameters")
    
    # Strategy Two recommendations
    if 'today_executions_two' in locals() and len(today_executions_two) > 0:
        narrow_failures_two = sum(1 for e in today_executions_two 
                                if not e['trade_placed'] and not e['conditions'].get('narrow', True))
        channel_failures_two = sum(1 for e in today_executions_two 
                                 if not e['trade_placed'] and not e['conditions'].get('channel', True))
        
        print("🟡 Strategy Two (AUDUSD M15):")
        if narrow_failures_two > len(today_executions_two) * 0.8:
            print("   📋 Consider adjusting percentile_num - narrow condition failing frequently")
        if channel_failures_two > len(today_executions_two) * 0.5:
            print("   📋 Consider reviewing channel detection logic")
        if 'trades_today_two' in locals() and trades_today_two == 0:
            print("   📋 No trades today - review market conditions and strategy parameters")
    
    # Combined recommendations
    if total_trades > 8:
        print("⚠️ High total trade frequency today - monitor overall risk management")
    elif total_trades == 0:
        print("📋 No trades from either strategy today - consider market conditions")
    
    print(f"\n⏰ Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

def save_daily_report():
    """Save daily report to file"""
    today = datetime.now().strftime("%Y-%m-%d")
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"daily_summary_{today}.txt")
    
    # Redirect stdout to file
    import sys
    original_stdout = sys.stdout
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            generate_daily_summary()
        
        sys.stdout = original_stdout
        print(f"📄 Daily report saved to: {report_file}")
        
    except Exception as e:
        sys.stdout = original_stdout
        print(f"❌ Error saving report: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate daily strategy summary for both strategies')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    
    args = parser.parse_args()
    
    if args.save:
        save_daily_report()
    else:
        generate_daily_summary() 