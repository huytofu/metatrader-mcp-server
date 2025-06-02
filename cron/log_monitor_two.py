#!/usr/bin/env python3
"""
Log Monitor and Analysis Tool for Range Straddle Strategy Two (AUDUSD M15)
Helps parse and analyze strategy execution logs
"""

import os
import re
import pandas as pd
from datetime import datetime, timedelta
import argparse
import json

class StrategyLogAnalyzer:
    """Analyze strategy execution logs for strat_two"""
    
    def __init__(self, log_dir=None):
        if log_dir is None:
            self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        else:
            self.log_dir = log_dir
        
        self.log_file = os.path.join(self.log_dir, 'range_straddle_strategy_two.log')
        
    def get_latest_executions(self, num_executions=10):
        """Get the latest N strategy executions"""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return []
        
        executions = []
        current_execution = None
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                # Check for execution start
                if "STRATEGY EXECUTION START" in line:
                    if current_execution:
                        executions.append(current_execution)
                    
                    # Extract execution ID and timestamp
                    execution_id_match = re.search(r'ID: (\d{8}_\d{6})', line)
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    
                    current_execution = {
                        'id': execution_id_match.group(1) if execution_id_match else 'Unknown',
                        'start_time': timestamp_match.group(1) if timestamp_match else 'Unknown',
                        'status': 'Running',
                        'reason': '',
                        'symbol': '',
                        'timeframe': '',
                        'logs': [line.strip()],
                        'trade_placed': False,
                        'conditions': {},
                        'errors': []
                    }
                
                elif current_execution:
                    current_execution['logs'].append(line.strip())
                    
                    # Extract symbol and timeframe
                    if "Symbol:" in line and not current_execution['symbol']:
                        symbol_match = re.search(r'Symbol: (\w+)', line)
                        if symbol_match:
                            current_execution['symbol'] = symbol_match.group(1)
                    
                    if "Timeframe:" in line and not current_execution['timeframe']:
                        tf_match = re.search(r'Timeframe: (\w+)', line)
                        if tf_match:
                            current_execution['timeframe'] = tf_match.group(1)
                    
                    # Check for trade placement
                    if "OCO TRADE SETUP COMPLETED SUCCESSFULLY" in line:
                        current_execution['trade_placed'] = True
                    
                    # Extract condition results
                    if "Narrow condition:" in line:
                        narrow_match = re.search(r'Narrow condition: (\w+)', line)
                        if narrow_match:
                            current_execution['conditions']['narrow'] = narrow_match.group(1) == 'True'
                    
                    if "Channel condition:" in line:
                        channel_match = re.search(r'Channel condition: (\w+)', line)
                        if channel_match:
                            current_execution['conditions']['channel'] = channel_match.group(1) == 'True'
                    
                    if "Min range condition:" in line:
                        min_range_match = re.search(r'Min range condition: (\w+)', line)
                        if min_range_match:
                            current_execution['conditions']['min_range'] = min_range_match.group(1) == 'True'
                    
                    # Check for errors
                    if "❌" in line or "ERROR" in line.upper():
                        current_execution['errors'].append(line.strip())
                    
                    # Check for execution end
                    if "STRATEGY EXECUTION END" in line:
                        status_match = re.search(r'END - ID: \w+ - (\w+)', line)
                        if status_match:
                            current_execution['status'] = status_match.group(1)
                        
                        reason_match = re.search(r'Reason: (.+)', line)
                        if reason_match:
                            current_execution['reason'] = reason_match.group(1)
            
            # Add the last execution if it exists
            if current_execution:
                executions.append(current_execution)
            
            # Return the latest N executions
            return executions[-num_executions:] if len(executions) > num_executions else executions
            
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            return []
    
    def print_execution_summary(self, executions):
        """Print a summary of executions"""
        print("=" * 100)
        print("📊 STRATEGY TWO EXECUTION SUMMARY (AUDUSD M15)")
        print("=" * 100)
        
        if not executions:
            print("❌ No executions found")
            return
        
        for i, execution in enumerate(reversed(executions), 1):
            status_emoji = "✅" if execution['status'] == 'SUCCESS' else "❌" if execution['status'] == 'FAILED' else "⏳"
            trade_emoji = "💰" if execution['trade_placed'] else "⏸️"
            
            print(f"\n{i}. {status_emoji} Execution ID: {execution['id']}")
            print(f"   📅 Time: {execution['start_time']}")
            print(f"   💎 Symbol: {execution['symbol']} {execution['timeframe']}")
            print(f"   📋 Status: {execution['status']}")
            print(f"   {trade_emoji} Trade Placed: {'YES' if execution['trade_placed'] else 'NO'}")
            
            if execution['reason']:
                print(f"   📝 Reason: {execution['reason']}")
            
            # Show conditions if available
            if execution['conditions']:
                conditions = execution['conditions']
                narrow = "✅" if conditions.get('narrow') else "❌"
                channel = "✅" if conditions.get('channel') else "❌"
                min_range = "✅" if conditions.get('min_range') else "❌"
                print(f"   🧪 Conditions: Narrow{narrow} Channel{channel} MinRange{min_range}")
            
            # Show errors if any
            if execution['errors']:
                print(f"   ⚠️ Errors: {len(execution['errors'])} error(s)")
        
        # Statistics
        total = len(executions)
        successful = sum(1 for e in executions if e['status'] == 'SUCCESS')
        trades_placed = sum(1 for e in executions if e['trade_placed'])
        
        print(f"\n📈 STATISTICS:")
        print(f"   Total Executions: {total}")
        print(f"   Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"   Trades Placed: {trades_placed}/{total} ({trades_placed/total*100:.1f}%)")
        
    def print_detailed_execution(self, execution_id):
        """Print detailed logs for a specific execution"""
        executions = self.get_latest_executions(50)  # Get more executions to search
        
        target_execution = None
        for execution in executions:
            if execution['id'] == execution_id:
                target_execution = execution
                break
        
        if not target_execution:
            print(f"❌ Execution ID {execution_id} not found")
            return
        
        print("=" * 100)
        print(f"📋 DETAILED EXECUTION LOG - ID: {execution_id} (STRATEGY TWO)")
        print("=" * 100)
        
        for log_line in target_execution['logs']:
            print(log_line)
    
    def analyze_condition_failures(self, num_executions=20):
        """Analyze why conditions are failing"""
        executions = self.get_latest_executions(num_executions)
        
        condition_stats = {
            'narrow': {'passed': 0, 'failed': 0},
            'channel': {'passed': 0, 'failed': 0},
            'min_range': {'passed': 0, 'failed': 0}
        }
        
        no_opportunity_count = 0
        
        for execution in executions:
            if not execution['trade_placed'] and execution['status'] == 'SUCCESS':
                no_opportunity_count += 1
                
                conditions = execution['conditions']
                for condition_name in ['narrow', 'channel', 'min_range']:
                    if condition_name in conditions:
                        if conditions[condition_name]:
                            condition_stats[condition_name]['passed'] += 1
                        else:
                            condition_stats[condition_name]['failed'] += 1
        
        print("=" * 80)
        print("🔍 CONDITION FAILURE ANALYSIS - STRATEGY TWO (AUDUSD M15)")
        print("=" * 80)
        print(f"📊 Analyzed {num_executions} recent executions")
        print(f"⏸️ No trading opportunities: {no_opportunity_count}/{num_executions} ({no_opportunity_count/num_executions*100:.1f}%)")
        print()
        
        for condition_name, stats in condition_stats.items():
            total = stats['passed'] + stats['failed']
            if total > 0:
                pass_rate = stats['passed'] / total * 100
                print(f"📋 {condition_name.upper()} Condition:")
                print(f"   ✅ Passed: {stats['passed']}/{total} ({pass_rate:.1f}%)")
                print(f"   ❌ Failed: {stats['failed']}/{total} ({100-pass_rate:.1f}%)")
                print()
    
    def tail_logs(self, num_lines=50):
        """Show the last N lines of the log file"""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print("=" * 100)
            print(f"📜 LAST {num_lines} LOG LINES - STRATEGY TWO")
            print("=" * 100)
            
            for line in lines[-num_lines:]:
                print(line.rstrip())
                
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
    
    def watch_logs(self):
        """Watch logs in real-time (basic implementation)"""
        print("👀 Watching Strategy Two logs... (Press Ctrl+C to stop)")
        print("=" * 80)
        
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        import time
                        time.sleep(1)
                        
        except KeyboardInterrupt:
            print("\n👋 Stopped watching logs")
        except Exception as e:
            print(f"❌ Error watching logs: {e}")

def main():
    parser = argparse.ArgumentParser(description='Strategy Two Log Monitor and Analyzer (AUDUSD M15)')
    parser.add_argument('--summary', '-s', type=int, nargs='?', const=10, default=None,
                       help='Show summary of last N executions (default: 10)')
    parser.add_argument('--detail', '-d', type=str, 
                       help='Show detailed logs for specific execution ID')
    parser.add_argument('--conditions', '-c', type=int, nargs='?', const=20, default=None,
                       help='Analyze condition failures for last N executions (default: 20)')
    parser.add_argument('--tail', '-t', type=int, nargs='?', const=50, default=None,
                       help='Show last N lines of log file (default: 50)')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Watch logs in real-time')
    parser.add_argument('--log-dir', type=str,
                       help='Custom log directory path')
    
    args = parser.parse_args()
    
    analyzer = StrategyLogAnalyzer(args.log_dir)
    
    if args.watch:
        analyzer.watch_logs()
    elif args.detail:
        analyzer.print_detailed_execution(args.detail)
    elif args.conditions is not None:
        analyzer.analyze_condition_failures(args.conditions)
    elif args.tail is not None:
        analyzer.tail_logs(args.tail)
    elif args.summary is not None:
        executions = analyzer.get_latest_executions(args.summary)
        analyzer.print_execution_summary(executions)
    else:
        # Default: show summary of last 10 executions
        executions = analyzer.get_latest_executions(10)
        analyzer.print_execution_summary(executions)

if __name__ == "__main__":
    main() 