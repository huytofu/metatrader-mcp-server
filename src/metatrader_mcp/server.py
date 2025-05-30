#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional, Union

from metatrader_mcp.utils import init, get_client, initialize_client

# ────────────────────────────────────────────────────────────────────────────────
# 1) Lifespan context definition
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class AppContext:
	client: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
	client = None
	try:
		client = init(os.getenv("login"), os.getenv("password"), os.getenv("server"), os.getenv("mt5_path"))
		yield AppContext(client=client)
	finally:
		client.disconnect()

# ────────────────────────────────────────────────────────────────────────────────
# 2) Instantiate FastMCP as `mcp` (must be named `mcp`, `server`, or `app`)
# ────────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
	"metatrader",
	lifespan=app_lifespan,
	dependencies=[],
)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Register tools with @mcp.tool()
# ────────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_account_info(ctx: Context) -> dict:
	"""Get account information (balance, equity, profit, margin level, free margin, account type, leverage, currency).
	Args:
		ctx (Context): The context object.
	Returns:
		dict: A dictionary containing the account information.
	"""
	client = get_client(ctx)
	return client.account.get_trade_statistics()

@mcp.tool()
def get_deals(ctx: Context, from_date: Optional[str] = None, to_date: Optional[str] = None, symbol: Optional[str] = None) -> pd.DataFrame:
	"""Get historical deals.
	Args:
		ctx (Context): The context object.
		from_date (Optional[str]): The start date of the range to get deals from (format: 'YYYY-MM-DD').
		to_date (Optional[str]): The end date of the range to get deals to (format: 'YYYY-MM-DD').
		symbol (Optional[str]): The symbol to get deals for.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the historical deals.
	"""
	client = get_client(ctx)
	df = client.history.get_deals_as_dataframe(from_date=from_date, to_date=to_date, group=symbol)
	return df

@mcp.tool()
def get_orders(ctx: Context, from_date: Optional[str] = None, to_date: Optional[str] = None, symbol: Optional[str] = None) -> pd.DataFrame:
	"""Get historical orders.
	Args:
		ctx (Context): The context object.
		from_date (Optional[str]): The start date of the range to get orders from (format: 'YYYY-MM-DD').
		to_date (Optional[str]): The end date of the range to get orders to (format: 'YYYY-MM-DD').
		symbol (Optional[str]): The symbol to get orders for.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the historical orders.
	"""
	client = get_client(ctx)
	df = client.history.get_orders_as_dataframe(from_date=from_date, to_date=to_date, group=symbol)
	return df

def get_candles_by_date(ctx: Context, symbol_name: str, timeframe: str, from_date: str = None, to_date: str = None) -> pd.DataFrame:
	"""Get candle data for a symbol in a given timeframe and date range.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		from_date (str): The start date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' or ISO 8601).
		to_date (str): The end date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' or ISO 8601).
	Returns:
		pd.DataFrame: A pandas DataFrame containing the candle data.
		Columns: time, open, high, low, close, tick_volume, spread and real_volume
	"""
	client = get_client(ctx)
	df = client.market.get_candles_by_date(symbol_name=symbol_name, timeframe=timeframe, from_date=from_date, to_date=to_date)
	return df

@mcp.tool()
def get_candles_latest(ctx: Context, symbol_name: str, timeframe: str, count: int = 100) -> pd.DataFrame:
	"""Get the latest N candles for a symbol and timeframe.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		count (int): The number of candles to get.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the candle data.
		Columns: time, open, high, low, close, tick_volume, spread and real_volume
	"""
	client = get_client(ctx)
	df = client.market.get_candles_latest(symbol_name=symbol_name, timeframe=timeframe, count=count)
	return df

@mcp.tool()
def calculate_candles_range(ctx: Context, symbol_name: str, timeframe: str, count: int = 100) -> pd.DataFrame:
	"""Calculate the range of the latest N candles for a symbol and timeframe.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		count (int): The number of candles to get.
	Returns:
		dict: A dictionary containing the min, max and range of the latest N candles for the symbol and timeframe.
	"""
	client = get_client(ctx)
	df = client.market.get_candles_latest(symbol_name=symbol_name, timeframe=timeframe, count=count)
	min_price = df['low'].min()
	max_price = df['high'].max()
	range = max_price - min_price
	return {
		"min_price": min_price,
		"max_price": max_price,
		"range": range
	}

@mcp.tool()
def check_if_range_is_channel(ctx: Context, symbol_name: str, timeframe: str, count: int = 100) -> bool:
	"""Check if the range of the latest N candles for a symbol and timeframe is a channel.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		count (int): The number of candles to get. Only relevant if candles_df is not provided.
	Returns:
		bool: True if the range is a channel, False otherwise.
	"""
	client = get_client(ctx)
	df = client.market.get_candles_latest(symbol_name=symbol_name, timeframe=timeframe, count=count)	
	channel_range = df["high"].max() - df["low"].min()
	two_top_peaks = df.nlargest(2, "high")
	two_bottom_peaks = df.nsmallest(2, "low")

	top_peaks_diff = two_top_peaks["high"].diff()
	bottom_peaks_diff = two_bottom_peaks["low"].diff()

	if abs(top_peaks_diff.iloc[1]) <= channel_range*0.2 and abs(bottom_peaks_diff.iloc[1]) <= channel_range*0.2:
		return True
	else:
		return False

@mcp.tool()
def calculate_range_percentile(ctx: Context, symbol_name: str, timeframe: str, count: int = 100) -> dict:
	"""Calculate the range percentile of the latest N candles for a symbol and timeframe.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		count (int): The number of candles to get.
	Returns:
		dict: A dictionary containing the range percentile of the latest N candles for the symbol and timeframe.
	"""
	client = get_client(ctx)
	df = client.market.get_candles_latest(symbol_name=symbol_name, timeframe=timeframe, count=count)
	df["range"] = df["high"] - df["low"]
	percentiles = {}
	for percentile in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
		range_percentile = df["range"].quantile(percentile)
		percentiles[percentile] = range_percentile
	return percentiles

@mcp.tool()
def historical_test_get_position_outcome(ctx: Context, symbol_name: str, timeframe: str, order_trigger_date: str, position_type: str = "BUY", take_profit: float = 0.0, stop_loss: float = 0.0) -> dict:
	"""Get the outcome of a hypothetical position during a historical test.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		order_trigger_date (str): The date of the order trigger. (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' or ISO 8601)
		position_type (str): The type of position to test. (Must be one of the following: 'BUY', 'SELL')
		take_profit (float): The take profit price.
		stop_loss (float): The stop loss price.
	Returns:
		dict: A dictionary containing the outcome of the position and the date of the outcome.
		Keys: 'outcome' (str): The outcome of the position. One of the following: 'Win', 'Lose', 'Ambiguous', 'Pending'
		'outcome_date' (str): The date of the outcome. (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' or ISO 8601)
	"""
	client = get_client(ctx)
	future_date = datetime.strptime(order_trigger_date, "%Y-%m-%d %H:%M") + timedelta(days=7)
	future_date = future_date.strftime("%Y-%m-%d %H:%M")
	df = client.market.get_candles_by_date(symbol_name=symbol_name, timeframe=timeframe, from_date=order_trigger_date, to_date=future_date)
	df = df[df["time"] >= order_trigger_date]
	for index, row in df.iterrows():
		is_tp_hit = False
		is_sl_hit = False
		if position_type == "BUY":
			if row["high"] > take_profit:
				is_tp_hit = True
			if row["low"] < stop_loss:
				is_sl_hit = True
		elif position_type == "SELL":
			if row["low"] < take_profit:
				is_tp_hit = True
			if row["high"] > stop_loss:
				is_sl_hit = True
		if is_tp_hit and is_sl_hit:
			outcome_date = row["time"]
			return {
				"outcome": "Ambiguous",
				"outcome_date": outcome_date
			}
		elif is_tp_hit:
			outcome_date = row["time"]
			return {
				"outcome": "Win",
				"outcome_date": outcome_date
			}
		elif is_sl_hit:
			outcome_date = row["time"]
			return {
				"outcome": "Lose",
				"outcome_date": outcome_date
			}
		else:
			continue
	return {
		"outcome": "Pending",
		"outcome_date": None
	}

@mcp.tool()
def historical_test_get_position_trigger_date(ctx: Context, symbol_name: str, timeframe: str, order_placement_date: str, position_type: str = "BUY", entry_price_buy: float = 0.0, entry_price_sell: float = 0.0) -> str:
	"""Get the trigger date of a hypothetical position during a historical test.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
		timeframe (str): The timeframe. (Must be one of the following: 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1')
		order_placement_date (str): The date of the order placement. (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' or ISO 8601)
		position_type (str): The type of position to test. (Must be one of the following: 'LIMIT_BUY', 'LIMIT_SELL', 'BUY_STOP', 'SELL_STOP', 'LIMIT_OCO', 'OCO_STOP')
		entry_price_buy (float): The entry price of the position. None for LIMIT_SELL and SELL_STOP.
		entry_price_sell (float): The entry price of the position. None for LIMIT_BUY and BUY_STOP.
	Returns:
		dict: A dictionary containing the trigger date of the position. There will be two keys: 'buy' and 'sell'. Values can be None or a datetime object.
		Both keys' values will be None if the position is not triggered. 
		Both keys will have values if the position type is LIMIT_OCO or OCO_STOP.
	"""
	client = get_client(ctx)
	future_date = datetime.strptime(order_placement_date, "%Y-%m-%d %H:%M") + timedelta(days=7)
	future_date = future_date.strftime("%Y-%m-%d %H:%M")
	df = client.market.get_candles_by_date(symbol_name=symbol_name, timeframe=timeframe, from_date=order_placement_date, to_date=future_date)
	df = df[df["time"] >= order_placement_date]
	if position_type in ["LIMIT_BUY", "BUY_STOP"]:
		for index, row in df.iterrows():
			if row["low"] < entry_price_buy and row["high"] > entry_price_buy:
				return row["time"]
			break
	elif position_type in ["LIMIT_SELL", "SELL_STOP"]:
		for index, row in df.iterrows():
			if row["low"] < entry_price_sell and row["high"] > entry_price_sell:
				return row["time"]
			break
	elif position_type in ["LIMIT_OCO", "OCO_STOP"]:
		time_buy = None
		time_sell = None
		should_break = False
		count = 0
		for index, row in df.iterrows():
			count = index
			if row["low"] < entry_price_buy and row["high"] > entry_price_buy:
				time_buy = time_buy or row["time"]
				should_break = True
			if row["low"] < entry_price_sell and row["high"] > entry_price_sell:
				time_sell = time_sell or row["time"]
				should_break = True
			if should_break:
				break
		
		one_more_candle = df.iloc[count + 1]
		if one_more_candle["low"] < entry_price_buy and one_more_candle["high"] > entry_price_buy:
			time_buy = time_buy or one_more_candle["time"]
		if one_more_candle["low"] < entry_price_sell and one_more_candle["high"] > entry_price_sell:
			time_sell = time_sell or one_more_candle["time"]

		return {
			"buy": time_buy,
			"sell": time_sell
		}

@mcp.tool()
def get_symbol_price(ctx: Context, symbol_name: str) -> dict:
	"""Get the latest price info for a symbol as a dictionary.
	Args:
		ctx (Context): The context object.
		symbol_name (str): The symbol name. Example: 'EURUSD'
	Returns:
		dict: A dictionary containing the latest price info for the symbol.
	"""
	client = get_client(ctx)
	return client.market.get_symbol_price(symbol_name=symbol_name)

@mcp.tool()
def get_all_symbols(ctx: Context) -> list:
	"""Get a list of all available market symbols.
	Args:
		ctx (Context): The context object.
	Returns:
		list: A list of all available market symbols.
	"""
	client = get_client(ctx)
	return client.market.get_symbols()

@mcp.tool()
def get_symbols(ctx: Context, group: Optional[str] = None) -> list:
	"""
	Get a list of available market symbols. Filter symbols by group pattern (e.g., '*USD*').
	Args:
		ctx (Context): The context object.
		group (Optional[str]): The group pattern to filter symbols by. Example: '*USD*'
	Returns:
		list: A list of available market symbols.
	"""
	client = get_client(ctx)
	return client.market.get_symbols(group=group)

# ────────────────────────────────────────────────────────────────────────────────
# Order module tools
# ────────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_all_positions(ctx: Context) -> pd.DataFrame:
	"""Get all open positions.
	Args:
		ctx (Context): The context object.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the open positions.
	"""
	client = get_client(ctx)
	df = client.order.get_all_positions()
	return df

@mcp.tool()
def get_positions_by_symbol(ctx: Context, symbol: str) -> pd.DataFrame:
	"""Get open positions for a specific symbol.
	Args:
		ctx (Context): The context object.
		symbol (str): The symbol name. Example: 'EURUSD'
	Returns:
		pd.DataFrame: A pandas DataFrame containing the open positions for the specific symbol.
	"""
	client = get_client(ctx)
	df = client.order.get_positions_by_symbol(symbol=symbol)
	return df

@mcp.tool()
def get_positions_by_id(ctx: Context, id: Union[int, str]) -> pd.DataFrame:
	"""Get open positions by ID.
	Args:
		ctx (Context): The context object.
		id (Union[int, str]): The ID of the position.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the open positions for the specific ID.
	"""
	client = get_client(ctx)
	df = client.order.get_positions_by_id(id=id)
	return df

@mcp.tool()
def get_all_pending_orders(ctx: Context) -> pd.DataFrame:
	"""Get all pending orders.
	Args:
		ctx (Context): The context object.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the pending orders.
	"""
	client = get_client(ctx)
	df = client.order.get_all_pending_orders()
	return df

@mcp.tool()
def get_pending_orders_by_symbol(ctx: Context, symbol: str) -> pd.DataFrame:
	"""Get pending orders for a specific symbol.
	Args:
		ctx (Context): The context object.
		symbol (str): The symbol name. Example: 'EURUSD'
	Returns:
		pd.DataFrame: A pandas DataFrame containing the pending orders for the specific symbol.
	"""
	client = get_client(ctx)
	df = client.order.get_pending_orders_by_symbol(symbol=symbol)
	return df

@mcp.tool()
def get_pending_orders_by_id(ctx: Context, id: Union[int, str]) -> pd.DataFrame:
	"""Get pending orders by id.
	Args:
		ctx (Context): The context object.
		id (Union[int, str]): The ID of the pending order.
	Returns:
		pd.DataFrame: A pandas DataFrame containing the pending orders for the specific ID.
	"""
	client = get_client(ctx)
	df = client.order.get_pending_orders_by_id(id=id)
	return df

@mcp.tool()
def place_market_order(ctx: Context, symbol: str, volume: float, type: str) -> dict:
	"""
	Place a market order. Parameters:
		symbol: Symbol name (e.g., 'EURUSD')
		volume: Lot size. (e.g. 1.5)
		type: Order type ('BUY' or 'SELL')
	"""
	client = get_client(ctx)
	return client.order.place_market_order(symbol=symbol, volume=volume, type=type)

@mcp.tool()
def place_pending_order(ctx: Context, symbol: str, volume: float, type: str, price: float, stop_loss: Optional[Union[int, float]] = 0.0, take_profit: Optional[Union[int, float]] = 0.0) -> dict:
	"""
	Place a pending order. Parameters:
		symbol: Symbol name (e.g., 'EURUSD')
		volume: Lot size. (e.g. 1.5)
		type: Order type ('BUY', 'SELL').
		price: Pending order price.
		stop_loss (optional): Stop loss price.
		take_profit (optional): Take profit price.
	"""
	client = get_client(ctx)
	return client.order.place_pending_order(symbol=symbol, volume=volume, type=type, price=price, stop_loss=stop_loss, take_profit=take_profit)

@mcp.tool()
def modify_position(ctx: Context, id: Union[int, str], stop_loss: Optional[Union[int, float]] = None, take_profit: Optional[Union[int, float]] = None) -> dict:
	"""Modify an open position by ID."""
	client = get_client(ctx)
	return client.order.modify_position(id=id, stop_loss=stop_loss, take_profit=take_profit)

@mcp.tool()
def modify_pending_order(ctx: Context, id: Union[int, str], price: Optional[Union[int, float]] = None, stop_loss: Optional[Union[int, float]] = None, take_profit: Optional[Union[int, float]] = None) -> dict:
	"""Modify a pending order by ID."""
	client = get_client(ctx)
	return client.order.modify_pending_order(id=id, price=price, stop_loss=stop_loss, take_profit=take_profit)

@mcp.tool()
def close_position(ctx: Context, id: Union[int, str]) -> dict:
	"""Close an open position by ID."""
	client = get_client(ctx)
	return client.order.close_position(id=id)

@mcp.tool()
def cancel_pending_order(ctx: Context, id: Union[int, str]) -> dict:
	"""Cancel a pending order by ID."""
	client = get_client(ctx)
	return client.order.cancel_pending_order(id=id)

@mcp.tool()
def close_all_positions(ctx: Context) -> dict:
	"""Close all open positions."""
	client = get_client(ctx)
	return client.order.close_all_positions()

@mcp.tool()
def close_all_positions_by_symbol(ctx: Context, symbol: str) -> dict:
	"""Close all open positions for a specific symbol."""
	client = get_client(ctx)
	return client.order.close_all_positions_by_symbol(symbol=symbol)

@mcp.tool()
def close_all_profitable_positions(ctx: Context) -> dict:
	"""Close all profitable positions."""
	client = get_client(ctx)
	return client.order.close_all_profitable_positions()

@mcp.tool()
def close_all_losing_positions(ctx: Context) -> dict:
	"""Close all losing positions."""
	client = get_client(ctx)
	return client.order.close_all_losing_positions()

@mcp.tool()
def cancel_all_pending_orders(ctx: Context) -> dict:
	"""Cancel all pending orders."""
	client = get_client(ctx)
	return client.order.cancel_all_pending_orders()

@mcp.tool()
def cancel_pending_orders_by_symbol(ctx: Context, symbol: str) -> dict:
	"""Cancel all pending orders for a specific symbol."""
	client = get_client(ctx)
	return client.order.cancel_pending_orders_by_symbol(symbol=symbol)

if __name__ == "__main__":
	load_dotenv()
	parser = argparse.ArgumentParser(description="MetaTrader MCP Server")
	parser.add_argument("--login",    type=str, help="MT5 login")
	parser.add_argument("--password", type=str, help="MT5 password")
	parser.add_argument("--server",   type=str, help="MT5 server name")
	
	args = parser.parse_args()

	# inject into lifespan via env vars
	if args.login:    os.environ["login"]    = args.login
	if args.password: os.environ["password"] = args.password
	if args.server:   os.environ["server"]   = args.server

	# run the MCP server (must call mcp.run)
	mcp.run(
		transport="stdio"
	)
