from typing import Any, Optional, Union
from metatrader_client import client
import os

def init(
	login: Optional[Union[str, int]],
	password: Optional[str],
	server: Optional[str],
	path: Optional[str],
) -> Optional[client.MT5Client]:
	"""
	Initialize MT5Client

	Args:
		login (Optional[Union[str, int]]): Login ID
		password (Optional[str]): Password
		server (Optional[str]): Server name
		path (Optional[str]): Path to the MetaTrader 5 terminal executable

	Returns:
		Optional[client.MT5Client]: MT5Client instance if all parameters are provided, None otherwise
	"""
	
	if login and password and server and path:
		try:
			print("Let's connect to MetaTrader 5")
			config = {
				"login": int(login),
				"password": password,
				"server": server,
				"path": path,
			}
			mt5_client = client.MT5Client(config=config)
			print(config)
			mt5_client.connect()
			return mt5_client
		except Exception as e:
			print(f"Error connecting to MetaTrader 5: {e}")
			return None

	return None
	
def get_client(ctx: Any) -> Optional[client.MT5Client]:
	try:	
		return ctx.request_context.lifespan_context.client
	except:
		return ctx.client

def initialize_client() -> Optional[client.MT5Client]:
	return init(os.getenv("login"), os.getenv("password"), os.getenv("server"), os.getenv("mt5_path"))