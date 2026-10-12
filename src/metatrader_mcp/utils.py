from typing import Any, Optional, Union
from metatrader_client import client
import os
import dotenv

dotenv.load_dotenv()

def _config_from_env() -> Optional[dict]:
	login = os.getenv("login") or os.getenv("LOGIN")
	password = os.getenv("password") or os.getenv("PASSWORD")
	server = os.getenv("server") or os.getenv("SERVER")
	path = os.getenv("mt5_path") or os.getenv("MT5_PATH")
	if not (login and password and server and path):
		return None
	return {
		"login": int(login),
		"password": password,
		"server": server,
		"path": path,
	}

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
	elif (env_config := _config_from_env()):
		try:
			mt5_client = client.MT5Client(config=env_config)
			print(env_config)
			mt5_client.connect()
			return mt5_client
		except Exception as e:
			print(f"Error connecting to MetaTrader 5: {e}")
			return None
	return None
	
def get_client(ctx: Any) -> Optional[client.MT5Client]:
	try:	
		client = ctx.request_context.lifespan_context.client
		if client:
			return client
		else:
			return ctx.client
	except:
		return ctx.client

def initialize_client() -> Optional[client.MT5Client]:
	return init(os.getenv("login"), os.getenv("password"), os.getenv("server"), os.getenv("mt5_path"))