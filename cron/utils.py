from metatrader_mcp.server import get_pending_orders_by_symbol
from metatrader_mcp.server import Context

def check_no_pending_orders(ctx: Context, symbol: str) -> bool:
    """
    Check if there are no pending orders for a given symbol
    """
    pending_orders_df = get_pending_orders_by_symbol(ctx, symbol)
    if pending_orders_df.empty:
        return True
    else:
        return False


