from metatrader_mcp.server import get_pending_orders_by_symbol, get_account_info, get_symbol_price
from metatrader_mcp.server import Context
import logging
import traceback

# Get logger for this module
logger = logging.getLogger(__name__)

def check_no_pending_orders(ctx: Context, symbol: str) -> bool:
    """
    Check if there are no pending orders for a given symbol
    """
    pending_orders_df = get_pending_orders_by_symbol(ctx, symbol)
    if pending_orders_df.empty:
        return True
    else:
        return False

def calculate_position_size(ctx, symbol, stop_loss_distance, risk_per_trade):
    """Calculate position size based on risk management with logging"""
    try:
        logger.info(f"💰 Calculating position size...")
        logger.info(f"   Stop loss distance: {stop_loss_distance:.5f}")
        logger.info(f"   Risk per trade: {risk_per_trade * 100}%")
        
        # Get current account information
        logger.info(f"🔍 Getting account information...")
        account_info = get_account_info(ctx=ctx)
        
        if not account_info:
            logger.error(f"❌ Failed to get account information")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        account_balance = account_info.get('balance', 0)
        account_currency = account_info.get('currency', 'USD')
        logger.info(f"💵 Account balance: {account_balance:,.2f} {account_currency}")
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_per_trade
        logger.info(f"💸 Risk amount: {risk_amount:,.2f} {account_currency}")
        
        # Get current symbol price to calculate pip value
        symbol_price_info = get_symbol_price(ctx=ctx, symbol_name=symbol)
        
        if not symbol_price_info:
            logger.error(f"❌ Failed to get symbol price information for {symbol}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        # Use bid price for calculation
        current_price = symbol_price_info.get('bid', 0)
        if current_price <= 0:
            logger.error(f"❌ Invalid current price: {current_price}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        logger.info(f"📊 Current {symbol} price: {current_price:.5f}")
        
        # Calculate pip value based on symbol
        # For standard lots (100,000 units):
        # - For pairs like EUR/USD: 1 pip = 0.0001, pip value = 100,000 * 0.0001 = $10
        # - For JPY pairs: 1 pip = 0.01, pip value = 100,000 * 0.01 / current_price
        
        if 'JPY' in symbol:
            # JPY pairs: pip = 0.01
            pip_size = 0.01
            pip_value_per_lot = 100000 * pip_size / current_price  # In account currency per lot
        else:
            # Standard pairs: pip = 0.0001  
            pip_size = 0.0001
            if symbol.endswith('USD') and account_currency == 'USD':
                # For USD base pairs like EUR/USD when account is USD
                pip_value_per_lot = 100000 * pip_size  # $10 per lot
            else:
                # For other pairs, approximate calculation
                pip_value_per_lot = 100000 * pip_size
        
        # Convert stop loss distance to pips
        stop_loss_pips = stop_loss_distance / pip_size
        logger.info(f"📏 Stop loss distance: {stop_loss_pips:.1f} pips")
        
        # Calculate position size
        # Risk = Position_Size * Pip_Value_Per_Lot * Stop_Loss_Pips
        # Position_Size = Risk / (Pip_Value_Per_Lot * Stop_Loss_Pips)
        
        if stop_loss_pips <= 0:
            logger.error(f"❌ Invalid stop loss pips: {stop_loss_pips}")
            logger.warning(f"⚠️ Using minimum lot size as fallback")
            return 0.01
        
        calculated_lot_size = risk_amount / (pip_value_per_lot * stop_loss_pips)
        
        # Apply position size constraints
        min_lot_size = 0.01
        max_lot_size = 10.0  # Conservative maximum
        
        # Round to valid lot size increments (0.01)
        lot_size = round(calculated_lot_size, 2)
        
        # Apply constraints
        lot_size = max(min_lot_size, min(lot_size, max_lot_size))
        
        # Calculate actual risk with final lot size
        actual_risk = lot_size * pip_value_per_lot * stop_loss_pips
        actual_risk_percentage = (actual_risk / account_balance) * 100
        
        logger.info(f"💰 POSITION SIZE CALCULATION:")
        logger.info(f"   Pip value per lot: {pip_value_per_lot:.2f} {account_currency}")
        logger.info(f"   Calculated lot size: {calculated_lot_size:.4f}")
        logger.info(f"   Final lot size: {lot_size:.2f} lots")
        logger.info(f"   Actual risk: {actual_risk:.2f} {account_currency} ({actual_risk_percentage:.2f}%)")
        
        if lot_size == min_lot_size:
            logger.warning(f"⚠️ Using minimum lot size - calculated size was too small")
        elif lot_size == max_lot_size:
            logger.warning(f"⚠️ Using maximum lot size - calculated size was too large")
        
        return lot_size
        
    except Exception as e:
        logger.error(f"❌ Error calculating position size: {str(e)}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        logger.warning(f"⚠️ Using minimum lot size as fallback")
        return 0.01  # Minimum lot size as fallback


