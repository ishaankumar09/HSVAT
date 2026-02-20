from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logging import log

def get_alpaca_client():
    config = load_config()
    
    api_key = config.get("ALPACA_API_KEY")
    secret_key = config.get("ALPACA_SECRET_KEY")
    base_url = config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key or not secret_key:
        log("Alpaca API credentials not configured")
        return None
    
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        return client
    except ImportError:
        log("alpaca-py not installed. Run: pip install alpaca-py")
        return None

def get_account_info() -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {}
    
    try:
        account = client.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity)
        }
    except Exception as e:
        log(f"Error getting account info: {e}")
        return {}

def place_market_order(ticker: str, qty: int, side: str) -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {"error": "Client not configured"}
    
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        
        order = client.submit_order(order_data)
        
        log(f"Order placed: {side} {qty} {ticker}")
        
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "status": str(order.status)
        }
    except Exception as e:
        log(f"Error placing order: {e}")
        return {"error": str(e)}

def place_bracket_order(ticker: str, qty: int, side: str, take_profit: float, stop_loss: float) -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {"error": "Client not configured"}
    
    take_profit = round(take_profit, 2)
    stop_loss = round(stop_loss, 2)
    
    try:
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        

        positions = get_positions()
        existing_pos = next((p for p in positions if p["symbol"] == ticker), None)
        if existing_pos:
            log(f"Warning: {ticker} already has an open position ({existing_pos['side']}), skipping bracket order")
            return {"error": f"Position already exists for {ticker}"}

        if side == "buy":
            order_side = OrderSide.BUY
            log(f"Placing LONG bracket order: BUY {qty} {ticker} TP={take_profit} SL={stop_loss}")
        elif side == "sell":
            order_side = OrderSide.SELL
            log(f"Placing SHORT bracket order: SELL {qty} {ticker} TP={take_profit} SL={stop_loss}")
        else:
            return {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'"}
        
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss)
        )
        
        order = client.submit_order(order_data)
        
        log(f"Bracket order placed successfully: {side.upper()} {qty} {ticker} TP=${take_profit:.2f} SL=${stop_loss:.2f} (Order ID: {order.id})")
        
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "status": str(order.status),
            "take_profit": take_profit,
            "stop_loss": stop_loss
        }
    except Exception as e:
        log(f"Error placing bracket order for {ticker}: {e}")
        return {"error": str(e)}

def get_positions() -> list:
    client = get_alpaca_client()
    
    if not client:
        return []
    
    try:
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": str(p.qty),
                "side": str(p.side),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl)
            }
            for p in positions
        ]
    except Exception as e:
        log(f"Error getting positions: {e}")
        return []

def cancel_orders_for_symbol(ticker: str) -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {"error": "Client not configured"}
    
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(request)
        
        cancelled_count = 0
        for order in orders:
            if order.symbol == ticker:
                try:
                    client.cancel_order_by_id(order.id)
                    cancelled_count += 1
                    log(f"Cancelled order {order.id} for {ticker}")
                except Exception as e:
                    log(f"Error cancelling order {order.id}: {e}")
        
        if cancelled_count > 0:
            log(f"Cancelled {cancelled_count} open order(s) for {ticker}")
        
        return {"status": "cancelled", "count": cancelled_count}
    except Exception as e:
        log(f"Error cancelling orders for {ticker}: {e}")
        return {"error": str(e)}

def close_position(ticker: str) -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {"error": "Client not configured"}
    
    try:
        cancel_result = cancel_orders_for_symbol(ticker)
        
        import time
        time.sleep(0.5)
        
        order = client.close_position(ticker)
        log(f"Position closed: {ticker}")
        return {"status": "closed", "symbol": ticker, "orders_cancelled": cancel_result.get("count", 0)}
    except Exception as e:
        log(f"Error closing position: {e}")
        return {"error": str(e)}

def close_all_positions() -> dict:
    client = get_alpaca_client()
    
    if not client:
        return {"error": "Client not configured"}
    
    try:
        client.close_all_positions(cancel_orders=True)
        log("All positions closed")
        return {"status": "all_closed"}
    except Exception as e:
        log(f"Error closing all positions: {e}")
        return {"error": str(e)}

def get_orders(status: str = "open") -> list:
    client = get_alpaca_client()
    
    if not client:
        return []
    
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        if status == "open":
            query_status = QueryOrderStatus.OPEN
        elif status == "closed":
            query_status = QueryOrderStatus.CLOSED
        else:
            query_status = QueryOrderStatus.ALL
        
        request = GetOrdersRequest(status=query_status)
        orders = client.get_orders(request)
        
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": str(o.qty),
                "side": str(o.side),
                "status": str(o.status),
                "created_at": str(o.created_at)
            }
            for o in orders
        ]
    except Exception as e:
        log(f"Error getting orders: {e}")
        return []
