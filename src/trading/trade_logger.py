import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

spreadsheet_path= project_root / "data" / "logs" / "trades_spreadsheet.csv"

def init_spreadsheet():
    if not spreadsheet_path.exists():
        df = pd.DataFrame(columns = [
           "Timestamp",
            "Ticker",
            "Action",
            "Prediction",
            "Entry Price",
            "Quantity",
            "Order ID",
            "Status",
            "Take Profit",
            "Stop Loss",
            "Account Balance",
            "Notes" 
        ])
        df.to_csv(spreadsheet_path, index = False)

def log_trade(trade_data: dict):
    init_spreadsheet()

    df = pd.read_csv(spreadsheet_path)

    new_row = {
        "Timestamp": trade_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "Ticker": trade_data.get("ticker", ""),
        "Action": trade_data.get("action", ""),
        "Prediction": trade_data.get("prediction", ""),
        "Entry Price": trade_data.get("price", ""),
        "Quantity": trade_data.get("qty", ""),
        "Order ID": trade_data.get("order_id", ""),
        "Status": trade_data.get("status", ""),
        "Take Profit": trade_data.get("take_profit", ""),
        "Stop Loss": trade_data.get("stop_loss", ""),
        "Account Balance": trade_data.get("account_balance", ""),
        "Notes": trade_data.get("notes", "")
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(spreadsheet_path, index=False)

    return str(spreadsheet_path)

def upd_trade_status(order_id: str, status: str, notes: str = ""):
    init_spreadsheet
    df = pd.read_csv(spreadsheet_path)

    if "Order ID" in df.columns:
        mask = df["Order ID"] == order_id
        if mask.any():
            df.loc[mask, "Status"] = status
            if notes:
                df.loc[mask, "Notes"] = notes
            df.to_csv(spreadsheet_path, index=False)
            return True
    return False

def get_all_trades() -> pd.DataFrame:
    init_spreadsheet()
    return pd.read_csv(spreadsheet_path)



