import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

def load_price_data(ticker: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        return pd.DataFrame()
    
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc= True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def save_price_data(ticker: str, interval: str = "1m", period: str = "1d") -> str:
    df = load_price_data(ticker, interval, period)

    data_dir = project_root / "data" / "raw" / "price_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    filename = f"{ticker}_{interval}_{date_str}.csv"
    filepath = data_dir / filename

    df.to_csv(filepath, index=False)
    return str(filepath) 