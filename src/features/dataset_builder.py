import pandas as pd
from pathlib import Path
import sys
import glob

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.sentiment_aggregator import floor_timestamp_to_bucket

def load_latest_price_file(ticker: str) ->pd.DataFrame:

    data_dir = project_root / "data" / "raw" / "price_data"
    csv_files = glob.glob(str(data_dir / f"{ticker}_*.csv"))

    if not csv_files:
        return pd.DataFrame()
    
    latest_file = csv_files[-1]
    return pd.read_csv(latest_file)

def load_latest_sentiment_agg() -> pd.DataFrame:
    data_dir = project_root / "data" / "processed" / "sentiment"

    # Prefer ticker-specific aggregated files (have a ticker column)
    ticker_agg_files = glob.glob(str(data_dir / "sentiment_agg_by_ticker_*.csv"))
    if ticker_agg_files:
        latest_file = max(ticker_agg_files, key=lambda f: Path(f).stat().st_mtime)
        return pd.read_csv(latest_file)

    # Fall back to general aggregated files
    general_agg_files = glob.glob(str(data_dir / "sentiment_agg_*.csv"))
    if not general_agg_files:
        return pd.DataFrame()

    latest_file = max(general_agg_files, key=lambda f: Path(f).stat().st_mtime)
    return pd.read_csv(latest_file)

def merge_sentiment_with_price(sentiment_df: pd.DataFrame, price_df: pd.DataFrame, bucket: str = "15min", ticker: str = None) -> pd.DataFrame:
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()

    # If sentiment data has a ticker column, filter to the specific ticker only
    if ticker and "ticker" in sentiment_df.columns:
        sentiment_df = sentiment_df[sentiment_df["ticker"] == ticker].drop(columns=["ticker"])

    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    price_df['timestamp'] = price_df["timestamp"].apply(lambda x: floor_timestamp_to_bucket(x, bucket))

    sentiment_df["bucket_start"] = pd.to_datetime(sentiment_df["bucket_start"], utc=True)

    sentiment_df = sentiment_df.sort_values("bucket_start")
    sentiment_df["bucket_start"] = sentiment_df["bucket_start"].shift(-1)
    sentiment_df = sentiment_df.dropna(subset=["bucket_start"])

    merged = pd.merge(price_df, sentiment_df, left_on="timestamp", right_on="bucket_start", how="left")

    merged = merged.fillna(0)
    return merged

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, moving averages, momentum, volume ratio for richer price context."""
    df = df.copy()
    
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    
    df["sma_20"] = close.rolling(window=20, min_periods=1).mean().fillna(close)
    df["sma_50"] = close.rolling(window=50, min_periods=1).mean().fillna(close)
    
    df["momentum"] = close.pct_change(periods=10).fillna(0)
    
    volume_sma = volume.rolling(window=20, min_periods=1).mean().replace(0, 1e-10)
    df["volume_ratio"] = (volume / volume_sma).fillna(1.0)
    
    return df

def add_target_column(
    df: pd.DataFrame,
    threshold: float = 0.0,
    forward_look_bars: int = 1,
) -> pd.DataFrame:
    """Add next-period return and a simple binary direction (used per-ticker; overwritten in build_full_dataset with mean+kσ)."""
    df = df.copy()

    df["close_shifted"] = df["close"].shift(-forward_look_bars)
    df["pct_change"] = (df["close_shifted"] - df["close"]) / df["close"].replace(0, 1e-10)
    df["target_direction"] = (df["pct_change"] > 0).astype(int)

    df = df.dropna(subset=["close_shifted"])
    return df

def build_train_and_test_datasets(train_ratio: float = 0.7) -> tuple:

    sentiment_df = load_latest_sentiment_agg()
    price_df = load_latest_price_file("AAPL")
    
    if sentiment_df.empty or price_df.empty:
        return "", ""
    
    merged = merge_sentiment_with_price(sentiment_df, price_df)
    merged = add_technical_indicators(merged)
    merged = add_target_column(merged)
    
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    split_idx = int(len(merged) * train_ratio)
    
    train_df = merged.iloc[:split_idx]
    test_df = merged.iloc[split_idx:]
    
    data_dir = project_root / "data" / "processed" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "train_dataset.csv"
    test_path = data_dir / "test_dataset.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return str(train_path), str(test_path)

def build_full_dataset(
    tickers: list = None,
    bucket: str = "15min",
    forward_look_bars: int = 3,
    sigma_k: float = None,
    target_quantile: float = 0.50,
) -> str:

    from src.utils.logging import log
    from src.utils.config_loader import load_watchlist

    if tickers is None:
        tickers = load_watchlist()

    sentiment_df = load_latest_sentiment_agg()

    if sentiment_df.empty:
        log("Missing sentiment data. Cannot build dataset.")
        return ""

    all_ticker_data = []

    for ticker in tickers:
        price_df = load_latest_price_file(ticker)

        if price_df.empty:
            log(f" No price data for {ticker}, skipping")
            continue

        merged = merge_sentiment_with_price(sentiment_df, price_df, bucket, ticker=ticker)
        merged = add_technical_indicators(merged)
        merged = add_target_column(merged, forward_look_bars=forward_look_bars)

        merged["ticker"] = ticker
        all_ticker_data.append(merged)
        log(f" {ticker}: {len(merged)} samples")

    if not all_ticker_data:
        log("No valid ticker data found. Cannot build dataset.")
        return ""

    full_df = pd.concat(all_ticker_data, ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    pc = full_df["pct_change"]
    if target_quantile is not None:
        # Percentile-based: UP = top (1 - target_quantile) of returns → controllable balance
        threshold = pc.quantile(target_quantile)
        full_df["target_direction"] = (full_df["pct_change"] >= threshold).astype(int)
        pct_up = 100 * full_df["target_direction"].mean()
        log(f"Target threshold: quantile({target_quantile}) = {threshold:.6f} → ~{100 - target_quantile*100:.0f}% UP (actual {pct_up:.1f}%)")
    else:
        mean_ret = pc.mean()
        std_ret = pc.std()
        if std_ret == 0 or pd.isna(std_ret):
            std_ret = 1e-10
        k = sigma_k if sigma_k is not None else 1.5
        threshold = mean_ret + k * std_ret
        full_df["target_direction"] = (full_df["pct_change"] > threshold).astype(int)
        log(f"Target threshold: mean + {k}*sigma = {threshold:.6f} (mean={mean_ret:.6f}, std={std_ret:.6f})")

    data_dir = project_root / "data" / "processed" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    full_path = data_dir / "full_dataset.csv"
    full_df.to_csv(full_path, index=False)

    log(f"Full dataset saved: {len(full_df)} samples from {len(all_ticker_data)} tickers")
    log(f"Target distribution: {full_df['target_direction'].value_counts().to_dict()}")

    return str(full_path)



