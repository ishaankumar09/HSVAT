import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import glob
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def floor_timestamp_to_bucket(ts: pd.Timestamp, freq: str = "15Min") -> pd.Timestamp:
    if isinstance(ts, datetime):
        ts = pd.Timestamp(ts)
    return ts.floor(freq)

def load_latest_sentiment() -> pd.DataFrame:
    data_dir = project_root / "data" / "processed" / "sentiment"

    ticker_files = sorted(glob.glob(str(data_dir / "sentiment_by_ticker_*.csv")))
    annotated_files = sorted(glob.glob(str(data_dir / "sentiment_annotated_*.csv")))
    legacy_files = sorted(glob.glob(str(data_dir / "reddit_sentiment_*.csv")))

    all_files = ticker_files + annotated_files + legacy_files

    if not all_files:
        return pd.DataFrame()

    latest_file = max(all_files, key=lambda f: Path(f).stat().st_mtime)
    return pd.read_csv(latest_file)

def aggregate_sentiment(df: pd.DataFrame, bucket: str = "15min") -> pd.DataFrame:
    """Aggregate all posts together by time bucket (general market sentiment)."""
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["bucket_start"] = df["created_utc"].apply(lambda x: floor_timestamp_to_bucket(x, bucket))

    agg = df.groupby("bucket_start").agg(
        num_posts=("id", "count"),
        num_pos=("sentiment_label", lambda x: (x == "positive").sum()),
        num_neg=("sentiment_label", lambda x: (x == "negative").sum()),
        num_neu=("sentiment_label", lambda x: (x == "neutral").sum()),
        mean_sentiment_score=("sentiment_score", "mean")
    ).reset_index()

    return agg

def aggregate_sentiment_by_ticker(df: pd.DataFrame, bucket: str = "15min") -> pd.DataFrame:
    """Aggregate posts per ticker per time bucket (ticker-specific sentiment)."""
    if "ticker" not in df.columns or "created_utc" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["created_utc"])
    df["bucket_start"] = df["created_utc"].apply(lambda x: floor_timestamp_to_bucket(x, bucket))

    agg = df.groupby(["ticker", "bucket_start"]).agg(
        num_posts=("id", "count"),
        num_pos=("sentiment_label", lambda x: (x == "positive").sum()),
        num_neg=("sentiment_label", lambda x: (x == "negative").sum()),
        num_neu=("sentiment_label", lambda x: (x == "neutral").sum()),
        mean_sentiment_score=("sentiment_score", "mean")
    ).reset_index()

    return agg

def save_aggregated_sentiment(bucket: str = "15min") -> str:
    data_dir = project_root / "data" / "processed" / "sentiment"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    # Prefer ticker-specific aggregation when data is available
    ticker_files = sorted(glob.glob(str(data_dir / "sentiment_by_ticker_*.csv")))
    if ticker_files:
        df = pd.read_csv(ticker_files[-1])
        if not df.empty and "ticker" in df.columns and "created_utc" in df.columns:
            agg_df = aggregate_sentiment_by_ticker(df, bucket)
            if not agg_df.empty:
                filename = f"sentiment_agg_by_ticker_{timestamp}.csv"
                file_path = data_dir / filename
                agg_df.to_csv(file_path, index=False)
                return str(file_path)

    # Fall back to general aggregation
    df = load_latest_sentiment()
    if df.empty:
        return ""

    agg_df = aggregate_sentiment(df, bucket)
    filename = f"sentiment_agg_{timestamp}.csv"
    file_path = data_dir / filename
    agg_df.to_csv(file_path, index=False)
    return str(file_path)