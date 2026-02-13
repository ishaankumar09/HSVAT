import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import glob
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def floor_timestamp_to_bucket(ts: pd.Timestamp, freq: str = "15Min") -> pd.Timestamp:
    return ts.floor(freq)

def load_latest_sentiment()-> pd.DataFrame:
    
    data_dir = project_root / "data" / "processed" / "sentiment"

    ticker_files = sorted(glob.glob(str(data_dir / "sentiment_by_ticker_*.csv")))
    annotated_files = sorted(glob.glob(str(data_dir / "sentiment_annotated_*.csv")))
    legacy_files = sorted(glob.glob(str(data_dir / "reddit_sentiment_*.csv")))

    all_files = ticker_files + annotated_files + legacy_files

    if not all_files:
        return pd.DataFrame()
    
    latest_file = all_files[-1]
    return pd.read_csv(latest_file)

def aggregate_sentiment(df: pd.DataFrame, bucket: str = "15min") -> pd.DataFrame:
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

def save_aggregated_sentiment(bucket: str = "15min") -> str:
    df = load_latest_sentiment()
    
    if df.empty:
        return ""
    
    agg_df = aggregate_sentiment(df, bucket)

    data_dir = project_root / "data" / "processed" / "sentiment"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"sentiment_agg_{timestamp}.csv"
    file_path = data_dir / filename

    agg_df.to_csv(file_path, index=False)

    return str(file_path)