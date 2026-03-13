from pathlib import Path
import sys
import argparse
import glob
import shutil

import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_collection.reddit_scraper import save_reddit_posts
from src.data_collection.twitter_scraper import save_tweets, stocks
from src.data_collection.price_loader import save_price_data
from src.features.sentiment_annotator import save_sentiment
from src.features.ticker_sorter import save_via_ticker
from src.features.sentiment_aggregator import save_aggregated_sentiment
from src.features.dataset_builder import build_full_dataset
from src.models.train_lstm import train_lstm_model, PRICE_ONLY_COLS
from src.utils.logging import log


def merge_all_sentiment():
    sentiment_dir = project_root / "data" / "processed" / "sentiment"
    files = sorted(glob.glob(str(sentiment_dir / "sentiment_agg_by_ticker_*.csv")))

    if not files:
        log("No sentiment aggregation files found")
        return 0

    dfs = []
    for f in files:
        if "_ALL" in f:
            continue
        dfs.append(pd.read_csv(f))

    if not dfs:
        log("No sentiment files to merge")
        return 0

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    out_path = sentiment_dir / "sentiment_agg_by_ticker_ALL.csv"
    combined.to_csv(out_path, index=False)
    log(f"Merged {len(files)} sentiment files -> {len(combined)} rows")
    return len(combined)


def merge_all_price_data():
    price_dir = project_root / "data" / "raw" / "price_data"

    if not price_dir.exists():
        log("No price data directory found")
        return 0

    all_files = glob.glob(str(price_dir / "*.csv"))
    tickers = set()
    for f in all_files:
        name = Path(f).stem
        if "_ALL" in name:
            continue
        ticker = name.split("_")[0]
        tickers.add(ticker)

    merged_count = 0
    for ticker in sorted(tickers):
        files = sorted(glob.glob(str(price_dir / f"{ticker}_*.csv")))
        files = [f for f in files if "_ALL" not in f]
        if len(files) < 1:
            continue

        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        combined.to_csv(price_dir / f"{ticker}_1m_ALL.csv", index=False)
        merged_count += 1
        log(f"  {ticker}: {len(files)} files -> {len(combined)} rows")

    log(f"Merged price data for {merged_count} tickers")
    return merged_count


def retrain_on_collected(bucket: str = "15min", price_only: bool = False):
    training_tickers = stocks

    mode = "price-only" if price_only else "sentiment+price"
    model_file = "lstm_price_only.pt" if price_only else "lstm_volatility.pt"
    
    log(f"Retraining on all collected data ({mode})")
    log(f"Tickers: {len(training_tickers)} major stocks")
    log(f"Model will be saved as: {model_file}")

    existing = project_root / "models" / model_file
    if existing.exists():
        backup = project_root / "models" / model_file.replace(".pt", "_backup.pt")
        shutil.copy2(existing, backup)
        log(f"Backed up existing model to {backup.name}")

    log("[1/4] Merging all sentiment files...")
    sent_rows = merge_all_sentiment()
    if sent_rows == 0:
        log("No sentiment data to train on, aborting")
        return

    log("[2/4] Merging all price data...")
    merge_all_price_data()

    log("[3/4] Building full dataset...")
    build_full_dataset(bucket=bucket, tickers=training_tickers)

    log("[4/4] Training LSTM model...")
    if price_only:
        train_lstm_model(epochs=25, seq_len=20, dropout=0.3, feature_cols=PRICE_ONLY_COLS, model_name=model_file)
    else:
        train_lstm_model(epochs=25, seq_len=20, dropout=0.3, model_name=model_file)

    log(f"Retrained {mode} model on all collected data ({sent_rows} sentiment rows)")


def setup_and_train(bucket: str = "15min"):
    training_tickers = stocks

    log("Trading bot setup and training")
    log(f"Training on {len(training_tickers)} major stocks")
    log(f"Stocks: {', '.join(training_tickers[:10])}{'...' if len(training_tickers) > 10 else ''}")
    log("Collecting 325 posts for optimal sentiment coverage...")
    log("[1/7] Collecting Reddit posts...")
    save_reddit_posts(limit=200)
    log("[2/7] Collecting Twitter posts...")
    save_tweets(limit=125)
    log("[3/7] Fetching historical price data for training tickers...")
    for i, ticker in enumerate(training_tickers, 1):
        log(f"  [{i}/{len(training_tickers)}] Fetching {ticker}...")
        try:
            save_price_data(ticker=ticker, interval="1m", period="7d")
        except Exception as e:
            log(f" Error fetching {ticker}: {e}")
    
    log("[4/7] Annotating sentiment with FinBERT...")
    save_sentiment(use_finbert=True)
    
    log("[5/7] Sorting posts by ticker with GPT-4o...")
    save_via_ticker(use_gpt4=True)
    
    log("[6/7] Aggregating sentiment...")
    save_aggregated_sentiment(bucket=bucket)

    log("[7/7] Building full dataset...")
    build_full_dataset(bucket=bucket, tickers=training_tickers)

    log("[8/8] Training LSTM model...")
    train_lstm_model(epochs=25, seq_len=20, dropout=0.3)

    log(f"Model trained on {len(training_tickers)} major stocks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and Training of Model")
    parser.add_argument("--bucket", type=str, default="15min", help="Time bucket")
    parser.add_argument("--retrain", action="store_true", help="Retrain using all collected data")
    parser.add_argument("--price-only", action="store_true", help="Train with price features only (no sentiment)")
    
    args = parser.parse_args()
    
    if args.retrain:
        retrain_on_collected(bucket=args.bucket, price_only=args.price_only)
    else:
        setup_and_train(bucket=args.bucket)

