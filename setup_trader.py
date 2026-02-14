from pathlib import Path
import sys
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_collection.reddit_scraper import save_reddit_posts
from src.data_collection.twitter_scraper import save_tweets, stocks
from src.data_collection.price_loader import save_price_data
from src.features.sentiment_annotator import save_sentiment
from src.features.ticker_sorter import save_via_ticker
from src.features.sentiment_aggregator import save_aggregated_sentiment
from src.features.dataset_builder import build_full_dataset
from src.models.train_lstm import train_lstm_model
from src.utils.logging import log

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
            save_price_data(ticker=ticker, interval="1m", period="5d")
        except Exception as e:
            log(f" Error fetching {ticker}: {e}")
    
    log("[4/7] Annotating sentiment with FinBERT...")
    save_sentiment(use_finbert=True)
    
    log("[5/7] Sorting posts by ticker with GPT-4o...")
    save_via_ticker(use_gpt4=True)
    
    log("[6/7] Aggregating sentiment...")
    save_aggregated_sentiment(bucket=bucket)

    log("[8/8] Training LSTM model...")
    train_lstm_model(epochs=10, dropout=0.2)

    log(f"Model trained on {len(training_tickers)} major stocks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and Training of Model")
    parser.add_argument("--bucket", type=str, default="15min", help="Time bucket")
    
    args = parser.parse_args()
    
    setup_and_train(bucket=args.bucket)
    
