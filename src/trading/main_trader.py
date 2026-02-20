import time
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import glob

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.reddit_scraper import save_reddit_posts
from src.data_collection.twitter_scraper import save_tweets
from src.data_collection.price_loader import save_price_data, load_price_data
from src.features.sentiment_annotator import save_sentiment
from src.features.ticker_sorter import save_via_ticker
from src.features.sentiment_aggregator import save_aggregated_sentiment
from src.models.predict_lstm import load_trained_model, predict_direction_labels, predict_action, predict_sequences
from src.models.train_lstm import FEATURE_COLS, create_sequences
from src.trading.simulator import execute_paper_trade
from src.trading.alpaca import get_account_info, get_positions, close_position, close_all_positions
from src.trading.trade_logger import log_trade
from src.utils.logging import log
from src.utils.market_hours import is_market_open, wait_until_market_open, is_near_market_close
from src.utils.stock_filter import is_blue_chip, has_sufficient_sentiment
from src.features.sentiment_aggregator import floor_timestamp_to_bucket

def get_ticker_from_sentiment() -> list:
    data_dir = project_root / "data" / "processed" / "sentiment"
    ticker_files = sorted(glob.glob(str(data_dir / "sentiment_by_ticker_*.csv")))

    if not ticker_files:
        log("No sentiment data found")
        return []
    
    latest_file = ticker_files[-1]
    df = pd.read_csv(latest_file)

    if df.empty or "ticker" not in df.columns:
        return []
    
    ticker_sentiment = df.groupby("ticker").agg({
        "sentiment_score": "mean",
        "id": "count"
    }).reset_index()

    ticker_sentiment.columns = ["ticker", "avg_sentiment", "num_posts"]

    ticker_sentiment["abs_sentiment"] = ticker_sentiment["avg_sentiment"].abs()
    ticker_sentiment = ticker_sentiment.sort_values("abs_sentiment", ascending=False)

    log(f"Found {len(ticker_sentiment)} unique tickers in sentiment data")
    if len(ticker_sentiment) > 0:
        top_5 = ticker_sentiment.head(5)
        log("Top sentiment tickers:")
        for _, row in top_5.iterrows():
            log(f"  {row['ticker']}: {row['avg_sentiment']:.2f} ({int(row['num_posts'])} posts)")

    return ticker_sentiment[["ticker", "avg_sentiment", "num_posts"]].to_dict("records")

def get_latest_features_lightweight(ticker: str, bucket: str = "15min", seq_len: int = 20) -> tuple:

    from src.features.dataset_builder import (
        load_latest_sentiment_agg,
        load_latest_price_file,
        merge_sentiment_with_price,
        add_technical_indicators,
    )

    sentiment_df = load_latest_sentiment_agg()
    price_df = load_latest_price_file(ticker)

    if sentiment_df.empty or price_df.empty:
        return None, None
    
    merged = merge_sentiment_with_price(sentiment_df, price_df, bucket, ticker=ticker)
    merged = add_technical_indicators(merged)

    if len(merged) < seq_len:
        return None, None
    
    merged = merged.tail(seq_len)

    available_features = [c for c in FEATURE_COLS if c in merged.columns]
    features = merged[available_features].values

    try: 
        model, mean, std, input_dim, output_dim = load_trained_model()
        if mean is not None and std is not None:
            features = (features - mean) / std
        else: 
            raise ValueError("Model loaded but no normalization parameters found")
        
    except Exception as e:
        log(f"Warning: Could not load saved normalization parameters: {e}")
        log("Using live data statistics (NOT RECOMMENDED - may cause data leakage)")
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1
        features = (features - mean) / std

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    current_price = merged["close"].iloc[-1]

    return features_tensor, current_price

def run_trading(bucket: str = "15min", max_tickers: int = 5, duration_hours: float = None):
    if duration_hours:
        log(f"Trading Bot running for {duration_hours} hours")
    else: 
        log("Trading Bot running 24/7 (only trade during market hours)")

    model_path = project_root / "models" / "lstm_volatility.pt"
    if not model_path.exists():
        log("Model not found, run setup")
        return
    
    wait_until_market_open()

    try:
        model, mean, std, input_dim, output_dim = load_trained_model()
        log(" Model loaded successfully")
    except Exception as e:
        log(f" Error loading model: {e}")
        return
    
    start_time = datetime.now(timezone.utc)
    if duration_hours:
        end_time = start_time + timedelta(hours=duration_hours)
    else: 
        end_time = None
    
    if end_time:
        log(f" Start: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log(f" End:   {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    last_candle_time = None
    cycle_count = 0
    position_entry_times = {}

    while True:
        if end_time and datetime.now(timezone.utc) >= end_time:
            log("TIME LIMIT REACHED")
            log(f"Ran for {duration_hours} hours")
            log(f"Total cycles completed: {cycle_count}")
            log("Bot stopping...")
            break
        try:
            if not is_market_open():
                log("Market closed. Waiting...")
                wait_until_market_open()
                continue
            
            if is_near_market_close(minutes_before_close=10):
                positions = get_positions()
                if positions:
                    log(f"Near market close - closing all {len(positions)} open position(s)")
                    close_result = close_all_positions()
                    if "error" not in close_result:
                        position_entry_times.clear()
                        log("All positions closed before market close")
                    else:
                        log(f"Error closing all positions: {close_result.get('error')}")
                else:
                    log("Near market close - no open positions to close")
            
            current_time = datetime.now(timezone.utc)
            current_candle = floor_timestamp_to_bucket(current_time, bucket)
            
            if last_candle_time is None or current_candle > last_candle_time:
                cycle_count += 1
                log(f"CYCLE #{cycle_count} - {current_candle.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                try:
                    if True:
                        log("Step 1: Collecting Reddit posts (every other cycle)...")
                        save_reddit_posts(limit=150)
                        
                        log("Step 2: Collecting Twitter posts...")
                        try:
                            save_tweets(limit=60)
                        except Exception as e:
                            log(f"Twitter collection failed (will use Reddit only): {e}")
                        
                        log("Step 3: Annotating sentiment with FinBERT...")
                        save_sentiment(use_finbert=True)
                        
                        log("Step 4: Sorting posts by ticker with GPT-4o...")
                        save_via_ticker(use_gpt4=True)
                        
                        log("Step 5: Aggregating sentiment...")
                        save_aggregated_sentiment(bucket=bucket)
                    
                    all_tickers = get_ticker_from_sentiment()
                    
                    if not all_tickers:
                        log("No tickers found in sentiment data, skipping this cycle")
                        time.sleep(30)
                        continue
                    
                    log(f"\nFiltering {len(all_tickers)} tickers (blue chip + min 2 posts)...")
                    blue_chip_tickers = []
                    skipped_low_posts = 0
                    for entry in all_tickers:
                        ticker = entry["ticker"]
                        num_posts = int(entry["num_posts"])
                        avg_sentiment = float(entry["avg_sentiment"])

                        if not has_sufficient_sentiment(num_posts, avg_sentiment):
                            skipped_low_posts += 1
                            continue

                        if is_blue_chip(ticker):
                            blue_chip_tickers.append(ticker)
                            if len(blue_chip_tickers) >= max_tickers:
                                break

                    if skipped_low_posts:
                        log(f"  Skipped {skipped_low_posts} tickers with insufficient sentiment data")

                    if not blue_chip_tickers:
                        log("No blue chip stocks with sufficient sentiment this cycle")
                        time.sleep(30)
                        continue
                    
                    tickers_to_process = blue_chip_tickers
                    log(f"\n Found {len(tickers_to_process)} blue chip stocks to analyze:")
                    log(f"  {', '.join(tickers_to_process)}")
                    
                    account = get_account_info()
                    account_balance = account.get("buying_power", 0) if account else 0

                    positions = get_positions()
                    open_symbols = {p["symbol"] for p in positions}

                    for pos in positions:
                        sym = pos["symbol"]
                        pos_side = str(pos["side"]).lower()
                        should_close = False
                        close_reason = ""

                        entry_time = position_entry_times.get(sym)
                        if entry_time:
                            age_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                            if age_minutes >= 60:
                                should_close = True
                                close_reason = f"time-based exit ({age_minutes:.0f} min open)"

                        if not should_close:
                            try:
                                save_price_data(ticker=sym, interval="1m", period="1d")
                                feat_tensor, _ = get_latest_features_lightweight(sym, bucket)
                                if feat_tensor is not None:
                                    probs = predict_sequences(model, feat_tensor)
                                    pred = torch.argmax(probs, dim=1).numpy()[0]
                                    conf = probs[0][pred].item()
                                    if "long" in pos_side and pred == 0 and conf > 0.57:
                                        should_close = True
                                        close_reason = f"model flipped SHORT (conf={conf:.1%})"
                                    elif "short" in pos_side and pred == 1 and conf > 0.57:
                                        should_close = True
                                        close_reason = f"model flipped BUY (conf={conf:.1%})"
                            except Exception as e:
                                log(f"{sym}: Model re-evaluation failed: {e}")

                        if should_close:
                            log(f"{sym}: Closing position — {close_reason}")
                            close_result = close_position(sym)
                            if "error" not in close_result:
                                position_entry_times.pop(sym, None)
                                open_symbols.discard(sym)
                            else:
                                log(f"{sym}: Failed to close: {close_result.get('error')}")

                    for ticker in tickers_to_process:
                        try:
                            log(f"\n--- Analyzing {ticker} ---")

                            if ticker in open_symbols:
                                log(f"{ticker}: Already have an open position, skipping")
                                continue

                            save_price_data(ticker=ticker, interval="1m", period="1d")
                            price_df = load_price_data(ticker=ticker, interval="1m", period="1d")
                            
                            features_tensor, current_price = get_latest_features_lightweight(ticker, bucket)
                            
                            if features_tensor is None:
                                log(f"{ticker}: Insufficient feature data, skipping")
                                continue
                            
                            probs = predict_sequences(model, features_tensor)
                            prediction = torch.argmax(probs, dim=1).numpy()
                            
                            action = predict_action(prediction[0], probs[0], confidence_threshold=0.57)
                            
                            confidence = probs[0][prediction[0]].item()
                            log(f"{ticker}: {action.upper()} (class={prediction[0]}, confidence={confidence:.1%}) at ${current_price:.2f}")
                            
                            if action != "stay":
                                result = execute_paper_trade(ticker, int(prediction[0]), current_price, price_df)
                                
                                if "error" not in result:
                                    position_entry_times[ticker] = datetime.now(timezone.utc)
                                    open_symbols.add(ticker)
                                    order_id = result.get("id", "")
                                    status = result.get("status", "")
                                    
                                    trade_data = {
                                        "timestamp": current_candle.isoformat(),
                                        "ticker": ticker,
                                        "action": action,
                                        "prediction": int(prediction[0]),
                                        "price": current_price,
                                        "qty": result.get("qty", ""),
                                        "order_id": order_id,
                                        "status": status,
                                        "take_profit": result.get("take_profit", ""),
                                        "stop_loss": result.get("stop_loss", ""),
                                        "account_balance": account_balance,
                                        "notes": f"Cycle #{cycle_count}"
                                    }
                                    
                                    spreadsheet_path = log_trade(trade_data)
                                    log(f"{ticker} trade logged: Order {order_id}")
                                else:
                                    log(f"{ticker} trade failed: {result.get('error', 'Unknown')}")
                            
                            time.sleep(2)
                            
                        except Exception as e:
                            log(f"Error processing {ticker}: {e}")
                            continue
                    
                    log(f"\nOpen positions: {len(open_symbols)}")
                    log(f"Account balance: ${account_balance:,.2f}")
                    
                except Exception as e:
                    log(f"Error in trading cycle: {e}")
                    import traceback
                    log(traceback.format_exc())
                
                last_candle_time = current_candle
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            log("Trading bot stopped by user")
            break
        except Exception as e:
            log(f"Fatal error: {e}")
            import traceback
            log(traceback.format_exc())
            log("Waiting 60 seconds before retry...")
            time.sleep(60)

        


