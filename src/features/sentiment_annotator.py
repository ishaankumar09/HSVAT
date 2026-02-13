import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys
import glob

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import log

_finbert_model = None
_finbert_tokenizer = None

def load_finbert():

    global _finbert_model, _finbert_tokenizer

    if _finbert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ProsusAI/finbert"
            log(f"Loading FinBERT model...")
            _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _finbert_model.eval()
            log(f"FinBERT model loaded successfully.")

        except Exception as e:
            log(f"Error loading FinBERT model: {e}")
            return None, None

def analyze_sentiment(text: str) -> tuple:
    import time
    start_time = time.time()

    model, tokenizer = load_finbert()
    if model is None or tokenizer is None:
        return _classify_sentiment_keywords(text)
    
    try:
        import torch
        
        max_length = 512
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim =-1)

        labels = ["negative", "neutral", "positive"]
        predicted_idx = torch.argmax(predictions, dim=-1).item()
        predicted_label = labels[predicted_idx]
        confidence = predictions[0][predicted_idx].item()

        score_map = {"negative": -1, "neutral": 0, "positive": 1}
        score = score_map[predicted_label] * confidence

        elapsed = time.time() - start_time

        if elapsed > 5:
            log(f"Warning: FinBERT sentiment analysis took {elapsed:.2f} seconds, which is longer than expected.")

        return predicted_label, score
    
    except Exception as e:
        log(f"Error during FinBERT sentiment analysis: {e}")
        return _classify_sentiment_keywords(text)
    
def _classify_sentiment_keywords(text: str) -> tuple:
    positive_words = {"bullish", "buy", "long", "moon", "rocket", "gain", "profit", "up", "rally", "surge", "breakout", "growth", "calls"}
    negative_words = {"bearish", "sell", "short", "crash", "dump", "loss", "down", "drop", "fall", "decline", "tank", "plunge", "puts"}

    text_lower = text.lower()
    pos_count = sum(word in positive_words for word in text_lower)
    neg_count = sum(word in negative_words for word in text_lower)

    if pos_count > neg_count:
        return "positive", 1.0
    elif neg_count > pos_count:
        return "negative", -1.0
    else:
        return "neutral", 0.0
    
def annotate_sentiment_for_df(df: pd.DataFrame, use_finbert: bool = True) -> pd.DataFrame:
    df = df.copy()
    
    sentiment_labels = []
    sentiment_scores = []
    
    total_posts = len(df)
    log(f"Processing {total_posts} posts through {'FinBERT' if use_finbert else 'keyword'} sentiment analysis...")
    
    for idx, row in df.iterrows():
        if (idx + 1) % 5 == 0 or idx == 0:
            log(f"  Processing post {idx + 1}/{total_posts} ({(idx+1)/total_posts*100:.1f}%)")
        
        text = f"{row.get('title', '')} {row.get('selftext', '')} {row.get('text', '')}".strip()
        
        if not text:
            sentiment_labels.append("neutral")
            sentiment_scores.append(0.0)
            continue
        
        if use_finbert:
            label, score = analyze_sentiment(text)
        else:
            label, score = _classify_sentiment_keywords(text)
        
        sentiment_labels.append(label)
        sentiment_scores.append(score)
    
    log(f"Completed sentiment analysis for {total_posts} posts")
    
    df["sentiment_label"] = sentiment_labels
    df["sentiment_score"] = sentiment_scores
    
    return df

def save_sentiment(use_finbert: bool = True) -> str:
    reddit_files = sorted(glob.glob(str(project_root / "data" / "raw" / "reddit" / "*.csv")))
    twitter_files = sorted(glob.glob(str(project_root / "data" / "raw" / "twitter" / "*.csv")))
    
    if not reddit_files and not twitter_files:
        log("No Reddit or Twitter data files found.")
        return ""
    
    dfs = []

    if reddit_files:
        latest_reddit = reddit_files[-1]
        log(f"Loading Reddit data from: {Path(latest_reddit).name}")
        df_reddit = pd.read_csv(latest_reddit)
        log(f"  Loaded {len(df_reddit)} Reddit posts")
        dfs.append(df_reddit)
    
    if twitter_files:
        latest_twitter = twitter_files[-1]
        log(f"Loading Twitter data from: {Path(latest_twitter).name}")
        df_twitter = pd.read_csv(latest_twitter)
        log(f"  Loaded {len(df_twitter)} Twitter posts")
        dfs.append(df_twitter)
    
    df = pd.concat(dfs, ignore_index=True)
    log(f"Total posts to process: {len(df)}")
    
    if df.empty:
        log("No posts to process")
        return ""
    
    df = annotate_sentiment_for_df(df, use_finbert=use_finbert)
    
    data_dir = project_root / "data" / "processed" / "sentiment"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"sentiment_annotated_{timestamp}.csv"
    filepath = data_dir / filename
    
    df.to_csv(filepath, index=False)
    log(f"Saved annotated sentiment data to {filepath}")
    
    return str(filepath)

if __name__ == "__main__":
    log("Starting sentiment annotation...")
    filepath = save_sentiment(use_finbert=True)
    
    if filepath:
        log(f"Successfully saved annotated data to: {filepath}")
    else:
        log("Failed to process sentiment data")