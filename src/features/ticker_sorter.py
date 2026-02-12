import pandas as pd 
from datetime import datetime, timezone
from pathlib import Path
import glob
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import log
from src.utils.config_loader import load_config

def sort_posts(df: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    api_key = config.get("OPENAI_API_KEY")

    if not api_key:
        log("OPENAI_API_KEY not found in config")
        return pd.DataFrame(
        )
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        posts_data= []

        for idx, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('selftext', '')} {row.get('text', '')}".strip()
            sentiment = row.get("sentiment_label", "neutral")

            if text:
                posts_data.append({
                    "index": idx,
                    "text": text[:500],
                    "sentiment": sentiment
                })

        if not posts_data:
            return pd.DataFrame()
        
        batch_size = 20
        results = []

        for i in range(0, len(posts_data), batch_size):
            batch = posts_data[i:i+batch_size]
            
            prompt_text = "Extract stock tickers from the following financial posts. For each post, return:\n"
            prompt_text += "POST_INDEX|TICKER|SENTIMENT\n"
            prompt_text += "If no ticker is found, use 'NONE' for the ticker.\n"
            prompt_text += "Only return valid stock tickers (1-5 uppercase letters).\n\n"

            for post in batch:
                prompt_text += f"Post {post['index']}: {post['text']}\n"
                prompt_text += f"Sentiment: {post['sentiment']}\n\n"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial data processor. Extract stock tickers from posts and return them in the format: POST_INDEX|TICKER|SENTIMENT. One line per post."
                        },
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ],
                    max_tokens=500,
                    temperature=0
                )

                result_text = response.choices[0].message.content.strip()

                for line in result_text.split("\n"):
                    if "|" in line:
                        parts = line.split("|")
                        if len(parts) >= 2:
                            try:
                                post_idx = int(parts[0].strip())
                                ticker = parts[1].strip().upper()
                                sentiment = parts[2].strip().lower() if len(parts) > 2 else batch[post_idx % len(batch)]['sentiment']

                                if ticker != "NONE" and len(ticker) <= 5:
                                    results.append({
                                        "index": post_idx,
                                        "ticker": ticker,
                                        "sentiment": sentiment
                                    })
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                log(f"Error processing batch with GPT-4o: {e}")
                continue
        
        if results:
            ticker_map = {r["index"]: r["ticker"] for r in results}
            df["ticker"] = df.index.map(lambda x: ticker_map.get(x, "NONE"))
            df = df[df["ticker"] != "NONE"].copy()
        else:
            log("No tickers extracted by GPT-4o-mini")
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        log(f"Error in sort_posts: {e}")
        log("Check OpenAI API key and usage limits")
        return pd.DataFrame()

def save_via_ticker(use_gpt4: bool = True ) -> str:
    data_dir = project_root / "data" / "processed" / "sentiment"
    csv_files = sorted(glob.glob(str(data_dir / "sentiment_annotated_*.csv")))

    if not csv_files:
        log("No annotated sentiment files found.")
        return ""
    
    latest_file = csv_files[-1]
    df = pd.read_csv(latest_file)

    if df.empty:
        return ""
    
    log(f"Sorting {len(df)} posts by ticker...")

    df_sorted = sort_posts(df)

    if df_sorted.empty:
        log("No posts with valid tickers found after sorting.")
        return ""
    
    output_dir = project_root / "data" / "processed" / "sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"sentiment_by_ticker_{timestamp}.csv"
    filepath = output_dir / filename

    columns_to_save = ["ticker", "sentiment_label", "sentiment_score"]

    for col in ['id', 'title', 'selftext', 'text', 'created_utc', 'score']:
        if col in df_sorted.columns:
            columns_to_save.append(col)

    df_sorted[columns_to_save].to_csv(filepath, index=False)
    log(f"Saved {len(df_sorted)} sorted posts to {filepath}")

    return str(filepath)