import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try: 
    from src.utils.logging import log
except:
    def log(message):
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {message}")

hashtags = ["techstocks", "tech", "stocks", "trading", "stockmarket", "investing", "defensestocks"]  

def build_query(use_watchlist: bool = True) -> str:
    if use_watchlist:
        try:
            from src.utils.config_loader import load_watchlist
            watchlist = load_watchlist()
            if watchlist:
                ticker_queries = [f"${ticker}" for ticker in watchlist[:20]]
                hashtag_queries = [f"#{tag}" for tag in hashtags]
                all_terms = ticker_queries + hashtag_queries
                query = " OR ".join(all_terms)
                return query
        except:
            pass

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"]
    ticker_queries = [f"${ticker}" for ticker in tickers]
    hashtag_queries = [f"#{tag}" for tag in hashtags]
    all_terms = ticker_queries + hashtag_queries
    query = " OR ".join(all_terms)
    return query

def scrape_tweets(query: str, limit: int) -> list:
    from src.utils.config_loader import load_config
    config = load_config()
    apify_token = config.get("APIFY_API_TOKEN")

    if not apify_token:
        log("APIFY_API_TOKEN not found in config.")
        return []
    
    try: 
        from apify_client import ApifyClient
        client = ApifyClient(apify_token)

        run_input = {
            "maxItems": limit,
            "query": query,
            "replies": "exclude",
            "retweets": "exclude",
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            }
        }

        log(f"Starting Twitter scraper, limit={limit}...")

        run = client.actor("G8hR9sp2nkjI2om8X").call(run_input)

        tweets = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            text = item.get("text", "")
            if not text:
                continue

            created_at_str = item.get("createdAt", "")
            if created_at_str:
                try:
                    created_utc = pd.to_datetime(created_at_str).timestamp()
                except:
                    created_utc = datetime.now(timezone.utc).timestamp()
            else:
                created_utc = datetime.now(timezone.utc).timestamp()

            tweets.append({
                "id": str(item.get("id", "")),
                "text": text,
                "created_utc": created_utc,
                "author_id": (item.get("username", "")),
                "likes": item.get("likes", 0),
                "retweets": item.get("retweets", 0),
                "replies": item.get("comments", 0),
            })
        
        log(f"Scraped {len(tweets)} tweets.")
        return tweets
    
    except ImportError:
        log("install apify-client.")
        return []
    except Exception as e:
        log(f"Error scraping tweets: {e}")
        return []
    
def save_tweets(limit: int = 50, use_watchlist: bool = True) -> str:
    query = build_query(use_watchlist)
    log(f"Twitter search query: {query[:200]}...")

    tweets = scrape_tweets(query, limit)

    if not tweets:
        log("No tweets to save.")
        return ""

    df = pd.DataFrame(tweets)
    
    data_dir = project_root / "data" / "raw" / "twitter"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"twitter_data_{timestamp}.csv"
    file_path = data_dir / filename

    df.to_csv(file_path, index=False)
    log(f"Saved {len(df)} tweets to {file_path}")

    return str(file_path)