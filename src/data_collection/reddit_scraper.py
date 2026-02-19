import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from src.utils.config_loader import load_config

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def scrape_reddit(limit: int = 100) -> list:
    config = load_config()
    apify_token = config.get('APIFY_API_TOKEN')

    if not apify_token:
        print("APIFY_API_TOKEN not found")
        return []
    
    try:
        from apify_client import ApifyClient
        client = ApifyClient(apify_token)

        run_input = {
            "startUrls": [
                            {"url": "https://www.reddit.com/r/stocks/"},
                            {"url": "https://www.reddit.com/r/wallstreetbets/"},
                            {"url": "https://www.reddit.com/r/investing/"},
                            {"url": "https://www.reddit.com/r/technology/"},
                            {"url": "https://www.reddit.com/r/options/"},
                            {"url": "https://www.reddit.com/r/Daytrading/"},
                            {"url": "https://www.reddit.com/r/StockMarket/"},
                        ],
            "maxItems": limit,
            "maxPostCount": limit,
            "maxComments": 0,
            "skipComments": True,
            "scrollTimeout": 40,
            "proxy": {
                "useApifyProxy": True
            }
        }

        print ("scraping {limit} reddit posts...")
        run = client.actor("trudax/reddit-scraper-lite").call(run_input=run_input)

        posts = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            if item.get("dataType") == "post":
                posts.append({
                    "id": item.get("parsedId", item.get("id", "")),
                    "title": item.get("title", ""),
                    "selftext": item.get("body", ""),
                    "score": item.get("upVotes", 0),
                    "created_utc": pd.to_datetime(item.get("createdAt", datetime.now(timezone.utc).isoformat())).timestamp(),
                    "url": item.get("url", "")
                })

        print(f"Scraped {len(posts)} posts from Reddit.")
        return posts
    
    except ImportError:
        print("Apify client library not installed")
        return []
    
    except Exception as e:
        print(f"Error scraping Reddit: {e}")
        return []

def save_reddit_posts(limit: int) -> str:
    posts = scrape_reddit(limit)

    if not posts:
        return ""
    
    df = pd.DataFrame(posts)
    
    data_dir = project_root / "data" / "raw" / "reddit"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"reddit_{timestamp}.csv"
    filepath = data_dir / filename
    
    df.to_csv(filepath, index=False)
    
    print(f"Saved {len(posts)} posts to {filepath}")
    
    return str(filepath)
    
