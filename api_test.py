from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging import log
from src.utils.config_loader import load_config

def test_api_keys():
    log("Testing api keys")
    config = load_config()

    checks = {
        "APIFY_API_TOKEN": config.get("APIFY_API_TOKEN"),
        "OPENAI_API_KEY": config.get("OPENAI_API_KEY"),
        "ALPACA_API_KEY": config.get("ALPACA_API_KEY"),
        "ALPACA_SECRET_KEY": config.get("ALPACA_SECRET_KEY"),
    }

    all_good = True
    for key, value in checks.items():
        if not value:
            log(f"Error: {key} is missing.")
            all_good = False
        else:
            log(f"{key} is good.")

    if not all_good:
        log("\nPlease set the missing API keys in the .env file.")
        return False
    
    log("\nAll API keys are present.")
    return True

def test_reddit_api():
    try:
        from src.data_collection.reddit_scraper import scrape_reddit

        
        log("Attempting to fetch 5 posts from r/stocks...")
        posts = scrape_reddit(limit=5)
        
        if posts and len(posts) > 0:
            log(f"SUCCESS: Fetched {len(posts)} Reddit posts")
            log(f"  Sample post: {posts[0].get('title', '')[:80]}...")
            return True
        else:
            log("FAILED: No posts returned")
            log(" Check: Apify subscription")
            return False
            
    except Exception as e:
        log(f"FAILED: {e}")
        log("Check: APIFY_API_TOKEN")
        return False
    
def test_twitter_api():
    try:
        from src.data_collection.twitter_scraper import scrape_twitter

        
        log("Attempting to fetch 5 tweets about $AAPL...")
        tweets = scrape_twitter(query="$AAPL", limit=5)
        
        if tweets and len(tweets) > 0:
            log(f"SUCCESS: Fetched {len(tweets)} tweets")
            log(f"  Sample tweet: {tweets[0].get('text', '')[:80]}...")
            return True
        else:
            log("FAILED: No tweets returned")
            log(" Check: Apify subscription")
            return False
            
    except Exception as e:
        log(f"FAILED: {e}")
        log("Check: APIFY_API_TOKEN")
        return False
    
def main():
    log("Checking API keys ...")
    
    results = {}
    
    if not test_api_keys():
        log("RESULT: API keys not configured properly")
        return
    
    results['Reddit'] = test_reddit_api()
    results['Twitter'] = test_twitter_api()
    
    
    for api, status in results.items():
        log(f"{api}: {'PASSED' if status else 'FAILED'}")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n\nTest stopped by user")
    except Exception as e:
        log(f"\n\nFatal error: {e}")
        import traceback
        log(traceback.format_exc())
