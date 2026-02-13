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
        from src.data_collection.twitter_scraper import scrape_tweets

        
        log("Attempting to fetch 5 tweets about $AAPL...")
        tweets = scrape_tweets(query="$AAPL", limit=5)
        
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
    
def test_openai_api():
    try:
        from src.utils.config_loader import load_config
        from openai import OpenAI

        config = load_config()
        client = OpenAI(api_key=config.get("OPENAI_API_KEY"))

        log("sending test prompt to OpenAI...")

        response = client.chat.completions.create(
             model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' in exactly those words."}
            ],
            max_tokens=20
        )

        result = response.choices[0].message.content

        if "API test successful" in result or "successful" in result.lower():
            log("SUCCESS: OpenAI API is working")
            log(f"  Response: {result}")
            return True
        else:
            log(f"SUCCESS: OpenAI API responded (unusual response: {result})")
            return True
        
    except Exception as e:
        log(f"FAILED: {e}")
        log("Check: OPENAI_API_KEY")
        return False
    
def test_alpaca_api():
    try:
        from src.trading.alpaca import get_account_info

        log("Attempting to fetch Alpaca account info...")
        account = get_account_info()

        if account:
            balance = account.get("buying_power", 0)
            log(f"SUCCESS: Connected to Alpaca")
            log(f"  Paper trading balance: ${balance:,.2f}")
            return True
        else:
            log("FAILED: No account info returned")
            log("Check: Alpaca API keys")
            return False
    
    except Exception as e:
        log(f"FAILED: {e}")
        log("Check: ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return False
 
def main():
    log("Checking API keys ...")
    
    results = {}
    
    if not test_api_keys():
        log("RESULT: API keys not configured properly")
        return
    
    results['Reddit'] = test_reddit_api()
    results['Twitter'] = test_twitter_api()
    results['OpenAI'] = test_openai_api()
    results['Alpaca'] = test_alpaca_api()
    
    
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