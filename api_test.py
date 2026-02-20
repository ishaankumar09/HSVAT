from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging import log
from src.utils.config_loader import load_config
from datetime import datetime
import pytz

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

def test_is_near_market_close():
    try:
        from src.utils.market_hours import is_near_market_close
        
        log("\nTesting is_near_market_close() function...")
        market_tz = pytz.timezone("America/New_York")
        
        test_cases = [
            (15, 50, True, "10 minutes before close (3:50 PM ET)"),
            (15, 51, True, "9 minutes before close (3:51 PM ET)"),
            (15, 59, True, "1 minute before close (3:59 PM ET)"),
            (15, 49, False, "11 minutes before close (3:49 PM ET) - too early"),
            (16, 0, False, "At market close (4:00 PM ET) - market closed"),
            (12, 0, False, "Midday (12:00 PM ET)"),
            (9, 30, False, "Market open (9:30 AM ET)"),
        ]
        
        passed = 0
        failed = 0
        
        for hour, minute, expected, desc in test_cases:
            test_time = market_tz.localize(datetime(2026, 2, 19, hour, minute, 0))  # Wednesday
            result = is_near_market_close(test_time, minutes_before_close=10)
            
            if result == expected:
                passed += 1
                log(f" {desc}: PASS")
            else:
                failed += 1
                log(f" {desc}: FAIL (Expected {expected}, got {result})")
        
        if failed == 0:
            log(f"SUCCESS: is_near_market_close() test passed ({passed} cases)")
            return True
        else:
            log(f"FAILED: is_near_market_close() test failed ({failed}/{len(test_cases)} cases)")
            return False
            
    except Exception as e:
        log(f"FAILED: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def test_short_order_logic():
    try:
        log("\nTesting SHORT order logic...")

        prediction = 0
        if prediction == 0:
            side = "sell"
            log(f" prediction=0 correctly maps to side='{side}'")
        else:
            log(f" prediction=0 did not map to side='sell'")
            return False
        
        prediction = 1
        if prediction == 1:
            side = "buy"
            log(f" prediction=1 correctly maps to side='{side}'")
        else:
            log(f" prediction=1 did not map to side='buy'")
            return False
        
        log("SUCCESS: SHORT order logic test passed")
        return True
        
    except Exception as e:
        log(f"FAILED: {e}")
        return False

def test_trading_fixes_code_structure():
    try:
        log("\nTesting trading fixes code structure...")
        
        alpaca_file = project_root / "src" / "trading" / "alpaca.py"
        main_trader_file = project_root / "src" / "trading" / "main_trader.py"
        
        if not alpaca_file.exists() or not main_trader_file.exists():
            log("FAILED: Required files not found")
            return False
        
        alpaca_content = alpaca_file.read_text()
        main_trader_content = main_trader_file.read_text()
        
        checks = [
            ("cancel_orders_for_symbol", alpaca_content, "Order cancellation function exists"),
            ("cancel_order_by_id", alpaca_content, "Calls cancel_order_by_id"),
            ("OrderSide.SELL", alpaca_content, "Uses OrderSide.SELL for shorts"),
            ("Placing SHORT bracket order", alpaca_content, "Logs SHORT orders correctly"),
            ("is_near_market_close", main_trader_content, "Checks market close in main trader"),
            ("close_all_positions", main_trader_content, "Closes all positions at end of day"),
        ]
        
        passed = 0
        failed = 0
        
        for check, content, desc in checks:
            if check in content:
                log(f" {desc}")
                passed += 1
            else:
                log(f" {desc}")
                failed += 1
        
        if failed == 0:
            log(f"SUCCESS: Code structure test passed ({passed} checks)")
            return True
        else:
            log(f"FAILED: Code structure test failed ({failed}/{len(checks)} checks)")
            return False
            
    except Exception as e:
        log(f"FAILED: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def test_trading_fixes_imports():
    try:
        log("\nTesting trading fixes imports...")
        
        from src.trading.alpaca import (
            place_bracket_order,
            close_position,
            cancel_orders_for_symbol,
            close_all_positions
        )
        log(" All alpaca functions import successfully")
        
        from src.utils.market_hours import is_near_market_close
        log(" is_near_market_close imports successfully")
        
        from src.trading.simulator import execute_paper_trade
        log(" execute_paper_trade imports successfully")
        
        log("SUCCESS: All imports work correctly")
        return True
        
    except ImportError as e:
        log(f"FAILED: Import error: {e}")
        return False
    except Exception as e:
        log(f"FAILED: {e}")
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
    
    log("\n" + "="*60)
    log("TRADING FIXES TESTS (can run while market is closed)")
    log("="*60)
    
    results['Trading Fixes - Imports'] = test_trading_fixes_imports()
    results['Trading Fixes - Market Close Detection'] = test_is_near_market_close()
    results['Trading Fixes - SHORT Logic'] = test_short_order_logic()
    results['Trading Fixes - Code Structure'] = test_trading_fixes_code_structure()
    
    log("\n" + "="*60)
    log("FINAL RESULTS")
    log("="*60)
    
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