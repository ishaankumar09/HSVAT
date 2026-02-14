from pathlib import Path
import sys 

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.main_trader import run_trading
from src.utils.config_loader import load_config
from src.utils.logging import log
import argparse

def validate_setup():

    config= load_config()
    required_keys = [
        "APIFY_API_TOKEN",
        "OPENAI_API_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY"
    ]

    missing = []

    for key in required_keys:
        if not config.get(key):
            missing.append(key)

    if missing:
        log("Missing API keys in .env")
        for key in missing:
            log(f"  - {key}")
        return False
        
    log("All API Keys available")
    return True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "HSVAT Research Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", type=str, default="15min", 
                       help="Time bucket for analysis (default: 15min)")
    parser.add_argument("--max-tickers", type=int, default=5, 
                       help="Max tickers to process per cycle (default: 5)")
    parser.add_argument("--duration", type=float, default=None, 
                       help="Run for specified hours then stop (e.g., 4, 0.5). If not specified, runs 24/7.")
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip API key validation")

    args = parser.parse_args()

    if not args.skip_validation:
        if not validate_setup():
            sys.exit(1)
    
    log("Hybrid Sentiment and Volatility Analysis Trader")
    log(f"Max tickers per cycle: {args.max_tickers}")
    log(f"Time bucket: {args.bucket}")

    if args.duration:
        log(f"Running for ({args.duration} hours)")
    else: 
        log(f"Running Continously (Ctrl+C to Stop)")

    log("")

    try: 
        run_trading(
            bucket=args.bucket,
            max_tickers=args.max_tickers, 
            duration_hours=args.duration
        )
    except KeyboardInterrupt:
        log("\n Bot stopped by user.")
        sys.exit(0)


    
