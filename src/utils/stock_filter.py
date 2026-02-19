import yfinance as yf
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import log

def is_blue_chip(ticker: str, min_market_cap: float = 1.5e9, min_volume: int = 5_000_000) -> bool:
    try: 
        stock = yf.Ticker(ticker)
        info = stock.info
        
        market_cap = info.get('marketCap', 0)
        avg_volume = info.get('averageVolume', 0)
        exchange = info.get('exchange', '')
        
        is_large_cap = market_cap > min_market_cap
        is_liquid = avg_volume > min_volume
        is_major_exchange = exchange in ['NMS', 'NYQ', 'NGM', 'NAS']
        
        if is_large_cap and is_liquid and is_major_exchange:
            return True
        else:
            return False
            
    except Exception as e:
        log(f" {ticker} validation failed: {e}")
        return False
    
def has_sufficient_sentiment(num_posts: int, mean_sentiment: float, min_posts: int = 2, min_sentiment_strength: float = 0.3) -> bool:
    if num_posts < min_posts:
        return False
    
    if abs(mean_sentiment) < min_sentiment_strength:
        return False
    
    return True