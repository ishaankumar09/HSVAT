from datetime import datetime, time, timedelta
import pytz

market_timezone = pytz.timezone("America/New_York")
market_open = time(9, 30)
market_close = time(16, 0)

def is_market_open(current_time: datetime = None) -> bool:
    if now is None:
        now = datetime.now(market_timezone)

    else: 
        if now.tzinfo is None:
            now = market_timezone.localize(now)
        else:
            now = now.astimezone(market_timezone)
    
    if now.weekday() >= 5:
        return False

    current_time = now.time()
    return (market_open <= current_time < market_close)

def get_next_market_open(now: datetime = None) -> datetime:
    if now is None:
        now = datetime.now(market_timezone)
    else:
        if now.tzinfo is None:
            now = market_timezone.localize(now)
        else:
            now = now.astimezone(market_timezone)

    next_open = now.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)
    
    if now.time() >= market_open:
        next_open += timedelta(days=1)
    
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    
    return next_open

def wait_until_market_open():
    from time import sleep

    while (not is_market_open()):
        next_open = get_next_market_open()
        now = datetime.now(market_timezone)
        wait_seconds = (next_open - now).total_seconds()
        if wait_seconds > 0:
            hours = int(wait_seconds // 3600)
            minutes = int((wait_seconds % 3600) // 60)
            print(f"Market closed. Waiting until {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')} ({hours}h {minutes}m)")
            sleep(min(3600, wait_seconds))
            
    print("Market is now open!")