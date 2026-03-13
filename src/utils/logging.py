from datetime import datetime, timezone
from pathlib import Path

_log_filename = "pipeline.log"

def set_log_file(filename: str):
    global _log_filename
    _log_filename = filename

def log(messge: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {messge}"
    
    print(formatted_message)

    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "data"/ "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / _log_filename
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_message + '\n')
