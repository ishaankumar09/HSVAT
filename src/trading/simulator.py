import pandas as pd
import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))

from src.models.predict_lstm import load_trained_model, predict_direction_labels, predict_action
from src.models.train_lstm import FEATURE_COLS, create_sequences
from src.trading.alpaca import place_bracket_order, get_account_info
from src.utils.logging import log

def fixed_fraction_position(balance: float, fraction: float = 0.05) -> float:
    return balance * fraction

def stop_loss(entry_price: float, pct: float = 0.02, side: str ="long" ) -> float:
    if side == "long":
        return entry_price * (1 - pct)
    else:
        return entry_price * (1 + pct)
    
def calc_take_profit(entry_price: float, pct: float = 0.04, side: str ="long" ) -> float:
    if side == "long":
        return entry_price * (1 + pct)
    else:
        return entry_price * (1 - pct)
    
def calc_atr(price_df: pd.DataFrame, period: int = 14) -> float:
    if len(price_df) < period:
        return (price_df['high'].max() - price_df['low'].min()) / period
    
    high = price_df['high'].values
    low = price_df['low'].values
    close = price_df['close'].shift(1).fillna(price_df['close']).values

    tr1 = high - low
    tr2 = np.abs(high - close)
    tr3 = np.abs(low - close)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    
    return atr if not np.isnan(atr) else 0.02 * price_df['close'].iloc[-1]

def run_backtest(transaction_cost_pct: float = 0.001) -> None:
    
    full_path =  project_root / "data" / "processed" / "datasets" / "full_dataset.csv"

    if not full_path.exists():
        log("Full dataset not found. Building dataset...")
        from src.features.dataset_builder import build_full_dataset
        build_full_dataset()

    df = pd.read_csv(full_path)

    test_start = int(len(df) * 0.85)
    df = df.iloc[test_start:].reset_index(drop=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    features = df[available_features].values

    model, mean, std, input_dim, output_dim = load_trained_model()

    if mean is not None and std is not None:
        features = (features - mean) / std

    else: 
        log("Warning: No normalization parameters found. Using raw features.")
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(df['target_direction'].values, dtype=torch.long)

    seq_len = 20
    X, y = create_sequences(features_tensor, targets_tensor, seq_len)

    predictions = predict_direction_labels(model, X)

    balance = 100000.0
    balance_gross = 100000.0
    position_fraction = 0.05
    trades = []
    total_transaction_costs = 0.0

    closes = df['close'].values[seq_len:]

    for i in range(len(predictions)-1):
        pred = predictions[i]
        entry_price = closes[i]
        exit_price = closes[i+1]

        if pred == 2:
            action = "buy"
            size = fixed_fraction_position(balance, position_fraction)
            shares = size / entry_price
            pnl_gross = shares * (exit_price - entry_price)

            entry_cost = size * transaction_cost_pct
            exit_cost = shares * exit_price * transaction_cost_pct
            transaction_cost = entry_cost + exit_cost
            pnl_net = pnl_gross - transaction_cost

        elif pred == 0:
            action = "short"
            size = fixed_fraction_position(balance, position_fraction)
            shares = size / entry_price
            pnl_gross = shares * (entry_price - exit_price)

            entry_cost = size * transaction_cost_pct
            exit_cost = shares * exit_price * transaction_cost_pct
            transaction_cost = entry_cost + exit_cost
            pnl_net = pnl_gross - transaction_cost

        else:
            action = "stay"
            size = 0
            pnl_gross = 0
            pnl_net = 0
            transaction_cost = 0
        
        balance += pnl_net
        balance_gross += pnl_gross
        total_transaction_costs += transaction_cost

        trades.append({
            "time": i,
            "action": action,
            "prediction": int(pred),
            "actual": int(y[i]),
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_gross": pnl_gross,
            "transaction_cost": transaction_cost,
            "pnl_net": pnl_net,
            "balance_net": balance,
            "balance_gross": balance_gross
        })

    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trades_df = pd.DataFrame(trades)
    trades_path = log_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    
    initial_balance = 10000.0
    total_return_gross = ((balance_gross - initial_balance) / initial_balance) * 100
    total_return_net = ((balance - initial_balance) / initial_balance) * 100
    cost_drag = total_return_gross - total_return_net
    winning_trades = trades_df[trades_df["pnl_net"] > 0]
    win_rate = len(winning_trades) / len(trades_df[trades_df["action"] != "stay"]) * 100 if len(trades_df[trades_df["action"] != "stay"]) > 0 else 0

    log(f"Backtest results (with {transaction_cost_pct*100:.2f}% transaction costs)")
    log(f"Initial balance: ${initial_balance:,.2f}")
    log(f"Final balance (gross): ${balance_gross:,.2f}")
    log(f"Final balance (net): ${balance:,.2f}")
    log(f"Total return (gross): {total_return_gross:.2f}%")
    log(f"Total return (net): {total_return_net:.2f}%")
    log(f"Transaction costs: ${total_transaction_costs:.2f}")
    log(f"Cost drag: {cost_drag:.2f}%")
    log(f"Total trades: {len(trades_df[trades_df['action'] != 'stay'])}")
    log(f"Win rate: {win_rate:.2f}%")
    log(f"Trades saved to {trades_path}")

def execute_paper_trade(ticker: str, prediction: int, current_price: float, price_df: pd.DataFrame = None) -> dict:
    
    account = get_account_info()
    
    if not account:
        log("Cannot execute: Alpaca not configured")
        return {"error": "Alpaca not configured"}
    
    balance = account.get("buying_power", 0)
    position_size = fixed_fraction_position(balance, 0.05)  # 5% position size
    qty = int(position_size / current_price)
    
    if qty < 1:
        log("Position size too small")
        return {"error": "Position size too small"}
    
    if price_df is not None and len(price_df) > 14:
        atr = calc_atr(price_df, period=14)
        atr_pct = atr / current_price
        
        sl_pct = max(0.005, min(0.03, atr_pct * 1.5))
        tp_pct = sl_pct * 2
        
        log(f"  ATR-based stops: SL={sl_pct*100:.2f}%, TP={tp_pct*100:.2f}%")
    else:
        sl_pct = 0.01  
        tp_pct = 0.02  
        log(f"  Fixed stops: SL={sl_pct*100:.1f}%, TP={tp_pct*100:.1f}%")

    if prediction == 2: 
        side = "buy"
        tp = calc_take_profit(current_price, tp_pct, "long")
        sl = stop_loss(current_price, sl_pct, "long")

    elif prediction == 0: 
        side = "sell"
        tp = calc_take_profit(current_price, tp_pct, "short")
        sl = stop_loss(current_price, sl_pct, "short")
        
    else:  
        log("Prediction is neutral, no trade")
        return {"action": "stay"}
    
    result = place_bracket_order(ticker, qty, side, tp, sl)
    
    return result



