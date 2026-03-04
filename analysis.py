import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf

from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import sys
import re

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

output_dir = project_root / "data" / "analysis"
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"

from src.utils.config_loader import load_config

def get_alpaca_client():
    config = load_config()
    from alpaca.trading.client import TradingClient
    return TradingClient(
        config["ALPACA_API_KEY"],
        config["ALPACA_SECRET_KEY"],
        paper=True
    )

def fetch_closed_orders(start: datetime, end: datetime = None) -> pd.DataFrame:
    client = get_alpaca_client()
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    all_orders = []
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        after = start.isoformat(),
        until = end.isoformat() if end else None,
        limit = 500
    )
    orders = client.get_orders(request)

    for o in orders:
        if str(o.status) != "filled":
            continue
        all_orders.append({
            "order_id": str(o.id),
            "symbol": o.symbol,
            "side": str(o.side),
            "qty": float(o.filled_qty),
            "filled_price": float(o.filled_avg_price),
            "filled_at": pd.Timestamp(o.filled_at).tz_convert("US/Eastern"),
            "order_type": str(o.order_class),
        })

    df = pd.DataFrame(all_orders)
    if not df.empty:
        df = df.sort_values("filled_at").reset_index(drop=True)
    return df

def build_trades(orders_df: pd.DataFrame) -> pd.DataFrame:
    if orders_df.empty:
        return pd.DataFrame()

    trades = []
    for symbol, group in orders_df.groupby("symbol"):
        group = group.sort_values("filled_at").reset_index(drop=True)

        position = None

        for _, order in group.iterrows():
            side = order["side"]

            if position is None:
                position = {
                    "symbol": symbol,
                    "entry_side": side,
                    "entry_price": order["filled_price"],
                    "entry_time": order["filled_at"],
                    "qty": order["qty"],
                    "direction": "long" if side == "buy" else "short",
                }
            else:
                is_closing = (
                    (position["direction"] == "long" and side == "sell") or
                    (position["direction"] == "short" and side == "buy")
                )

                if is_closing:
                    exit_price = order["filled_price"]
                    entry_price = position["entry_price"]
                    qty = min(position["qty"], order["qty"])

                    if position["direction"] == "long":
                        pnl = (exit_price - entry_price) * qty
                    else: 
                        pnl = (entry_price - exit_price) * qty
                    
                    pnl_pct = pnl / (entry_price * qty) * 100
                    hold_minutes = (order["filled_at"] - position["entry_time"]).total_seconds()/60

                    trades.append({
                        "symbol": symbol,
                        "direction": position["direction"],
                        "entry_time": position["entry_time"],
                        "exit_time": order["filled_at"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl_dollars": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 4),
                        "hold_minutes": round(hold_minutes, 1)
                    })

                    remaining_qty = position["qty"] - qty

                    if remaining_qty > 0:
                        position["qty"] = remaining_qty
                    else: 
                        position = None
                else:
                    total_qty = position["qty"] + order["qty"]
                    position["entry_price"] = (
                        (position["entry_price"] * position["qty"] + order["filled_price"] * order["qty"]) / total_qty
                    )
                    position["qty"] = total_qty

    return pd.DataFrame(trades)

def load_all_sentiment() -> pd.DataFrame:
    import glob
    sentiment_dir = project_root / "data" / "processed" / "sentiment"
    files = glob.glob(str(sentiment_dir / "sentiment_agg_by_ticker_*.csv"))

    if not files:
        print ("no sentiment agrregation files found")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    if "bucket_start" in combined.columns:
        combined["bucket_start"] = pd.to_datetime(combined["bucket_start"], utc=True)
    return combined

def match_sentiment_to_trades(trades_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df.empty or trades_df.empty:
        trades_df["avg_sentiment"] = np.nan
        return trades_df
    
    sentiment = []
    for _, trade in trades_df.iterrows():
        ticker = trade["symbol"]
        entry_time = trade["entry_time"]

        mask = (
            (sentiment_df["ticker"] == ticker) &
            (sentiment_df["bucket_start"] <= entry_time) &
            (sentiment_df["bucket_start"] >= entry_time - timedelta(minutes=30))
        )
        relevant = sentiment_df[mask]

        if not relevant.empty and "mean_sentiment_score" in relevant.columns:
            sentiment.append(relevant["mean_sentiment_score"].mean())
        else:
            sentiment.append(np.nan)

    trades_df["avg_sentiment"] = sentiment
    return trades_df

def parse_confidence_from_logs() -> dict:
    log_path = project_root / "data" / "logs" / "pipeline.log"
    if not log_path.exists():
        return {}

    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+): (?:BUY|SHORT) \(class=\d, confidence=(\d+\.\d+)%\)"
    )
    confidences = {}
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ts = pd.Timestamp(m.group(1), tz="UTC").tz_convert("US/Eastern")
                confidences[(m.group(2), ts)] = float(m.group(3))
    return confidences


def match_confidence_to_trades(trades_df: pd.DataFrame, confidences: dict) -> pd.DataFrame:
    if not confidences or trades_df.empty:
        trades_df["confidence"] = np.nan
        return trades_df

    conf_list = []
    for _, trade in trades_df.iterrows():
        best_conf, best_diff = np.nan, timedelta(minutes=5)
        for (t, ts), conf in confidences.items():
            if t == trade["symbol"]:
                diff = abs(trade["entry_time"] - ts)
                if diff < best_diff:
                    best_diff, best_conf = diff, conf
        conf_list.append(best_conf)

    trades_df["confidence"] = conf_list
    return trades_df


def fetch_spy_benchmark(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    spy = yf.Ticker("SPY")
    start = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    df = spy.history(start=start, end=end, interval="1h")
    if df.empty:
        df = spy.history(start=start, end=end, interval="1d")
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["timestamp"] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert("US/Eastern")
    return df[["timestamp", "Close"]].rename(columns={"Close": "spy_close"})


def fetch_portfolio_history(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    client = get_alpaca_client()
    days = (end_date - start_date).days + 1
    try:
        history = client.get_portfolio_history(period=f"{max(days, 1)}D", timeframe="1H")
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(t, unit="s", tz="US/Eastern") for t in history.timestamp],
            "equity": history.equity,
            "pnl": history.profit_loss,
            "pnl_pct": history.profit_loss_pct,
        })
    except Exception as e:
        print(f"Warning: Could not fetch portfolio history: {e}")
        return pd.DataFrame()


def compute_summary_stats(trades_df: pd.DataFrame, initial_balance: float = 200000.0) -> dict:
    if trades_df.empty:
        return {}

    total_pnl = trades_df["pnl_dollars"].sum()
    n_trades = len(trades_df)
    winners = trades_df[trades_df["pnl_dollars"] > 0]
    losers = trades_df[trades_df["pnl_dollars"] < 0]
    win_rate = len(winners) / n_trades * 100

    profit_factor = (
        abs(winners["pnl_dollars"].sum() / losers["pnl_dollars"].sum())
        if len(losers) > 0 and losers["pnl_dollars"].sum() != 0 else float("inf")
    )

    daily_returns = trades_df.groupby(trades_df["entry_time"].dt.date)["pnl_dollars"].sum() / initial_balance
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if n_trades > 1 and daily_returns.std() > 0 else 0

    cumulative = trades_df["pnl_dollars"].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()

    longs = trades_df[trades_df["direction"] == "long"]
    shorts = trades_df[trades_df["direction"] == "short"]
    avg_conf_w = winners["confidence"].mean() if "confidence" in trades_df.columns else np.nan
    avg_conf_l = losers["confidence"].mean() if "confidence" in trades_df.columns else np.nan

    return {
        "Total Trades": n_trades,
        "Winners": len(winners),
        "Losers": len(losers),
        "Breakeven": len(trades_df[trades_df["pnl_dollars"] == 0]),
        "Win Rate (%)": f"{win_rate:.1f}",
        "Total P&L ($)": f"{total_pnl:,.2f}",
        "Total Return (%)": f"{(total_pnl / initial_balance) * 100:.2f}",
        "Avg Win ($)": f"{winners['pnl_dollars'].mean():,.2f}" if len(winners) > 0 else "N/A",
        "Avg Loss ($)": f"{losers['pnl_dollars'].mean():,.2f}" if len(losers) > 0 else "N/A",
        "Largest Win ($)": f"{trades_df['pnl_dollars'].max():,.2f}",
        "Largest Loss ($)": f"{trades_df['pnl_dollars'].min():,.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Sharpe Ratio (ann.)": f"{sharpe:.2f}",
        "Max Drawdown ($)": f"{max_dd:,.2f}",
        "Max Drawdown (%)": f"{(max_dd / initial_balance) * 100:.2f}",
        "Avg Hold Time (min)": f"{trades_df['hold_minutes'].mean():.0f}",
        "Avg Confidence - Winners (%)": f"{avg_conf_w:.1f}" if not np.isnan(avg_conf_w) else "N/A",
        "Avg Confidence - Losers (%)": f"{avg_conf_l:.1f}" if not np.isnan(avg_conf_l) else "N/A",
        "Long Trades": len(longs),
        "Long Win Rate (%)": f"{len(longs[longs['pnl_dollars'] > 0]) / len(longs) * 100:.1f}" if len(longs) > 0 else "N/A",
        "Short Trades": len(shorts),
        "Short Win Rate (%)": f"{len(shorts[shorts['pnl_dollars'] > 0]) / len(shorts) * 100:.1f}" if len(shorts) > 0 else "N/A",
    }


def compute_per_stock_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    stats = trades_df.groupby("symbol").agg(
        Trades=("pnl_dollars", "count"),
        avg_sentiment=("avg_sentiment", "mean"),
        avg_confidence=("confidence", "mean"),
        total_pnl=("pnl_dollars", "sum"),
        avg_pnl=("pnl_dollars", "mean"),
        win_rate=("pnl_dollars", lambda x: (x > 0).sum() / len(x) * 100),
        avg_hold_min=("hold_minutes", "mean"),
    ).sort_values("total_pnl", ascending=False).reset_index()
    stats.columns = ["Ticker", "Trades", "Avg Sentiment", "Avg Conf (%)", "Total P&L ($)", "Avg P&L ($)", "Win Rate (%)", "Avg Hold (min)"]
    return stats.round(2)


def save_fig(fig, name: str):
    fig.savefig(figures_dir / f"{name}.pdf", format="pdf")
    fig.savefig(figures_dir / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  Saved {name}.pdf")


def setup_plot_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6), "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12, "axes.titlesize": 14,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def fig_portfolio_vs_spy(portfolio_df: pd.DataFrame, spy_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not portfolio_df.empty:
        portfolio_df["cum_return"] = (portfolio_df["equity"] / portfolio_df["equity"].iloc[0] - 1) * 100
        ax.plot(portfolio_df["timestamp"], portfolio_df["cum_return"], label="HSVAT Portfolio", color="#2563eb", linewidth=2)
    if not spy_df.empty:
        spy_df["cum_return"] = (spy_df["spy_close"] / spy_df["spy_close"].iloc[0] - 1) * 100
        ax.plot(spy_df["timestamp"], spy_df["cum_return"], label="S&P 500 (SPY)", color="#64748b", linewidth=2, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Portfolio Performance vs. S&P 500")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=45)
    save_fig(fig, "portfolio_vs_spy")


def fig_cumulative_pnl(trades_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    cum_pnl = trades_df["pnl_dollars"].cumsum()
    colors = ["#22c55e" if p > 0 else "#ef4444" for p in trades_df["pnl_dollars"]]
    ax.bar(range(len(cum_pnl)), trades_df["pnl_dollars"], color=colors, alpha=0.7, label="Per-trade P&L")
    ax.plot(range(len(cum_pnl)), cum_pnl.values, color="#2563eb", linewidth=2, label="Cumulative P&L")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("P&L ($)")
    ax.set_title("Per-Trade and Cumulative P&L")
    ax.legend()
    save_fig(fig, "cumulative_pnl")


def fig_confidence_vs_pnl(trades_df: pd.DataFrame):
    df = trades_df.dropna(subset=["confidence"])
    if df.empty:
        print("  Skipping confidence_vs_pnl (no confidence data)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#22c55e" if p > 0 else "#ef4444" for p in df["pnl_dollars"]]
    ax.scatter(df["confidence"], df["pnl_dollars"], c=colors, alpha=0.6, s=60, edgecolors="black", linewidth=0.5)
    z = np.polyfit(df["confidence"], df["pnl_dollars"], 1)
    x_range = np.linspace(df["confidence"].min(), df["confidence"].max(), 100)
    ax.plot(x_range, np.poly1d(z)(x_range), "--", color="#64748b", linewidth=1.5, label=f"Trend (slope={z[0]:.1f})")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Model Confidence (%)")
    ax.set_ylabel("Trade P&L ($)")
    ax.set_title("Model Confidence vs. Trade Outcome")
    ax.legend()
    save_fig(fig, "confidence_vs_pnl")


def fig_pnl_by_hour(trades_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    trades_df["entry_hour"] = trades_df["entry_time"].dt.hour
    hourly = trades_df.groupby("entry_hour").agg(avg_pnl=("pnl_dollars", "mean"), count=("pnl_dollars", "count")).reset_index()
    colors = ["#22c55e" if p > 0 else "#ef4444" for p in hourly["avg_pnl"]]
    bars = ax.bar(hourly["entry_hour"], hourly["avg_pnl"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, hourly["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={count}", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Hour of Day (ET)")
    ax.set_ylabel("Average P&L ($)")
    ax.set_title("Trading Performance by Time of Day")
    ax.set_xticks(hourly["entry_hour"])
    ax.set_xticklabels([f"{h}:00" for h in hourly["entry_hour"]])
    save_fig(fig, "pnl_by_hour")


def fig_sentiment_vs_accuracy(trades_df: pd.DataFrame):
    df = trades_df.dropna(subset=["avg_sentiment"])
    if df.empty:
        print("  Skipping sentiment_vs_accuracy (no sentiment data)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    df["abs_sentiment"] = df["avg_sentiment"].abs()
    df["correct"] = (df["pnl_dollars"] > 0).astype(int)
    df["sentiment_bin"] = pd.cut(df["abs_sentiment"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                  labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    binned = df.groupby("sentiment_bin", observed=True).agg(accuracy=("correct", "mean"), count=("correct", "count")).reset_index()
    bars = ax.bar(range(len(binned)), binned["accuracy"] * 100, color="#2563eb", alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, binned["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={count}", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=50, color="red", linewidth=1, linestyle="--", label="Random (50%)")
    ax.set_xlabel("Absolute Sentiment Score")
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("Prediction Accuracy by Sentiment Strength")
    ax.set_xticks(range(len(binned)))
    ax.set_xticklabels(binned["sentiment_bin"])
    ax.legend()
    save_fig(fig, "sentiment_vs_accuracy")


def fig_long_vs_short(trades_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, direction in enumerate(["long", "short"]):
        subset = trades_df[trades_df["direction"] == direction]
        if subset.empty:
            continue
        axes[i].bar(["Winners", "Losers"],
                    [len(subset[subset["pnl_dollars"] > 0]), len(subset[subset["pnl_dollars"] <= 0])],
                    color=["#22c55e", "#ef4444"], alpha=0.8, edgecolor="black")
        axes[i].set_title(f"{direction.title()} Trades (n={len(subset)})")
        axes[i].set_ylabel("Count")
    fig.suptitle("Long vs. Short Trade Outcomes")
    fig.tight_layout()
    save_fig(fig, "long_vs_short")


def export_trades_csv(trades_df: pd.DataFrame):
    trades_df.to_csv(OUTPUT_DIR / "trades_complete.csv", index=False)
    print("  Saved trades_complete.csv")


def export_summary_txt(stats: dict, stock_stats: pd.DataFrame, trades_df: pd.DataFrame):
    lines = ["HSVAT Post-Hoc Analysis Summary", "=" * 50, "", "SUMMARY STATISTICS", "-" * 30]
    lines += [f"  {k}: {v}" for k, v in stats.items()]
    lines += ["", "PER-STOCK BREAKDOWN", "-" * 30]
    if not stock_stats.empty:
        lines.append(stock_stats.to_string(index=False))
    lines += [
        "",
        f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total trading days: {trades_df['entry_time'].dt.date.nunique()}",
        f"Date range: {trades_df['entry_time'].min()} to {trades_df['exit_time'].max()}",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines))
    print("  Saved analysis_summary.txt")


def main():
    parser = argparse.ArgumentParser(description="HSVAT Post-Hoc Trading Analysis")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--balance", type=float, default=200000.0)
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.start else datetime.now(timezone.utc) - timedelta(days=args.days)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    for d in [output_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"HSVAT Post-Hoc Analysis")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial balance: ${args.balance:,.2f}\n")

    print("[1/6] Fetching orders from Alpaca...")
    orders_df = fetch_closed_orders(start_date, end_date)
    print(f"  Found {len(orders_df)} filled orders")

    print("[2/6] Building trade pairs...")
    trades_df = build_trades(orders_df)
    print(f"  Matched {len(trades_df)} complete trades")

    if trades_df.empty:
        print("\nNo trades found in this period. Exiting.")
        return

    print("[3/6] Loading sentiment data...")
    sentiment_df = load_all_sentiment()
    trades_df = match_sentiment_to_trades(trades_df, sentiment_df)
    print(f"  Matched sentiment for {trades_df['avg_sentiment'].notna().sum()}/{len(trades_df)} trades")

    print("[4/6] Parsing confidence from pipeline logs...")
    confidences = parse_confidence_from_logs()
    trades_df = match_confidence_to_trades(trades_df, confidences)
    print(f"  Matched confidence for {trades_df['confidence'].notna().sum()}/{len(trades_df)} trades")

    print("[5/6] Computing statistics...")
    summary_stats = compute_summary_stats(trades_df, args.balance)
    stock_stats = compute_per_stock_stats(trades_df)
    for key, val in summary_stats.items():
        print(f"  {key}: {val}")

    print("[6/6] Generating outputs...")
    setup_plot_style()
    spy_df = fetch_spy_benchmark(start_date, end_date)
    portfolio_df = fetch_portfolio_history(start_date, end_date)
    fig_portfolio_vs_spy(portfolio_df, spy_df)
    fig_cumulative_pnl(trades_df)
    fig_confidence_vs_pnl(trades_df)
    fig_pnl_by_hour(trades_df)
    fig_sentiment_vs_accuracy(trades_df)
    fig_long_vs_short(trades_df)
    export_trades_csv(trades_df)
    export_summary_txt(summary_stats, stock_stats, trades_df)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()








