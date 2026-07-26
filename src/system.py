import argparse
from src.optimize import run_optimize
from src.predictor import predict

def system_run(ticker, start="2010-01-01", end=None):
    print(f"\n=== OPTIMIZING MODEL FOR {ticker} ===")
    best_params, val_acc = run_optimize(ticker, start=start, end=end)
    print(f"Best hyperparameters: {best_params}")
    print(f"Validation accuracy: {val_acc:.4f}")

    print(f"\n=== DAILY PREDICTION FOR {ticker} ===")
    out = predict(ticker, start=start, end=end)
    print(f"Date: {out['date']}")
    print(f"Prob UP: {out['prob_up']:.4f}")
    print(f"Prob DOWN: {out['prob_down']:.4f}")
    print(f"Signal: {out['signal']}")
    print(f"Model accuracy (historical): {out['model_accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    system_run(
        ticker=args.ticker,
        start=args.start,
        end=args.end
    )
