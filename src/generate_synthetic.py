"""Generate synthetic regression dataset for pipeline demo."""
import argparse
import os
import pandas as pd
from sklearn.datasets import make_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--repr", required=True, help="Representative sample CSV path")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    X, y = make_regression(n_samples=args.samples, n_features=args.features, noise=args.noise, random_state=42)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # Representative sample
    r = df.sample(n=min(200, len(df)), random_state=1)
    os.makedirs(os.path.dirname(args.repr) or ".", exist_ok=True)
    r.to_csv(args.repr, index=False)

    print("Wrote dataset:", args.out)
    print("Wrote representative sample:", args.repr)


if __name__ == "__main__":
    main()
