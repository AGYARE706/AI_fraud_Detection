"""
Generate a small synthetic creditcard.csv for local testing when Kaggle data is unavailable.
Not for production — schema matches the real dataset for pipeline QA.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "creditcard.csv"


def main(n_rows: int = 10000, fraud_rate: float = 0.02):
    rng = np.random.default_rng(42)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    def block(n, fraud: bool):
        amount = rng.exponential(88 if not fraud else 120, n)
        time = np.sort(rng.uniform(0, 172800, n))
        V = rng.normal(0, 1.5 if not fraud else 2.5, (n, 28))
        if fraud:
            V[:, :5] += rng.normal(2, 1, (n, 5))
        return pd.DataFrame(
            {
                "Time": time,
                "Amount": amount,
                **{f"V{i}": V[:, i - 1] for i in range(1, 29)},
                "Class": int(fraud),
            }
        )

    df = pd.concat([block(n_legit, False), block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT} (fraud rate {df['Class'].mean():.2%})")


if __name__ == "__main__":
    main()
