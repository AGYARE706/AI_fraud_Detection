"""
Train the fraud detection model on data/creditcard.csv.

Run: python train_model.py
"""

from src.train_model import print_training_report, train_pipeline


def main():
    metadata = train_pipeline()
    print_training_report(metadata)


if __name__ == "__main__":
    main()
