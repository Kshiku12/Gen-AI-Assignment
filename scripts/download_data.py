"""
Standalone Script to Download and Preprocess the English-French Parallel Dataset.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import download_and_prepare_dataset


if __name__ == "__main__":
    bundle = download_and_prepare_dataset(
        data_dir="data",
        total_samples=12000,
        min_words=4,
        max_words=18,
        seed=42,
    )
    meta = bundle["metadata"]
    print("\n=== Dataset Preparation Summary ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\nSample parallel pairs (Train):")
    for i in range(3):
        pair = bundle["train"][i]
        print(f"  [{i+1}] EN: {pair['en']}  -->  FR: {pair['fr']}")

