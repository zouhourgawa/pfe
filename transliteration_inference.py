import os
import logging
import argparse
import pandas as pd
from models import TransliterationModel
from utils import load_transliteration_dataset, load_sentiment_dataset
from set_seed import set_seed

logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
set_seed()

def main():
    parser = argparse.ArgumentParser(description="Transliterate sentences from CSV files.")
    args = parser.parse_args()

    # Load the transliteration model with pre-loaded weights
    dataset, (known, known_idx) = load_transliteration_dataset()
    transliterate_model = TransliterationModel(dataset, load_weights=True, known=known, known_idx=known_idx)

    # Load data
    train, test = load_sentiment_dataset()

    # Perform transliteration and ensure output length matches input
    train['text_arabic'] = transliterate_model.transliterate_list(train.text.tolist())
    test['text_arabic'] = transliterate_model.transliterate_list(test.text.tolist())

    # Check for any length mismatch and handle it
    assert len(train['text_arabic']) == len(train.text), "Mismatch in train lengths"
    assert len(test['text_arabic']) == len(test.text), "Mismatch in test lengths"

    # Save the results
    train.to_csv("data/interim/Train.csv", index=False)
    test.to_csv("data/interim/Test.csv", index=False)

if __name__ == "__main__":
    main()
