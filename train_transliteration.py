"""

Trains transliteration model using command line arguments (CLI)

Example:

    python train_transliteration.py --epochs 150 --batch_size 16 --save_folder .

"""

import torch

import random
import numpy as np
import pandas as pd

import os
import logging
import argparse

from utils import load_transliteration_dataset
from models import TransliterationModel
from set_seed import set_seed


#La fonction logging.basicConfig configure la manière dont les messages d'information seront enregistrés, incluant l'heure et le format
logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
set_seed()

def main():

    parser = argparse.ArgumentParser(description="Train the transliteration transformer model,\
                                         if there is a model already trained, this command would replace it")


    parser.add_argument("--epochs", help="Number for epochs. Default: 100", default=100, type=int)
    parser.add_argument("--batch_size", help="Batch size. Default: 32", default=32, type=int)
    parser.add_argument("--d_model", help="Transformer model dimension. Default 128", default=128, type=int)
    parser.add_argument("--save_folder", help="Folder to save model in. Default: `checkpoint`", default="checkpoint")
    parser.add_argument("--val_frac", help="Fraction of that data that would constitute as \
            the validation set. Default: 0.1", default=0.1, type=float)
    parser.add_argument("--dataset_dir", help="Transliteration dataset directory. Default: `data/external/transliteration`"
            ,default="data/external/transliteration")


    args = parser.parse_args()
    
#charge le jeu de données et Elle retourne le dataset, un dictionnaire de mots connus (known), et leurs indices (known_idx).
    dataset, (known, known_idx) = load_transliteration_dataset(dataset_dir=args.dataset_dir)
#La classe TransliterationModel est initialisée en utilisant le dataset chargé et les arguments passés.
    transliterate_model = TransliterationModel(dataset, load_weights=False, d_model=args.d_model, validation_frac=args.val_frac,
            known=known, known_idx=known_idx, checkpoint_folder=args.save_folder)

    transliterate_model.train_model(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
