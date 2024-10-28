import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime
from pathlib import Path

def predict(data):
    print("Predicting")

def load_data(year):
    print("Loading data")

def main(year):
    print("hello world")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    args = parser.parse_args()
    if args.year is None:
        main(datetime.now().year - 1) # Default to the previous season
    main(args.year)