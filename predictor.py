import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime
from pathlib import Path
"""
FOR THE MODEL

- just use a basic linear or polynomial regression model for now
- load the data from the csv files (player_stats.csv)
- train the model
- predict the values
- save the model
- return the predicted values in a spreadsheet
- using the data from the csv files (season_stats.csv), calculate the loss
- try to get to around 10% loss at max
"""

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