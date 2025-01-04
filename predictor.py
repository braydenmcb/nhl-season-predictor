import pandas as pd
import numpy as np
# import torch
import argparse
from datetime import datetime
from pathlib import Path

# incase the data loaded is not in the 'data' folder
import scraper

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
def get_features(data):
    print("Getting features")

    features = {
                'Player Name': data["Player Name"],
                'Season': data['Season'],
                'Age': data['Age'],
                # scoring stats
                'Points_Per_Game': data['PTS'] / data['GP'],
                'Goals_Per_Game': data['G'] / data['GP'],
                "Goals": data['G'],
                'PowerPlay_Goals': data['PPG'],
                'Even_Strength_Goals': data['EVG'],
                'Assists_Per_Game': data['A'] / data['GP'],
                'Assists': data['A'],
                'Shots_Per_Game': data['SOG'] / data['GP'],
                'Shooting_Percentage': data['SPCT'],
                'Avg_Time_On_Ice': data['ATOI'],
                'Takeaways': data['TAKE'],
                'Giveaways': data['GIVE'],
                'Hits': data['HIT'],
            }
    features_df = pd.DataFrame(features)
    return features_df


def predict(data):
    print("Predicting")

def calculate_loss(y, y_pred): # y is the actual values, y_pred is the predicted values
    # might set to its own program file later
    print("Calculating loss")

def load_data(year):
    print("Loading data")

    try: # get the training data from the player_stats csv file
        data = pd.read_csv(f'data/player_stats_{year}.csv')
    except FileNotFoundError:
        print(f"Data for {year} not found, scraping data")
        scraper.scrape_season(year)
        data = pd.read_csv(f'data/player_stats_{year}.csv')

    y_year = year + 1
    try: # get the prediction data from the season_stats csv file (to measure loss)
        y = pd.read_csv(f'data/season_stats_{y_year}.csv')
    except FileNotFoundError:
        print(f"Data for the next season {y_year} not found, scraping data")
        scraper.scrape_season(year + 1)
        y = pd.read_csv(f'data/season_stats_{y_year}.csv')
    
    return data, y
    

def main(year):
    """
    #TODO
    - load the data (DONE)
    - get the features (DONE)
    - train the model
        - using the features, parse the data to get the values for each player
        - then with this new little dataset, train the model to predict that player's stats
    - return predicted values into a new dataset and return as a csv file

    - calculate the loss of the predicted values, (Could be a separate program)
        - figure out what the best loss function would be
    """
    data, y = load_data(year)

    # print(data.head())
    # print(y.head())

    headers = data.columns
    # print(headers)
    # clean the data and make easier to work with
    y = y.rename(columns={"Player": "Player Name"})
    data= data.rename(columns={"Player": "Player Name"})

    data["Player Name"] = data["Player Name"].str.strip()
    y["Player Name"] = y["Player Name"].str.strip()
    data["ATOI"] = '00:' + data["ATOI"]
    # Convert the updated 'ATOI' strings to timedelta
    data['ATOI'] = pd.to_timedelta(data['ATOI'], errors='coerce')
    data['ATOI'] = data['ATOI'].dt.total_seconds() / 60

    print(data.head())
    features = get_features(data)
    # print(features)
    print(features.iloc[1])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    args = parser.parse_args()
    if args.year is None:
        main(datetime.now().year - 1) # Default to the previous season
    main(args.year)