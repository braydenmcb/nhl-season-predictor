import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

# incase the data loaded is not in the 'data' folder
import scraper

# extra imports
import sys
import argparse

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

###################################################################################
########################### DATA MANIPULATION FUNCTIONS ###########################
###################################################################################

def process_data(data1, data2):
    """
    This function will process the data to make it easier to work with
    by trimming unnecessary text in names and converting time on ice to seconds
    """
    print("Processing data . . .")

    data1 = data1.rename(columns={"Player": "Player Name"})
    data2 = data2.rename(columns={"Player": "Player Name"})

    # process X
    data1["Player Name"] = data1["Player Name"].str.strip()
    data1["Avg_Time_On_Ice"] = '00:' + data1["Avg_Time_On_Ice"]
    # Convert the updated 'Avg_Time_On_Ice' strings to timedelta
    data1['Avg_Time_On_Ice'] = pd.to_timedelta(data1['Avg_Time_On_Ice'], errors='coerce')
    data1['Avg_Time_On_Ice'] = data1['Avg_Time_On_Ice'].dt.total_seconds() / 60

    data1['nAvg_Time_On_Ice'] = pd.to_timedelta(data1['nAvg_Time_On_Ice'], errors='coerce')
    data1['nAvg_Time_On_Ice'] = data1['nAvg_Time_On_Ice'].dt.total_seconds() / 60

    # process y
    data2["Player Name"] = data2["Player Name"].str.strip()
    # Ensure ATOI is properly formatted before conversion
    data2["ATOI"] = data2["ATOI"].astype(str)  # Convert to string if it's not already

    # Check if ATOI has valid values before prefixing '00:'
    data2["ATOI"] = data2["ATOI"].apply(lambda x: '00:' + x if ':' not in x else x)

    # Convert the updated 'ATOI' strings to timedelta
    data2['ATOI'] = pd.to_timedelta(data2['ATOI'], errors='coerce')

    # Convert to minutes, replacing NaNs with a default value (e.g., 0)
    data2['ATOI'] = data2['ATOI'].dt.total_seconds() / 60
    data2['ATOI'].fillna(0, inplace=True)  # Replace NaNs with 0 or another appropriate value


    print(data1.iloc[2])
    return data1, data2

def get_training_data(data):
    """
    this function prepares the data for training the model
    It creates the featuers needed to train the model
    and groups the data by player name
    """
    print("Getting training data . . .")

    training_data = []

    for player, player_data in data.groupby('Player Name'):
        if len(player_data) < 2:
            continue

        for i in range(len(player_data) - 1):
            current_season = player_data.iloc[i]
            next_season = player_data.iloc[i + 1]

            # Calculate the difference between the current and next season
            features = {
                # name and seasonal info (stuff that doesnt/shouldnt change much)
                'Player': player,
                'Season': current_season['Season'],
                'Next_Season': next_season['Season'],
                'Age': current_season['Age'],
                'Games_Played': current_season['GP'],
                'Position': current_season['Pos'],

                # scoring stats
                'Points_Per_Game': current_season['PTS'] / current_season['GP'],
                'Goals_Per_Game': current_season['G'] / current_season['GP'],
                'Assists_Per_Game': current_season['A'] / current_season['GP'],
                'Shots_Per_Game': current_season['SOG'] / current_season['GP'],
                
                # efficiency stats
                'Shooting_Percentage': current_season['SPCT'],
                'Faceoff_Percentage': current_season['FO%'],
                'Avg_Time_On_Ice': current_season['ATOI'],

                # special teams stats
                'PowerPlay_Goals': current_season['PPG'],
                'ShortHanded_Goals': current_season['SHG'],
                'Even_Strength_Goals': current_season['EVG'],

                # defensive stats
                'Takeaways': current_season['TAKE'],
                'Giveaways': current_season['GIVE'],
                'Hits': current_season['HIT'],       
            }

            targets = {
                # scoring stats
                'nPoints_Per_Game': next_season['PTS'] / next_season['GP'],
                'nGoals_Per_Game': next_season['G'] / next_season['GP'],
                'nAssists_Per_Game': next_season['A'] / next_season['GP'],
                'nShots_Per_Game': next_season['SOG'] / next_season['GP'],
                
                # efficiency stats
                'nShooting_Percentage': next_season['SPCT'],
                'nFaceoff_Percentage': next_season['FO%'],
                'nAvg_Time_On_Ice': next_season['ATOI'],

                # special teams stats
                'nPowerPlay_Goals': next_season['PPG'],
                'nShortHanded_Goals': next_season['SHG'],
                'nEven_Strength_Goals': next_season['EVG'],

                # defensive stats
                'nTakeaways': next_season['TAKE'],
                'nGiveaways': next_season['GIVE'],
                'nHits': next_season['HIT'],       
            }

            training_data.append({**features, **targets})
        
        training_dataframe = pd.DataFrame(training_data)
    
    print(training_dataframe.head())
    return training_dataframe

def load_data(year):
    print("Loading data . . .")

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

############################################################################
########################### PREDICTION FUNCTIONS ###########################
############################################################################

def gradient_descent(X, y, epochs=1000, step_size=0.01):

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    X = np.c_[np.ones(n), X]

    output_dim = y.shape[1]
    w = np.random.randn(d + 1, output_dim)

    for _ in range(epochs):
        predictions = X @ w
        gradient = (2/n) * (X.T @ (predictions - y))
        w -= step_size * gradient

    def predictor(X_new):
        X_new = np.c_[np.ones(X_new.shape[0]), X_new]
        X_new = np.array(X_new, dtype=float)
        return X_new @ w

    return predictor, w

def predict(data):
    features = ['Points_Per_Game', 'Goals_Per_Game', 'Assists_Per_Game', 'Shots_Per_Game', 'Shooting_Percentage', 
                'Faceoff_Percentage', 'Avg_Time_On_Ice','PowerPlay_Goals', 'ShortHanded_Goals', 'Even_Strength_Goals', 
                'Takeaways', 'Giveaways', 'Hits']
    
    targets = ['nPoints_Per_Game', 'nGoals_Per_Game', 'nAssists_Per_Game', 'nShots_Per_Game', 'nShooting_Percentage', 
                'nFaceoff_Percentage', 'nAvg_Time_On_Ice','nPowerPlay_Goals', 'nShortHanded_Goals', 'nEven_Strength_Goals', 
                'nTakeaways', 'nGiveaways', 'nHits']

    predictions = []

    for player_name, player_df in data.groupby('Player Name'):
        print(f"Predicting for {player_name} . . .")
        print(player_df.head())
        player_data = player_df[features].copy()
        player_y = player_df[targets].copy()

        print("before gradient descent")
        player_predictor, player_weights = gradient_descent(player_data, player_y)
        player_predictions = player_predictor(player_data)
        print("after gradient descent")
        for i in range(len(player_predictions)):
            pred_dict = {f'Predicted_{target}': player_predictions[i][j] for j, target in enumerate(targets)}
            predictions.append({
                'Player': player_name,
                "Sesaon": player_df.iloc[i]['Season'],
                **pred_dict
            }) 
            
        
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df.head())
    

#####################################################################
########################### LOSS FUNCTION ###########################
#####################################################################

def calculate_loss(y, y_pred): # y is the actual values, y_pred is the predicted values
    # might set to its own program file later
    print("Calculating loss")



############################################################
########################### MAIN ###########################
############################################################

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
    # load and process the data
    X, y = load_data(year)
    
    # drop the awards column
    X.drop(columns=["Awards"], inplace=True)
    y.drop(columns=["Awards"], inplace=True)

    training_data = get_training_data(X)
    data, y = process_data(training_data, y)

    # predict the values
    predict(data)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    args = parser.parse_args()
    if args.year is None:
        main(datetime.now().year - 1) # Default to the previous season
    main(args.year)