import argparse
from datetime import datetime

from data_processor import load_data, process_data, get_training_data, handle_missing_data
from engineering.predictor import predict
from engineering.evaluator import evaluate_predictions, compare_predictions


def main(year):
    """
    Main function to orchestrate the prediction pipeline.
    
    TODO:
    - load the data (DONE)
    - get the features (DONE)
    - train the model
        - using the features, parse the data to get the values for each player
        - then with this new little dataset, train the model to predict that player's stats
    - return predicted values into a new dataset and return as a csv file

    - calculate the loss of the predicted values, (Could be a separate program)
        - figure out what the best loss function would be
    """    
    # Load and process the data
    X, y = load_data(year)
    
    # Handle missing data (HITS, TAKE, GIVE before 2005, etc.)
    X = handle_missing_data(X)
    
    # Create training examples from historical data
    training_data = get_training_data(X)
    data, y = process_data(training_data, y)

    # Predict the values
    predictions = predict(data)
    
    # Save predictions to CSV
    predictions.to_csv(f'predictions/predictions_{year}_to_{year+1}.csv', index=False)
    print(f"\nPredictions saved to predictions_{year}_to_{year+1}.csv")
    
    # TODO: Calculate loss if actual data is available
    # if y is not None:
    #     evaluation = evaluate_predictions(y, predictions)
    #     print("\nEvaluation Metrics:")
    #     print(evaluation)
    #     evaluation.to_csv(f'evaluation_{year}_to_{year+1}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    args = parser.parse_args()
    
    if args.year is None:
        year = datetime.now().year - 1  # Default to the previous season
    else:
        year = args.year
    
    main(year)
