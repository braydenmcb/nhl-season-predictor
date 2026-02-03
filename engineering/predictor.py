import pandas as pd
import numpy as np


def gradient_descent(X, y, epochs=1000, step_size=0.01):
    """
    Train a linear regression model using gradient descent.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_targets)
        epochs: Number of training iterations
        step_size: Learning rate
        
    Returns:
        tuple: (predictor_function, weights)
            - predictor_function: Function that takes X_new and returns predictions
            - weights: Trained weight matrix
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    X = np.c_[np.ones(n), X]  # Add bias term

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


def predict(data, model_fn=gradient_descent):
    """
    Predict next season stats for players in the data.
    
    TODO: fix the issue where some stats are NaN since they weren't tracked 
    when some players were playing. Also getting some int overflow in matmul 
    and some invalid values in subtraction in gradient_descent()
    
    Args:
        data: DataFrame with training data containing features and targets
        model_fn: Function to train the model (default: gradient_descent)
        
    Returns:
        DataFrame: Predictions for each player
    """
    features = ['Points_Per_Game', 'Goals_Per_Game', 'Assists_Per_Game', 'Shots_Per_Game', 'Shooting_Percentage', 
                'Faceoff_Percentage', 'Avg_Time_On_Ice','PowerPlay_Goals', 'ShortHanded_Goals', 'Even_Strength_Goals', 
                'Takeaways', 'Giveaways', 'Hits']
    
    targets = ['nPoints_Per_Game', 'nGoals_Per_Game', 'nAssists_Per_Game', 'nShots_Per_Game', 'nShooting_Percentage', 
                'nFaceoff_Percentage', 'nAvg_Time_On_Ice','nPowerPlay_Goals', 'nShortHanded_Goals', 'nEven_Strength_Goals', 
                'nTakeaways', 'nGiveaways', 'nHits']

    predictions = []

    test_names = ["Connor McDavid", "Leon Draisaitl", "Sidney Crosby", "Brent Burns"]
    test_data = data[data["Player Name"].isin(test_names)].reset_index(drop=True)
    print(test_data)

    
    for player_name, player_df in test_data.groupby('Player Name'):
        print(f"Predicting for player: {player_name}")
        player_data = player_df[features].copy()
        player_y = player_df[targets].copy()

        player_predictor, player_weights = model_fn(player_data, player_y)
        player_predictions = player_predictor(player_data)

        for i in range(len(player_predictions)):
            pred_dict = {f'Predicted_{target}': player_predictions[i][j] for j, target in enumerate(targets)}
            predictions.append({
                'Player': player_name,
                "Season": player_df.iloc[i]['Season'],
                **pred_dict
            })
            
        
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df.head())
    
    return predictions_df
