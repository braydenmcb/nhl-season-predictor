import pandas as pd
import numpy as np


def calculate_loss(y_true, y_pred):
    """
    Calculate various loss metrics between actual and predicted values.
    
    Args:
        y_true: Actual values (DataFrame or array)
        y_pred: Predicted values (DataFrame or array)
        
    Returns:
        dict: Dictionary containing various loss metrics
    """
    print("Calculating loss...")
    
    # Convert to numpy arrays if DataFrames
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_predictions(y_true, y_pred, stat_names=None):
    """
    Evaluate predictions for multiple statistics.
    
    Args:
        y_true: DataFrame with actual values
        y_pred: DataFrame with predicted values
        stat_names: List of stat names to evaluate (optional)
        
    Returns:
        DataFrame: Evaluation metrics for each statistic
    """
    if stat_names is None:
        # Default stat names (remove 'n' prefix from target names)
        stat_names = ['Points_Per_Game', 'Goals_Per_Game', 'Assists_Per_Game', 
                     'Shots_Per_Game', 'Shooting_Percentage', 'Faceoff_Percentage', 
                     'Avg_Time_On_Ice', 'PowerPlay_Goals', 'ShortHanded_Goals', 
                     'Even_Strength_Goals', 'Takeaways', 'Giveaways', 'Hits']
    
    results = []
    
    for stat in stat_names:
        true_col = f'n{stat}' if not stat.startswith('n') else stat
        pred_col = f'Predicted_n{stat}' if not stat.startswith('n') else f'Predicted_{stat}'
        
        if true_col in y_true.columns and pred_col in y_pred.columns:
            true_values = y_true[true_col].values
            pred_values = y_pred[pred_col].values
            
            # Remove NaN values for comparison
            mask = ~(np.isnan(true_values) | np.isnan(pred_values))
            true_values = true_values[mask]
            pred_values = pred_values[mask]
            
            if len(true_values) > 0:
                metrics = calculate_loss(true_values, pred_values)
                metrics['Stat'] = stat
                results.append(metrics)
    
    return pd.DataFrame(results)


def compare_predictions(actual_df, predicted_df, player_col='Player'):
    """
    Compare predicted vs actual values for each player.
    
    Args:
        actual_df: DataFrame with actual values
        predicted_df: DataFrame with predicted values
        player_col: Column name for player names
        
    Returns:
        DataFrame: Comparison of predicted vs actual for each player
    """
    # Merge on player name
    comparison = pd.merge(
        actual_df, 
        predicted_df, 
        on=player_col, 
        how='inner',
        suffixes=('_actual', '_predicted')
    )
    
    return comparison
