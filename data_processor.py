import pandas as pd
import numpy as np
import scraper

def handle_missing_data(data):
    """
    Handle missing data by processing player-by-player.
    This makes it easier to:
    - Use player's own historical data (forward/backward fill)
    - Calculate position-based medians
    - Handle pre-2005 stats that weren't tracked
    
    Args:
        data: DataFrame with player stats (must have 'Player Name' column)
        
    Returns:
        DataFrame: DataFrame with missing data handled
    """
    print("Handling missing data . . .")
    
    # Stats that weren't tracked until after 2005 season
    stats_introduced_2005 = ['HIT', 'TAKE', 'GIVE']
    
    # Ensure we have Player Name column
    if 'Player Name' not in data.columns and 'Player' in data.columns:
        data = data.rename(columns={'Player': 'Player Name'})
    
    # Extract year from season (e.g., "2022-23" -> 2022)
    if 'Season' in data.columns:
        data['Year'] = data['Season'].str[:4].astype(int, errors='ignore')
    else:
        # If no Season column, try to infer from other columns or set to 2006 (after stats introduced)
        data['Year'] = 2006
    
    # Convert stat columns to numeric (they might be strings from CSV)
    for stat in stats_introduced_2005:
        if stat in data.columns:
            # Convert to numeric, coercing errors to NaN
            data[stat] = pd.to_numeric(data[stat], errors='coerce')
    
    # Calculate position-based medians for fallback (do this once, outside the loop)
    position_medians = {}
    if 'Pos' in data.columns:
        for stat in stats_introduced_2005:
            if stat in data.columns:
                # Only calculate median if column is numeric
                if pd.api.types.is_numeric_dtype(data[stat]):
                    position_medians[stat] = data.groupby('Pos')[stat].median().to_dict()
    
    # Process each player's data separately
    processed_players = []
    
    for player_name, player_df in data.groupby('Player Name'):
        player_df = player_df.copy()
        player_df = player_df.sort_values('Year' if 'Year' in player_df.columns else 'Season')
        
        # Get player's position (use most common position if multiple)
        player_position = None
        if 'Pos' in player_df.columns:
            player_position = player_df['Pos'].mode()[0] if len(player_df['Pos'].mode()) > 0 else None
        
        # Handle each stat that might be missing
        for stat in stats_introduced_2005:
            if stat not in player_df.columns:
                continue
            
            # Convert to numeric if not already
            if not pd.api.types.is_numeric_dtype(player_df[stat]):
                player_df[stat] = pd.to_numeric(player_df[stat], errors='coerce')
            
            # Step 1: Fill pre-2005 seasons with 0 (stats weren't tracked)
            pre_2005_mask = player_df['Year'] < 2005
            if pre_2005_mask.any():
                player_df.loc[pre_2005_mask, stat] = 0
            
            # Step 2: Forward fill using player's own history (use next available value)
            player_df[stat] = player_df[stat].ffill()
            
            # Step 3: Backward fill using player's own history (use previous available value)
            player_df[stat] = player_df[stat].bfill()
            
            # Step 4: If still missing, use position-based median
            if player_df[stat].isna().any() and player_position and player_position in position_medians.get(stat, {}):
                position_median = position_medians[stat][player_position]
                if not pd.isna(position_median):
                    player_df[stat] = player_df[stat].fillna(position_median)
            
            # Step 5: Final fallback - use overall median for that stat
            if player_df[stat].isna().any():
                if pd.api.types.is_numeric_dtype(data[stat]):
                    overall_median = data[stat].median()
                    player_df[stat] = player_df[stat].fillna(overall_median if not pd.isna(overall_median) else 0)
                else:
                    # If still can't convert, just fill with 0
                    player_df[stat] = player_df[stat].fillna(0)
            
            # Ensure final result is numeric
            player_df[stat] = pd.to_numeric(player_df[stat], errors='coerce').fillna(0)
        
        processed_players.append(player_df)
    
    # Combine all processed player data
    result = pd.concat(processed_players, ignore_index=True)
    
    # Drop the temporary Year column if we created it
    if 'Year' in result.columns and 'Year' not in data.columns:
        result = result.drop(columns=['Year'])
    
    return result

def load_data(year):
    """
    Load training data and actual results data for a given year.
    
    Args:
        year: The year to load data for (e.g., 2023)
        
    Returns:
        tuple: (training_data, actual_results)
            - training_data: DataFrame with player stats for the given year
            - actual_results: DataFrame with actual stats for the next season
    """
    print("Loading data . . .")

    try:  # get the training data from the player_stats csv file
        data = pd.read_csv(f'data/player_stats_{year}.csv')
    except FileNotFoundError:
        print(f"Data for {year} not found, scraping data")
        scraper.scrape_season(year)
        data = pd.read_csv(f'data/player_stats_{year}.csv')

    y_year = year + 1
    try:  # get the prediction data from the season_stats csv file (to measure loss)
        y = pd.read_csv(f'data/season_stats_{y_year}.csv')
    except FileNotFoundError:
        print(f"Data for the next season {y_year} not found, scraping data")
        scraper.scrape_season(year + 1)
        y = pd.read_csv(f'data/season_stats_{y_year}.csv')
    
    return data, y


def process_data(data1, data2):
    """
    Process the data to make it easier to work with by trimming unnecessary 
    text in names and converting time on ice to minutes.
    
    Args:
        data1: Training data DataFrame
        data2: Actual results DataFrame
        
    Returns:
        tuple: (processed_data1, processed_data2)
    """
    print("Processing data . . .")

    data1 = data1.rename(columns={"Player": "Player Name"})
    data2 = data2.rename(columns={"Player": "Player Name"})

    def toi_to_minutes(values: pd.Series) -> pd.Series:
        """
        Convert a TOI column to minutes.
        - If already numeric, return as float minutes.
        - If string like 'MM:SS', convert to minutes.
        - Anything unparseable becomes 0.
        """
        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce").fillna(0).astype(float)

        s = values.astype(str).str.strip()

        # If format is MM:SS, prefix with 00: to make it HH:MM:SS for pandas
        def normalize_time(x: str) -> str:
            if not x or x.lower() == "nan":
                return ""
            if ":" not in x:
                return ""
            # MM:SS -> 00:MM:SS
            if x.count(":") == 1:
                return "00:" + x
            return x

        s = s.map(normalize_time)
        td = pd.to_timedelta(s, errors="coerce")
        return (td.dt.total_seconds() / 60).fillna(0)

    # process X
    if "Player Name" in data1.columns:
        data1["Player Name"] = data1["Player Name"].astype(str).str.strip()

    # training_data already has numeric minutes for TOI; raw scraped data has strings.
    if "Avg_Time_On_Ice" in data1.columns:
        data1["Avg_Time_On_Ice"] = toi_to_minutes(data1["Avg_Time_On_Ice"])
    if "nAvg_Time_On_Ice" in data1.columns:
        data1["nAvg_Time_On_Ice"] = toi_to_minutes(data1["nAvg_Time_On_Ice"])

    # process y
    if "Player Name" in data2.columns:
        data2["Player Name"] = data2["Player Name"].astype(str).str.strip()

    if "ATOI" in data2.columns:
        data2["ATOI"] = toi_to_minutes(data2["ATOI"])

    return data1, data2


def get_training_data(data):
    """
    Prepare the data for training the model by creating features and targets.
    It creates the features needed to train the model and groups the data by player name.
    Processes player-by-player, making it easier to handle missing values.
    
    Args:
        data: DataFrame with player stats (should already have missing data handled)
        
    Returns:
        DataFrame: Training data with features and targets
    """
    print("Getting training data . . .")

    training_data = []

    for player, player_data in data.groupby('Player Name'):
        if len(player_data) < 2:
            continue

        # Sort by season to ensure chronological order
        if 'Season' in player_data.columns:
            player_data = player_data.sort_values('Season')

        for i in range(len(player_data) - 1):
            current_season = player_data.iloc[i]
            next_season = player_data.iloc[i + 1]

            # Skip if games played is 0 or missing (avoid division by zero)
            def safe_num(value, default=0.0) -> float:
                """
                Convert a scalar to float safely (handles strings/NaN).
                """
                num = pd.to_numeric(value, errors='coerce')
                if pd.isna(num):
                    return float(default)
                return float(num)

            current_gp = safe_num(current_season.get('GP', 0), 0.0)
            next_gp = safe_num(next_season.get('GP', 0), 0.0)
            
            if pd.isna(current_gp) or current_gp == 0 or pd.isna(next_gp) or next_gp == 0:
                continue

            # Helper function to safely divide (avoid division by zero)
            def safe_divide(numerator, denominator):
                numerator = safe_num(numerator, 0.0)
                denominator = safe_num(denominator, 0.0)
                if denominator == 0:
                    return 0.0
                return numerator / denominator

            # Calculate the difference between the current and next season
            features = {
                # name and seasonal info (stuff that doesnt/shouldnt change much)
                'Player': player,
                'Season': current_season.get('Season', ''),
                'Next_Season': next_season.get('Season', ''),
                'Age': safe_num(current_season.get('Age', 0), 0.0),
                'Games_Played': current_gp,
                'Position': current_season.get('Pos', ''),

                # scoring stats (per game to normalize)
                'Points_Per_Game': safe_divide(current_season.get('PTS', 0), current_gp),
                'Goals_Per_Game': safe_divide(current_season.get('G', 0), current_gp),
                'Assists_Per_Game': safe_divide(current_season.get('A', 0), current_gp),
                'Shots_Per_Game': safe_divide(current_season.get('SOG', 0), current_gp),
                
                # efficiency stats
                'Shooting_Percentage': safe_num(current_season.get('SPCT', 0), 0.0),
                'Faceoff_Percentage': safe_num(current_season.get('FO%', 0), 0.0),
                'Avg_Time_On_Ice': safe_num(current_season.get('ATOI', 0), 0.0),

                # special teams stats
                'PowerPlay_Goals': safe_num(current_season.get('PPG', 0), 0.0),
                'ShortHanded_Goals': safe_num(current_season.get('SHG', 0), 0.0),
                'Even_Strength_Goals': safe_num(current_season.get('EVG', 0), 0.0),

                # defensive stats (these should be handled by handle_missing_data)
                'Takeaways': safe_num(current_season.get('TAKE', 0), 0.0),
                'Giveaways': safe_num(current_season.get('GIVE', 0), 0.0),
                'Hits': safe_num(current_season.get('HIT', 0), 0.0),       
            }

            targets = {
                # scoring stats
                'nPoints_Per_Game': safe_divide(next_season.get('PTS', 0), next_gp),
                'nGoals_Per_Game': safe_divide(next_season.get('G', 0), next_gp),
                'nAssists_Per_Game': safe_divide(next_season.get('A', 0), next_gp),
                'nShots_Per_Game': safe_divide(next_season.get('SOG', 0), next_gp),
                
                # efficiency stats
                'nShooting_Percentage': safe_num(next_season.get('SPCT', 0), 0.0),
                'nFaceoff_Percentage': safe_num(next_season.get('FO%', 0), 0.0),
                'nAvg_Time_On_Ice': safe_num(next_season.get('ATOI', 0), 0.0),

                # special teams stats
                'nPowerPlay_Goals': safe_num(next_season.get('PPG', 0), 0.0),
                'nShortHanded_Goals': safe_num(next_season.get('SHG', 0), 0.0),
                'nEven_Strength_Goals': safe_num(next_season.get('EVG', 0), 0.0),

                # defensive stats
                'nTakeaways': safe_num(next_season.get('TAKE', 0), 0.0),
                'nGiveaways': safe_num(next_season.get('GIVE', 0), 0.0),
                'nHits': safe_num(next_season.get('HIT', 0), 0.0),       
            }

            training_data.append({**features, **targets})
        
    training_dataframe = pd.DataFrame(training_data)
    
    # Final check: fill any remaining NaN values with 0
    training_dataframe = training_dataframe.fillna(0)
    
    return training_dataframe
