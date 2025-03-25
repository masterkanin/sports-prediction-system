"""
Feature Engineering for Sports Prediction System

This module implements advanced feature engineering techniques for the sports prediction system,
including player embeddings, team chemistry metrics, and sport-specific advanced metrics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout

class FeatureEngineering:
    """
    Feature engineering class that handles the creation and transformation of features
    for the sports prediction system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering module.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.player_embedding_model = None
        self.team_chemistry_model = None
        self.scalers = {}
        
    def create_sequential_features(self, player_game_stats, window_sizes=[1, 3, 5, 10]):
        """
        Create sequential features from player game statistics with multiple time windows.
        
        Args:
            player_game_stats (pd.DataFrame): DataFrame containing player game statistics
            window_sizes (list): List of window sizes for rolling averages
            
        Returns:
            pd.DataFrame: DataFrame with sequential features
        """
        features = pd.DataFrame()
        
        # Sort by player and date
        sorted_stats = player_game_stats.sort_values(['player_id', 'game_date'])
        
        # Calculate rolling averages for different window sizes
        for window in window_sizes:
            window_features = sorted_stats.groupby('player_id').rolling(window=window, min_periods=1).mean()
            window_features = window_features.reset_index(level=0, drop=True)
            
            # Rename columns to indicate window size
            renamed_columns = {col: f"{col}_last_{window}" for col in window_features.columns 
                              if col not in ['player_id', 'game_id', 'game_date']}
            window_features = window_features.rename(columns=renamed_columns)
            
            # Add to features DataFrame
            if features.empty:
                features = window_features
            else:
                # Only add the new window columns, not the metadata columns
                for col in renamed_columns.values():
                    features[col] = window_features[col]
        
        # Calculate trend features (difference between short and long term averages)
        for stat in [col for col in sorted_stats.columns if col not in ['player_id', 'game_id', 'game_date']]:
            if f"{stat}_last_3" in features.columns and f"{stat}_last_10" in features.columns:
                features[f"{stat}_trend"] = features[f"{stat}_last_3"] - features[f"{stat}_last_10"]
        
        # Calculate volatility features (standard deviation over last N games)
        for window in [5, 10]:
            volatility_features = sorted_stats.groupby('player_id').rolling(window=window, min_periods=2).std()
            volatility_features = volatility_features.reset_index(level=0, drop=True)
            
            # Rename columns to indicate volatility
            renamed_columns = {col: f"{col}_volatility_{window}" for col in volatility_features.columns 
                              if col not in ['player_id', 'game_id', 'game_date']}
            volatility_features = volatility_features.rename(columns=renamed_columns)
            
            # Add to features DataFrame
            for col in renamed_columns.values():
                features[col] = volatility_features[col]
        
        return features
    
    def create_matchup_features(self, player_stats, opponent_stats):
        """
        Create matchup-specific features based on player and opponent statistics.
        
        Args:
            player_stats (pd.DataFrame): DataFrame containing player statistics
            opponent_stats (pd.DataFrame): DataFrame containing opponent team statistics
            
        Returns:
            pd.DataFrame: DataFrame with matchup features
        """
        matchup_features = pd.DataFrame()
        
        # Merge player and opponent stats
        merged = pd.merge(
            player_stats, 
            opponent_stats, 
            left_on=['game_id', 'opponent_id'], 
            right_on=['game_id', 'team_id'], 
            suffixes=('_player', '_opponent')
        )
        
        # Calculate matchup-specific features
        for stat in [col for col in player_stats.columns if col.endswith('_last_5')]:
            base_stat = stat.replace('_last_5', '')
            opponent_stat = f"{base_stat}_opponent_last_5"
            
            if opponent_stat in merged.columns:
                # Matchup advantage (player stat relative to opponent's average allowed)
                matchup_features[f"{base_stat}_matchup_advantage"] = (
                    merged[stat] / merged[opponent_stat]
                )
        
        # Historical performance against specific opponents
        historical_vs_opponent = player_stats.groupby(['player_id', 'opponent_id']).mean()
        historical_vs_opponent = historical_vs_opponent.reset_index()
        
        # Rename columns to indicate vs_opponent
        renamed_columns = {col: f"{col}_vs_opponent" for col in historical_vs_opponent.columns 
                          if col not in ['player_id', 'opponent_id']}
        historical_vs_opponent = historical_vs_opponent.rename(columns=renamed_columns)
        
        # Merge with matchup features
        matchup_features = pd.merge(
            matchup_features,
            historical_vs_opponent,
            on=['player_id', 'opponent_id'],
            how='left'
        )
        
        return matchup_features
    
    def create_player_embeddings(self, player_stats, embedding_dim=32):
        """
        Create player embeddings that capture playing style and skill level.
        
        Args:
            player_stats (pd.DataFrame): DataFrame containing player statistics
            embedding_dim (int): Dimension of the embeddings
            
        Returns:
            dict: Dictionary mapping player_id to embedding vector
        """
        if self.player_embedding_model is None:
            # Create and train the embedding model if it doesn't exist
            self._train_player_embedding_model(player_stats, embedding_dim)
        
        # Generate embeddings for all players
        player_embeddings = {}
        for player_id in player_stats['player_id'].unique():
            player_data = player_stats[player_stats['player_id'] == player_id]
            
            # Prepare input for the model (select only numeric columns)
            numeric_cols = player_data.select_dtypes(include=[np.number]).columns
            X = player_data[numeric_cols].values
            
            # Generate embedding
            embedding = self.player_embedding_model.predict(X.mean(axis=0).reshape(1, -1))
            player_embeddings[player_id] = embedding.flatten()
        
        return player_embeddings
    
    def _train_player_embedding_model(self, player_stats, embedding_dim):
        """
        Train a model to generate player embeddings.
        
        Args:
            player_stats (pd.DataFrame): DataFrame containing player statistics
            embedding_dim (int): Dimension of the embeddings
        """
        # Select only numeric columns for training
        numeric_cols = player_stats.select_dtypes(include=[np.number]).columns
        X = player_stats[numeric_cols].values
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a simple autoencoder for embeddings
        input_dim = X_scaled.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(embedding_dim, activation='relu', name='embedding')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train the autoencoder
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
        
        # Extract the encoder part
        self.player_embedding_model = Model(inputs=input_layer, outputs=encoded)
    
    def create_team_chemistry_metrics(self, lineup_data, player_embeddings):
        """
        Create team chemistry metrics based on lineup combinations.
        
        Args:
            lineup_data (pd.DataFrame): DataFrame containing lineup information
            player_embeddings (dict): Dictionary mapping player_id to embedding vector
            
        Returns:
            pd.DataFrame: DataFrame with team chemistry metrics
        """
        chemistry_metrics = pd.DataFrame()
        
        # Group by lineup
        for lineup_id, lineup in lineup_data.groupby('lineup_id'):
            player_ids = lineup['player_id'].tolist()
            
            # Get embeddings for all players in the lineup
            lineup_embeddings = [player_embeddings.get(pid) for pid in player_ids]
            
            # Skip if any player doesn't have an embedding
            if None in lineup_embeddings or len(lineup_embeddings) < 2:
                continue
            
            # Calculate chemistry metrics
            
            # 1. Embedding similarity (cosine similarity between players)
            similarities = []
            for i in range(len(lineup_embeddings)):
                for j in range(i+1, len(lineup_embeddings)):
                    similarity = self._cosine_similarity(lineup_embeddings[i], lineup_embeddings[j])
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            # 2. Embedding diversity (standard deviation of embeddings)
            stacked_embeddings = np.vstack(lineup_embeddings)
            diversity = np.mean(np.std(stacked_embeddings, axis=0))
            
            # Add to chemistry metrics
            chemistry_metrics = chemistry_metrics.append({
                'lineup_id': lineup_id,
                'avg_player_similarity': avg_similarity,
                'min_player_similarity': min_similarity,
                'lineup_diversity': diversity
            }, ignore_index=True)
        
        return chemistry_metrics
    
    def _cosine_similarity(self, a, b):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a (np.ndarray): First vector
            b (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def create_advanced_metrics(self, game_stats, sport):
        """
        Create sport-specific advanced metrics.
        
        Args:
            game_stats (pd.DataFrame): DataFrame containing game statistics
            sport (str): Sport name (nba, nfl, mlb, nhl)
            
        Returns:
            pd.DataFrame: DataFrame with advanced metrics
        """
        advanced_metrics = pd.DataFrame()
        
        if sport.lower() == 'nba':
            advanced_metrics = self._create_nba_advanced_metrics(game_stats)
        elif sport.lower() == 'nfl':
            advanced_metrics = self._create_nfl_advanced_metrics(game_stats)
        elif sport.lower() == 'mlb':
            advanced_metrics = self._create_mlb_advanced_metrics(game_stats)
        elif sport.lower() == 'nhl':
            advanced_metrics = self._create_nhl_advanced_metrics(game_stats)
        
        return advanced_metrics
    
    def _create_nba_advanced_metrics(self, game_stats):
        """
        Create advanced metrics for NBA.
        
        Args:
            game_stats (pd.DataFrame): DataFrame containing NBA game statistics
            
        Returns:
            pd.DataFrame: DataFrame with NBA advanced metrics
        """
        advanced = pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['player_id', 'game_id', 'minutes', 'points', 'field_goals_made', 
                         'field_goals_attempted', 'three_pointers_made', 'three_pointers_attempted',
                         'free_throws_made', 'free_throws_attempted', 'rebounds', 'assists',
                         'steals', 'blocks', 'turnovers']
        
        missing_cols = [col for col in required_cols if col not in game_stats.columns]
        if missing_cols:
            print(f"Warning: Missing columns for NBA advanced metrics: {missing_cols}")
            return advanced
        
        # Copy basic info
        advanced['player_id'] = game_stats['player_id']
        advanced['game_id'] = game_stats['game_id']
        
        # True Shooting Percentage (TS%)
        advanced['true_shooting_pct'] = game_stats['points'] / (2 * (game_stats['field_goals_attempted'] + 0.44 * game_stats['free_throws_attempted']))
        
        # Effective Field Goal Percentage (eFG%)
        advanced['effective_fg_pct'] = (game_stats['field_goals_made'] + 0.5 * game_stats['three_pointers_made']) / game_stats['field_goals_attempted']
        
        # Usage Rate (USG%)
        advanced['usage_rate'] = (game_stats['field_goals_attempted'] + 0.44 * game_stats['free_throws_attempted'] + game_stats['turnovers']) / game_stats['minutes']
        
        # Assist to Turnover Ratio
        advanced['assist_to_turnover'] = game_stats['assists'] / (game_stats['turnovers'] + 1)  # Add 1 to avoid division by zero
        
        # Stocks (Steals + Blocks)
        advanced['stocks'] = game_stats['steals'] + game_stats['blocks']
        
        # Box Plus/Minus approximation (simplified)
        advanced['box_plus_minus'] = (
            game_stats['points'] + 2.2 * game_stats['rebounds'] + 
            2 * game_stats['assists'] + 3 * game_stats['steals'] + 
            3 * game_stats['blocks'] - 2 * game_stats['turnovers']
        ) / game_stats['minutes']
        
        return advanced
    
    def _create_nfl_advanced_metrics(self, game_stats):
        """
        Create advanced metrics for NFL.
        
        Args:
            game_stats (pd.DataFrame): DataFrame containing NFL game statistics
            
        Returns:
            pd.DataFrame: DataFrame with NFL advanced metrics
        """
        advanced = pd.DataFrame()
        
        # Copy basic info
        advanced['player_id'] = game_stats['player_id']
        advanced['game_id'] = game_stats['game_id']
        
        # Check if QB stats are available
        if all(col in game_stats.columns for col in ['passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns', 'interceptions']):
            # Passer Rating
            a = ((game_stats['passing_completions'] / game_stats['passing_attempts']) - 0.3) * 5
            b = ((game_stats['passing_yards'] / game_stats['passing_attempts']) - 3) * 0.25
            c = (game_stats['passing_touchdowns'] / game_stats['passing_attempts']) * 20
            d = 2.375 - ((game_stats['interceptions'] / game_stats['passing_attempts']) * 25)
            
            a = a.clip(0, 2.375)
            b = b.clip(0, 2.375)
            c = c.clip(0, 2.375)
            d = d.clip(0, 2.375)
            
            advanced['passer_rating'] = ((a + b + c + d) / 6) * 100
            
            # Adjusted Yards per Attempt
            advanced['adjusted_yards_per_attempt'] = (game_stats['passing_yards'] + 20 * game_stats['passing_touchdowns'] - 45 * game_stats['interceptions']) / game_stats['passing_attempts']
        
        # Check if rushing stats are available
        if all(col in game_stats.columns for col in ['rushing_attempts', 'rushing_yards', 'rushing_touchdowns']):
            # Yards per Carry
            advanced['yards_per_carry'] = game_stats['rushing_yards'] / game_stats['rushing_attempts']
            
            # Rushing Success Rate (simplified)
            if 'rushing_first_downs' in game_stats.columns:
                advanced['rushing_success_rate'] = game_stats['rushing_first_downs'] / game_stats['rushing_attempts']
        
        # Check if receiving stats are available
        if all(col in game_stats.columns for col in ['targets', 'receptions', 'receiving_yards', 'receiving_touchdowns']):
            # Catch Rate
            advanced['catch_rate'] = game_stats['receptions'] / game_stats['targets']
            
            # Yards per Target
            advanced['yards_per_target'] = game_stats['receiving_yards'] / game_stats['targets']
            
            # Yards per Reception
            advanced['yards_per_reception'] = game_stats['receiving_yards'] / game_stats['receptions']
        
        return advanced
    
    def _create_mlb_advanced_metrics(self, game_stats):
        """
        Create advanced metrics for MLB.
        
        Args:
            game_stats (pd.DataFrame): DataFrame containing MLB game statistics
            
        Returns:
            pd.DataFrame: DataFrame with MLB advanced metrics
        """
        advanced = pd.DataFrame()
        
        # Copy basic info
        advanced['player_id'] = game_stats['player_id']
        advanced['game_id'] = game_stats['game_id']
        
        # Check if batting stats are available
        if all(col in game_stats.columns for col in ['at_bats', 'hits', 'doubles', 'triples', 'home_runs', 'walks']):
            # Batting Average
            advanced['batting_avg'] = game_stats['hits'] / game_stats['at_bats']
            
            # On-Base Percentage (OBP)
            if 'hit_by_pitch' in game_stats.columns:
                advanced['on_base_pct'] = (game_stats['hits'] + game_stats['walks'] + game_stats['hit_by_pitch']) / (game_stats['at_bats'] + game_stats['walks'] + game_stats['hit_by_pitch'])
            else:
                advanced['on_base_pct'] = (game_stats['hits'] + game_stats['walks']) / (game_stats['at_bats'] + game_stats['walks'])
            
            # Slugging Percentage (SLG)
            advanced['slugging_pct'] = (game_stats['hits'] - game_stats['doubles'] - game_stats['triples'] - game_stats['home_runs'] + 
                                       2 * game_stats['doubles'] + 3 * game_stats['triples'] + 4 * game_stats['home_runs']) / game_stats['at_bats']
            
            # On-Base Plus Slugging (OPS)
            advanced['ops'] = advanced['on_base_pct'] + advanced['slugging_pct']
            
            # Isolated Power (ISO)
            advanced['iso'] = advanced['slugging_pct'] - advanced['batting_avg']
        
        # Check if pitching stats are available
        if all(col in game_stats.columns for col in ['innings_pitched', 'earned_runs', 'strikeouts', 'walks_allowed']):
            # Earned Run Average (ERA)
            advanced['era'] = 9 * game_stats['earned_runs'] / game_stats['innings_pitched']
            
            # WHIP (Walks and Hits per Inning Pitched)
            if 'hits_allowed' in game_stats.columns:
                advanced['whip'] = (game_stats['walks_allowed'] + game_stats['hits_allowed']) / game_stats['innings_pitched']
            
            # Strikeout Rate (K/9)
            advanced['k_per_9'] = 9 * game_stats['strikeouts'] / game_stats['innings_pitched']
            
            # Walk Rate (BB/9)
            advanced['bb_per_9'] = 9 * game_stats['walks_allowed'] / game_stats['innings_pitched']
            
            # Strikeout to Walk Ratio
            advanced['k_to_bb'] = game_stats['strikeouts'] / game_stats['walks_allowed']
        
        return advanced
    
    def _create_nhl_advanced_metrics(self, game_stats):
        """
        Create advanced metrics for NHL.
        
        Args:
            game_stats (pd.DataFrame): DataFrame containing NHL game statistics
            
        Returns:
            pd.DataFrame: DataFrame with NHL advanced metrics
        """
        advanced = pd.DataFrame()
        
        # Copy basic info
        advanced['player_id'] = game_stats['player_id']
        advanced['game_id'] = game_stats['game_id']
        
        # Check if required columns exist
        if all(col in game_stats.columns for col in ['goals', 'assists', 'shots', 'time_on_ice']):
            # Points
            advanced['points'] = game_stats['goals'] + game_stats['assists']
            
            # Shooting Percentage
            advanced['shooting_pct'] = game_stats['goals'] / game_stats['shots']
            
            # Points per 60 minutes
            minutes = game_stats['time_on_ice'] / 60  # Convert seconds to minutes
            advanced['points_per_60'] = 60 * advanced['points'] / minutes
            
            # Goals per 60 minutes
            advanced['goals_per_60'] = 60 * game_stats['goals'] / minutes
            
            # Assists per 60 minutes
            advanced['assists_per_60'] = 60 * game_stats['assists'] / minutes
        
        # Check if additional stats are available for advanced metrics
        if all(col in game_stats.columns for col in ['blocked_shots', 'hits', 'takeaways', 'giveaways']):
            # Defensive contribution
            advanced['defensive_score'] = game_stats['blocked_shots'] + game_stats['hits'] + game_stats['takeaways'] - game_stats['giveaways']
        
        # Goalie stats
        if all(col in game_stats.columns for col in ['saves', 'shots_against', 'goals_against']):
            # Save Percentage
            advanced['save_pct'] = game_stats['saves'] / game_stats['shots_against']
            
            # Goals Against Average (GAA)
            if 'minutes_played' in game_stats.columns:
                advanced['gaa'] = 60 * game_stats['goals_against'] / game_stats['minutes_played']
        
        return advanced
    
    def normalize_features(self, features, feature_group=None):
        """
        Normalize features using StandardScaler.
        
        Args:
            features (pd.DataFrame): DataFrame containing features
            feature_group (str): Name of the feature group for scaler tracking
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        # Select only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Create a copy to avoid modifying the original
        normalized = features.copy()
        
        # Create or get scaler
        group_key = feature_group or 'default'
        if group_key not in self.scalers:
            self.scalers[group_key] = StandardScaler()
            normalized[numeric_cols] = self.scalers[group_key].fit_transform(features[numeric_cols])
        else:
            normalized[numeric_cols] = self.scalers[group_key].transform(features[numeric_cols])
        
        return normalized
    
    def reduce_dimensions(self, features, n_components=10):
        """
        Reduce dimensions of features using PCA.
        
        Args:
            features (pd.DataFrame): DataFrame containing features
            n_components (int): Number of components for PCA
            
        Returns:
            pd.DataFrame: DataFrame with reduced dimensions
        """
        # Select only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features[numeric_cols])
        
        # Create DataFrame with reduced dimensions
        reduced_df = pd.DataFrame(
            reduced, 
            columns=[f'pca_{i}' for i in range(n_components)],
            index=features.index
        )
        
        # Add non-numeric columns back
        for col in features.columns:
            if col not in numeric_cols:
                reduced_df[col] = features[col]
        
        return reduced_df
