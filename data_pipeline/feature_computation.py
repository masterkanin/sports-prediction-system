"""
Feature Computation Pipeline for Sports Prediction System

This module implements the feature computation pipeline for the sports prediction system,
including pre-computing advanced metrics, generating rolling averages, and calculating
matchup-specific features.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query
from neural_network.feature_engineering import PlayerEmbedding, TeamChemistryMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_computation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('feature_computation')

class FeatureComputation:
    """
    Class for computing features for the sports prediction system.
    """
    
    def __init__(self):
        """
        Initialize the feature computation pipeline.
        """
        self.player_embedding = PlayerEmbedding()
        self.team_chemistry = TeamChemistryMetrics()
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
    
    def compute_player_features(self, player_id: str, game_id: str, sport: str) -> Dict[str, Any]:
        """
        Compute features for a player prediction.
        
        Args:
            player_id: Player ID
            game_id: Game ID
            sport: Sport code
            
        Returns:
            Dictionary of computed features
        """
        try:
            # Get player information
            player_info = self._get_player_info(player_id)
            
            # Get game information
            game_info = self._get_game_info(game_id)
            
            # Get player's team
            team_id = player_info.get('team_id')
            
            # Get opponent team
            opponent_id = game_info.get('away_team_id') if game_info.get('home_team_id') == team_id else game_info.get('home_team_id')
            
            # Get player's recent stats
            recent_stats = self._get_player_recent_stats(player_id, days=30)
            
            # Get player's historical matchup stats against opponent
            matchup_stats = self._get_player_matchup_stats(player_id, opponent_id)
            
            # Get player's home/away splits
            is_home = game_info.get('home_team_id') == team_id
            home_away_stats = self._get_player_home_away_stats(player_id, is_home)
            
            # Get team's recent performance
            team_stats = self._get_team_recent_stats(team_id, days=30)
            
            # Get opponent's recent performance
            opponent_stats = self._get_team_recent_stats(opponent_id, days=30)
            
            # Get team's defensive stats against player's position
            position_defense = self._get_position_defense_stats(opponent_id, player_info.get('position'))
            
            # Get rest days
            rest_days = self._calculate_rest_days(player_id, game_info.get('game_date'))
            
            # Get travel distance
            travel_distance = self._calculate_travel_distance(team_id, game_info.get('venue'))
            
            # Get weather conditions (for outdoor sports)
            weather = {}
            if sport.lower() in ['nfl', 'mlb', 'soccer']:
                weather = self._get_weather_conditions(game_info.get('venue'), game_info.get('game_date'))
            
            # Get player news sentiment
            news_sentiment = self._get_player_news_sentiment(player_info.get('name'))
            
            # Get player embedding
            embedding = self.player_embedding.get_player_embedding(player_id)
            
            # Get team chemistry
            chemistry = self.team_chemistry.get_lineup_chemistry(team_id, self._get_expected_lineup(team_id, game_id))
            
            # Get advanced metrics based on sport
            advanced_metrics = self._get_sport_specific_advanced_metrics(player_id, sport)
            
            # Combine all features
            features = {
                'player_info': player_info,
                'game_info': game_info,
                'recent_stats': recent_stats,
                'matchup_stats': matchup_stats,
                'home_away_stats': home_away_stats,
                'team_stats': team_stats,
                'opponent_stats': opponent_stats,
                'position_defense': position_defense,
                'rest_days': rest_days,
                'travel_distance': travel_distance,
                'weather': weather,
                'news_sentiment': news_sentiment,
                'player_embedding': embedding,
                'team_chemistry': chemistry,
                'advanced_metrics': advanced_metrics
            }
            
            return features
        except Exception as e:
            logger.error(f"Error computing player features: {e}")
            raise
    
    def compute_rolling_averages(self, player_id: str, stat_type: str, windows: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Compute rolling averages for a player's statistic.
        
        Args:
            player_id: Player ID
            stat_type: Statistic type
            windows: List of window sizes
            
        Returns:
            Dictionary of rolling averages
        """
        try:
            # Get player's game stats
            query = """
            SELECT g.game_date, ar.actual_value
            FROM actual_results ar
            JOIN games g ON ar.game_id = g.game_id
            WHERE ar.player_id = %s AND ar.stat_type = %s
            ORDER BY g.game_date DESC
            """
            
            params = (player_id, stat_type)
            results = execute_query(query, params)
            
            if not results:
                logger.warning(f"No stats found for player {player_id}, stat {stat_type}")
                return {f"rolling_avg_{window}": None for window in windows}
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Compute rolling averages
            rolling_avgs = {}
            for window in windows:
                if len(df) >= window:
                    rolling_avgs[f"rolling_avg_{window}"] = df['actual_value'].head(window).mean()
                else:
                    rolling_avgs[f"rolling_avg_{window}"] = df['actual_value'].mean()
            
            return rolling_avgs
        except Exception as e:
            logger.error(f"Error computing rolling averages: {e}")
            return {f"rolling_avg_{window}": None for window in windows}
    
    def compute_trend_features(self, player_id: str, stat_type: str) -> Dict[str, float]:
        """
        Compute trend features for a player's statistic.
        
        Args:
            player_id: Player ID
            stat_type: Statistic type
            
        Returns:
            Dictionary of trend features
        """
        try:
            # Get player's game stats
            query = """
            SELECT g.game_date, ar.actual_value
            FROM actual_results ar
            JOIN games g ON ar.game_id = g.game_id
            WHERE ar.player_id = %s AND ar.stat_type = %s
            ORDER BY g.game_date DESC
            LIMIT 10
            """
            
            params = (player_id, stat_type)
            results = execute_query(query, params)
            
            if not results or len(results) < 3:
                logger.warning(f"Insufficient stats found for player {player_id}, stat {stat_type}")
                return {
                    'trend_slope': None,
                    'trend_direction': None,
                    'volatility': None
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate trend slope (simple linear regression)
            x = np.arange(len(df))
            y = df['actual_value'].values
            slope, _ = np.polyfit(x, y, 1)
            
            # Calculate trend direction
            if slope > 0.1:
                direction = 'increasing'
            elif slope < -0.1:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Calculate volatility (standard deviation)
            volatility = df['actual_value'].std()
            
            return {
                'trend_slope': slope,
                'trend_direction': direction,
                'volatility': volatility
            }
        except Exception as e:
            logger.error(f"Error computing trend features: {e}")
            return {
                'trend_slope': None,
                'trend_direction': None,
                'volatility': None
            }
    
    def compute_matchup_features(self, player_id: str, opponent_id: str, stat_type: str) -> Dict[str, float]:
        """
        Compute matchup-specific features.
        
        Args:
            player_id: Player ID
            opponent_id: Opponent team ID
            stat_type: Statistic type
            
        Returns:
            Dictionary of matchup features
        """
        try:
            # Get player's stats against this opponent
            query = """
            SELECT ar.actual_value
            FROM actual_results ar
            JOIN games g ON ar.game_id = g.game_id
            JOIN players p ON ar.player_id = p.player_id
            WHERE ar.player_id = %s 
            AND ar.stat_type = %s
            AND (g.home_team_id = %s OR g.away_team_id = %s)
            ORDER BY g.game_date DESC
            """
            
            params = (player_id, stat_type, opponent_id, opponent_id)
            results = execute_query(query, params)
            
            if not results:
                logger.warning(f"No matchup stats found for player {player_id} vs team {opponent_id}")
                return {
                    'matchup_avg': None,
                    'matchup_max': None,
                    'matchup_min': None,
                    'matchup_games': 0
                }
            
            # Convert to list of values
            values = [r['actual_value'] for r in results]
            
            return {
                'matchup_avg': sum(values) / len(values),
                'matchup_max': max(values),
                'matchup_min': min(values),
                'matchup_games': len(values)
            }
        except Exception as e:
            logger.error(f"Error computing matchup features: {e}")
            return {
                'matchup_avg': None,
                'matchup_max': None,
                'matchup_min': None,
                'matchup_games': 0
            }
    
    def compute_contextual_features(self, player_id: str, game_id: str) -> Dict[str, Any]:
        """
        Compute contextual features for a game.
        
        Args:
            player_id: Player ID
            game_id: Game ID
            
        Returns:
            Dictionary of contextual features
        """
        try:
            # Get game information
            game_info = self._get_game_info(game_id)
            
            # Get player information
            player_info = self._get_player_info(player_id)
            
            # Get player's team
            team_id = player_info.get('team_id')
            
            # Determine if home or away
            is_home = game_info.get('home_team_id') == team_id
            
            # Get team's record
            team_record = self._get_team_record(team_id)
            
            # Get opponent's record
            opponent_id = game_info.get('away_team_id') if is_home else game_info.get('home_team_id')
            opponent_record = self._get_team_record(opponent_id)
            
            # Calculate team strength differential
            team_strength_diff = (team_record.get('win_pct', 0.5) - opponent_record.get('win_pct', 0.5)) * 100
            
            # Get game importance (e.g., playoff implications)
            game_importance = self._calculate_game_importance(team_id, opponent_id, game_info.get('game_date'))
            
            # Get back-to-back status
            back_to_back = self._is_back_to_back(team_id, game_info.get('game_date'))
            
            # Get player's minutes trend
            minutes_trend = self._get_player_minutes_trend(player_id)
            
            # Get player's usage rate
            usage_rate = self._get_player_usage_rate(player_id)
            
            # Get player's injury status
            injury_status = self._get_player_injury_status(player_id)
            
            # Get team's pace
            team_pace = self._get_team_pace(team_id)
            
            # Get opponent's pace
            opponent_pace = self._get_team_pace(opponent_id)
            
            # Get expected game pace
            expected_pace = (team_pace + opponent_pace) / 2
            
            # Get expected game total (points)
            expected_total = self._get_expected_game_total(team_id, opponent_id)
            
            # Get expected game spread
            expected_spread = self._get_expected_game_spread(team_id, opponent_id)
            
            return {
                'is_home': is_home,
                'team_record': team_record,
                'opponent_record': opponent_record,
                'team_strength_diff': team_strength_diff,
                'game_importance': game_importance,
                'back_to_back': back_to_back,
                'minutes_trend': minutes_trend,
                'usage_rate': usage_rate,
                'injury_status': injury_status,
                'team_pace': team_pace,
                'opponent_pace': opponent_pace,
                'expected_pace': expected_pace,
                'expected_total': expected_total,
                'expected_spread': expected_spread
            }
        except Exception as e:
            logger.error(f"Error computing contextual features: {e}")
            return {}
    
    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize features for model input.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Dictionary of normalized features
        """
        try:
            # Extract numerical features
            numerical_features = {}
            for category, values in features.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)) and value is not None:
                            numerical_features[f"{category}_{key}"] = value
                elif isinstance(values, (int, float)) and values is not None:
                    numerical_features[category] = values
            
            # Convert to DataFrame
            df = pd.DataFrame([numerical_features])
            
            # Fill missing values
            df = df.fillna(df.mean())
            
            # Normalize using StandardScaler
            normalized_df = pd.DataFrame(
                self.standard_scaler.fit_transform(df),
                columns=df.columns
            )
            
            # Convert back to dictionary
            normalized_features = normalized_df.iloc[0].to_dict()
            
            # Add back non-numerical features
            for category, values in features.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if not isinstance(value, (int, float)) or value is None:
                            normalized_features[f"{category}_{key}"] = value
                elif not isinstance(values, (int, float)) or values is None:
                    normalized_features[category] = values
            
            return normalized_features
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features
    
    def reduce_dimensions(self, features: Dict[str, Any], n_components: int = 50) -> Dict[str, Any]:
        """
        Reduce dimensionality of features using PCA.
        
        Args:
            features: Dictionary of features
            n_components: Number of components to keep
            
        Returns:
            Dictionary of reduced features
        """
        try:
            # Extract numerical features
            numerical_features = {}
            for category, values in features.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)) and value is not None:
                            numerical_features[f"{category}_{key}"] = value
                elif isinstance(values, (int, float)) and values is not None:
                    numerical_features[category] = values
            
            # Convert to DataFrame
            df = pd.DataFrame([numerical_features])
            
            # Fill missing values
            df = df.fillna(df.mean())
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, len(df.columns)))
            reduced_df = pd.DataFrame(
                pca.fit_transform(df),
                columns=[f"PC{i+1}" for i in range(min(n_components, len(df.columns)))]
            )
            
            # Convert back to dictionary
            reduced_features = reduced_df.iloc[0].to_dict()
            
            # Add explained variance
            reduced_features['explained_variance'] = sum(pca.explained_variance_ratio_)
            
            # Add back non-numerical features
            for category, values in features.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if not isinstance(value, (int, float)) or value is None:
                            reduced_features[f"{category}_{key}"] = value
                elif not isinstance(values, (int, float)) or values is None:
                    reduced_features[category] = values
            
            return reduced_features
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            return features
    
    def prepare_model_input(self, features: Dict[str, Any], stat_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare features for model input.
        
        Args:
            features: Dictionary of features
            stat_type: Statistic type
            
        Returns:
            Tuple of (model input array, feature metadata)
        """
        try:
            # Extract relevant features for this stat type
            relevant_features = self._select_relevant_features(features, stat_type)
            
            # Normalize features
            normalized_features = self.normalize_features(relevant_features)
            
            # Convert to flat array
            feature_array = self._flatten_features(normalized_features)
            
            # Create feature metadata
            feature_metadata = {
                'feature_names': list(normalized_features.keys()),
                'original_features': features,
                'stat_type': stat_type
            }
            
            return feature_array, feature_metadata
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            raise
    
    def _get_player_info(self, player_id: str) -> Dict[str, Any]:
        """
        Get player information from the database.
        
        Args:
            player_id: Player ID
            
        Returns:
            Player information
        """
        query = "SELECT * FROM players WHERE player_id = %s"
        params = (player_id,)
        results = execute_query(query, params)
        
        if results:
            return results[0]
        return {}
    
    def _get_game_info(self, game_id: str) -> Dict[str, Any]:
        """
        Get game information from the database.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game information
        """
        query = "SELECT * FROM games WHERE game_id = %s"
        params = (game_id,)
        results = execute_query(query, params)
        
        if results:
            return results[0]
        return {}
    
    def _get_player_recent_stats(self, player_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get player's recent stats.
        
        Args:
            player_id: Player ID
            days: Number of days to look back
            
        Returns:
            List of recent stats
        """
        query = """
        SELECT ar.*, g.game_date
        FROM actual_results ar
        JOIN games g ON ar.game_id = g.game_id
        WHERE ar.player_id = %s
        AND g.game_date >= %s
        ORDER BY g.game_date DESC
        """
        
        params = (player_id, (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
        return execute_query(query, params)
    
    def _get_player_matchup_stats(self, player_id: str, opponent_id: str) -> List[Dict[str, Any]]:
        """
        Get player's historical matchup stats against opponent.
        
        Args:
            player_id: Player ID
            opponent_id: Opponent team ID
            
        Returns:
            List of matchup stats
        """
        query = """
        SELECT ar.*, g.game_date
        FROM actual_results ar
        JOIN games g ON ar.game_id = g.game_id
        WHERE ar.player_id = %s
        AND (g.home_team_id = %s OR g.away_team_id = %s)
        ORDER BY g.game_date DESC
        """
        
        params = (player_id, opponent_id, opponent_id)
        return execute_query(query, params)
    
    def _get_player_home_away_stats(self, player_id: str, is_home: bool) -> Dict[str, Any]:
        """
        Get player's home/away splits.
        
        Args:
            player_id: Player ID
            is_home: Whether the player is at home
            
        Returns:
            Home/away stats
        """
        query = """
        SELECT 
            ar.stat_type,
            AVG(ar.actual_value) as avg_value,
            COUNT(*) as games
        FROM actual_results ar
        JOIN games g ON ar.game_id = g.game_id
        JOIN players p ON ar.player_id = p.player_id
        WHERE ar.player_id = %s
        AND (
            (g.home_team_id = p.team_id AND %s = TRUE) OR
            (g.away_team_id = p.team_id AND %s = FALSE)
        )
        GROUP BY ar.stat_type
        """
        
        params = (player_id, is_home, is_home)
        results = execute_query(query, params)
        
        # Convert to dictionary
        stats = {}
        for row in results:
            stats[row['stat_type']] = {
                'avg_value': row['avg_value'],
                'games': row['games']
            }
        
        return stats
    
    def _get_team_recent_stats(self, team_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get team's recent performance.
        
        Args:
            team_id: Team ID
            days: Number of days to look back
            
        Returns:
            Team stats
        """
        query = """
        SELECT 
            g.*,
            CASE 
                WHEN g.home_team_id = %s THEN g.home_score
                ELSE g.away_score
            END as team_score,
            CASE 
                WHEN g.home_team_id = %s THEN g.away_score
                ELSE g.home_score
            END as opponent_score
        FROM games g
        WHERE (g.home_team_id = %s OR g.away_team_id = %s)
        AND g.game_date >= %s
        AND g.status = 'completed'
        ORDER BY g.game_date DESC
        """
        
        params = (
            team_id, team_id, team_id, team_id,
            (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        )
        results = execute_query(query, params)
        
        # Calculate aggregate stats
        if not results:
            return {}
        
        wins = sum(1 for r in results if (r['home_team_id'] == team_id and r['home_score'] > r['away_score']) or
                                         (r['away_team_id'] == team_id and r['away_score'] > r['home_score']))
        
        return {
            'games': len(results),
            'wins': wins,
            'losses': len(results) - wins,
            'win_pct': wins / len(results) if results else 0,
            'avg_score': sum(r['team_score'] for r in results) / len(results) if results else 0,
            'avg_opponent_score': sum(r['opponent_score'] for r in results) / len(results) if results else 0,
            'point_diff': sum(r['team_score'] - r['opponent_score'] for r in results) / len(results) if results else 0
        }
    
    def _get_position_defense_stats(self, team_id: str, position: str) -> Dict[str, Any]:
        """
        Get team's defensive stats against a position.
        
        Args:
            team_id: Team ID
            position: Player position
            
        Returns:
            Position defense stats
        """
        query = """
        SELECT 
            ar.stat_type,
            AVG(ar.actual_value) as avg_value,
            COUNT(*) as games
        FROM actual_results ar
        JOIN games g ON ar.game_id = g.game_id
        JOIN players p ON ar.player_id = p.player_id
        WHERE p.position = %s
        AND (g.home_team_id = %s OR g.away_team_id = %s)
        AND p.team_id != %s
        GROUP BY ar.stat_type
        """
        
        params = (position, team_id, team_id, team_id)
        results = execute_query(query, params)
        
        # Convert to dictionary
        stats = {}
        for row in results:
            stats[row['stat_type']] = {
                'avg_value': row['avg_value'],
                'games': row['games']
            }
        
        return stats
    
    def _calculate_rest_days(self, player_id: str, game_date: str) -> int:
        """
        Calculate rest days for a player.
        
        Args:
            player_id: Player ID
            game_date: Game date
            
        Returns:
            Number of rest days
        """
        query = """
        SELECT MAX(g.game_date) as last_game_date
        FROM actual_results ar
        JOIN games g ON ar.game_id = g.game_id
        WHERE ar.player_id = %s
        AND g.game_date < %s
        """
        
        params = (player_id, game_date)
        results = execute_query(query, params)
        
        if not results or not results[0]['last_game_date']:
            return 7  # Default to a week if no previous game
        
        last_game = datetime.strptime(results[0]['last_game_date'], '%Y-%m-%d')
        current_game = datetime.strptime(game_date, '%Y-%m-%d')
        
        return (current_game - last_game).days
    
    def _calculate_travel_distance(self, team_id: str, venue: str) -> float:
        """
        Calculate travel distance for a team.
        
        Args:
            team_id: Team ID
            venue: Venue name
            
        Returns:
            Travel distance in miles
        """
        # This is a placeholder - in a real implementation, this would use
        # geolocation data to calculate actual distances
        return 0.0
    
    def _get_weather_conditions(self, venue: str, game_date: str) -> Dict[str, Any]:
        """
        Get weather conditions for a game.
        
        Args:
            venue: Venue name
            game_date: Game date
            
        Returns:
            Weather conditions
        """
        # This is a placeholder - in a real implementation, this would use
        # the WeatherAPI to get actual weather data
        return {}
    
    def _get_player_news_sentiment(self, player_name: str) -> Dict[str, float]:
        """
        Get sentiment analysis of recent news about a player.
        
        Args:
            player_name: Player name
            
        Returns:
            News sentiment scores
        """
        # This is a placeholder - in a real implementation, this would use
        # the NewsAPI and sentiment analysis
        return {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }
    
    def _get_expected_lineup(self, team_id: str, game_id: str) -> List[str]:
        """
        Get expected lineup for a team in a game.
        
        Args:
            team_id: Team ID
            game_id: Game ID
            
        Returns:
            List of player IDs in the expected lineup
        """
        # This is a placeholder - in a real implementation, this would use
        # historical lineup data and injury reports
        return []
    
    def _get_sport_specific_advanced_metrics(self, player_id: str, sport: str) -> Dict[str, Any]:
        """
        Get sport-specific advanced metrics for a player.
        
        Args:
            player_id: Player ID
            sport: Sport code
            
        Returns:
            Advanced metrics
        """
        # This is a placeholder - in a real implementation, this would use
        # sport-specific APIs and calculations
        if sport.lower() == 'nba':
            return {
                'raptor': 0.0,
                'per': 0.0,
                'ts_pct': 0.0,
                'usage_rate': 0.0
            }
        elif sport.lower() == 'nfl':
            return {
                'qbr': 0.0,
                'dvoa': 0.0,
                'epa': 0.0
            }
        elif sport.lower() == 'mlb':
            return {
                'war': 0.0,
                'woba': 0.0,
                'ops_plus': 0.0
            }
        elif sport.lower() == 'nhl':
            return {
                'corsi': 0.0,
                'fenwick': 0.0,
                'pdo': 0.0
            }
        else:
            return {}
    
    def _get_team_record(self, team_id: str) -> Dict[str, Any]:
        """
        Get team's current record.
        
        Args:
            team_id: Team ID
            
        Returns:
            Team record
        """
        query = """
        SELECT 
            COUNT(*) as games,
            SUM(CASE 
                WHEN (g.home_team_id = %s AND g.home_score > g.away_score) OR
                     (g.away_team_id = %s AND g.away_score > g.home_score)
                THEN 1 ELSE 0 END) as wins
        FROM games g
        WHERE (g.home_team_id = %s OR g.away_team_id = %s)
        AND g.status = 'completed'
        """
        
        params = (team_id, team_id, team_id, team_id)
        results = execute_query(query, params)
        
        if not results:
            return {'games': 0, 'wins': 0, 'losses': 0, 'win_pct': 0.0}
        
        games = results[0]['games']
        wins = results[0]['wins']
        
        return {
            'games': games,
            'wins': wins,
            'losses': games - wins,
            'win_pct': wins / games if games > 0 else 0.0
        }
    
    def _calculate_game_importance(self, team_id: str, opponent_id: str, game_date: str) -> float:
        """
        Calculate the importance of a game.
        
        Args:
            team_id: Team ID
            opponent_id: Opponent team ID
            game_date: Game date
            
        Returns:
            Game importance score (0-1)
        """
        # This is a placeholder - in a real implementation, this would use
        # standings, playoff implications, rivalries, etc.
        return 0.5
    
    def _is_back_to_back(self, team_id: str, game_date: str) -> bool:
        """
        Check if a game is the second of a back-to-back.
        
        Args:
            team_id: Team ID
            game_date: Game date
            
        Returns:
            True if back-to-back, False otherwise
        """
        query = """
        SELECT COUNT(*) as count
        FROM games g
        WHERE (g.home_team_id = %s OR g.away_team_id = %s)
        AND g.game_date = %s::date - interval '1 day'
        """
        
        params = (team_id, team_id, game_date)
        results = execute_query(query, params)
        
        return results[0]['count'] > 0 if results else False
    
    def _get_player_minutes_trend(self, player_id: str) -> Dict[str, float]:
        """
        Get player's minutes trend.
        
        Args:
            player_id: Player ID
            
        Returns:
            Minutes trend information
        """
        query = """
        SELECT pgs.minutes_played, g.game_date
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE pgs.player_id = %s
        ORDER BY g.game_date DESC
        LIMIT 10
        """
        
        params = (player_id,)
        results = execute_query(query, params)
        
        if not results or len(results) < 3:
            return {'avg_minutes': 0, 'trend': 0}
        
        minutes = [r['minutes_played'] for r in results if r['minutes_played'] is not None]
        
        if not minutes:
            return {'avg_minutes': 0, 'trend': 0}
        
        avg_minutes = sum(minutes) / len(minutes)
        
        # Calculate trend (simple linear regression)
        x = np.arange(len(minutes))
        y = np.array(minutes)
        slope, _ = np.polyfit(x, y, 1)
        
        return {
            'avg_minutes': avg_minutes,
            'trend': slope
        }
    
    def _get_player_usage_rate(self, player_id: str) -> float:
        """
        Get player's usage rate.
        
        Args:
            player_id: Player ID
            
        Returns:
            Usage rate
        """
        # This is a placeholder - in a real implementation, this would use
        # advanced stats calculations
        return 0.0
    
    def _get_player_injury_status(self, player_id: str) -> str:
        """
        Get player's injury status.
        
        Args:
            player_id: Player ID
            
        Returns:
            Injury status
        """
        # This is a placeholder - in a real implementation, this would use
        # injury reports
        return 'healthy'
    
    def _get_team_pace(self, team_id: str) -> float:
        """
        Get team's pace.
        
        Args:
            team_id: Team ID
            
        Returns:
            Team pace
        """
        # This is a placeholder - in a real implementation, this would use
        # advanced stats calculations
        return 0.0
    
    def _get_expected_game_total(self, team_id: str, opponent_id: str) -> float:
        """
        Get expected game total (points).
        
        Args:
            team_id: Team ID
            opponent_id: Opponent team ID
            
        Returns:
            Expected game total
        """
        # This is a placeholder - in a real implementation, this would use
        # team stats and betting lines
        return 0.0
    
    def _get_expected_game_spread(self, team_id: str, opponent_id: str) -> float:
        """
        Get expected game spread.
        
        Args:
            team_id: Team ID
            opponent_id: Opponent team ID
            
        Returns:
            Expected game spread
        """
        # This is a placeholder - in a real implementation, this would use
        # team stats and betting lines
        return 0.0
    
    def _select_relevant_features(self, features: Dict[str, Any], stat_type: str) -> Dict[str, Any]:
        """
        Select relevant features for a specific stat type.
        
        Args:
            features: Dictionary of features
            stat_type: Statistic type
            
        Returns:
            Dictionary of relevant features
        """
        # This is a simplified implementation - in a real system, this would be more sophisticated
        # and would select different features for different stat types
        return features
    
    def _flatten_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Flatten nested feature dictionary into a 1D array.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Flattened feature array
        """
        # Extract numerical features
        flat_features = {}
        for category, values in features.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float)) and value is not None:
                        flat_features[f"{category}_{key}"] = value
            elif isinstance(values, (int, float)) and values is not None:
                flat_features[category] = values
        
        # Convert to array
        return np.array(list(flat_features.values()))


class FeatureStore:
    """
    Class for storing and retrieving pre-computed features.
    """
    
    def __init__(self):
        """
        Initialize the feature store.
        """
        pass
    
    def store_features(self, player_id: str, game_id: str, stat_type: str, features: Dict[str, Any]) -> bool:
        """
        Store features in the database.
        
        Args:
            player_id: Player ID
            game_id: Game ID
            stat_type: Statistic type
            features: Dictionary of features
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            INSERT INTO feature_store (
                player_id, game_id, stat_type, features, created_at
            ) VALUES (
                %s, %s, %s, %s, %s
            ) ON CONFLICT (player_id, game_id, stat_type) DO UPDATE SET
                features = EXCLUDED.features,
                created_at = EXCLUDED.created_at
            """
            
            params = (
                player_id,
                game_id,
                stat_type,
                json.dumps(features),
                datetime.now()
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Stored features for player {player_id}, game {game_id}, stat {stat_type}")
            return True
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False
    
    def get_features(self, player_id: str, game_id: str, stat_type: str) -> Optional[Dict[str, Any]]:
        """
        Get features from the database.
        
        Args:
            player_id: Player ID
            game_id: Game ID
            stat_type: Statistic type
            
        Returns:
            Dictionary of features or None if not found
        """
        try:
            query = """
            SELECT features
            FROM feature_store
            WHERE player_id = %s AND game_id = %s AND stat_type = %s
            """
            
            params = (player_id, game_id, stat_type)
            results = execute_query(query, params)
            
            if results and results[0]['features']:
                return json.loads(results[0]['features'])
            return None
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            return None
    
    def delete_old_features(self, days: int = 30) -> bool:
        """
        Delete old features from the database.
        
        Args:
            days: Number of days to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            DELETE FROM feature_store
            WHERE created_at < %s
            """
            
            params = ((datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Deleted features older than {days} days")
            return True
        except Exception as e:
            logger.error(f"Error deleting old features: {e}")
            return False
