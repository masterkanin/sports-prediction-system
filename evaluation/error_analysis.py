"""
Error Analysis and Systematic Bias Detection for Sports Prediction System

This module implements error analysis and systematic bias detection for the sports prediction system,
providing insights into model performance and areas for improvement.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query
from database.versioning import DataVersioning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/error_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('error_analysis')

class ErrorAnalyzer:
    """
    Class for analyzing prediction errors and detecting systematic biases.
    """
    
    def __init__(self):
        """
        Initialize the error analyzer.
        """
        self.data_versioning = DataVersioning()
        self.reports_dir = os.path.join('reports', 'error_analysis')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def analyze_errors_by_player(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                               sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors by player.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of player error analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Analyzing player errors from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date,
                pl.name as player_name, pl.position, pl.team_id,
                t.name as team_name
            FROM 
                predictions p
            JOIN 
                actual_results ar ON p.player_id = ar.player_id 
                AND p.game_id = ar.game_id 
                AND p.stat_type = ar.stat_type
            JOIN 
                games g ON p.game_id = g.game_id
            JOIN 
                players pl ON p.player_id = pl.player_id
            JOIN 
                teams t ON pl.team_id = t.team_id
            WHERE 
                g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            if model_version:
                query += " AND p.model_version = %s"
                params.append(model_version)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate error metrics
            df['error'] = df['predicted_value'] - df['actual_value']
            df['absolute_error'] = np.abs(df['error'])
            df['squared_error'] = df['error'] ** 2
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Group by player
            player_errors = df.groupby(['player_id', 'player_name', 'position', 'team_name']).agg({
                'error': ['mean', 'std'],
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            })
            
            player_errors.columns = ['_'.join(col).strip() for col in player_errors.columns.values]
            player_errors = player_errors.rename(columns={
                'error_mean': 'bias',
                'error_std': 'error_std',
                'absolute_error_mean': 'mae',
                'squared_error_mean': 'mse',
                'percentage_error_mean': 'mape',
                'over_under_correct_mean': 'over_under_accuracy',
                'prediction_id_count': 'count'
            })
            
            player_errors['rmse'] = np.sqrt(player_errors['mse'])
            
            # Sort by count and error
            player_errors = player_errors.sort_values(['count', 'mae'], ascending=[False, True])
            
            # Reset index for easier serialization
            player_errors = player_errors.reset_index()
            
            # Generate plots
            self._generate_player_error_plots(df, player_errors, sport, model_version)
            
            # Save analysis to file
            analysis_file = os.path.join(
                self.reports_dir, 
                f"player_errors_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    'player_errors': player_errors.to_dict(orient='records')
                }, f, indent=2, default=str)
            
            logger.info(f"Saved player error analysis to {analysis_file}")
            
            return {
                'success': True,
                'player_errors': player_errors.to_dict(orient='records'),
                'analysis_file': analysis_file
            }
        except Exception as e:
            logger.error(f"Error analyzing player errors: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_errors_by_team(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors by team.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of team error analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Analyzing team errors from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date, g.home_team_id, g.away_team_id,
                pl.name as player_name, pl.position, pl.team_id,
                ht.name as home_team_name,
                at.name as away_team_name
            FROM 
                predictions p
            JOIN 
                actual_results ar ON p.player_id = ar.player_id 
                AND p.game_id = ar.game_id 
                AND p.stat_type = ar.stat_type
            JOIN 
                games g ON p.game_id = g.game_id
            JOIN 
                players pl ON p.player_id = pl.player_id
            JOIN 
                teams ht ON g.home_team_id = ht.team_id
            JOIN 
                teams at ON g.away_team_id = at.team_id
            WHERE 
                g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            if model_version:
                query += " AND p.model_version = %s"
                params.append(model_version)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate error metrics
            df['error'] = df['predicted_value'] - df['actual_value']
            df['absolute_error'] = np.abs(df['error'])
            df['squared_error'] = df['error'] ** 2
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Add team context
            df['is_home_team'] = df['team_id'] == df['home_team_id']
            df['team_name'] = np.where(df['is_home_team'], df['home_team_name'], df['away_team_name'])
            df['opponent_team_id'] = np.where(df['is_home_team'], df['away_team_id'], df['home_team_id'])
            df['opponent_team_name'] = np.where(df['is_home_team'], df['away_team_name'], df['home_team_name'])
            
            # Group by team
            team_errors = df.groupby(['team_id', 'team_name']).agg({
                'error': ['mean', 'std'],
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            })
            
            team_errors.columns = ['_'.join(col).strip() for col in team_errors.columns.values]
            team_errors = team_errors.rename(columns={
                'error_mean': 'bias',
                'error_std': 'error_std',
                'absolute_error_mean': 'mae',
                'squared_error_mean': 'mse',
                'percentage_error_mean': 'mape',
                'over_under_correct_mean': 'over_under_accuracy',
                'prediction_id_count': 'count'
            })
            
            team_errors['rmse'] = np.sqrt(team_errors['mse'])
            
            # Sort by count and error
            team_errors = team_errors.sort_values(['count', 'mae'], ascending=[False, True])
            
            # Reset index for easier serialization
            team_errors = team_errors.reset_index()
            
            # Generate plots
            self._generate_team_error_plots(df, team_errors, sport, model_version)
            
            # Save analysis to file
            analysis_file = os.path.join(
                self.reports_dir, 
                f"team_errors_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    'team_errors': team_errors.to_dict(orient='records')
                }, f, indent=2, default=str)
            
            logger.info(f"Saved team error analysis to {analysis_file}")
            
            return {
                'success': True,
                'team_errors': team_errors.to_dict(orient='records'),
                'analysis_file': analysis_file
            }
        except Exception as e:
            logger.error(f"Error analyzing team errors: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_errors_by_stat_type(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors by stat type.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of stat type error analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Analyzing stat type errors from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date
            FROM 
                predictions p
            JOIN 
                actual_results ar ON p.player_id = ar.player_id 
                AND p.game_id = ar.game_id 
                AND p.stat_type = ar.stat_type
            JOIN 
                games g ON p.game_id = g.game_id
            WHERE 
                g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            if model_version:
                query += " AND p.model_version = %s"
                params.append(model_version)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate error metrics
            df['error'] = df['predicted_value'] - df['actual_value']
            df['absolute_error'] = np.abs(df['error'])
            df['squared_error'] = df['error'] ** 2
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Group by stat type
            stat_type_errors = df.groupby(['stat_type', 'sport']).agg({
                'error': ['mean', 'std'],
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            })
            
            stat_type_errors.columns = ['_'.join(col).strip() for col in stat_type_errors.columns.values]
            stat_type_errors = stat_type_errors.rename(columns={
                'error_mean': 'bias',
                'error_std': 'error_std',
                'absolute_error_mean': 'mae',
                'squared_error_mean': 'mse',
                'percentage_error_mean': 'mape',
                'over_under_correct_mean': 'over_under_accuracy',
                'prediction_id_count': 'count'
            })
            
            stat_type_errors['rmse'] = np.sqrt(stat_type_errors['mse'])
            
            # Sort by count and error
            stat_type_errors = stat_type_errors.sort_values(['count', 'mae'], ascending=[False, True])
            
            # Reset index for easier serialization
            stat_type_errors = stat_type_errors.reset_index()
            
            # Generate plots
            self._generate_stat_type_error_plots(df, stat_type_errors, sport, model_version)
            
            # Save analysis to file
            analysis_file = os.path.join(
                self.reports_dir, 
                f"stat_type_errors_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    'stat_type_errors': stat_type_errors.to_dict(orient='records')
                }, f, indent=2, default=str)
            
            logger.info(f"Saved stat type error analysis to {analysis_file}")
            
            return {
                'success': True,
                'stat_type_errors': stat_type_errors.to_dict(orient='records'),
                'analysis_file': analysis_file
            }
        except Exception as e:
            logger.error(f"Error analyzing stat type errors: {e}")
            return {'success': False, 'error': str(e)}
    
    def detect_systematic_biases(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                               sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect systematic biases in the model.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of bias detection results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Detecting systematic biases from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date, g.home_team_id, g.away_team_id,
                pl.name as player_name, pl.position, pl.team_id,
                ht.name as home_team_name,
                at.name as away_team_name
            FROM 
                predictions p
            JOIN 
                actual_results ar ON p.player_id = ar.player_id 
                AND p.game_id = ar.game_id 
                AND p.stat_type = ar.stat_type
            JOIN 
                games g ON p.game_id = g.game_id
            JOIN 
                players pl ON p.player_id = pl.player_id
            JOIN 
                teams ht ON g.home_team_id = ht.team_id
            JOIN 
                teams at ON g.away_team_id = at.team_id
            WHERE 
                g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            if model_version:
                query += " AND p.model_version = %s"
                params.append(model_version)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate error metrics
            df['error'] = df['predicted_value'] - df['actual_value']
            df['absolute_error'] = np.abs(df['error'])
            df['squared_error'] = df['error'] ** 2
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Add team context
            df['is_home_team'] = df['team_id'] == df['home_team_id']
            df['team_name'] = np.where(df['is_home_team'], df['home_team_name'], df['away_team_name'])
            df['opponent_team_id'] = np.where(df['is_home_team'], df['away_team_id'], df['home_team_id'])
            df['opponent_team_name'] = np.where(df['is_home_team'], df['away_team_name'], df['home_team_name'])
            
            # Identify biases
            biases = []
            
            # Bias by home/away
            home_away_bias = self._test_bias(df, 'is_home_team', 'Home/Away')
            if home_away_bias:
                biases.append(home_away_bias)
            
            # Bias by position
            position_bias = self._test_categorical_bias(df, 'position', 'Position')
            biases.extend(position_bias)
            
            # Bias by stat type
            stat_type_bias = self._test_categorical_bias(df, 'stat_type', 'Stat Type')
            biases.extend(stat_type_bias)
            
            # Bias by actual value (high vs low)
            df['actual_value_bucket'] = pd.qcut(df['actual_value'], q=5, labels=False)
            actual_value_bias = self._test_categorical_bias(df, 'actual_value_bucket', 'Actual Value Bucket')
            biases.extend(actual_value_bias)
            
            # Bias by predicted value (high vs low)
            df['predicted_value_bucket'] = pd.qcut(df['predicted_value'], q=5, labels=False)
            predicted_value_bias = self._test_categorical_bias(df, 'predicted_value_bucket', 'Predicted Value Bucket')
            biases.extend(predicted_value_bias)
            
            # Bias by confidence score
            df['confidence_bucket'] = pd.cut(df['confidence_score'], bins=[0, 25, 50, 75, 90, 100], 
                                           labels=['0-25', '25-50', '50-75', '75-90', '90-100'])
            confidence_bias = self._test_categorical_bias(df, 'confidence_bucket', 'Confidence Score')
            biases.extend(confidence_bias)
            
            # Bias by over/under probability
            df['over_under_bucket'] = pd.cut(df['over_probability'], bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                           labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0'])
            over_under_bias = self._test_categorical_bias(df, 'over_under_bucket', 'Over/Under Probability')
            biases.extend(over_under_bias)
            
            # Bias by day of week
            df['day_of_week'] = pd.to_datetime(df['game_date']).dt.day_name()
            day_of_week_bias = self._test_categorical_bias(df, 'day_of_week', 'Day of Week')
            biases.extend(day_of_week_bias)
            
            # Sort biases by significance
            biases = sorted(biases, key=lambda x: x['p_value'])
            
            # Generate plots
            self._generate_bias_plots(df, biases, sport, model_version)
            
            # Save biases to file
            biases_file = os.path.join(
                self.reports_dir, 
                f"systematic_biases_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(biases_file, 'w') as f:
                json.dump({
                    'biases': biases
                }, f, indent=2, default=str)
            
            logger.info(f"Saved systematic biases to {biases_file}")
            
            return {
                'success': True,
                'biases': biases,
                'biases_file': biases_file
            }
        except Exception as e:
            logger.error(f"Error detecting systematic biases: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_player_error_plots(self, df: pd.DataFrame, player_errors: pd.DataFrame, 
                                   sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate player error plots.
        
        Args:
            df: DataFrame of prediction results
            player_errors: DataFrame of player errors
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot top 20 players by MAE (with at least 10 predictions)
            top_players = player_errors[player_errors['count'] >= 10].head(20)
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='player_name', y='mae', data=top_players)
            plt.title(f"Top 20 Players by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Player')
            plt.ylabel('MAE')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            player_mae_file = os.path.join(
                plots_dir, 
                f"top_players_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(player_mae_file)
            plt.close()
            
            # Plot top 20 players by bias (with at least 10 predictions)
            top_bias_players = player_errors[player_errors['count'] >= 10].sort_values('bias', key=abs, ascending=False).head(20)
            
            plt.figure(figsize=(14, 8))
            colors = ['red' if x > 0 else 'blue' for x in top_bias_players['bias']]
            plt.bar(top_bias_players['player_name'], top_bias_players['bias'], color=colors)
            plt.title(f"Top 20 Players by Bias{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Player')
            plt.ylabel('Bias (Predicted - Actual)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            player_bias_file = os.path.join(
                plots_dir, 
                f"top_players_by_bias{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(player_bias_file)
            plt.close()
            
            # Plot top 20 players by over/under accuracy (with at least 10 predictions)
            top_accuracy_players = player_errors[player_errors['count'] >= 10].sort_values('over_under_accuracy', ascending=False).head(20)
            
            plt.figure(figsize=(14, 8))
            plt.bar(top_accuracy_players['player_name'], top_accuracy_players['over_under_accuracy'], color='green')
            plt.title(f"Top 20 Players by Over/Under Accuracy{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Player')
            plt.ylabel('Over/Under Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.tight_layout()
            
            player_accuracy_file = os.path.join(
                plots_dir, 
                f"top_players_by_accuracy{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(player_accuracy_file)
            plt.close()
            
            # Plot error distribution by position
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='position', y='error', data=df)
            plt.title(f"Error Distribution by Position{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Position')
            plt.ylabel('Error (Predicted - Actual)')
            plt.grid(True)
            plt.tight_layout()
            
            position_error_file = os.path.join(
                plots_dir, 
                f"error_by_position{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(position_error_file)
            plt.close()
            
            logger.info(f"Generated player error plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating player error plots: {e}")
    
    def _generate_team_error_plots(self, df: pd.DataFrame, team_errors: pd.DataFrame, 
                                 sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate team error plots.
        
        Args:
            df: DataFrame of prediction results
            team_errors: DataFrame of team errors
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot all teams by MAE
            plt.figure(figsize=(14, 8))
            sns.barplot(x='team_name', y='mae', data=team_errors)
            plt.title(f"Teams by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Team')
            plt.ylabel('MAE')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            team_mae_file = os.path.join(
                plots_dir, 
                f"teams_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(team_mae_file)
            plt.close()
            
            # Plot all teams by bias
            plt.figure(figsize=(14, 8))
            colors = ['red' if x > 0 else 'blue' for x in team_errors['bias']]
            plt.bar(team_errors['team_name'], team_errors['bias'], color=colors)
            plt.title(f"Teams by Bias{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Team')
            plt.ylabel('Bias (Predicted - Actual)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            team_bias_file = os.path.join(
                plots_dir, 
                f"teams_by_bias{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(team_bias_file)
            plt.close()
            
            # Plot all teams by over/under accuracy
            plt.figure(figsize=(14, 8))
            plt.bar(team_errors['team_name'], team_errors['over_under_accuracy'], color='green')
            plt.title(f"Teams by Over/Under Accuracy{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Team')
            plt.ylabel('Over/Under Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.tight_layout()
            
            team_accuracy_file = os.path.join(
                plots_dir, 
                f"teams_by_accuracy{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(team_accuracy_file)
            plt.close()
            
            # Plot error distribution by home/away
            if 'is_home_team' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='is_home_team', y='error', data=df)
                plt.title(f"Error Distribution by Home/Away{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Is Home Team')
                plt.ylabel('Error (Predicted - Actual)')
                plt.grid(True)
                plt.tight_layout()
                
                home_away_error_file = os.path.join(
                    plots_dir, 
                    f"error_by_home_away{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(home_away_error_file)
                plt.close()
            
            logger.info(f"Generated team error plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating team error plots: {e}")
    
    def _generate_stat_type_error_plots(self, df: pd.DataFrame, stat_type_errors: pd.DataFrame, 
                                      sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate stat type error plots.
        
        Args:
            df: DataFrame of prediction results
            stat_type_errors: DataFrame of stat type errors
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Filter by sport if specified
            if sport:
                stat_type_errors = stat_type_errors[stat_type_errors['sport'] == sport]
            
            # Plot stat types by MAE
            plt.figure(figsize=(14, 8))
            sns.barplot(x='stat_type', y='mae', data=stat_type_errors)
            plt.title(f"Stat Types by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Stat Type')
            plt.ylabel('MAE')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            stat_type_mae_file = os.path.join(
                plots_dir, 
                f"stat_types_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(stat_type_mae_file)
            plt.close()
            
            # Plot stat types by bias
            plt.figure(figsize=(14, 8))
            colors = ['red' if x > 0 else 'blue' for x in stat_type_errors['bias']]
            plt.bar(stat_type_errors['stat_type'], stat_type_errors['bias'], color=colors)
            plt.title(f"Stat Types by Bias{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Stat Type')
            plt.ylabel('Bias (Predicted - Actual)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            stat_type_bias_file = os.path.join(
                plots_dir, 
                f"stat_types_by_bias{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(stat_type_bias_file)
            plt.close()
            
            # Plot stat types by over/under accuracy
            plt.figure(figsize=(14, 8))
            plt.bar(stat_type_errors['stat_type'], stat_type_errors['over_under_accuracy'], color='green')
            plt.title(f"Stat Types by Over/Under Accuracy{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Stat Type')
            plt.ylabel('Over/Under Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.tight_layout()
            
            stat_type_accuracy_file = os.path.join(
                plots_dir, 
                f"stat_types_by_accuracy{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(stat_type_accuracy_file)
            plt.close()
            
            # Plot error distribution by stat type
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='stat_type', y='error', data=df)
            plt.title(f"Error Distribution by Stat Type{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Stat Type')
            plt.ylabel('Error (Predicted - Actual)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            stat_type_error_file = os.path.join(
                plots_dir, 
                f"error_by_stat_type{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(stat_type_error_file)
            plt.close()
            
            logger.info(f"Generated stat type error plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating stat type error plots: {e}")
    
    def _generate_bias_plots(self, df: pd.DataFrame, biases: List[Dict[str, Any]], 
                           sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate bias plots.
        
        Args:
            df: DataFrame of prediction results
            biases: List of identified biases
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot top biases
            top_biases = biases[:10]
            
            plt.figure(figsize=(14, 8))
            bias_names = [f"{b['category']}: {b['value']}" for b in top_biases]
            bias_values = [b['mean_error'] for b in top_biases]
            
            colors = ['red' if v > 0 else 'blue' for v in bias_values]
            
            plt.bar(bias_names, bias_values, color=colors)
            plt.title(f"Top Systematic Biases{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Category')
            plt.ylabel('Mean Error')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            bias_file = os.path.join(
                plots_dir, 
                f"top_biases{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(bias_file)
            plt.close()
            
            # Plot bias by home/away
            if 'is_home_team' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='is_home_team', y='error', data=df)
                plt.title(f"Error by Home/Away{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Is Home Team')
                plt.ylabel('Error')
                plt.grid(True)
                plt.tight_layout()
                
                home_away_file = os.path.join(
                    plots_dir, 
                    f"error_by_home_away{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(home_away_file)
                plt.close()
            
            # Plot bias by position
            if 'position' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='position', y='error', data=df)
                plt.title(f"Error by Position{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Position')
                plt.ylabel('Error')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True)
                plt.tight_layout()
                
                position_file = os.path.join(
                    plots_dir, 
                    f"error_by_position{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(position_file)
                plt.close()
            
            # Plot bias by confidence bucket
            if 'confidence_bucket' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='confidence_bucket', y='error', data=df)
                plt.title(f"Error by Confidence Bucket{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Confidence Bucket')
                plt.ylabel('Error')
                plt.grid(True)
                plt.tight_layout()
                
                confidence_file = os.path.join(
                    plots_dir, 
                    f"error_by_confidence{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(confidence_file)
                plt.close()
            
            # Plot bias by day of week
            if 'day_of_week' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='day_of_week', y='error', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                plt.title(f"Error by Day of Week{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Day of Week')
                plt.ylabel('Error')
                plt.grid(True)
                plt.tight_layout()
                
                day_of_week_file = os.path.join(
                    plots_dir, 
                    f"error_by_day_of_week{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(day_of_week_file)
                plt.close()
            
            logger.info(f"Generated bias plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating bias plots: {e}")
    
    def _test_bias(self, df: pd.DataFrame, column: str, category_name: str) -> Optional[Dict[str, Any]]:
        """
        Test for bias in a binary column.
        
        Args:
            df: DataFrame of prediction results
            column: Column to test
            category_name: Name of the category
            
        Returns:
            Dictionary of bias results or None if no bias detected
        """
        try:
            # Group by column
            grouped = df.groupby(column)['error'].agg(['mean', 'std', 'count'])
            
            if len(grouped) != 2:
                return None
            
            # Perform t-test
            group_0 = df[df[column] == grouped.index[0]]['error']
            group_1 = df[df[column] == grouped.index[1]]['error']
            
            t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
            
            # Check if bias is significant
            if p_value < 0.05:
                return {
                    'category': category_name,
                    'value': str(grouped.index[0]),
                    'mean_error': float(grouped['mean'][0]),
                    'std_error': float(grouped['std'][0]),
                    'count': int(grouped['count'][0]),
                    'comparison_value': str(grouped.index[1]),
                    'comparison_mean_error': float(grouped['mean'][1]),
                    'comparison_std_error': float(grouped['std'][1]),
                    'comparison_count': int(grouped['count'][1]),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': True
                }
            
            return None
        except Exception as e:
            logger.error(f"Error testing bias for {column}: {e}")
            return None
    
    def _test_categorical_bias(self, df: pd.DataFrame, column: str, category_name: str) -> List[Dict[str, Any]]:
        """
        Test for bias in a categorical column.
        
        Args:
            df: DataFrame of prediction results
            column: Column to test
            category_name: Name of the category
            
        Returns:
            List of bias results
        """
        try:
            # Group by column
            grouped = df.groupby(column)['error'].agg(['mean', 'std', 'count'])
            
            if len(grouped) <= 1:
                return []
            
            # Perform ANOVA
            groups = [df[df[column] == value]['error'] for value in grouped.index]
            f_stat, p_value = stats.f_oneway(*groups)
            
            biases = []
            
            # Check if overall bias is significant
            if p_value < 0.05:
                # Perform pairwise t-tests
                for i, value_i in enumerate(grouped.index):
                    group_i = df[df[column] == value_i]['error']
                    
                    for j, value_j in enumerate(grouped.index):
                        if i >= j:
                            continue
                        
                        group_j = df[df[column] == value_j]['error']
                        
                        t_stat, p_value = stats.ttest_ind(group_i, group_j, equal_var=False)
                        
                        if p_value < 0.05:
                            biases.append({
                                'category': category_name,
                                'value': str(value_i),
                                'mean_error': float(grouped['mean'][i]),
                                'std_error': float(grouped['std'][i]),
                                'count': int(grouped['count'][i]),
                                'comparison_value': str(value_j),
                                'comparison_mean_error': float(grouped['mean'][j]),
                                'comparison_std_error': float(grouped['std'][j]),
                                'comparison_count': int(grouped['count'][j]),
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': True
                            })
            
            return biases
        except Exception as e:
            logger.error(f"Error testing categorical bias for {column}: {e}")
            return []


if __name__ == "__main__":
    # Run error analysis
    analyzer = ErrorAnalyzer()
    player_results = analyzer.analyze_errors_by_player()
    team_results = analyzer.analyze_errors_by_team()
    stat_type_results = analyzer.analyze_errors_by_stat_type()
    bias_results = analyzer.detect_systematic_biases()
    
    # Log results
    logger.info(f"Error analysis and bias detection completed")
