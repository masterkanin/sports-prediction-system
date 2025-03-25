"""
Performance Tracking for Sports Prediction System

This module implements comprehensive performance tracking for the sports prediction system,
including accuracy tracking, error analysis, and bias detection.
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
        logging.FileHandler('logs/performance_tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('performance_tracking')

class PerformanceTracker:
    """
    Class for tracking prediction performance over time.
    """
    
    def __init__(self):
        """
        Initialize the performance tracker.
        """
        self.data_versioning = DataVersioning()
        self.reports_dir = os.path.join('reports', 'performance')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def track_accuracy_over_time(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                                sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Track prediction accuracy over time.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of accuracy metrics over time
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Tracking accuracy from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date,
                pl.name as player_name, pl.position
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
            
            query += " ORDER BY g.game_date"
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate accuracy metrics
            df['absolute_error'] = np.abs(df['predicted_value'] - df['actual_value'])
            df['squared_error'] = (df['predicted_value'] - df['actual_value']) ** 2
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Group by date
            df['date'] = pd.to_datetime(df['game_date']).dt.date
            daily_metrics = df.groupby('date').agg({
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            }).rename(columns={
                'absolute_error': 'mae',
                'squared_error': 'mse',
                'percentage_error': 'mape',
                'over_under_correct': 'over_under_accuracy',
                'prediction_id': 'count'
            })
            
            daily_metrics['rmse'] = np.sqrt(daily_metrics['mse'])
            
            # Calculate overall metrics
            overall_metrics = {
                'mae': float(df['absolute_error'].mean()),
                'mse': float(df['squared_error'].mean()),
                'rmse': float(np.sqrt(df['squared_error'].mean())),
                'mape': float(df['percentage_error'].mean()),
                'over_under_accuracy': float(df['over_under_correct'].mean()),
                'count': len(df),
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Calculate metrics by sport
            sport_metrics = {}
            if not sport:
                for sport_name, sport_df in df.groupby('sport'):
                    sport_metrics[sport_name] = {
                        'mae': float(sport_df['absolute_error'].mean()),
                        'mse': float(sport_df['squared_error'].mean()),
                        'rmse': float(np.sqrt(sport_df['squared_error'].mean())),
                        'mape': float(sport_df['percentage_error'].mean()),
                        'over_under_accuracy': float(sport_df['over_under_correct'].mean()),
                        'count': len(sport_df)
                    }
            
            # Calculate metrics by stat type
            stat_type_metrics = {}
            for stat_type, stat_df in df.groupby('stat_type'):
                stat_type_metrics[stat_type] = {
                    'mae': float(stat_df['absolute_error'].mean()),
                    'mse': float(stat_df['squared_error'].mean()),
                    'rmse': float(np.sqrt(stat_df['squared_error'].mean())),
                    'mape': float(stat_df['percentage_error'].mean()),
                    'over_under_accuracy': float(stat_df['over_under_correct'].mean()),
                    'count': len(stat_df)
                }
            
            # Generate plots
            self._generate_accuracy_plots(daily_metrics, sport, model_version)
            
            # Save metrics to file
            metrics_file = os.path.join(
                self.reports_dir, 
                f"accuracy_metrics_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'overall': overall_metrics,
                    'by_sport': sport_metrics,
                    'by_stat_type': stat_type_metrics,
                    'daily': daily_metrics.to_dict(orient='index')
                }, f, indent=2, default=str)
            
            logger.info(f"Saved accuracy metrics to {metrics_file}")
            
            return {
                'success': True,
                'overall': overall_metrics,
                'by_sport': sport_metrics,
                'by_stat_type': stat_type_metrics,
                'daily': daily_metrics.reset_index().to_dict(orient='records'),
                'metrics_file': metrics_file
            }
        except Exception as e:
            logger.error(f"Error tracking accuracy: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_errors(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                      sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors by player, team, and stat type.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of error analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Analyzing errors from {start_date} to {end_date}")
            
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
            
            # Analyze errors by player
            player_errors = df.groupby(['player_id', 'player_name']).agg({
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
            
            # Analyze errors by team
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
            
            # Analyze errors by stat type
            stat_type_errors = df.groupby('stat_type').agg({
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
            
            # Analyze errors by position
            position_errors = df.groupby('position').agg({
                'error': ['mean', 'std'],
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            })
            
            position_errors.columns = ['_'.join(col).strip() for col in position_errors.columns.values]
            position_errors = position_errors.rename(columns={
                'error_mean': 'bias',
                'error_std': 'error_std',
                'absolute_error_mean': 'mae',
                'squared_error_mean': 'mse',
                'percentage_error_mean': 'mape',
                'over_under_correct_mean': 'over_under_accuracy',
                'prediction_id_count': 'count'
            })
            
            position_errors['rmse'] = np.sqrt(position_errors['mse'])
            
            # Sort by count and error
            position_errors = position_errors.sort_values(['count', 'mae'], ascending=[False, True])
            
            # Analyze errors by confidence score
            df['confidence_bucket'] = pd.cut(df['confidence_score'], bins=[0, 25, 50, 75, 90, 100], 
                                           labels=['0-25', '25-50', '50-75', '75-90', '90-100'])
            
            confidence_errors = df.groupby('confidence_bucket').agg({
                'error': ['mean', 'std'],
                'absolute_error': 'mean',
                'squared_error': 'mean',
                'percentage_error': 'mean',
                'over_under_correct': 'mean',
                'prediction_id': 'count'
            })
            
            confidence_errors.columns = ['_'.join(col).strip() for col in confidence_errors.columns.values]
            confidence_errors = confidence_errors.rename(columns={
                'error_mean': 'bias',
                'error_std': 'error_std',
                'absolute_error_mean': 'mae',
                'squared_error_mean': 'mse',
                'percentage_error_mean': 'mape',
                'over_under_correct_mean': 'over_under_accuracy',
                'prediction_id_count': 'count'
            })
            
            confidence_errors['rmse'] = np.sqrt(confidence_errors['mse'])
            
            # Generate plots
            self._generate_error_analysis_plots(df, player_errors, team_errors, stat_type_errors, position_errors, confidence_errors, sport, model_version)
            
            # Save analysis to file
            analysis_file = os.path.join(
                self.reports_dir, 
                f"error_analysis_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    'by_player': player_errors.reset_index().to_dict(orient='records'),
                    'by_team': team_errors.reset_index().to_dict(orient='records'),
                    'by_stat_type': stat_type_errors.reset_index().to_dict(orient='records'),
                    'by_position': position_errors.reset_index().to_dict(orient='records'),
                    'by_confidence': confidence_errors.reset_index().to_dict(orient='records')
                }, f, indent=2, default=str)
            
            logger.info(f"Saved error analysis to {analysis_file}")
            
            return {
                'success': True,
                'by_player': player_errors.reset_index().to_dict(orient='records'),
                'by_team': team_errors.reset_index().to_dict(orient='records'),
                'by_stat_type': stat_type_errors.reset_index().to_dict(orient='records'),
                'by_position': position_errors.reset_index().to_dict(orient='records'),
                'by_confidence': confidence_errors.reset_index().to_dict(orient='records'),
                'analysis_file': analysis_file
            }
        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            return {'success': False, 'error': str(e)}
    
    def identify_systematic_biases(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                 sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify systematic biases in the model.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of bias analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Identifying systematic biases from {start_date} to {end_date}")
            
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
            logger.error(f"Error identifying systematic biases: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_feature_importance(self, model_version: Optional[str] = None, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze feature importance for the model.
        
        Args:
            model_version: Model version to analyze (optional)
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of feature importance analysis results
        """
        try:
            logger.info(f"Analyzing feature importance for model version {model_version}")
            
            # Get model information
            if model_version:
                model_info = self.data_versioning.get_model_version(model_version)
                if not model_info:
                    logger.warning(f"Model version {model_version} not found")
                    return {'success': False, 'error': f"Model version {model_version} not found"}
            else:
                # Get latest active model version
                model_info = self.data_versioning.get_active_model_version(sport)
                if not model_info:
                    logger.warning(f"No active model version found for {sport}")
                    return {'success': False, 'error': f"No active model version found for {sport}"}
                
                model_version = model_info['version_id']
            
            logger.info(f"Analyzing feature importance for model version {model_version}")
            
            # Get feature importance from model
            feature_importance = self._get_feature_importance(model_version)
            
            if not feature_importance:
                logger.warning(f"No feature importance available for model version {model_version}")
                return {'success': False, 'error': f"No feature importance available for model version {model_version}"}
            
            # Sort by importance
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            
            # Generate plots
            self._generate_feature_importance_plots(feature_importance, model_version, sport)
            
            # Save feature importance to file
            importance_file = os.path.join(
                self.reports_dir, 
                f"feature_importance_{model_version}{'_' + sport if sport else ''}.json"
            )
            
            with open(importance_file, 'w') as f:
                json.dump({
                    'model_version': model_version,
                    'feature_importance': feature_importance
                }, f, indent=2)
            
            logger.info(f"Saved feature importance to {importance_file}")
            
            return {
                'success': True,
                'model_version': model_version,
                'feature_importance': feature_importance,
                'importance_file': importance_file
            }
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {'success': False, 'error': str(e)}
    
    def detect_anomalies(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                        sport: Optional[str] = None, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in player performances.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary of anomaly detection results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Detecting anomalies from {start_date} to {end_date}")
            
            # Build query
            query = """
            SELECT 
                ar.player_id, ar.game_id, ar.stat_type, ar.actual_value,
                g.sport, g.game_date, g.home_team_id, g.away_team_id,
                pl.name as player_name, pl.position, pl.team_id,
                ht.name as home_team_name,
                at.name as away_team_name
            FROM 
                actual_results ar
            JOIN 
                games g ON ar.game_id = g.game_id
            JOIN 
                players pl ON ar.player_id = pl.player_id
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
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No player performances found for the specified criteria")
                return {'success': False, 'error': 'No player performances found'}
            
            logger.info(f"Found {len(results)} player performances")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Add team context
            df['is_home_team'] = df['team_id'] == df['home_team_id']
            df['team_name'] = np.where(df['is_home_team'], df['home_team_name'], df['away_team_name'])
            df['opponent_team_id'] = np.where(df['is_home_team'], df['away_team_id'], df['home_team_id'])
            df['opponent_team_name'] = np.where(df['is_home_team'], df['away_team_name'], df['home_team_name'])
            
            # Detect anomalies
            anomalies = []
            
            # Group by player and stat type
            for (player_id, stat_type), group in df.groupby(['player_id', 'stat_type']):
                if len(group) < 5:
                    continue
                
                # Calculate z-scores
                mean = group['actual_value'].mean()
                std = group['actual_value'].std()
                
                if std == 0:
                    continue
                
                group['z_score'] = (group['actual_value'] - mean) / std
                
                # Identify anomalies
                anomaly_mask = np.abs(group['z_score']) > threshold
                anomaly_performances = group[anomaly_mask]
                
                for _, row in anomaly_performances.iterrows():
                    anomalies.append({
                        'player_id': player_id,
                        'player_name': row['player_name'],
                        'position': row['position'],
                        'team_name': row['team_name'],
                        'game_id': row['game_id'],
                        'game_date': row['game_date'],
                        'opponent_team_name': row['opponent_team_name'],
                        'stat_type': stat_type,
                        'actual_value': float(row['actual_value']),
                        'z_score': float(row['z_score']),
                        'is_home_team': bool(row['is_home_team']),
                        'direction': 'above' if row['z_score'] > 0 else 'below'
                    })
            
            # Sort anomalies by absolute z-score
            anomalies = sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
            
            # Generate plots
            self._generate_anomaly_plots(anomalies, sport)
            
            # Save anomalies to file
            anomalies_file = os.path.join(
                self.reports_dir, 
                f"anomalies_{start_date}_to_{end_date}{'_' + sport if sport else ''}.json"
            )
            
            with open(anomalies_file, 'w') as f:
                json.dump({
                    'anomalies': anomalies
                }, f, indent=2, default=str)
            
            logger.info(f"Saved anomalies to {anomalies_file}")
            
            return {
                'success': True,
                'anomalies': anomalies,
                'anomalies_file': anomalies_file
            }
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_accuracy_plots(self, daily_metrics: pd.DataFrame, sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate accuracy plots.
        
        Args:
            daily_metrics: DataFrame of daily metrics
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot MAE over time
            plt.figure(figsize=(12, 6))
            plt.plot(daily_metrics.index, daily_metrics['mae'], marker='o')
            plt.title(f"Mean Absolute Error Over Time{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Date')
            plt.ylabel('MAE')
            plt.grid(True)
            plt.tight_layout()
            
            mae_plot_file = os.path.join(
                plots_dir, 
                f"mae_over_time{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(mae_plot_file)
            plt.close()
            
            # Plot RMSE over time
            plt.figure(figsize=(12, 6))
            plt.plot(daily_metrics.index, daily_metrics['rmse'], marker='o', color='orange')
            plt.title(f"Root Mean Squared Error Over Time{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Date')
            plt.ylabel('RMSE')
            plt.grid(True)
            plt.tight_layout()
            
            rmse_plot_file = os.path.join(
                plots_dir, 
                f"rmse_over_time{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(rmse_plot_file)
            plt.close()
            
            # Plot over/under accuracy over time
            plt.figure(figsize=(12, 6))
            plt.plot(daily_metrics.index, daily_metrics['over_under_accuracy'], marker='o', color='green')
            plt.title(f"Over/Under Accuracy Over Time{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Date')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.tight_layout()
            
            accuracy_plot_file = os.path.join(
                plots_dir, 
                f"over_under_accuracy_over_time{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(accuracy_plot_file)
            plt.close()
            
            # Plot prediction count over time
            plt.figure(figsize=(12, 6))
            plt.bar(daily_metrics.index, daily_metrics['count'], color='blue')
            plt.title(f"Prediction Count Over Time{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.grid(True)
            plt.tight_layout()
            
            count_plot_file = os.path.join(
                plots_dir, 
                f"prediction_count_over_time{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(count_plot_file)
            plt.close()
            
            logger.info(f"Generated accuracy plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating accuracy plots: {e}")
    
    def _generate_error_analysis_plots(self, df: pd.DataFrame, player_errors: pd.DataFrame, team_errors: pd.DataFrame, 
                                     stat_type_errors: pd.DataFrame, position_errors: pd.DataFrame, confidence_errors: pd.DataFrame,
                                     sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate error analysis plots.
        
        Args:
            df: DataFrame of prediction results
            player_errors: DataFrame of player errors
            team_errors: DataFrame of team errors
            stat_type_errors: DataFrame of stat type errors
            position_errors: DataFrame of position errors
            confidence_errors: DataFrame of confidence errors
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot error distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(df['error'], kde=True)
            plt.title(f"Error Distribution{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Error (Predicted - Actual)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            
            error_dist_file = os.path.join(
                plots_dir, 
                f"error_distribution{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(error_dist_file)
            plt.close()
            
            # Plot top 10 players by MAE (with at least 10 predictions)
            top_players = player_errors[player_errors['count'] >= 10].head(10)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_players.index.get_level_values('player_name'), y=top_players['mae'])
            plt.title(f"Top 10 Players by MAE{' for ' + sport.upper() if sport else ''}")
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
            
            # Plot top 10 teams by MAE
            top_teams = team_errors.head(10)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_teams.index.get_level_values('team_name'), y=top_teams['mae'])
            plt.title(f"Top 10 Teams by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Team')
            plt.ylabel('MAE')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            team_mae_file = os.path.join(
                plots_dir, 
                f"top_teams_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(team_mae_file)
            plt.close()
            
            # Plot stat types by MAE
            plt.figure(figsize=(12, 6))
            sns.barplot(x=stat_type_errors.index, y=stat_type_errors['mae'])
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
            
            # Plot positions by MAE
            plt.figure(figsize=(12, 6))
            sns.barplot(x=position_errors.index, y=position_errors['mae'])
            plt.title(f"Positions by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Position')
            plt.ylabel('MAE')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            position_mae_file = os.path.join(
                plots_dir, 
                f"positions_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(position_mae_file)
            plt.close()
            
            # Plot confidence buckets by MAE
            plt.figure(figsize=(12, 6))
            sns.barplot(x=confidence_errors.index, y=confidence_errors['mae'])
            plt.title(f"Confidence Buckets by MAE{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Confidence Bucket')
            plt.ylabel('MAE')
            plt.grid(True)
            plt.tight_layout()
            
            confidence_mae_file = os.path.join(
                plots_dir, 
                f"confidence_buckets_by_mae{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(confidence_mae_file)
            plt.close()
            
            # Plot confidence buckets by over/under accuracy
            plt.figure(figsize=(12, 6))
            sns.barplot(x=confidence_errors.index, y=confidence_errors['over_under_accuracy'])
            plt.title(f"Confidence Buckets by Over/Under Accuracy{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Confidence Bucket')
            plt.ylabel('Over/Under Accuracy')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.tight_layout()
            
            confidence_accuracy_file = os.path.join(
                plots_dir, 
                f"confidence_buckets_by_accuracy{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(confidence_accuracy_file)
            plt.close()
            
            # Plot actual vs predicted values
            plt.figure(figsize=(12, 6))
            plt.scatter(df['actual_value'], df['predicted_value'], alpha=0.5)
            plt.plot([df['actual_value'].min(), df['actual_value'].max()], 
                    [df['actual_value'].min(), df['actual_value'].max()], 'r--')
            plt.title(f"Actual vs Predicted Values{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Actual Value')
            plt.ylabel('Predicted Value')
            plt.grid(True)
            plt.tight_layout()
            
            actual_vs_predicted_file = os.path.join(
                plots_dir, 
                f"actual_vs_predicted{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(actual_vs_predicted_file)
            plt.close()
            
            logger.info(f"Generated error analysis plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating error analysis plots: {e}")
    
    def _generate_bias_plots(self, df: pd.DataFrame, biases: List[Dict[str, Any]], sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
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
            
            plt.figure(figsize=(12, 6))
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
                plt.figure(figsize=(12, 6))
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
                plt.figure(figsize=(12, 6))
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
                plt.figure(figsize=(12, 6))
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
                plt.figure(figsize=(12, 6))
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
    
    def _generate_feature_importance_plots(self, feature_importance: List[Dict[str, Any]], model_version: str, sport: Optional[str] = None) -> None:
        """
        Generate feature importance plots.
        
        Args:
            feature_importance: List of feature importance values
            model_version: Model version
            sport: Sport to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot top 20 features
            top_features = feature_importance[:20]
            
            plt.figure(figsize=(12, 8))
            feature_names = [f['feature'] for f in top_features]
            importance_values = [f['importance'] for f in top_features]
            
            plt.barh(feature_names, importance_values, color='skyblue')
            plt.title(f"Top 20 Feature Importance{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.grid(True)
            plt.tight_layout()
            
            importance_file = os.path.join(
                plots_dir, 
                f"feature_importance_{model_version}{'_' + sport if sport else ''}.png"
            )
            plt.savefig(importance_file)
            plt.close()
            
            logger.info(f"Generated feature importance plot in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {e}")
    
    def _generate_anomaly_plots(self, anomalies: List[Dict[str, Any]], sport: Optional[str] = None) -> None:
        """
        Generate anomaly plots.
        
        Args:
            anomalies: List of detected anomalies
            sport: Sport to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot top 20 anomalies
            top_anomalies = anomalies[:20]
            
            plt.figure(figsize=(12, 8))
            anomaly_names = [f"{a['player_name']} ({a['stat_type']})" for a in top_anomalies]
            z_scores = [a['z_score'] for a in top_anomalies]
            
            colors = ['red' if z > 0 else 'blue' for z in z_scores]
            
            plt.barh(anomaly_names, z_scores, color=colors)
            plt.title(f"Top 20 Performance Anomalies{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Z-Score')
            plt.ylabel('Player (Stat Type)')
            plt.grid(True)
            plt.tight_layout()
            
            anomaly_file = os.path.join(
                plots_dir, 
                f"performance_anomalies{'_' + sport if sport else ''}.png"
            )
            plt.savefig(anomaly_file)
            plt.close()
            
            logger.info(f"Generated anomaly plot in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating anomaly plot: {e}")
    
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
    
    def _get_feature_importance(self, model_version: str) -> List[Dict[str, Any]]:
        """
        Get feature importance from model.
        
        Args:
            model_version: Model version
            
        Returns:
            List of feature importance values
        """
        try:
            # Query feature importance from database
            query = """
            SELECT feature_name, importance
            FROM feature_importance
            WHERE model_version = %s
            ORDER BY importance DESC
            """
            
            params = (model_version,)
            results = execute_query(query, params)
            
            if not results:
                logger.warning(f"No feature importance found for model version {model_version}")
                return []
            
            # Format results
            feature_importance = []
            
            for result in results:
                feature_importance.append({
                    'feature': result['feature_name'],
                    'importance': float(result['importance'])
                })
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return []


class ContinuousImprovement:
    """
    Class for implementing continuous improvement mechanisms.
    """
    
    def __init__(self):
        """
        Initialize the continuous improvement system.
        """
        self.performance_tracker = PerformanceTracker()
        self.data_versioning = DataVersioning()
    
    def analyze_model_performance(self, model_version: Optional[str] = None, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze model performance and generate improvement recommendations.
        
        Args:
            model_version: Model version to analyze (optional)
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of analysis results and recommendations
        """
        try:
            logger.info(f"Analyzing model performance for version {model_version}")
            
            # Get model information
            if model_version:
                model_info = self.data_versioning.get_model_version(model_version)
                if not model_info:
                    logger.warning(f"Model version {model_version} not found")
                    return {'success': False, 'error': f"Model version {model_version} not found"}
            else:
                # Get latest active model version
                model_info = self.data_versioning.get_active_model_version(sport)
                if not model_info:
                    logger.warning(f"No active model version found for {sport}")
                    return {'success': False, 'error': f"No active model version found for {sport}"}
                
                model_version = model_info['version_id']
            
            # Track accuracy
            accuracy_results = self.performance_tracker.track_accuracy_over_time(
                model_version=model_version, sport=sport
            )
            
            if not accuracy_results.get('success', False):
                logger.warning(f"Error tracking accuracy: {accuracy_results.get('error')}")
                return {'success': False, 'error': accuracy_results.get('error')}
            
            # Analyze errors
            error_results = self.performance_tracker.analyze_errors(
                model_version=model_version, sport=sport
            )
            
            if not error_results.get('success', False):
                logger.warning(f"Error analyzing errors: {error_results.get('error')}")
                return {'success': False, 'error': error_results.get('error')}
            
            # Identify systematic biases
            bias_results = self.performance_tracker.identify_systematic_biases(
                model_version=model_version, sport=sport
            )
            
            if not bias_results.get('success', False):
                logger.warning(f"Error identifying biases: {bias_results.get('error')}")
                return {'success': False, 'error': bias_results.get('error')}
            
            # Analyze feature importance
            importance_results = self.performance_tracker.analyze_feature_importance(
                model_version=model_version, sport=sport
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                accuracy_results, error_results, bias_results, importance_results
            )
            
            # Save analysis to file
            analysis_file = os.path.join(
                'reports', 'performance', 
                f"model_analysis_{model_version}{'_' + sport if sport else ''}.json"
            )
            
            with open(analysis_file, 'w') as f:
                json.dump({
                    'model_version': model_version,
                    'sport': sport,
                    'accuracy': accuracy_results.get('overall', {}),
                    'errors': {
                        'by_player': error_results.get('by_player', [])[:10],
                        'by_team': error_results.get('by_team', [])[:10],
                        'by_stat_type': error_results.get('by_stat_type', []),
                        'by_position': error_results.get('by_position', [])
                    },
                    'biases': bias_results.get('biases', [])[:10],
                    'feature_importance': importance_results.get('feature_importance', [])[:20],
                    'recommendations': recommendations
                }, f, indent=2, default=str)
            
            logger.info(f"Saved model analysis to {analysis_file}")
            
            return {
                'success': True,
                'model_version': model_version,
                'sport': sport,
                'accuracy': accuracy_results.get('overall', {}),
                'errors': {
                    'by_player': error_results.get('by_player', [])[:10],
                    'by_team': error_results.get('by_team', [])[:10],
                    'by_stat_type': error_results.get('by_stat_type', []),
                    'by_position': error_results.get('by_position', [])
                },
                'biases': bias_results.get('biases', [])[:10],
                'feature_importance': importance_results.get('feature_importance', [])[:20],
                'recommendations': recommendations,
                'analysis_file': analysis_file
            }
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {'success': False, 'error': str(e)}
    
    def implement_reinforcement_learning(self, model_version: Optional[str] = None, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Implement reinforcement learning to optimize prediction strategy.
        
        Args:
            model_version: Model version to optimize (optional)
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of optimization results
        """
        try:
            logger.info(f"Implementing reinforcement learning for version {model_version}")
            
            # Get model information
            if model_version:
                model_info = self.data_versioning.get_model_version(model_version)
                if not model_info:
                    logger.warning(f"Model version {model_version} not found")
                    return {'success': False, 'error': f"Model version {model_version} not found"}
            else:
                # Get latest active model version
                model_info = self.data_versioning.get_active_model_version(sport)
                if not model_info:
                    logger.warning(f"No active model version found for {sport}")
                    return {'success': False, 'error': f"No active model version found for {sport}"}
                
                model_version = model_info['version_id']
            
            # Build query to get prediction data
            query = """
            SELECT 
                p.prediction_id, p.player_id, p.game_id, p.stat_type, 
                p.predicted_value, p.confidence_score, p.over_probability, p.line_value,
                p.created_at, p.model_version,
                ar.actual_value,
                g.sport, g.game_date,
                pl.name as player_name, pl.position
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
            WHERE 
                p.model_version = %s
            """
            
            params = [model_version]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning("No prediction results found for the specified criteria")
                return {'success': False, 'error': 'No prediction results found'}
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate over/under accuracy
            df['over_under_correct'] = ((df['actual_value'] > df['line_value']) == (df['over_probability'] > 0.5)).astype(int)
            
            # Optimize confidence threshold
            thresholds = np.arange(0.5, 1.0, 0.01)
            accuracies = []
            counts = []
            
            for threshold in thresholds:
                # Filter predictions with high confidence
                high_conf_over = df[df['over_probability'] >= threshold]
                high_conf_under = df[df['over_probability'] <= (1 - threshold)]
                high_conf = pd.concat([high_conf_over, high_conf_under])
                
                if len(high_conf) > 0:
                    accuracy = high_conf['over_under_correct'].mean()
                    accuracies.append(accuracy)
                    counts.append(len(high_conf))
                else:
                    accuracies.append(0)
                    counts.append(0)
            
            # Find optimal threshold
            optimal_idx = np.argmax(accuracies)
            optimal_threshold = thresholds[optimal_idx]
            optimal_accuracy = accuracies[optimal_idx]
            optimal_count = counts[optimal_idx]
            
            logger.info(f"Optimal confidence threshold: {optimal_threshold}, accuracy: {optimal_accuracy}, count: {optimal_count}")
            
            # Optimize line adjustment
            adjustments = np.arange(-0.5, 0.5, 0.05)
            adj_accuracies = []
            
            for adjustment in adjustments:
                # Adjust line value
                df['adjusted_line'] = df['line_value'] + adjustment
                df['adjusted_correct'] = ((df['actual_value'] > df['adjusted_line']) == (df['over_probability'] > 0.5)).astype(int)
                accuracy = df['adjusted_correct'].mean()
                adj_accuracies.append(accuracy)
            
            # Find optimal adjustment
            optimal_adj_idx = np.argmax(adj_accuracies)
            optimal_adjustment = adjustments[optimal_adj_idx]
            optimal_adj_accuracy = adj_accuracies[optimal_adj_idx]
            
            logger.info(f"Optimal line adjustment: {optimal_adjustment}, accuracy: {optimal_adj_accuracy}")
            
            # Generate plots
            self._generate_rl_plots(thresholds, accuracies, counts, adjustments, adj_accuracies, sport, model_version)
            
            # Save optimization results
            optimization_file = os.path.join(
                'reports', 'performance', 
                f"rl_optimization_{model_version}{'_' + sport if sport else ''}.json"
            )
            
            optimization_results = {
                'model_version': model_version,
                'sport': sport,
                'confidence_threshold': {
                    'optimal_threshold': float(optimal_threshold),
                    'optimal_accuracy': float(optimal_accuracy),
                    'optimal_count': int(optimal_count),
                    'thresholds': [float(t) for t in thresholds],
                    'accuracies': [float(a) for a in accuracies],
                    'counts': [int(c) for c in counts]
                },
                'line_adjustment': {
                    'optimal_adjustment': float(optimal_adjustment),
                    'optimal_accuracy': float(optimal_adj_accuracy),
                    'adjustments': [float(a) for a in adjustments],
                    'accuracies': [float(a) for a in adj_accuracies]
                }
            }
            
            with open(optimization_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            logger.info(f"Saved optimization results to {optimization_file}")
            
            # Register optimization strategy
            strategy_id = self._register_optimization_strategy(
                model_version, sport, optimal_threshold, optimal_adjustment
            )
            
            return {
                'success': True,
                'model_version': model_version,
                'sport': sport,
                'confidence_threshold': {
                    'optimal_threshold': float(optimal_threshold),
                    'optimal_accuracy': float(optimal_accuracy),
                    'optimal_count': int(optimal_count)
                },
                'line_adjustment': {
                    'optimal_adjustment': float(optimal_adjustment),
                    'optimal_accuracy': float(optimal_adj_accuracy)
                },
                'strategy_id': strategy_id,
                'optimization_file': optimization_file
            }
        except Exception as e:
            logger.error(f"Error implementing reinforcement learning: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_recommendations(self, accuracy_results: Dict[str, Any], error_results: Dict[str, Any], 
                                bias_results: Dict[str, Any], importance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on analysis results.
        
        Args:
            accuracy_results: Accuracy analysis results
            error_results: Error analysis results
            bias_results: Bias analysis results
            importance_results: Feature importance analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check overall accuracy
        overall = accuracy_results.get('overall', {})
        if overall:
            if overall.get('over_under_accuracy', 0) < 0.55:
                recommendations.append({
                    'category': 'Accuracy',
                    'issue': 'Low over/under accuracy',
                    'recommendation': 'Consider retraining the model with more features or different architecture',
                    'priority': 'High'
                })
            
            if overall.get('mape', 0) > 20:
                recommendations.append({
                    'category': 'Accuracy',
                    'issue': 'High percentage error',
                    'recommendation': 'Improve feature engineering for better regression accuracy',
                    'priority': 'Medium'
                })
        
        # Check for biases
        biases = bias_results.get('biases', [])
        if biases:
            for bias in biases[:3]:
                recommendations.append({
                    'category': 'Bias',
                    'issue': f"Systematic bias in {bias.get('category')}: {bias.get('value')}",
                    'recommendation': f"Add features to address {bias.get('category')} bias or use separate models",
                    'priority': 'High' if bias.get('p_value', 1) < 0.01 else 'Medium'
                })
        
        # Check for player-specific errors
        player_errors = error_results.get('by_player', [])
        if player_errors:
            high_error_players = [p for p in player_errors if p.get('mae', 0) > overall.get('mae', 0) * 1.5 and p.get('count', 0) > 10]
            if high_error_players:
                player_names = ', '.join([p.get('player_name', '') for p in high_error_players[:3]])
                recommendations.append({
                    'category': 'Player Errors',
                    'issue': f"High prediction errors for specific players: {player_names}",
                    'recommendation': 'Create player-specific features or models for these players',
                    'priority': 'Medium'
                })
        
        # Check for stat type errors
        stat_type_errors = error_results.get('by_stat_type', [])
        if stat_type_errors:
            high_error_stats = [s for s in stat_type_errors if s.get('mae', 0) > overall.get('mae', 0) * 1.5 and s.get('count', 0) > 10]
            if high_error_stats:
                stat_types = ', '.join([s.get('stat_type', '') for s in high_error_stats[:3]])
                recommendations.append({
                    'category': 'Stat Type Errors',
                    'issue': f"High prediction errors for specific stat types: {stat_types}",
                    'recommendation': 'Create separate models for these stat types',
                    'priority': 'Medium'
                })
        
        # Check feature importance
        feature_importance = importance_results.get('feature_importance', [])
        if feature_importance:
            top_features = [f.get('feature', '') for f in feature_importance[:5]]
            recommendations.append({
                'category': 'Feature Importance',
                'issue': 'Model relies heavily on specific features',
                'recommendation': f"Consider adding more features similar to top features: {', '.join(top_features)}",
                'priority': 'Low'
            })
        
        return recommendations
    
    def _generate_rl_plots(self, thresholds: np.ndarray, accuracies: List[float], counts: List[int], 
                         adjustments: np.ndarray, adj_accuracies: List[float], sport: Optional[str] = None, 
                         model_version: Optional[str] = None) -> None:
        """
        Generate reinforcement learning optimization plots.
        
        Args:
            thresholds: Array of confidence thresholds
            accuracies: List of accuracies for each threshold
            counts: List of prediction counts for each threshold
            adjustments: Array of line adjustments
            adj_accuracies: List of accuracies for each adjustment
            sport: Sport to filter by (optional)
            model_version: Model version (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join('reports', 'performance', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot confidence threshold vs accuracy
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Confidence Threshold')
            ax1.set_ylabel('Accuracy', color=color)
            ax1.plot(thresholds, accuracies, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Count', color=color)
            ax2.plot(thresholds, counts, marker='x', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f"Confidence Threshold Optimization{' for ' + sport.upper() if sport else ''}")
            plt.grid(True)
            plt.tight_layout()
            
            threshold_file = os.path.join(
                plots_dir, 
                f"confidence_threshold_optimization{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(threshold_file)
            plt.close()
            
            # Plot line adjustment vs accuracy
            plt.figure(figsize=(12, 6))
            plt.plot(adjustments, adj_accuracies, marker='o')
            plt.title(f"Line Adjustment Optimization{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Line Adjustment')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.tight_layout()
            
            adjustment_file = os.path.join(
                plots_dir, 
                f"line_adjustment_optimization{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(adjustment_file)
            plt.close()
            
            logger.info(f"Generated reinforcement learning plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating reinforcement learning plots: {e}")
    
    def _register_optimization_strategy(self, model_version: str, sport: Optional[str] = None, 
                                      confidence_threshold: float = 0.5, line_adjustment: float = 0.0) -> str:
        """
        Register optimization strategy in the database.
        
        Args:
            model_version: Model version
            sport: Sport to filter by (optional)
            confidence_threshold: Optimal confidence threshold
            line_adjustment: Optimal line adjustment
            
        Returns:
            Strategy ID
        """
        try:
            # Generate strategy ID
            strategy_id = f"strategy_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Insert strategy into database
            query = """
            INSERT INTO optimization_strategies (
                strategy_id, model_version, sport, confidence_threshold, line_adjustment, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
            """
            
            params = (
                strategy_id,
                model_version,
                sport,
                confidence_threshold,
                line_adjustment,
                datetime.now()
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Registered optimization strategy: {strategy_id}")
            
            return strategy_id
        except Exception as e:
            logger.error(f"Error registering optimization strategy: {e}")
            return ""


if __name__ == "__main__":
    # Run performance tracking
    tracker = PerformanceTracker()
    accuracy_results = tracker.track_accuracy_over_time()
    error_results = tracker.analyze_errors()
    bias_results = tracker.identify_systematic_biases()
    
    # Run continuous improvement
    improvement = ContinuousImprovement()
    analysis_results = improvement.analyze_model_performance()
    rl_results = improvement.implement_reinforcement_learning()
    
    # Log results
    logger.info(f"Performance tracking and continuous improvement completed")
