"""
Data Versioning for Sports Prediction System

This module implements data versioning capabilities for the sports prediction system,
allowing tracking of model versions, predictions, and historical data.
"""

import os
import sys
import logging
import json
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values, Json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, get_db_cursor, execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/versioning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('versioning')

class DataVersioning:
    """
    Class to handle data versioning for the sports prediction system.
    """
    
    @staticmethod
    def register_model_version(version_id, model_type, description, hyperparameters, 
                              training_data_range, validation_metrics, created_by=None, active=False):
        """
        Register a new model version.
        
        Args:
            version_id (str): Unique identifier for the model version
            model_type (str): Type of model (hybrid, lstm, transformer, etc.)
            description (str): Description of the model version
            hyperparameters (dict): Model hyperparameters
            training_data_range (dict): Start and end dates of training data
            validation_metrics (dict): Validation metrics (accuracy, MSE, etc.)
            created_by (str): User who created the model
            active (bool): Whether this is the active model version
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            query = """
            INSERT INTO model_versions (
                version_id, model_type, description, hyperparameters, 
                training_date, training_data_range, validation_metrics, 
                created_at, created_by, active
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            params = (
                version_id,
                model_type,
                description,
                Json(hyperparameters),
                datetime.now(),
                Json(training_data_range),
                Json(validation_metrics),
                datetime.now(),
                created_by,
                active
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                # If this model is active, deactivate all other models of the same type
                if active:
                    cursor.execute(
                        "UPDATE model_versions SET active = FALSE WHERE model_type = %s AND version_id != %s",
                        (model_type, version_id)
                    )
                
                conn.commit()
                cursor.close()
            
            logger.info(f"Registered model version: {version_id}")
            return True
        except Exception as e:
            logger.error(f"Error registering model version: {e}")
            return False
    
    @staticmethod
    def get_active_model_version(model_type=None):
        """
        Get the active model version.
        
        Args:
            model_type (str): Type of model (optional)
            
        Returns:
            dict: Active model version information
        """
        try:
            if model_type:
                query = "SELECT * FROM model_versions WHERE model_type = %s AND active = TRUE"
                params = (model_type,)
            else:
                query = "SELECT * FROM model_versions WHERE active = TRUE"
                params = None
            
            result = execute_query(query, params)
            
            if result:
                return result[0]
            else:
                logger.warning(f"No active model version found for type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Error getting active model version: {e}")
            return None
    
    @staticmethod
    def get_model_version_history(model_type=None, limit=10):
        """
        Get the history of model versions.
        
        Args:
            model_type (str): Type of model (optional)
            limit (int): Maximum number of versions to return
            
        Returns:
            list: Model version history
        """
        try:
            if model_type:
                query = """
                SELECT * FROM model_versions 
                WHERE model_type = %s 
                ORDER BY created_at DESC 
                LIMIT %s
                """
                params = (model_type, limit)
            else:
                query = """
                SELECT * FROM model_versions 
                ORDER BY created_at DESC 
                LIMIT %s
                """
                params = (limit,)
            
            result = execute_query(query, params)
            return result
        except Exception as e:
            logger.error(f"Error getting model version history: {e}")
            return []
    
    @staticmethod
    def store_prediction_with_version(player_id, game_id, stat_type, predicted_value, 
                                     confidence_score, prediction_range, over_probability, 
                                     line_value, model_version, top_factors=None, feature_importance=None):
        """
        Store a prediction with version information.
        
        Args:
            player_id (str): Player ID
            game_id (str): Game ID
            stat_type (str): Type of statistic being predicted
            predicted_value (float): Predicted value
            confidence_score (float): Confidence score (0-100)
            prediction_range (tuple): Low and high range of prediction
            over_probability (float): Probability of going over the line (0-1)
            line_value (float): Line value
            model_version (str): Model version ID
            top_factors (dict): Top factors that influenced the prediction
            feature_importance (dict): Importance of each feature in the prediction
            
        Returns:
            str: Prediction ID if successful, None otherwise
        """
        try:
            prediction_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO predictions (
                prediction_id, player_id, game_id, stat_type, predicted_value, 
                confidence_score, prediction_range_low, prediction_range_high, 
                over_probability, line_value, created_at, model_version, 
                top_factors, feature_importance
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING prediction_id
            """
            
            params = (
                prediction_id,
                player_id,
                game_id,
                stat_type,
                predicted_value,
                confidence_score,
                prediction_range[0] if prediction_range else None,
                prediction_range[1] if prediction_range else None,
                over_probability,
                line_value,
                datetime.now(),
                model_version,
                Json(top_factors) if top_factors else None,
                Json(feature_importance) if feature_importance else None
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
                conn.commit()
                cursor.close()
            
            logger.info(f"Stored prediction with ID: {prediction_id}")
            return prediction_id
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return None
    
    @staticmethod
    def get_prediction_history(player_id, stat_type, limit=10):
        """
        Get the history of predictions for a player and stat type.
        
        Args:
            player_id (str): Player ID
            stat_type (str): Type of statistic
            limit (int): Maximum number of predictions to return
            
        Returns:
            list: Prediction history
        """
        try:
            query = """
            SELECT p.*, m.model_type, m.description as model_description
            FROM predictions p
            JOIN model_versions m ON p.model_version = m.version_id
            WHERE p.player_id = %s AND p.stat_type = %s
            ORDER BY p.created_at DESC
            LIMIT %s
            """
            
            params = (player_id, stat_type, limit)
            result = execute_query(query, params)
            return result
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []
    
    @staticmethod
    def compare_model_versions(version_id_1, version_id_2, sport=None, stat_type=None):
        """
        Compare the performance of two model versions.
        
        Args:
            version_id_1 (str): First model version ID
            version_id_2 (str): Second model version ID
            sport (str): Sport to filter by (optional)
            stat_type (str): Stat type to filter by (optional)
            
        Returns:
            dict: Comparison results
        """
        try:
            # Get performance metrics for both models
            query_base = """
            SELECT 
                model_version,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN (p.over_probability > 0.5 AND ar.actual_value > p.line_value) OR 
                           (p.over_probability <= 0.5 AND ar.actual_value <= p.line_value) 
                    THEN 1 ELSE 0 END) as correct_predictions,
                ROUND(SUM(CASE WHEN (p.over_probability > 0.5 AND ar.actual_value > p.line_value) OR 
                                (p.over_probability <= 0.5 AND ar.actual_value <= p.line_value) 
                         THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) as accuracy_percentage,
                ROUND(AVG(ABS(p.predicted_value - ar.actual_value)), 2) as mean_absolute_error,
                ROUND(SQRT(AVG(POWER(p.predicted_value - ar.actual_value, 2))), 2) as root_mean_squared_error
            FROM 
                predictions p
            JOIN 
                actual_results ar ON p.player_id = ar.player_id AND p.game_id = ar.game_id AND p.stat_type = ar.stat_type
            JOIN 
                games g ON p.game_id = g.game_id
            WHERE 
                p.model_version = %s
                AND g.status = 'completed'
            """
            
            params_1 = [version_id_1]
            params_2 = [version_id_2]
            
            if sport:
                query_base += " AND g.sport = %s"
                params_1.append(sport)
                params_2.append(sport)
            
            if stat_type:
                query_base += " AND p.stat_type = %s"
                params_1.append(stat_type)
                params_2.append(stat_type)
            
            query_base += " GROUP BY p.model_version"
            
            # Execute queries
            metrics_1 = execute_query(query_base, tuple(params_1))
            metrics_2 = execute_query(query_base, tuple(params_2))
            
            # Get model information
            model_1 = execute_query("SELECT * FROM model_versions WHERE version_id = %s", (version_id_1,))
            model_2 = execute_query("SELECT * FROM model_versions WHERE version_id = %s", (version_id_2,))
            
            # Compile comparison results
            comparison = {
                'model_1': {
                    'version_id': version_id_1,
                    'model_type': model_1[0]['model_type'] if model_1 else None,
                    'description': model_1[0]['description'] if model_1 else None,
                    'metrics': metrics_1[0] if metrics_1 else None
                },
                'model_2': {
                    'version_id': version_id_2,
                    'model_type': model_2[0]['model_type'] if model_2 else None,
                    'description': model_2[0]['description'] if model_2 else None,
                    'metrics': metrics_2[0] if metrics_2 else None
                }
            }
            
            # Calculate differences
            if metrics_1 and metrics_2:
                m1 = metrics_1[0]
                m2 = metrics_2[0]
                
                comparison['differences'] = {
                    'accuracy_percentage': round(m1['accuracy_percentage'] - m2['accuracy_percentage'], 2),
                    'mean_absolute_error': round(m1['mean_absolute_error'] - m2['mean_absolute_error'], 2),
                    'root_mean_squared_error': round(m1['root_mean_squared_error'] - m2['root_mean_squared_error'], 2)
                }
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            return {}
    
    @staticmethod
    def setup_ab_test(name, description, model_a, model_b):
        """
        Set up an A/B test between two model versions.
        
        Args:
            name (str): Name of the A/B test
            description (str): Description of the A/B test
            model_a (str): First model version ID
            model_b (str): Second model version ID
            
        Returns:
            str: A/B test ID if successful, None otherwise
        """
        try:
            test_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO ab_testing (
                test_id, name, description, model_a, model_b, 
                start_date, status, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING test_id
            """
            
            params = (
                test_id,
                name,
                description,
                model_a,
                model_b,
                datetime.now(),
                'active',
                datetime.now(),
                datetime.now()
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
                conn.commit()
                cursor.close()
            
            logger.info(f"Set up A/B test with ID: {test_id}")
            return test_id
        except Exception as e:
            logger.error(f"Error setting up A/B test: {e}")
            return None
    
    @staticmethod
    def complete_ab_test(test_id, results):
        """
        Complete an A/B test and store the results.
        
        Args:
            test_id (str): A/B test ID
            results (dict): Test results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = """
            UPDATE ab_testing
            SET status = 'completed', end_date = %s, results = %s, updated_at = %s
            WHERE test_id = %s
            """
            
            params = (
                datetime.now(),
                Json(results),
                datetime.now(),
                test_id
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Completed A/B test with ID: {test_id}")
            return True
        except Exception as e:
            logger.error(f"Error completing A/B test: {e}")
            return False
    
    @staticmethod
    def get_historical_data_version(table_name, record_id, version_date=None):
        """
        Get a historical version of a record.
        
        Args:
            table_name (str): Table name
            record_id (str): Record ID
            version_date (datetime): Date of the version to retrieve (optional)
            
        Returns:
            dict: Historical record
        """
        try:
            # Determine the ID column name based on table name
            id_column = f"{table_name[:-1]}_id" if table_name.endswith('s') else f"{table_name}_id"
            
            if version_date:
                query = f"""
                SELECT * FROM versioning.{table_name}
                WHERE {id_column} = %s
                AND valid_from <= %s
                AND (valid_to > %s OR valid_to IS NULL)
                ORDER BY version_id DESC
                LIMIT 1
                """
                params = (record_id, version_date, version_date)
            else:
                query = f"""
                SELECT * FROM versioning.{table_name}
                WHERE {id_column} = %s
                ORDER BY version_id DESC
                LIMIT 1
                """
                params = (record_id,)
            
            result = execute_query(query, params)
            
            if result:
                return result[0]
            else:
                logger.warning(f"No historical version found for {table_name} with ID {record_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting historical data version: {e}")
            return None
    
    @staticmethod
    def get_version_history(table_name, record_id):
        """
        Get the version history of a record.
        
        Args:
            table_name (str): Table name
            record_id (str): Record ID
            
        Returns:
            list: Version history
        """
        try:
            # Determine the ID column name based on table name
            id_column = f"{table_name[:-1]}_id" if table_name.endswith('s') else f"{table_name}_id"
            
            query = f"""
            SELECT * FROM versioning.{table_name}
            WHERE {id_column} = %s
            ORDER BY version_id DESC
            """
            
            params = (record_id,)
            result = execute_query(query, params)
            return result
        except Exception as e:
            logger.error(f"Error getting version history: {e}")
            return []
