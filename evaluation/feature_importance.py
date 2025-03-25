"""
Feature Importance Analysis and Anomaly Detection for Sports Prediction System

This module implements feature importance analysis and anomaly detection for the sports prediction system,
providing insights into model behavior and unusual player performances.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query
from database.versioning import DataVersioning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_importance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('feature_importance')

class FeatureImportanceAnalyzer:
    """
    Class for analyzing feature importance in prediction models.
    """
    
    def __init__(self):
        """
        Initialize the feature importance analyzer.
        """
        self.data_versioning = DataVersioning()
        self.reports_dir = os.path.join('reports', 'feature_importance')
        os.makedirs(self.reports_dir, exist_ok=True)
    
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
                # If no stored feature importance, try to calculate it
                calculated_importance = self._calculate_feature_importance(model_version, sport)
                if calculated_importance:
                    feature_importance = calculated_importance
                else:
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
    
    def analyze_feature_interactions(self, model_version: Optional[str] = None, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze feature interactions for the model.
        
        Args:
            model_version: Model version to analyze (optional)
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of feature interaction analysis results
        """
        try:
            logger.info(f"Analyzing feature interactions for model version {model_version}")
            
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
            
            logger.info(f"Analyzing feature interactions for model version {model_version}")
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model_version)
            
            if not feature_importance:
                logger.warning(f"No feature importance available for model version {model_version}")
                return {'success': False, 'error': f"No feature importance available for model version {model_version}"}
            
            # Get top features
            top_features = [f['feature'] for f in sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]]
            
            # Get training data
            training_data = self._get_training_data(model_version, sport)
            
            if not training_data:
                logger.warning(f"No training data available for model version {model_version}")
                return {'success': False, 'error': f"No training data available for model version {model_version}"}
            
            # Calculate feature interactions
            interactions = self._calculate_feature_interactions(training_data, top_features)
            
            # Generate plots
            self._generate_interaction_plots(training_data, interactions, model_version, sport)
            
            # Save interactions to file
            interactions_file = os.path.join(
                self.reports_dir, 
                f"feature_interactions_{model_version}{'_' + sport if sport else ''}.json"
            )
            
            with open(interactions_file, 'w') as f:
                json.dump({
                    'model_version': model_version,
                    'interactions': interactions
                }, f, indent=2)
            
            logger.info(f"Saved feature interactions to {interactions_file}")
            
            return {
                'success': True,
                'model_version': model_version,
                'interactions': interactions,
                'interactions_file': interactions_file
            }
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_feature_drift(self, model_version: Optional[str] = None, sport: Optional[str] = None,
                            start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze feature drift over time.
        
        Args:
            model_version: Model version to analyze (optional)
            sport: Sport to filter by (optional)
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            
        Returns:
            Dictionary of feature drift analysis results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Analyzing feature drift from {start_date} to {end_date}")
            
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
            
            logger.info(f"Analyzing feature drift for model version {model_version}")
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model_version)
            
            if not feature_importance:
                logger.warning(f"No feature importance available for model version {model_version}")
                return {'success': False, 'error': f"No feature importance available for model version {model_version}"}
            
            # Get top features
            top_features = [f['feature'] for f in sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]]
            
            # Get feature data over time
            feature_data = self._get_feature_data_over_time(top_features, start_date, end_date, sport)
            
            if not feature_data:
                logger.warning(f"No feature data available for the specified criteria")
                return {'success': False, 'error': f"No feature data available for the specified criteria"}
            
            # Calculate drift metrics
            drift_metrics = self._calculate_drift_metrics(feature_data)
            
            # Generate plots
            self._generate_drift_plots(feature_data, drift_metrics, model_version, sport)
            
            # Save drift metrics to file
            drift_file = os.path.join(
                self.reports_dir, 
                f"feature_drift_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(drift_file, 'w') as f:
                json.dump({
                    'model_version': model_version,
                    'start_date': start_date,
                    'end_date': end_date,
                    'drift_metrics': drift_metrics
                }, f, indent=2, default=str)
            
            logger.info(f"Saved feature drift metrics to {drift_file}")
            
            return {
                'success': True,
                'model_version': model_version,
                'start_date': start_date,
                'end_date': end_date,
                'drift_metrics': drift_metrics,
                'drift_file': drift_file
            }
        except Exception as e:
            logger.error(f"Error analyzing feature drift: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_feature_importance(self, model_version: str) -> List[Dict[str, Any]]:
        """
        Get feature importance from database.
        
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
    
    def _calculate_feature_importance(self, model_version: str, sport: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calculate feature importance for a model.
        
        Args:
            model_version: Model version
            sport: Sport to filter by (optional)
            
        Returns:
            List of feature importance values
        """
        try:
            # Get model path
            model_info = self.data_versioning.get_model_version(model_version)
            if not model_info:
                logger.warning(f"Model version {model_version} not found")
                return []
            
            model_path = model_info.get('model_path')
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model file not found for version {model_version}")
                return []
            
            # Get training data
            training_data = self._get_training_data(model_version, sport)
            if not training_data:
                logger.warning(f"No training data available for model version {model_version}")
                return []
            
            # Extract features and targets
            features = training_data['features']
            targets = training_data['targets']
            feature_names = training_data['feature_names']
            
            # Determine model type
            model_type = model_info.get('model_type', '')
            
            if model_type in ['hybrid', 'lstm', 'transformer']:
                # For neural network models, use permutation importance
                importance_values = self._calculate_permutation_importance(model_path, features, targets, feature_names)
            elif model_type == 'ensemble':
                # For ensemble models, use built-in feature importance
                importance_values = self._calculate_ensemble_importance(model_path, features, feature_names)
            else:
                # Default to training a surrogate model
                importance_values = self._calculate_surrogate_importance(features, targets, feature_names)
            
            # Store feature importance in database
            self._store_feature_importance(model_version, importance_values)
            
            return importance_values
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _calculate_permutation_importance(self, model_path: str, features: np.ndarray, targets: np.ndarray, 
                                        feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate permutation importance for a neural network model.
        
        Args:
            model_path: Path to model file
            features: Feature matrix
            targets: Target values
            feature_names: List of feature names
            
        Returns:
            List of feature importance values
        """
        try:
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Calculate baseline score
            baseline_pred = model.predict(features)
            baseline_score = mean_squared_error(targets, baseline_pred)
            
            # Calculate permutation importance
            importance_values = []
            
            for i, feature_name in enumerate(feature_names):
                # Create a copy of the features
                features_permuted = features.copy()
                
                # Permute the feature
                features_permuted[:, i] = np.random.permutation(features_permuted[:, i])
                
                # Make predictions with permuted feature
                permuted_pred = model.predict(features_permuted)
                permuted_score = mean_squared_error(targets, permuted_pred)
                
                # Calculate importance
                importance = permuted_score - baseline_score
                
                importance_values.append({
                    'feature': feature_name,
                    'importance': float(importance)
                })
            
            # Normalize importance values
            max_importance = max(abs(v['importance']) for v in importance_values)
            if max_importance > 0:
                for v in importance_values:
                    v['importance'] = v['importance'] / max_importance
            
            # Sort by importance
            importance_values = sorted(importance_values, key=lambda x: x['importance'], reverse=True)
            
            return importance_values
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return []
    
    def _calculate_ensemble_importance(self, model_path: str, features: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate feature importance for an ensemble model.
        
        Args:
            model_path: Path to model file
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            List of feature importance values
        """
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                # Direct feature importance
                importances = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # Average feature importance from estimators
                importances = np.zeros(len(feature_names))
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                importances /= len(model.estimators_)
            else:
                # Train a surrogate model
                return self._calculate_surrogate_importance(features, model.predict(features), feature_names)
            
            # Format results
            importance_values = []
            
            for i, feature_name in enumerate(feature_names):
                importance_values.append({
                    'feature': feature_name,
                    'importance': float(importances[i])
                })
            
            # Sort by importance
            importance_values = sorted(importance_values, key=lambda x: x['importance'], reverse=True)
            
            return importance_values
        except Exception as e:
            logger.error(f"Error calculating ensemble importance: {e}")
            return []
    
    def _calculate_surrogate_importance(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate feature importance using a surrogate model.
        
        Args:
            features: Feature matrix
            targets: Target values
            feature_names: List of feature names
            
        Returns:
            List of feature importance values
        """
        try:
            # Train a surrogate model
            surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
            surrogate.fit(features, targets)
            
            # Get feature importance
            importances = surrogate.feature_importances_
            
            # Format results
            importance_values = []
            
            for i, feature_name in enumerate(feature_names):
                importance_values.append({
                    'feature': feature_name,
                    'importance': float(importances[i])
                })
            
            # Sort by importance
            importance_values = sorted(importance_values, key=lambda x: x['importance'], reverse=True)
            
            return importance_values
        except Exception as e:
            logger.error(f"Error calculating surrogate importance: {e}")
            return []
    
    def _store_feature_importance(self, model_version: str, importance_values: List[Dict[str, Any]]) -> bool:
        """
        Store feature importance in database.
        
        Args:
            model_version: Model version
            importance_values: List of feature importance values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing feature importance
            delete_query = """
            DELETE FROM feature_importance
            WHERE model_version = %s
            """
            
            delete_params = (model_version,)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(delete_query, delete_params)
                conn.commit()
                cursor.close()
            
            # Insert new feature importance
            insert_query = """
            INSERT INTO feature_importance (
                model_version, feature_name, importance, created_at
            ) VALUES (
                %s, %s, %s, %s
            )
            """
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for value in importance_values:
                    insert_params = (
                        model_version,
                        value['feature'],
                        value['importance'],
                        datetime.now()
                    )
                    
                    cursor.execute(insert_query, insert_params)
                
                conn.commit()
                cursor.close()
            
            logger.info(f"Stored feature importance for model version {model_version}")
            
            return True
        except Exception as e:
            logger.error(f"Error storing feature importance: {e}")
            return False
    
    def _get_training_data(self, model_version: str, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Get training data for a model.
        
        Args:
            model_version: Model version
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary with features, targets, and feature names
        """
        try:
            # Get model information
            model_info = self.data_versioning.get_model_version(model_version)
            if not model_info:
                logger.warning(f"Model version {model_version} not found")
                return {}
            
            # Get training data range
            training_data_range = model_info.get('training_data_range', {})
            start_date = training_data_range.get('start_date')
            end_date = training_data_range.get('end_date')
            
            if not start_date or not end_date:
                logger.warning(f"No training data range found for model version {model_version}")
                return {}
            
            # Build query to get feature data
            query = """
            SELECT fd.player_id, fd.game_id, fd.feature_data, fd.created_at,
                   g.sport, g.game_date
            FROM feature_data fd
            JOIN games g ON fd.game_id = g.game_id
            WHERE g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning(f"No feature data found for the specified criteria")
                return {}
            
            logger.info(f"Found {len(results)} feature data records")
            
            # Extract features and targets
            features_list = []
            targets_list = []
            feature_names = []
            
            for result in results:
                feature_data = json.loads(result['feature_data'])
                
                if not feature_names and 'feature_names' in feature_data:
                    feature_names = feature_data['feature_names']
                
                if 'features' in feature_data and 'target' in feature_data:
                    features_list.append(feature_data['features'])
                    targets_list.append(feature_data['target'])
            
            if not features_list or not targets_list:
                logger.warning(f"No valid feature data found")
                return {}
            
            # Convert to numpy arrays
            features = np.array(features_list)
            targets = np.array(targets_list)
            
            logger.info(f"Extracted {len(features)} training examples with {len(feature_names)} features")
            
            return {
                'features': features,
                'targets': targets,
                'feature_names': feature_names
            }
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return {}
    
    def _calculate_feature_interactions(self, training_data: Dict[str, Any], top_features: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate feature interactions.
        
        Args:
            training_data: Dictionary with features, targets, and feature names
            top_features: List of top feature names
            
        Returns:
            List of feature interaction values
        """
        try:
            features = training_data['features']
            targets = training_data['targets']
            feature_names = training_data['feature_names']
            
            # Get indices of top features
            top_indices = [feature_names.index(f) for f in top_features if f in feature_names]
            
            # Calculate interactions
            interactions = []
            
            for i, idx_i in enumerate(top_indices):
                for j, idx_j in enumerate(top_indices):
                    if i >= j:
                        continue
                    
                    # Calculate correlation
                    correlation = np.corrcoef(features[:, idx_i], features[:, idx_j])[0, 1]
                    
                    # Calculate interaction strength
                    interaction_strength = self._calculate_interaction_strength(
                        features, targets, idx_i, idx_j
                    )
                    
                    interactions.append({
                        'feature_1': feature_names[idx_i],
                        'feature_2': feature_names[idx_j],
                        'correlation': float(correlation),
                        'interaction_strength': float(interaction_strength)
                    })
            
            # Sort by interaction strength
            interactions = sorted(interactions, key=lambda x: abs(x['interaction_strength']), reverse=True)
            
            return interactions
        except Exception as e:
            logger.error(f"Error calculating feature interactions: {e}")
            return []
    
    def _calculate_interaction_strength(self, features: np.ndarray, targets: np.ndarray, idx_i: int, idx_j: int) -> float:
        """
        Calculate interaction strength between two features.
        
        Args:
            features: Feature matrix
            targets: Target values
            idx_i: Index of first feature
            idx_j: Index of second feature
            
        Returns:
            Interaction strength
        """
        try:
            # Train a model with individual features
            model_i = RandomForestRegressor(n_estimators=50, random_state=42)
            model_i.fit(features[:, [idx_i]], targets)
            pred_i = model_i.predict(features[:, [idx_i]])
            
            model_j = RandomForestRegressor(n_estimators=50, random_state=42)
            model_j.fit(features[:, [idx_j]], targets)
            pred_j = model_j.predict(features[:, [idx_j]])
            
            # Train a model with both features
            model_ij = RandomForestRegressor(n_estimators=50, random_state=42)
            model_ij.fit(features[:, [idx_i, idx_j]], targets)
            pred_ij = model_ij.predict(features[:, [idx_i, idx_j]])
            
            # Calculate errors
            error_i = mean_squared_error(targets, pred_i)
            error_j = mean_squared_error(targets, pred_j)
            error_ij = mean_squared_error(targets, pred_ij)
            
            # Calculate interaction strength
            interaction_strength = (error_i + error_j - error_ij) / max(error_i, error_j)
            
            return interaction_strength
        except Exception as e:
            logger.error(f"Error calculating interaction strength: {e}")
            return 0.0
    
    def _get_feature_data_over_time(self, top_features: List[str], start_date: str, end_date: str, 
                                  sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Get feature data over time.
        
        Args:
            top_features: List of top feature names
            start_date: Start date for analysis
            end_date: End date for analysis
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary with feature data over time
        """
        try:
            # Build query to get feature data
            query = """
            SELECT fd.player_id, fd.game_id, fd.feature_data, fd.created_at,
                   g.sport, g.game_date
            FROM feature_data fd
            JOIN games g ON fd.game_id = g.game_id
            WHERE g.game_date BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            if sport:
                query += " AND g.sport = %s"
                params.append(sport)
            
            # Execute query
            results = execute_query(query, params)
            
            if not results:
                logger.warning(f"No feature data found for the specified criteria")
                return {}
            
            logger.info(f"Found {len(results)} feature data records")
            
            # Extract feature data over time
            feature_data = {}
            
            for result in results:
                data = json.loads(result['feature_data'])
                
                if 'feature_names' not in data or 'features' not in data:
                    continue
                
                feature_names = data['feature_names']
                features = data['features']
                
                # Get date
                date = result['game_date']
                if isinstance(date, datetime):
                    date = date.strftime('%Y-%m-%d')
                
                # Initialize date entry
                if date not in feature_data:
                    feature_data[date] = {f: [] for f in top_features}
                
                # Add feature values
                for feature in top_features:
                    if feature in feature_names:
                        idx = feature_names.index(feature)
                        if idx < len(features):
                            feature_data[date][feature].append(features[idx])
            
            # Calculate statistics for each date
            feature_stats = {}
            
            for date, features in feature_data.items():
                feature_stats[date] = {}
                
                for feature, values in features.items():
                    if values:
                        feature_stats[date][feature] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'count': len(values)
                        }
            
            # Sort by date
            feature_stats = {k: feature_stats[k] for k in sorted(feature_stats.keys())}
            
            return feature_stats
        except Exception as e:
            logger.error(f"Error getting feature data over time: {e}")
            return {}
    
    def _calculate_drift_metrics(self, feature_data: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate drift metrics for features over time.
        
        Args:
            feature_data: Dictionary with feature data over time
            
        Returns:
            Dictionary with drift metrics
        """
        try:
            # Get dates and features
            dates = list(feature_data.keys())
            if len(dates) < 2:
                logger.warning(f"Not enough data points to calculate drift")
                return {}
            
            features = list(feature_data[dates[0]].keys())
            
            # Calculate drift metrics
            drift_metrics = {}
            
            for feature in features:
                # Get feature values over time
                values = []
                
                for date in dates:
                    if feature in feature_data[date]:
                        values.append(feature_data[date][feature]['mean'])
                
                if len(values) < 2:
                    continue
                
                # Calculate drift metrics
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                # Calculate trend
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                
                # Calculate drift score
                drift_score = abs(slope) * len(values) / (std_value if std_value > 0 else 1)
                
                drift_metrics[feature] = {
                    'mean': float(mean_value),
                    'std': float(std_value),
                    'slope': float(slope),
                    'drift_score': float(drift_score)
                }
            
            # Sort by drift score
            drift_metrics = {k: drift_metrics[k] for k in sorted(drift_metrics.keys(), key=lambda x: drift_metrics[x]['drift_score'], reverse=True)}
            
            return drift_metrics
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {e}")
            return {}
    
    def _generate_feature_importance_plots(self, feature_importance: List[Dict[str, Any]], 
                                         model_version: str, sport: Optional[str] = None) -> None:
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
            
            plt.figure(figsize=(14, 10))
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
    
    def _generate_interaction_plots(self, training_data: Dict[str, Any], interactions: List[Dict[str, Any]],
                                  model_version: str, sport: Optional[str] = None) -> None:
        """
        Generate feature interaction plots.
        
        Args:
            training_data: Dictionary with features, targets, and feature names
            interactions: List of feature interaction values
            model_version: Model version
            sport: Sport to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Get data
            features = training_data['features']
            targets = training_data['targets']
            feature_names = training_data['feature_names']
            
            # Plot top 5 interactions
            top_interactions = interactions[:5]
            
            for interaction in top_interactions:
                feature_1 = interaction['feature_1']
                feature_2 = interaction['feature_2']
                
                if feature_1 not in feature_names or feature_2 not in feature_names:
                    continue
                
                idx_1 = feature_names.index(feature_1)
                idx_2 = feature_names.index(feature_2)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(features[:, idx_1], features[:, idx_2], c=targets, cmap='viridis', alpha=0.5)
                plt.colorbar(label='Target Value')
                plt.title(f"Interaction: {feature_1} vs {feature_2}")
                plt.xlabel(feature_1)
                plt.ylabel(feature_2)
                plt.grid(True)
                plt.tight_layout()
                
                interaction_file = os.path.join(
                    plots_dir, 
                    f"interaction_{feature_1}_vs_{feature_2}_{model_version}{'_' + sport if sport else ''}.png"
                )
                plt.savefig(interaction_file)
                plt.close()
            
            # Plot interaction strength
            plt.figure(figsize=(14, 10))
            interaction_pairs = [f"{i['feature_1']} vs {i['feature_2']}" for i in top_interactions]
            interaction_strengths = [i['interaction_strength'] for i in top_interactions]
            
            plt.barh(interaction_pairs, interaction_strengths, color='lightgreen')
            plt.title(f"Top Feature Interaction Strengths{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Interaction Strength')
            plt.ylabel('Feature Pair')
            plt.grid(True)
            plt.tight_layout()
            
            strength_file = os.path.join(
                plots_dir, 
                f"interaction_strengths_{model_version}{'_' + sport if sport else ''}.png"
            )
            plt.savefig(strength_file)
            plt.close()
            
            logger.info(f"Generated interaction plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating interaction plots: {e}")
    
    def _generate_drift_plots(self, feature_data: Dict[str, Dict[str, Dict[str, float]]], 
                            drift_metrics: Dict[str, Dict[str, float]], model_version: str, 
                            sport: Optional[str] = None) -> None:
        """
        Generate feature drift plots.
        
        Args:
            feature_data: Dictionary with feature data over time
            drift_metrics: Dictionary with drift metrics
            model_version: Model version
            sport: Sport to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Get dates and features
            dates = list(feature_data.keys())
            
            # Plot top 5 drifting features
            top_features = list(drift_metrics.keys())[:5]
            
            for feature in top_features:
                # Get feature values over time
                values = []
                
                for date in dates:
                    if feature in feature_data[date]:
                        values.append(feature_data[date][feature]['mean'])
                
                if len(values) < 2:
                    continue
                
                # Plot feature drift
                plt.figure(figsize=(14, 8))
                plt.plot(dates, values, marker='o')
                
                # Add trend line
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                trend = slope * x + intercept
                plt.plot(dates, trend, 'r--', label=f"Trend (slope: {slope:.4f})")
                
                plt.title(f"Feature Drift: {feature}{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Date')
                plt.ylabel('Feature Value (Mean)')
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                drift_file = os.path.join(
                    plots_dir, 
                    f"feature_drift_{feature}_{model_version}{'_' + sport if sport else ''}.png"
                )
                plt.savefig(drift_file)
                plt.close()
            
            # Plot drift scores
            plt.figure(figsize=(14, 10))
            feature_names = top_features
            drift_scores = [drift_metrics[f]['drift_score'] for f in top_features]
            
            plt.barh(feature_names, drift_scores, color='salmon')
            plt.title(f"Top Feature Drift Scores{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Drift Score')
            plt.ylabel('Feature')
            plt.grid(True)
            plt.tight_layout()
            
            scores_file = os.path.join(
                plots_dir, 
                f"drift_scores_{model_version}{'_' + sport if sport else ''}.png"
            )
            plt.savefig(scores_file)
            plt.close()
            
            logger.info(f"Generated drift plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating drift plots: {e}")


class AnomalyDetector:
    """
    Class for detecting anomalies in player performances.
    """
    
    def __init__(self):
        """
        Initialize the anomaly detector.
        """
        self.reports_dir = os.path.join('reports', 'anomalies')
        os.makedirs(self.reports_dir, exist_ok=True)
    
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
    
    def detect_prediction_anomalies(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  sport: Optional[str] = None, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in prediction errors.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary of prediction anomaly detection results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Detecting prediction anomalies from {start_date} to {end_date}")
            
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
            df['percentage_error'] = df['absolute_error'] / np.maximum(df['actual_value'], 1e-10) * 100
            
            # Add team context
            df['is_home_team'] = df['team_id'] == df['home_team_id']
            df['team_name'] = np.where(df['is_home_team'], df['home_team_name'], df['away_team_name'])
            df['opponent_team_id'] = np.where(df['is_home_team'], df['away_team_id'], df['home_team_id'])
            df['opponent_team_name'] = np.where(df['is_home_team'], df['away_team_name'], df['home_team_name'])
            
            # Detect anomalies
            anomalies = []
            
            # Calculate z-scores for errors
            mean_error = df['error'].mean()
            std_error = df['error'].std()
            
            if std_error > 0:
                df['error_z_score'] = (df['error'] - mean_error) / std_error
                
                # Identify anomalies
                anomaly_mask = np.abs(df['error_z_score']) > threshold
                anomaly_predictions = df[anomaly_mask]
                
                for _, row in anomaly_predictions.iterrows():
                    anomalies.append({
                        'prediction_id': row['prediction_id'],
                        'player_id': row['player_id'],
                        'player_name': row['player_name'],
                        'position': row['position'],
                        'team_name': row['team_name'],
                        'game_id': row['game_id'],
                        'game_date': row['game_date'],
                        'opponent_team_name': row['opponent_team_name'],
                        'stat_type': row['stat_type'],
                        'predicted_value': float(row['predicted_value']),
                        'actual_value': float(row['actual_value']),
                        'error': float(row['error']),
                        'error_z_score': float(row['error_z_score']),
                        'confidence_score': float(row['confidence_score']),
                        'is_home_team': bool(row['is_home_team']),
                        'direction': 'over-predicted' if row['error'] > 0 else 'under-predicted'
                    })
            
            # Sort anomalies by absolute z-score
            anomalies = sorted(anomalies, key=lambda x: abs(x['error_z_score']), reverse=True)
            
            # Generate plots
            self._generate_prediction_anomaly_plots(anomalies, sport)
            
            # Save anomalies to file
            anomalies_file = os.path.join(
                self.reports_dir, 
                f"prediction_anomalies_{start_date}_to_{end_date}{'_' + sport if sport else ''}.json"
            )
            
            with open(anomalies_file, 'w') as f:
                json.dump({
                    'anomalies': anomalies
                }, f, indent=2, default=str)
            
            logger.info(f"Saved prediction anomalies to {anomalies_file}")
            
            return {
                'success': True,
                'anomalies': anomalies,
                'anomalies_file': anomalies_file
            }
        except Exception as e:
            logger.error(f"Error detecting prediction anomalies: {e}")
            return {'success': False, 'error': str(e)}
    
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
            
            plt.figure(figsize=(14, 10))
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
            
            # Plot anomalies by position
            position_counts = {}
            for anomaly in anomalies:
                position = anomaly['position']
                if position not in position_counts:
                    position_counts[position] = 0
                position_counts[position] += 1
            
            if position_counts:
                plt.figure(figsize=(14, 8))
                positions = list(position_counts.keys())
                counts = [position_counts[p] for p in positions]
                
                plt.bar(positions, counts, color='lightgreen')
                plt.title(f"Anomalies by Position{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Position')
                plt.ylabel('Count')
                plt.grid(True)
                plt.tight_layout()
                
                position_file = os.path.join(
                    plots_dir, 
                    f"anomalies_by_position{'_' + sport if sport else ''}.png"
                )
                plt.savefig(position_file)
                plt.close()
            
            # Plot anomalies by stat type
            stat_type_counts = {}
            for anomaly in anomalies:
                stat_type = anomaly['stat_type']
                if stat_type not in stat_type_counts:
                    stat_type_counts[stat_type] = 0
                stat_type_counts[stat_type] += 1
            
            if stat_type_counts:
                plt.figure(figsize=(14, 8))
                stat_types = list(stat_type_counts.keys())
                counts = [stat_type_counts[s] for s in stat_types]
                
                plt.bar(stat_types, counts, color='lightblue')
                plt.title(f"Anomalies by Stat Type{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Stat Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True)
                plt.tight_layout()
                
                stat_type_file = os.path.join(
                    plots_dir, 
                    f"anomalies_by_stat_type{'_' + sport if sport else ''}.png"
                )
                plt.savefig(stat_type_file)
                plt.close()
            
            logger.info(f"Generated anomaly plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating anomaly plots: {e}")
    
    def _generate_prediction_anomaly_plots(self, anomalies: List[Dict[str, Any]], sport: Optional[str] = None) -> None:
        """
        Generate prediction anomaly plots.
        
        Args:
            anomalies: List of detected prediction anomalies
            sport: Sport to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot top 20 prediction anomalies
            top_anomalies = anomalies[:20]
            
            plt.figure(figsize=(14, 10))
            anomaly_names = [f"{a['player_name']} ({a['stat_type']})" for a in top_anomalies]
            errors = [a['error'] for a in top_anomalies]
            
            colors = ['red' if e > 0 else 'blue' for e in errors]
            
            plt.barh(anomaly_names, errors, color=colors)
            plt.title(f"Top 20 Prediction Anomalies{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Error (Predicted - Actual)')
            plt.ylabel('Player (Stat Type)')
            plt.grid(True)
            plt.tight_layout()
            
            anomaly_file = os.path.join(
                plots_dir, 
                f"prediction_anomalies{'_' + sport if sport else ''}.png"
            )
            plt.savefig(anomaly_file)
            plt.close()
            
            # Plot anomalies by confidence score
            if anomalies:
                plt.figure(figsize=(14, 8))
                confidence_scores = [a['confidence_score'] for a in top_anomalies]
                error_z_scores = [abs(a['error_z_score']) for a in top_anomalies]
                
                plt.scatter(confidence_scores, error_z_scores, alpha=0.7)
                plt.title(f"Prediction Anomalies: Confidence vs Error{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Confidence Score')
                plt.ylabel('Error Z-Score (Absolute)')
                plt.grid(True)
                plt.tight_layout()
                
                confidence_file = os.path.join(
                    plots_dir, 
                    f"anomalies_by_confidence{'_' + sport if sport else ''}.png"
                )
                plt.savefig(confidence_file)
                plt.close()
            
            # Plot actual vs predicted for anomalies
            if anomalies:
                plt.figure(figsize=(14, 8))
                actual_values = [a['actual_value'] for a in top_anomalies]
                predicted_values = [a['predicted_value'] for a in top_anomalies]
                
                plt.scatter(actual_values, predicted_values, alpha=0.7)
                
                # Add diagonal line
                min_val = min(min(actual_values), min(predicted_values))
                max_val = max(max(actual_values), max(predicted_values))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.title(f"Prediction Anomalies: Actual vs Predicted{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Actual Value')
                plt.ylabel('Predicted Value')
                plt.grid(True)
                plt.tight_layout()
                
                actual_vs_predicted_file = os.path.join(
                    plots_dir, 
                    f"anomalies_actual_vs_predicted{'_' + sport if sport else ''}.png"
                )
                plt.savefig(actual_vs_predicted_file)
                plt.close()
            
            logger.info(f"Generated prediction anomaly plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating prediction anomaly plots: {e}")


if __name__ == "__main__":
    # Run feature importance analysis
    analyzer = FeatureImportanceAnalyzer()
    importance_results = analyzer.analyze_feature_importance()
    interaction_results = analyzer.analyze_feature_interactions()
    drift_results = analyzer.analyze_feature_drift()
    
    # Run anomaly detection
    detector = AnomalyDetector()
    anomaly_results = detector.detect_anomalies()
    prediction_anomaly_results = detector.detect_prediction_anomalies()
    
    # Log results
    logger.info(f"Feature importance analysis and anomaly detection completed")
