"""
Automated Model Training for Sports Prediction System

This module implements automated model training with Bayesian hyperparameter optimization,
model ensembling, and automated backtesting against historical data.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query
from database.versioning import DataVersioning
from neural_network.model_architecture import HybridSportsPredictionModel
from neural_network.multi_task_learning import MultiTaskSportsPredictionModel
from data_pipeline.feature_computation import FeatureComputation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_training')

class ModelTrainer:
    """
    Class for training and optimizing prediction models.
    """
    
    def __init__(self, model_type: str = 'hybrid', sport: str = 'nba'):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('hybrid', 'lstm', 'transformer', 'ensemble')
            sport: Sport to train for
        """
        self.model_type = model_type
        self.sport = sport
        self.feature_computation = FeatureComputation()
        self.data_versioning = DataVersioning()
        self.model = None
        self.hyperparameters = {}
        self.training_history = {}
        self.version_id = None
    
    def load_training_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data from the database.
        
        Args:
            start_date: Start date for training data (optional)
            end_date: End date for training data (optional)
            
        Returns:
            Tuple of (features, targets, stat_types)
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Loading training data from {start_date} to {end_date} for {self.sport}")
            
            # Get completed games
            query = """
            SELECT g.game_id, g.sport, g.game_date, g.home_team_id, g.away_team_id
            FROM games g
            WHERE g.sport = %s
            AND g.game_date BETWEEN %s AND %s
            AND g.status = 'completed'
            ORDER BY g.game_date
            """
            
            params = (self.sport, start_date, end_date)
            games = execute_query(query, params)
            
            if not games:
                logger.warning(f"No games found for {self.sport} between {start_date} and {end_date}")
                return np.array([]), np.array([]), []
            
            logger.info(f"Found {len(games)} completed games")
            
            # Get player stats for these games
            query = """
            SELECT ar.player_id, ar.game_id, ar.stat_type, ar.actual_value,
                   p.name as player_name, p.position, p.team_id,
                   g.game_date, g.home_team_id, g.away_team_id
            FROM actual_results ar
            JOIN players p ON ar.player_id = p.player_id
            JOIN games g ON ar.game_id = g.game_id
            WHERE g.sport = %s
            AND g.game_date BETWEEN %s AND %s
            AND g.status = 'completed'
            ORDER BY g.game_date, ar.player_id, ar.stat_type
            """
            
            params = (self.sport, start_date, end_date)
            stats = execute_query(query, params)
            
            if not stats:
                logger.warning(f"No player stats found for {self.sport} between {start_date} and {end_date}")
                return np.array([]), np.array([]), []
            
            logger.info(f"Found {len(stats)} player stat records")
            
            # Group by stat type
            stat_types = list(set(stat['stat_type'] for stat in stats))
            logger.info(f"Found stat types: {stat_types}")
            
            # Prepare features and targets
            features_list = []
            targets_list = []
            
            for stat in stats:
                # Compute features for this player and game
                try:
                    computed_features = self.feature_computation.compute_player_features(
                        stat['player_id'], stat['game_id'], self.sport
                    )
                    
                    # Prepare model input
                    feature_array, _ = self.feature_computation.prepare_model_input(
                        computed_features, stat['stat_type']
                    )
                    
                    # Add to lists
                    features_list.append(feature_array)
                    targets_list.append(stat['actual_value'])
                except Exception as e:
                    logger.error(f"Error computing features for player {stat['player_id']}, game {stat['game_id']}: {e}")
                    continue
            
            # Convert to numpy arrays
            features = np.array(features_list)
            targets = np.array(targets_list)
            
            logger.info(f"Prepared {len(features)} training examples with {features.shape[1]} features")
            
            return features, targets, stat_types
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, features: np.ndarray, targets: np.ndarray, stat_types: List[str]) -> Dict[str, Any]:
        """
        Train a model with the provided data.
        
        Args:
            features: Feature matrix
            targets: Target values
            stat_types: List of statistic types
            
        Returns:
            Dictionary of training results
        """
        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No training data available")
                return {'success': False, 'error': 'No training data available'}
            
            logger.info(f"Training {self.model_type} model for {self.sport}")
            
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Train model based on type
            if self.model_type == 'hybrid':
                return self._train_hybrid_model(X_train, y_train, X_val, y_val, stat_types)
            elif self.model_type == 'lstm':
                return self._train_lstm_model(X_train, y_train, X_val, y_val, stat_types)
            elif self.model_type == 'transformer':
                return self._train_transformer_model(X_train, y_train, X_val, y_val, stat_types)
            elif self.model_type == 'ensemble':
                return self._train_ensemble_model(X_train, y_train, X_val, y_val, stat_types)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return {'success': False, 'error': f"Unknown model type: {self.model_type}"}
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_hyperparameters(self, features: np.ndarray, targets: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            features: Feature matrix
            targets: Target values
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No data available for hyperparameter optimization")
                return {}
            
            logger.info(f"Optimizing hyperparameters for {self.model_type} model with {n_trials} trials")
            
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(),
                pruner=MedianPruner()
            )
            
            # Define objective function based on model type
            if self.model_type == 'hybrid':
                objective = lambda trial: self._hybrid_objective(trial, X_train, y_train, X_val, y_val)
            elif self.model_type == 'lstm':
                objective = lambda trial: self._lstm_objective(trial, X_train, y_train, X_val, y_val)
            elif self.model_type == 'transformer':
                objective = lambda trial: self._transformer_objective(trial, X_train, y_train, X_val, y_val)
            elif self.model_type == 'ensemble':
                objective = lambda trial: self._ensemble_objective(trial, X_train, y_train, X_val, y_val)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return {}
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Get best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Best hyperparameters: {best_params}, validation loss: {best_value}")
            
            # Store hyperparameters
            self.hyperparameters = best_params
            
            return best_params
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return {}
    
    def save_model(self, model_dir: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return ""
            
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Generate version ID if not already set
            if self.version_id is None:
                self.version_id = f"{self.model_type}_{self.sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model based on type
            if self.model_type in ['hybrid', 'lstm', 'transformer']:
                model_path = os.path.join(model_dir, f"{self.version_id}.h5")
                self.model.save(model_path)
            elif self.model_type == 'ensemble':
                model_path = os.path.join(model_dir, f"{self.version_id}.joblib")
                joblib.dump(self.model, model_path)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return ""
            
            # Save hyperparameters
            params_path = os.path.join(model_dir, f"{self.version_id}_params.json")
            with open(params_path, 'w') as f:
                json.dump(self.hyperparameters, f)
            
            # Save training history
            history_path = os.path.join(model_dir, f"{self.version_id}_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f)
            
            logger.info(f"Saved model to {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def register_model_version(self, validation_metrics: Dict[str, Any], description: str = "") -> bool:
        """
        Register the model version in the database.
        
        Args:
            validation_metrics: Validation metrics
            description: Model description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.version_id is None:
                logger.error("No version ID set")
                return False
            
            # Prepare training data range
            training_data_range = {
                'start_date': self.training_history.get('start_date'),
                'end_date': self.training_history.get('end_date')
            }
            
            # Register model version
            success = self.data_versioning.register_model_version(
                version_id=self.version_id,
                model_type=self.model_type,
                description=description or f"{self.model_type.capitalize()} model for {self.sport.upper()}",
                hyperparameters=self.hyperparameters,
                training_data_range=training_data_range,
                validation_metrics=validation_metrics,
                created_by="automated_training",
                active=True  # Set as active model
            )
            
            if success:
                logger.info(f"Registered model version: {self.version_id}")
            else:
                logger.error(f"Failed to register model version: {self.version_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error registering model version: {e}")
            return False
    
    def load_model(self, version_id: str, model_dir: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            version_id: Model version ID
            model_dir: Directory containing the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine model type from version ID
            model_type = version_id.split('_')[0]
            
            # Load model based on type
            if model_type in ['hybrid', 'lstm', 'transformer']:
                model_path = os.path.join(model_dir, f"{version_id}.h5")
                self.model = tf.keras.models.load_model(model_path)
            elif model_type == 'ensemble':
                model_path = os.path.join(model_dir, f"{version_id}.joblib")
                self.model = joblib.load(model_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Load hyperparameters
            params_path = os.path.join(model_dir, f"{version_id}_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.hyperparameters = json.load(f)
            
            # Load training history
            history_path = os.path.join(model_dir, f"{version_id}_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            # Set version ID and model type
            self.version_id = version_id
            self.model_type = model_type
            
            logger.info(f"Loaded model {version_id} from {model_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predictions
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return np.array([])
            
            # Make predictions based on model type
            if self.model_type in ['hybrid', 'lstm', 'transformer']:
                predictions = self.model.predict(features)
                
                # Extract predictions from model output
                if isinstance(predictions, list):
                    # Multi-output model
                    predictions = predictions[0]
                
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multiple outputs, take the first one (regression value)
                    predictions = predictions[:, 0]
            elif self.model_type == 'ensemble':
                predictions = self.model.predict(features)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return np.array([])
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def predict_with_uncertainty(self, features: np.ndarray, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            features: Feature matrix
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return np.array([]), np.array([])
            
            # Make predictions based on model type
            if self.model_type in ['hybrid', 'lstm', 'transformer']:
                # Enable dropout at inference time
                predictions = []
                
                # Perform multiple forward passes
                for _ in range(num_samples):
                    pred = self.model.predict(features, training=True)
                    
                    # Extract predictions from model output
                    if isinstance(pred, list):
                        # Multi-output model
                        pred = pred[0]
                    
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        # Multiple outputs, take the first one (regression value)
                        pred = pred[:, 0]
                    
                    predictions.append(pred)
                
                # Stack predictions
                stacked_preds = np.stack(predictions, axis=0)
                
                # Calculate mean and standard deviation
                mean_preds = np.mean(stacked_preds, axis=0)
                std_preds = np.std(stacked_preds, axis=0)
                
                return mean_preds, std_preds
            elif self.model_type == 'ensemble':
                # For ensemble models, use the built-in uncertainty estimation
                if hasattr(self.model, 'predict_with_uncertainty'):
                    return self.model.predict_with_uncertainty(features)
                else:
                    # Use standard deviation across ensemble members
                    predictions = []
                    
                    for estimator in self.model.estimators_:
                        pred = estimator.predict(features)
                        predictions.append(pred)
                    
                    # Stack predictions
                    stacked_preds = np.stack(predictions, axis=0)
                    
                    # Calculate mean and standard deviation
                    mean_preds = np.mean(stacked_preds, axis=0)
                    std_preds = np.std(stacked_preds, axis=0)
                    
                    return mean_preds, std_preds
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Error making predictions with uncertainty: {e}")
            return np.array([]), np.array([])
    
    def backtest(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """
        Backtest the model against historical data.
        
        Args:
            features: Feature matrix
            targets: Target values
            
        Returns:
            Dictionary of backtest results
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return {'success': False, 'error': 'No model loaded'}
            
            # Make predictions
            predictions = self.predict(features)
            
            if len(predictions) == 0:
                logger.error("No predictions generated")
                return {'success': False, 'error': 'No predictions generated'}
            
            # Calculate metrics
            mae = mean_absolute_error(targets, predictions)
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate over/under accuracy
            line_values = targets.mean()  # Use mean as a simple line value
            over_under_actual = (targets > line_values).astype(int)
            over_under_pred = (predictions > line_values).astype(int)
            accuracy = accuracy_score(over_under_actual, over_under_pred)
            
            # Calculate percentage of predictions within 10% of actual
            pct_diff = np.abs(predictions - targets) / np.maximum(targets, 1e-10)
            within_10_pct = np.mean(pct_diff <= 0.1)
            
            # Calculate percentage of predictions within 20% of actual
            within_20_pct = np.mean(pct_diff <= 0.2)
            
            results = {
                'success': True,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'over_under_accuracy': accuracy,
                'within_10_pct': within_10_pct,
                'within_20_pct': within_20_pct,
                'num_samples': len(targets)
            }
            
            logger.info(f"Backtest results: {results}")
            
            return results
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_hybrid_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, stat_types: List[str]) -> Dict[str, Any]:
        """
        Train a hybrid neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            stat_types: List of statistic types
            
        Returns:
            Dictionary of training results
        """
        try:
            # Get input shape
            input_shape = X_train.shape[1:]
            
            # Create model
            model = HybridSportsPredictionModel(
                input_shape=input_shape,
                lstm_units=self.hyperparameters.get('lstm_units', 64),
                transformer_dim=self.hyperparameters.get('transformer_dim', 64),
                transformer_heads=self.hyperparameters.get('transformer_heads', 4),
                dense_units=self.hyperparameters.get('dense_units', [128, 64]),
                dropout_rate=self.hyperparameters.get('dropout_rate', 0.3),
                learning_rate=self.hyperparameters.get('learning_rate', 0.001),
                use_lstm=self.hyperparameters.get('use_lstm', True),
                use_transformer=self.hyperparameters.get('use_transformer', True),
                use_attention=self.hyperparameters.get('use_attention', True)
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.hyperparameters.get('epochs', 50),
                batch_size=self.hyperparameters.get('batch_size', 32),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Store model and training history
            self.model = model
            self.training_history = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']],
                'epochs': len(history.history['loss']),
                'start_date': self.training_history.get('start_date'),
                'end_date': self.training_history.get('end_date')
            }
            
            # Evaluate model
            val_loss, val_mae = model.evaluate(X_val, y_val)
            
            # Generate version ID
            self.version_id = f"hybrid_{self.sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'success': True,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'version_id': self.version_id
            }
        except Exception as e:
            logger.error(f"Error training hybrid model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, stat_types: List[str]) -> Dict[str, Any]:
        """
        Train an LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            stat_types: List of statistic types
            
        Returns:
            Dictionary of training results
        """
        try:
            # Get input shape
            input_shape = X_train.shape[1:]
            
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((1, input_shape[0])),  # Reshape for LSTM
                tf.keras.layers.LSTM(
                    units=self.hyperparameters.get('lstm_units', 64),
                    return_sequences=True
                ),
                tf.keras.layers.LSTM(
                    units=self.hyperparameters.get('lstm_units_2', 32)
                ),
                tf.keras.layers.Dense(
                    units=self.hyperparameters.get('dense_units', 32),
                    activation='relu'
                ),
                tf.keras.layers.Dropout(self.hyperparameters.get('dropout_rate', 0.3)),
                tf.keras.layers.Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.hyperparameters.get('epochs', 50),
                batch_size=self.hyperparameters.get('batch_size', 32),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Store model and training history
            self.model = model
            self.training_history = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']],
                'epochs': len(history.history['loss']),
                'start_date': self.training_history.get('start_date'),
                'end_date': self.training_history.get('end_date')
            }
            
            # Evaluate model
            val_loss, val_mae = model.evaluate(X_val, y_val)
            
            # Generate version ID
            self.version_id = f"lstm_{self.sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'success': True,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'version_id': self.version_id
            }
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, stat_types: List[str]) -> Dict[str, Any]:
        """
        Train a Transformer model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            stat_types: List of statistic types
            
        Returns:
            Dictionary of training results
        """
        try:
            # Get input shape
            input_shape = X_train.shape[1:]
            
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((1, input_shape[0])),  # Reshape for Transformer
                tf.keras.layers.MultiHeadAttention(
                    num_heads=self.hyperparameters.get('num_heads', 4),
                    key_dim=self.hyperparameters.get('key_dim', 64)
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(
                    units=self.hyperparameters.get('dense_units', 64),
                    activation='relu'
                ),
                tf.keras.layers.Dropout(self.hyperparameters.get('dropout_rate', 0.3)),
                tf.keras.layers.Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.hyperparameters.get('epochs', 50),
                batch_size=self.hyperparameters.get('batch_size', 32),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Store model and training history
            self.model = model
            self.training_history = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']],
                'epochs': len(history.history['loss']),
                'start_date': self.training_history.get('start_date'),
                'end_date': self.training_history.get('end_date')
            }
            
            # Evaluate model
            val_loss, val_mae = model.evaluate(X_val, y_val)
            
            # Generate version ID
            self.version_id = f"transformer_{self.sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'success': True,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'version_id': self.version_id
            }
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, stat_types: List[str]) -> Dict[str, Any]:
        """
        Train an ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            stat_types: List of statistic types
            
        Returns:
            Dictionary of training results
        """
        try:
            # Create base models
            models = []
            
            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=self.hyperparameters.get('rf_n_estimators', 100),
                max_depth=self.hyperparameters.get('rf_max_depth', 10),
                min_samples_split=self.hyperparameters.get('rf_min_samples_split', 2),
                random_state=42
            )
            models.append(('rf', rf))
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=self.hyperparameters.get('gb_n_estimators', 100),
                max_depth=self.hyperparameters.get('gb_max_depth', 5),
                learning_rate=self.hyperparameters.get('gb_learning_rate', 0.1),
                random_state=42
            )
            models.append(('gb', gb))
            
            # Neural Network
            nn_model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units=self.hyperparameters.get('nn_dense_units', 64),
                    activation='relu',
                    input_shape=(X_train.shape[1],)
                ),
                tf.keras.layers.Dropout(self.hyperparameters.get('nn_dropout_rate', 0.3)),
                tf.keras.layers.Dense(
                    units=self.hyperparameters.get('nn_dense_units_2', 32),
                    activation='relu'
                ),
                tf.keras.layers.Dense(1)
            ])
            
            nn_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.get('nn_learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            # Train each model
            trained_models = []
            
            for name, model in models:
                logger.info(f"Training {name} model")
                model.fit(X_train, y_train)
                trained_models.append((name, model))
            
            # Train neural network
            logger.info("Training neural network model")
            nn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.hyperparameters.get('nn_epochs', 50),
                batch_size=self.hyperparameters.get('nn_batch_size', 32),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Create ensemble model
            class EnsembleModel:
                def __init__(self, models, nn_model, weights=None):
                    self.models = models
                    self.nn_model = nn_model
                    self.weights = weights or [1.0] * (len(models) + 1)
                    self.weights = [w / sum(self.weights) for w in self.weights]  # Normalize weights
                
                def predict(self, X):
                    predictions = []
                    
                    # Get predictions from traditional models
                    for _, model in self.models:
                        pred = model.predict(X)
                        predictions.append(pred)
                    
                    # Get predictions from neural network
                    nn_pred = self.nn_model.predict(X, verbose=0)
                    predictions.append(nn_pred.flatten())
                    
                    # Weighted average
                    weighted_pred = np.zeros_like(predictions[0])
                    for i, pred in enumerate(predictions):
                        weighted_pred += self.weights[i] * pred
                    
                    return weighted_pred
                
                def predict_with_uncertainty(self, X):
                    predictions = []
                    
                    # Get predictions from traditional models
                    for _, model in self.models:
                        if hasattr(model, 'estimators_'):
                            # For ensemble models, get predictions from each estimator
                            estimator_preds = []
                            for estimator in model.estimators_:
                                pred = estimator.predict(X)
                                estimator_preds.append(pred)
                            
                            # Stack predictions
                            stacked_preds = np.stack(estimator_preds, axis=0)
                            
                            # Calculate mean and standard deviation
                            mean_pred = np.mean(stacked_preds, axis=0)
                            std_pred = np.std(stacked_preds, axis=0)
                            
                            predictions.append((mean_pred, std_pred))
                        else:
                            # For non-ensemble models, just use the prediction
                            pred = model.predict(X)
                            predictions.append((pred, np.zeros_like(pred)))
                    
                    # Get predictions from neural network with dropout
                    nn_preds = []
                    for _ in range(10):  # Monte Carlo dropout
                        nn_pred = self.nn_model.predict(X, verbose=0)
                        nn_preds.append(nn_pred.flatten())
                    
                    # Stack predictions
                    stacked_nn_preds = np.stack(nn_preds, axis=0)
                    
                    # Calculate mean and standard deviation
                    mean_nn_pred = np.mean(stacked_nn_preds, axis=0)
                    std_nn_pred = np.std(stacked_nn_preds, axis=0)
                    
                    predictions.append((mean_nn_pred, std_nn_pred))
                    
                    # Weighted average for mean
                    weighted_mean = np.zeros_like(predictions[0][0])
                    for i, (mean_pred, _) in enumerate(predictions):
                        weighted_mean += self.weights[i] * mean_pred
                    
                    # Weighted average for variance (sum of weighted variances)
                    weighted_var = np.zeros_like(predictions[0][0])
                    for i, (_, std_pred) in enumerate(predictions):
                        weighted_var += (self.weights[i] * std_pred) ** 2
                    
                    # Convert variance to standard deviation
                    weighted_std = np.sqrt(weighted_var)
                    
                    return weighted_mean, weighted_std
            
            # Optimize ensemble weights
            def optimize_weights(models, nn_model, X, y):
                def objective(weights):
                    ensemble = EnsembleModel(models, nn_model, weights)
                    pred = ensemble.predict(X)
                    return mean_squared_error(y, pred)
                
                # Simple grid search for weights
                best_weights = [1.0] * (len(models) + 1)
                best_score = objective(best_weights)
                
                for i in range(len(models) + 1):
                    for w in [0.5, 1.0, 1.5, 2.0]:
                        weights = [1.0] * (len(models) + 1)
                        weights[i] = w
                        score = objective(weights)
                        
                        if score < best_score:
                            best_score = score
                            best_weights = weights
                
                return best_weights
            
            # Optimize weights
            logger.info("Optimizing ensemble weights")
            weights = optimize_weights(trained_models, nn_model, X_val, y_val)
            
            # Create final ensemble
            ensemble = EnsembleModel(trained_models, nn_model, weights)
            
            # Evaluate ensemble
            val_pred = ensemble.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            # Store model and training history
            self.model = ensemble
            self.training_history = {
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'weights': weights,
                'start_date': self.training_history.get('start_date'),
                'end_date': self.training_history.get('end_date')
            }
            
            # Generate version ID
            self.version_id = f"ensemble_{self.sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'success': True,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'version_id': self.version_id
            }
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _hybrid_objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization of hybrid model.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        # Define hyperparameters to optimize
        params = {
            'lstm_units': trial.suggest_int('lstm_units', 32, 128, 32),
            'transformer_dim': trial.suggest_int('transformer_dim', 32, 128, 32),
            'transformer_heads': trial.suggest_int('transformer_heads', 2, 8, 2),
            'dense_units': [
                trial.suggest_int('dense_units_1', 64, 256, 64),
                trial.suggest_int('dense_units_2', 32, 128, 32)
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'use_lstm': trial.suggest_categorical('use_lstm', [True, False]),
            'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50  # Fixed number of epochs with early stopping
        }
        
        # Ensure at least one of LSTM or Transformer is used
        if not params['use_lstm'] and not params['use_transformer']:
            params['use_lstm'] = True
        
        # Create and train model
        input_shape = X_train.shape[1:]
        
        model = HybridSportsPredictionModel(
            input_shape=input_shape,
            lstm_units=params['lstm_units'],
            transformer_dim=params['transformer_dim'],
            transformer_heads=params['transformer_heads'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            use_lstm=params['use_lstm'],
            use_transformer=params['use_transformer'],
            use_attention=params['use_attention']
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Return best validation loss
        return min(history.history['val_loss'])
    
    def _lstm_objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization of LSTM model.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        # Define hyperparameters to optimize
        params = {
            'lstm_units': trial.suggest_int('lstm_units', 32, 128, 32),
            'lstm_units_2': trial.suggest_int('lstm_units_2', 16, 64, 16),
            'dense_units': trial.suggest_int('dense_units', 16, 64, 16),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50  # Fixed number of epochs with early stopping
        }
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((1, X_train.shape[1])),  # Reshape for LSTM
            tf.keras.layers.LSTM(
                units=params['lstm_units'],
                return_sequences=True
            ),
            tf.keras.layers.LSTM(
                units=params['lstm_units_2']
            ),
            tf.keras.layers.Dense(
                units=params['dense_units'],
                activation='relu'
            ),
            tf.keras.layers.Dropout(params['dropout_rate']),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Return best validation loss
        return min(history.history['val_loss'])
    
    def _transformer_objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization of Transformer model.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        # Define hyperparameters to optimize
        params = {
            'num_heads': trial.suggest_int('num_heads', 2, 8, 2),
            'key_dim': trial.suggest_int('key_dim', 32, 128, 32),
            'dense_units': trial.suggest_int('dense_units', 32, 128, 32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50  # Fixed number of epochs with early stopping
        }
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((1, X_train.shape[1])),  # Reshape for Transformer
            tf.keras.layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_dim=params['key_dim']
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(
                units=params['dense_units'],
                activation='relu'
            ),
            tf.keras.layers.Dropout(params['dropout_rate']),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Return best validation loss
        return min(history.history['val_loss'])
    
    def _ensemble_objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization of ensemble model.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        # Define hyperparameters to optimize
        params = {
            # Random Forest parameters
            'rf_n_estimators': trial.suggest_int('rf_n_estimators', 50, 200, 50),
            'rf_max_depth': trial.suggest_int('rf_max_depth', 5, 20, 5),
            'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10, 2),
            
            # Gradient Boosting parameters
            'gb_n_estimators': trial.suggest_int('gb_n_estimators', 50, 200, 50),
            'gb_max_depth': trial.suggest_int('gb_max_depth', 3, 10, 1),
            'gb_learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.2, log=True),
            
            # Neural Network parameters
            'nn_dense_units': trial.suggest_int('nn_dense_units', 32, 128, 32),
            'nn_dense_units_2': trial.suggest_int('nn_dense_units_2', 16, 64, 16),
            'nn_dropout_rate': trial.suggest_float('nn_dropout_rate', 0.1, 0.5, step=0.1),
            'nn_learning_rate': trial.suggest_float('nn_learning_rate', 1e-4, 1e-2, log=True),
            'nn_batch_size': trial.suggest_categorical('nn_batch_size', [16, 32, 64]),
            'nn_epochs': 50  # Fixed number of epochs with early stopping
        }
        
        # Create base models
        models = []
        
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            random_state=42
        )
        models.append(('rf', rf))
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=params['gb_n_estimators'],
            max_depth=params['gb_max_depth'],
            learning_rate=params['gb_learning_rate'],
            random_state=42
        )
        models.append(('gb', gb))
        
        # Neural Network
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=params['nn_dense_units'],
                activation='relu',
                input_shape=(X_train.shape[1],)
            ),
            tf.keras.layers.Dropout(params['nn_dropout_rate']),
            tf.keras.layers.Dense(
                units=params['nn_dense_units_2'],
                activation='relu'
            ),
            tf.keras.layers.Dense(1)
        ])
        
        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['nn_learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train each model
        trained_models = []
        
        for name, model in models:
            model.fit(X_train, y_train)
            trained_models.append((name, model))
        
        # Train neural network
        nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['nn_epochs'],
            batch_size=params['nn_batch_size'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Simple ensemble with equal weights
        predictions = []
        
        # Get predictions from traditional models
        for _, model in trained_models:
            pred = model.predict(X_val)
            predictions.append(pred)
        
        # Get predictions from neural network
        nn_pred = nn_model.predict(X_val, verbose=0)
        predictions.append(nn_pred.flatten())
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calculate validation loss
        val_loss = mean_squared_error(y_val, ensemble_pred)
        
        return val_loss


class AutomatedTraining:
    """
    Class for automated model training and evaluation.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the automated training system.
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = model_dir
        self.data_versioning = DataVersioning()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train_all_sports(self, model_types: List[str] = ['hybrid'], n_trials: int = 50) -> Dict[str, Any]:
        """
        Train models for all supported sports.
        
        Args:
            model_types: List of model types to train
            n_trials: Number of hyperparameter optimization trials
            
        Returns:
            Dictionary of training results
        """
        sports = ['nba', 'nfl', 'mlb', 'nhl', 'soccer']
        results = {}
        
        for sport in sports:
            sport_results = {}
            
            for model_type in model_types:
                logger.info(f"Training {model_type} model for {sport}")
                
                try:
                    # Train model
                    result = self.train_sport_model(sport, model_type, n_trials)
                    sport_results[model_type] = result
                except Exception as e:
                    logger.error(f"Error training {model_type} model for {sport}: {e}")
                    sport_results[model_type] = {'success': False, 'error': str(e)}
            
            results[sport] = sport_results
        
        return results
    
    def train_sport_model(self, sport: str, model_type: str = 'hybrid', n_trials: int = 50) -> Dict[str, Any]:
        """
        Train a model for a specific sport.
        
        Args:
            sport: Sport to train for
            model_type: Type of model to train
            n_trials: Number of hyperparameter optimization trials
            
        Returns:
            Dictionary of training results
        """
        try:
            # Create trainer
            trainer = ModelTrainer(model_type, sport)
            
            # Load training data
            features, targets, stat_types = trainer.load_training_data()
            
            if len(features) == 0 or len(targets) == 0:
                logger.error(f"No training data available for {sport}")
                return {'success': False, 'error': 'No training data available'}
            
            # Optimize hyperparameters
            logger.info(f"Optimizing hyperparameters for {model_type} model for {sport}")
            hyperparameters = trainer.optimize_hyperparameters(features, targets, n_trials)
            
            # Train model with best hyperparameters
            logger.info(f"Training {model_type} model for {sport} with best hyperparameters")
            training_result = trainer.train_model(features, targets, stat_types)
            
            if not training_result.get('success', False):
                logger.error(f"Error training model: {training_result.get('error')}")
                return training_result
            
            # Save model
            model_path = trainer.save_model(self.model_dir)
            
            if not model_path:
                logger.error("Error saving model")
                return {'success': False, 'error': 'Error saving model'}
            
            # Backtest model
            logger.info(f"Backtesting {model_type} model for {sport}")
            backtest_result = trainer.backtest(features, targets)
            
            if not backtest_result.get('success', False):
                logger.error(f"Error backtesting model: {backtest_result.get('error')}")
                return {'success': False, 'error': backtest_result.get('error')}
            
            # Register model version
            logger.info(f"Registering {model_type} model version for {sport}")
            validation_metrics = {
                'mae': training_result.get('val_mae'),
                'mse': training_result.get('val_loss'),
                'rmse': np.sqrt(training_result.get('val_loss', 0)),
                'over_under_accuracy': backtest_result.get('over_under_accuracy'),
                'within_10_pct': backtest_result.get('within_10_pct'),
                'within_20_pct': backtest_result.get('within_20_pct')
            }
            
            description = f"{model_type.capitalize()} model for {sport.upper()} trained on {backtest_result.get('num_samples', 0)} samples"
            
            success = trainer.register_model_version(validation_metrics, description)
            
            if not success:
                logger.error("Error registering model version")
                return {'success': False, 'error': 'Error registering model version'}
            
            # Return results
            return {
                'success': True,
                'version_id': trainer.version_id,
                'model_path': model_path,
                'hyperparameters': hyperparameters,
                'validation_metrics': validation_metrics,
                'backtest_results': backtest_result
            }
        except Exception as e:
            logger.error(f"Error training sport model: {e}")
            return {'success': False, 'error': str(e)}
    
    def setup_daily_training(self) -> Dict[str, Any]:
        """
        Set up daily model training.
        
        Returns:
            Dictionary of setup results
        """
        try:
            # Create cron job for daily training
            cron_command = f"0 0 * * * cd {os.getcwd()} && python -m data_pipeline.automated_training > logs/daily_training.log 2>&1"
            
            # Write to crontab
            with open('daily_training.cron', 'w') as f:
                f.write(cron_command + '\n')
            
            # Install cron job
            os.system('crontab daily_training.cron')
            
            logger.info("Set up daily training cron job")
            
            return {
                'success': True,
                'cron_command': cron_command
            }
        except Exception as e:
            logger.error(f"Error setting up daily training: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_models(self, version_id_1: str, version_id_2: str, sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_id_1: First model version ID
            version_id_2: Second model version ID
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of comparison results
        """
        try:
            # Get comparison from data versioning
            comparison = self.data_versioning.compare_model_versions(version_id_1, version_id_2, sport)
            
            if not comparison:
                logger.error(f"Error comparing models {version_id_1} and {version_id_2}")
                return {'success': False, 'error': 'Error comparing models'}
            
            logger.info(f"Compared models {version_id_1} and {version_id_2}")
            
            return {
                'success': True,
                'comparison': comparison
            }
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {'success': False, 'error': str(e)}
    
    def setup_ab_test(self, name: str, description: str, model_a: str, model_b: str) -> Dict[str, Any]:
        """
        Set up an A/B test between two model versions.
        
        Args:
            name: Name of the A/B test
            description: Description of the A/B test
            model_a: First model version ID
            model_b: Second model version ID
            
        Returns:
            Dictionary of setup results
        """
        try:
            # Set up A/B test
            test_id = self.data_versioning.setup_ab_test(name, description, model_a, model_b)
            
            if not test_id:
                logger.error(f"Error setting up A/B test between {model_a} and {model_b}")
                return {'success': False, 'error': 'Error setting up A/B test'}
            
            logger.info(f"Set up A/B test {name} between {model_a} and {model_b}")
            
            return {
                'success': True,
                'test_id': test_id
            }
        except Exception as e:
            logger.error(f"Error setting up A/B test: {e}")
            return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run automated training
    automated_training = AutomatedTraining()
    results = automated_training.train_all_sports()
    
    # Log results
    logger.info(f"Automated training results: {results}")
