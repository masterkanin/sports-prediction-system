"""
Continuous Improvement System for Sports Prediction Model

This module implements continuous improvement mechanisms for the sports prediction system,
including reinforcement learning for prediction strategy optimization.
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
import tensorflow as tf
from tensorflow import keras
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query
from database.versioning import DataVersioning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_improvement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('continuous_improvement')

class ContinuousImprovement:
    """
    Class for implementing continuous improvement mechanisms for the sports prediction system.
    """
    
    def __init__(self):
        """
        Initialize the continuous improvement system.
        """
        self.data_versioning = DataVersioning()
        self.reports_dir = os.path.join('reports', 'continuous_improvement')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def optimize_prediction_strategy(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                   sport: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize prediction strategy using reinforcement learning.
        
        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            Dictionary of optimization results
        """
        try:
            # Default date range if not specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Optimizing prediction strategy from {start_date} to {end_date}")
            
            # Get prediction data
            prediction_data = self._get_prediction_data(start_date, end_date, sport, model_version)
            
            if not prediction_data:
                logger.warning(f"No prediction data found for the specified criteria")
                return {'success': False, 'error': 'No prediction data found'}
            
            # Train reinforcement learning model
            rl_model, optimization_results = self._train_rl_model(prediction_data)
            
            if not rl_model:
                logger.warning(f"Failed to train reinforcement learning model")
                return {'success': False, 'error': 'Failed to train reinforcement learning model'}
            
            # Save model
            model_file = os.path.join(
                self.reports_dir, 
                f"rl_model_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.pkl"
            )
            
            joblib.dump(rl_model, model_file)
            
            # Generate plots
            self._generate_optimization_plots(optimization_results, sport, model_version)
            
            # Save optimization results
            results_file = os.path.join(
                self.reports_dir, 
                f"optimization_results_{start_date}_to_{end_date}{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump({
                    'optimization_results': optimization_results
                }, f, indent=2, default=str)
            
            logger.info(f"Saved optimization results to {results_file}")
            
            return {
                'success': True,
                'model_file': model_file,
                'results_file': results_file,
                'optimization_results': optimization_results
            }
        except Exception as e:
            logger.error(f"Error optimizing prediction strategy: {e}")
            return {'success': False, 'error': str(e)}
    
    def implement_model_improvements(self, model_version: str, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implement improvements to the prediction model.
        
        Args:
            model_version: Model version to improve
            improvements: List of improvements to implement
            
        Returns:
            Dictionary of implementation results
        """
        try:
            logger.info(f"Implementing improvements for model version {model_version}")
            
            # Get model information
            model_info = self.data_versioning.get_model_version(model_version)
            if not model_info:
                logger.warning(f"Model version {model_version} not found")
                return {'success': False, 'error': f"Model version {model_version} not found"}
            
            # Get model path
            model_path = model_info.get('model_path')
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model file not found for version {model_version}")
                return {'success': False, 'error': f"Model file not found for version {model_version}"}
            
            # Determine model type
            model_type = model_info.get('model_type', '')
            
            # Implement improvements
            implementation_results = []
            
            for improvement in improvements:
                improvement_type = improvement.get('type')
                
                if improvement_type == 'hyperparameter':
                    result = self._implement_hyperparameter_improvement(model_version, model_path, model_type, improvement)
                elif improvement_type == 'feature':
                    result = self._implement_feature_improvement(model_version, model_path, model_type, improvement)
                elif improvement_type == 'architecture':
                    result = self._implement_architecture_improvement(model_version, model_path, model_type, improvement)
                else:
                    logger.warning(f"Unknown improvement type: {improvement_type}")
                    result = {'success': False, 'error': f"Unknown improvement type: {improvement_type}"}
                
                implementation_results.append({
                    'improvement': improvement,
                    'result': result
                })
            
            # Create new model version
            new_model_version = self.data_versioning.create_model_version(
                base_version=model_version,
                model_path=model_path,
                model_type=model_type,
                improvements=improvements
            )
            
            if not new_model_version:
                logger.warning(f"Failed to create new model version")
                return {'success': False, 'error': 'Failed to create new model version'}
            
            logger.info(f"Created new model version: {new_model_version}")
            
            # Save implementation results
            results_file = os.path.join(
                self.reports_dir, 
                f"implementation_results_{model_version}_to_{new_model_version}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump({
                    'base_version': model_version,
                    'new_version': new_model_version,
                    'implementation_results': implementation_results
                }, f, indent=2, default=str)
            
            logger.info(f"Saved implementation results to {results_file}")
            
            return {
                'success': True,
                'base_version': model_version,
                'new_version': new_model_version,
                'results_file': results_file,
                'implementation_results': implementation_results
            }
        except Exception as e:
            logger.error(f"Error implementing model improvements: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_ab_testing(self, model_a: str, model_b: str, duration_days: int = 7, 
                     sport: Optional[str] = None) -> Dict[str, Any]:
        """
        Run A/B testing between two model versions.
        
        Args:
            model_a: First model version
            model_b: Second model version
            duration_days: Duration of A/B testing in days
            sport: Sport to filter by (optional)
            
        Returns:
            Dictionary of A/B testing results
        """
        try:
            logger.info(f"Running A/B testing between model versions {model_a} and {model_b}")
            
            # Get model information
            model_a_info = self.data_versioning.get_model_version(model_a)
            if not model_a_info:
                logger.warning(f"Model version {model_a} not found")
                return {'success': False, 'error': f"Model version {model_a} not found"}
            
            model_b_info = self.data_versioning.get_model_version(model_b)
            if not model_b_info:
                logger.warning(f"Model version {model_b} not found")
                return {'success': False, 'error': f"Model version {model_b} not found"}
            
            # Set up A/B testing
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=duration_days)).strftime('%Y-%m-%d')
            
            ab_testing_id = self.data_versioning.create_ab_testing(
                model_a=model_a,
                model_b=model_b,
                start_date=start_date,
                end_date=end_date,
                sport=sport
            )
            
            if not ab_testing_id:
                logger.warning(f"Failed to create A/B testing")
                return {'success': False, 'error': 'Failed to create A/B testing'}
            
            logger.info(f"Created A/B testing with ID: {ab_testing_id}")
            
            # Save A/B testing setup
            setup_file = os.path.join(
                self.reports_dir, 
                f"ab_testing_setup_{ab_testing_id}.json"
            )
            
            with open(setup_file, 'w') as f:
                json.dump({
                    'ab_testing_id': ab_testing_id,
                    'model_a': model_a,
                    'model_b': model_b,
                    'start_date': start_date,
                    'end_date': end_date,
                    'sport': sport
                }, f, indent=2)
            
            logger.info(f"Saved A/B testing setup to {setup_file}")
            
            return {
                'success': True,
                'ab_testing_id': ab_testing_id,
                'model_a': model_a,
                'model_b': model_b,
                'start_date': start_date,
                'end_date': end_date,
                'sport': sport,
                'setup_file': setup_file
            }
        except Exception as e:
            logger.error(f"Error setting up A/B testing: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_ab_testing_results(self, ab_testing_id: str) -> Dict[str, Any]:
        """
        Analyze results of A/B testing.
        
        Args:
            ab_testing_id: A/B testing ID
            
        Returns:
            Dictionary of A/B testing analysis results
        """
        try:
            logger.info(f"Analyzing A/B testing results for ID: {ab_testing_id}")
            
            # Get A/B testing information
            ab_testing_info = self.data_versioning.get_ab_testing(ab_testing_id)
            if not ab_testing_info:
                logger.warning(f"A/B testing with ID {ab_testing_id} not found")
                return {'success': False, 'error': f"A/B testing with ID {ab_testing_id} not found"}
            
            # Get prediction results
            model_a = ab_testing_info.get('model_a')
            model_b = ab_testing_info.get('model_b')
            start_date = ab_testing_info.get('start_date')
            end_date = ab_testing_info.get('end_date')
            sport = ab_testing_info.get('sport')
            
            # Get prediction data for model A
            prediction_data_a = self._get_prediction_data(start_date, end_date, sport, model_a)
            
            # Get prediction data for model B
            prediction_data_b = self._get_prediction_data(start_date, end_date, sport, model_b)
            
            if not prediction_data_a or not prediction_data_b:
                logger.warning(f"No prediction data found for one or both models")
                return {'success': False, 'error': 'No prediction data found for one or both models'}
            
            # Calculate metrics
            metrics_a = self._calculate_metrics(prediction_data_a)
            metrics_b = self._calculate_metrics(prediction_data_b)
            
            # Compare metrics
            comparison = self._compare_metrics(metrics_a, metrics_b)
            
            # Determine winner
            winner = self._determine_winner(comparison)
            
            # Generate plots
            self._generate_ab_testing_plots(metrics_a, metrics_b, comparison, ab_testing_id)
            
            # Save analysis results
            results_file = os.path.join(
                self.reports_dir, 
                f"ab_testing_results_{ab_testing_id}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump({
                    'ab_testing_id': ab_testing_id,
                    'model_a': model_a,
                    'model_b': model_b,
                    'metrics_a': metrics_a,
                    'metrics_b': metrics_b,
                    'comparison': comparison,
                    'winner': winner
                }, f, indent=2, default=str)
            
            logger.info(f"Saved A/B testing results to {results_file}")
            
            # Update A/B testing with results
            self.data_versioning.update_ab_testing(
                ab_testing_id=ab_testing_id,
                results={
                    'metrics_a': metrics_a,
                    'metrics_b': metrics_b,
                    'comparison': comparison,
                    'winner': winner
                }
            )
            
            return {
                'success': True,
                'ab_testing_id': ab_testing_id,
                'model_a': model_a,
                'model_b': model_b,
                'metrics_a': metrics_a,
                'metrics_b': metrics_b,
                'comparison': comparison,
                'winner': winner,
                'results_file': results_file
            }
        except Exception as e:
            logger.error(f"Error analyzing A/B testing results: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_prediction_data(self, start_date: str, end_date: str, sport: Optional[str] = None,
                           model_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get prediction data for analysis.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
            
        Returns:
            List of prediction data
        """
        try:
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
                return []
            
            logger.info(f"Found {len(results)} prediction results")
            
            # Process results
            prediction_data = []
            
            for result in results:
                # Calculate error metrics
                actual_value = float(result['actual_value'])
                predicted_value = float(result['predicted_value'])
                line_value = float(result['line_value'])
                over_probability = float(result['over_probability'])
                
                error = predicted_value - actual_value
                absolute_error = abs(error)
                squared_error = error ** 2
                percentage_error = absolute_error / max(actual_value, 1e-10) * 100
                
                # Calculate over/under correctness
                actual_over = actual_value > line_value
                predicted_over = over_probability > 0.5
                over_under_correct = actual_over == predicted_over
                
                # Add to prediction data
                prediction_data.append({
                    'prediction_id': result['prediction_id'],
                    'player_id': result['player_id'],
                    'player_name': result['player_name'],
                    'position': result['position'],
                    'team_id': result['team_id'],
                    'team_name': result['team_name'] if 'team_name' in result else '',
                    'game_id': result['game_id'],
                    'game_date': result['game_date'],
                    'sport': result['sport'],
                    'stat_type': result['stat_type'],
                    'predicted_value': predicted_value,
                    'actual_value': actual_value,
                    'confidence_score': float(result['confidence_score']),
                    'over_probability': over_probability,
                    'line_value': line_value,
                    'error': error,
                    'absolute_error': absolute_error,
                    'squared_error': squared_error,
                    'percentage_error': percentage_error,
                    'over_under_correct': over_under_correct,
                    'model_version': result['model_version']
                })
            
            return prediction_data
        except Exception as e:
            logger.error(f"Error getting prediction data: {e}")
            return []
    
    def _train_rl_model(self, prediction_data: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a reinforcement learning model for prediction strategy optimization.
        
        Args:
            prediction_data: List of prediction data
            
        Returns:
            Tuple of (RL model, optimization results)
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(prediction_data)
            
            # Define state features
            state_features = [
                'confidence_score',
                'over_probability',
                'line_value'
            ]
            
            # Define actions
            # 0: Skip prediction (don't bet)
            # 1: Bet over
            # 2: Bet under
            
            # Define reward function
            def calculate_reward(row):
                # Skip prediction
                if row['action'] == 0:
                    return 0
                
                # Bet over
                if row['action'] == 1:
                    return 1 if row['actual_value'] > row['line_value'] else -1
                
                # Bet under
                if row['action'] == 2:
                    return 1 if row['actual_value'] < row['line_value'] else -1
                
                return 0
            
            # Create Q-learning model
            class QLearningModel:
                def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.learning_rate = learning_rate
                    self.discount_factor = discount_factor
                    
                    # Create Q-network
                    self.model = self._build_model()
                    
                def _build_model(self):
                    model = keras.Sequential([
                        keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
                        keras.layers.Dense(24, activation='relu'),
                        keras.layers.Dense(self.action_size, activation='linear')
                    ])
                    
                    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
                    
                    return model
                
                def train(self, states, actions, rewards, next_states, dones):
                    # Get current Q-values
                    q_values = self.model.predict(states)
                    
                    # Get next Q-values
                    next_q_values = self.model.predict(next_states)
                    
                    # Update Q-values
                    for i in range(len(states)):
                        if dones[i]:
                            q_values[i, actions[i]] = rewards[i]
                        else:
                            q_values[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
                    
                    # Train model
                    self.model.fit(states, q_values, epochs=1, verbose=0)
                
                def predict(self, state):
                    return self.model.predict(state)[0]
            
            # Prepare data for training
            states = df[state_features].values
            
            # Normalize states
            states_mean = np.mean(states, axis=0)
            states_std = np.std(states, axis=0)
            states = (states - states_mean) / (states_std + 1e-10)
            
            # Initialize model
            rl_model = QLearningModel(
                state_size=len(state_features),
                action_size=3,
                learning_rate=0.001,
                discount_factor=0.95
            )
            
            # Train model
            num_episodes = 100
            batch_size = 32
            
            episode_rewards = []
            episode_actions = []
            
            for episode in range(num_episodes):
                total_reward = 0
                action_counts = [0, 0, 0]
                
                # Shuffle data
                indices = np.random.permutation(len(df))
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    
                    # Get batch data
                    batch_states = states[batch_indices]
                    
                    # Get actions
                    batch_actions = []
                    batch_rewards = []
                    
                    for j, idx in enumerate(batch_indices):
                        # Get state
                        state = batch_states[j].reshape(1, -1)
                        
                        # Get action
                        if np.random.rand() < 0.1:  # Exploration
                            action = np.random.randint(0, 3)
                        else:  # Exploitation
                            q_values = rl_model.predict(state)
                            action = np.argmax(q_values)
                        
                        # Get reward
                        df.loc[idx, 'action'] = action
                        reward = calculate_reward(df.loc[idx])
                        
                        batch_actions.append(action)
                        batch_rewards.append(reward)
                        
                        total_reward += reward
                        action_counts[action] += 1
                    
                    # Get next states
                    next_batch_indices = indices[i+batch_size:i+2*batch_size]
                    if len(next_batch_indices) == 0:
                        next_batch_indices = indices[:batch_size]
                    
                    batch_next_states = states[next_batch_indices[:len(batch_indices)]]
                    
                    # Get dones
                    batch_dones = [False] * len(batch_indices)
                    
                    # Train model
                    rl_model.train(
                        batch_states,
                        np.array(batch_actions),
                        np.array(batch_rewards),
                        batch_next_states,
                        np.array(batch_dones)
                    )
                
                episode_rewards.append(total_reward)
                episode_actions.append(action_counts)
                
                logger.info(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Actions: {action_counts}")
            
            # Evaluate model
            df['rl_action'] = 0
            
            for i in range(len(df)):
                state = states[i].reshape(1, -1)
                q_values = rl_model.predict(state)
                action = np.argmax(q_values)
                df.loc[i, 'rl_action'] = action
            
            df['rl_reward'] = df.apply(calculate_reward, axis=1)
            
            # Calculate metrics
            total_predictions = len(df)
            skipped_predictions = sum(df['rl_action'] == 0)
            over_predictions = sum(df['rl_action'] == 1)
            under_predictions = sum(df['rl_action'] == 2)
            
            total_reward = df['rl_reward'].sum()
            average_reward = df['rl_reward'].mean()
            
            correct_predictions = sum((df['rl_action'] == 1) & (df['actual_value'] > df['line_value']) | 
                                     (df['rl_action'] == 2) & (df['actual_value'] < df['line_value']))
            
            accuracy = correct_predictions / (over_predictions + under_predictions) if (over_predictions + under_predictions) > 0 else 0
            
            # Prepare optimization results
            optimization_results = {
                'episode_rewards': episode_rewards,
                'episode_actions': episode_actions,
                'total_predictions': total_predictions,
                'skipped_predictions': int(skipped_predictions),
                'over_predictions': int(over_predictions),
                'under_predictions': int(under_predictions),
                'total_reward': float(total_reward),
                'average_reward': float(average_reward),
                'correct_predictions': int(correct_predictions),
                'accuracy': float(accuracy),
                'state_features': state_features,
                'states_mean': states_mean.tolist(),
                'states_std': states_std.tolist()
            }
            
            return rl_model, optimization_results
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            return None, {}
    
    def _implement_hyperparameter_improvement(self, model_version: str, model_path: str, 
                                            model_type: str, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement hyperparameter improvement to the model.
        
        Args:
            model_version: Model version
            model_path: Path to model file
            model_type: Type of model
            improvement: Improvement details
            
        Returns:
            Dictionary of implementation results
        """
        try:
            logger.info(f"Implementing hyperparameter improvement for model version {model_version}")
            
            # Get hyperparameter changes
            hyperparameters = improvement.get('hyperparameters', {})
            
            if not hyperparameters:
                logger.warning(f"No hyperparameters specified for improvement")
                return {'success': False, 'error': 'No hyperparameters specified'}
            
            # Load model
            if model_type in ['hybrid', 'lstm', 'transformer']:
                # Load Keras model
                model = tf.keras.models.load_model(model_path)
                
                # Update hyperparameters
                if 'learning_rate' in hyperparameters:
                    # Update optimizer learning rate
                    optimizer = model.optimizer
                    tf.keras.backend.set_value(optimizer.learning_rate, hyperparameters['learning_rate'])
                
                # Save updated model
                model.save(model_path)
            else:
                # Load scikit-learn model
                model = joblib.load(model_path)
                
                # Update hyperparameters
                for param, value in hyperparameters.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
                
                # Save updated model
                joblib.dump(model, model_path)
            
            logger.info(f"Updated hyperparameters for model version {model_version}")
            
            return {
                'success': True,
                'model_version': model_version,
                'model_path': model_path,
                'hyperparameters': hyperparameters
            }
        except Exception as e:
            logger.error(f"Error implementing hyperparameter improvement: {e}")
            return {'success': False, 'error': str(e)}
    
    def _implement_feature_improvement(self, model_version: str, model_path: str, 
                                     model_type: str, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement feature improvement to the model.
        
        Args:
            model_version: Model version
            model_path: Path to model file
            model_type: Type of model
            improvement: Improvement details
            
        Returns:
            Dictionary of implementation results
        """
        try:
            logger.info(f"Implementing feature improvement for model version {model_version}")
            
            # Get feature changes
            feature_changes = improvement.get('feature_changes', {})
            
            if not feature_changes:
                logger.warning(f"No feature changes specified for improvement")
                return {'success': False, 'error': 'No feature changes specified'}
            
            # Update feature configuration
            feature_config_path = os.path.join(os.path.dirname(model_path), 'feature_config.json')
            
            if os.path.exists(feature_config_path):
                with open(feature_config_path, 'r') as f:
                    feature_config = json.load(f)
            else:
                feature_config = {}
            
            # Apply feature changes
            for change_type, changes in feature_changes.items():
                if change_type == 'add':
                    for feature in changes:
                        if 'features' not in feature_config:
                            feature_config['features'] = []
                        
                        feature_config['features'].append(feature)
                
                elif change_type == 'remove':
                    if 'features' in feature_config:
                        feature_config['features'] = [f for f in feature_config['features'] if f not in changes]
                
                elif change_type == 'modify':
                    for feature, config in changes.items():
                        if 'feature_config' not in feature_config:
                            feature_config['feature_config'] = {}
                        
                        feature_config['feature_config'][feature] = config
            
            # Save updated feature configuration
            with open(feature_config_path, 'w') as f:
                json.dump(feature_config, f, indent=2)
            
            logger.info(f"Updated feature configuration for model version {model_version}")
            
            return {
                'success': True,
                'model_version': model_version,
                'feature_config_path': feature_config_path,
                'feature_changes': feature_changes
            }
        except Exception as e:
            logger.error(f"Error implementing feature improvement: {e}")
            return {'success': False, 'error': str(e)}
    
    def _implement_architecture_improvement(self, model_version: str, model_path: str, 
                                          model_type: str, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement architecture improvement to the model.
        
        Args:
            model_version: Model version
            model_path: Path to model file
            model_type: Type of model
            improvement: Improvement details
            
        Returns:
            Dictionary of implementation results
        """
        try:
            logger.info(f"Implementing architecture improvement for model version {model_version}")
            
            # Get architecture changes
            architecture_changes = improvement.get('architecture_changes', {})
            
            if not architecture_changes:
                logger.warning(f"No architecture changes specified for improvement")
                return {'success': False, 'error': 'No architecture changes specified'}
            
            # Update architecture configuration
            architecture_config_path = os.path.join(os.path.dirname(model_path), 'architecture_config.json')
            
            if os.path.exists(architecture_config_path):
                with open(architecture_config_path, 'r') as f:
                    architecture_config = json.load(f)
            else:
                architecture_config = {}
            
            # Apply architecture changes
            for component, changes in architecture_changes.items():
                if component not in architecture_config:
                    architecture_config[component] = {}
                
                for param, value in changes.items():
                    architecture_config[component][param] = value
            
            # Save updated architecture configuration
            with open(architecture_config_path, 'w') as f:
                json.dump(architecture_config, f, indent=2)
            
            logger.info(f"Updated architecture configuration for model version {model_version}")
            
            return {
                'success': True,
                'model_version': model_version,
                'architecture_config_path': architecture_config_path,
                'architecture_changes': architecture_changes
            }
        except Exception as e:
            logger.error(f"Error implementing architecture improvement: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_metrics(self, prediction_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for prediction data.
        
        Args:
            prediction_data: List of prediction data
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(prediction_data)
            
            # Calculate regression metrics
            mae = float(df['absolute_error'].mean())
            mse = float(df['squared_error'].mean())
            rmse = float(np.sqrt(mse))
            mape = float(df['percentage_error'].mean())
            
            # Calculate classification metrics
            over_under_accuracy = float(df['over_under_correct'].mean())
            
            # Calculate metrics by confidence level
            df['confidence_bucket'] = pd.cut(df['confidence_score'], bins=[0, 25, 50, 75, 90, 100], 
                                           labels=['0-25', '25-50', '50-75', '75-90', '90-100'])
            
            confidence_metrics = {}
            
            for bucket, group in df.groupby('confidence_bucket'):
                confidence_metrics[str(bucket)] = {
                    'count': len(group),
                    'mae': float(group['absolute_error'].mean()),
                    'rmse': float(np.sqrt(group['squared_error'].mean())),
                    'over_under_accuracy': float(group['over_under_correct'].mean())
                }
            
            # Calculate metrics by sport
            sport_metrics = {}
            
            for sport, group in df.groupby('sport'):
                sport_metrics[sport] = {
                    'count': len(group),
                    'mae': float(group['absolute_error'].mean()),
                    'rmse': float(np.sqrt(group['squared_error'].mean())),
                    'over_under_accuracy': float(group['over_under_correct'].mean())
                }
            
            # Calculate metrics by stat type
            stat_type_metrics = {}
            
            for stat_type, group in df.groupby('stat_type'):
                stat_type_metrics[stat_type] = {
                    'count': len(group),
                    'mae': float(group['absolute_error'].mean()),
                    'rmse': float(np.sqrt(group['squared_error'].mean())),
                    'over_under_accuracy': float(group['over_under_correct'].mean())
                }
            
            return {
                'count': len(df),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'over_under_accuracy': over_under_accuracy,
                'confidence_metrics': confidence_metrics,
                'sport_metrics': sport_metrics,
                'stat_type_metrics': stat_type_metrics
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _compare_metrics(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics between two models.
        
        Args:
            metrics_a: Metrics for model A
            metrics_b: Metrics for model B
            
        Returns:
            Dictionary of metric comparisons
        """
        try:
            comparison = {}
            
            # Compare overall metrics
            for metric in ['mae', 'rmse', 'mape', 'over_under_accuracy']:
                if metric in metrics_a and metric in metrics_b:
                    value_a = metrics_a[metric]
                    value_b = metrics_b[metric]
                    
                    if metric in ['mae', 'rmse', 'mape']:
                        # Lower is better
                        diff = value_a - value_b
                        pct_diff = diff / value_a * 100 if value_a != 0 else 0
                        better = 'B' if diff > 0 else 'A'
                    else:
                        # Higher is better
                        diff = value_b - value_a
                        pct_diff = diff / value_a * 100 if value_a != 0 else 0
                        better = 'B' if diff > 0 else 'A'
                    
                    comparison[metric] = {
                        'value_a': value_a,
                        'value_b': value_b,
                        'diff': float(diff),
                        'pct_diff': float(pct_diff),
                        'better': better
                    }
            
            # Compare confidence metrics
            comparison['confidence_metrics'] = {}
            
            for bucket in set(metrics_a.get('confidence_metrics', {}).keys()) | set(metrics_b.get('confidence_metrics', {}).keys()):
                bucket_metrics_a = metrics_a.get('confidence_metrics', {}).get(bucket, {})
                bucket_metrics_b = metrics_b.get('confidence_metrics', {}).get(bucket, {})
                
                if not bucket_metrics_a or not bucket_metrics_b:
                    continue
                
                comparison['confidence_metrics'][bucket] = {}
                
                for metric in ['mae', 'rmse', 'over_under_accuracy']:
                    if metric in bucket_metrics_a and metric in bucket_metrics_b:
                        value_a = bucket_metrics_a[metric]
                        value_b = bucket_metrics_b[metric]
                        
                        if metric in ['mae', 'rmse']:
                            # Lower is better
                            diff = value_a - value_b
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        else:
                            # Higher is better
                            diff = value_b - value_a
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        
                        comparison['confidence_metrics'][bucket][metric] = {
                            'value_a': value_a,
                            'value_b': value_b,
                            'diff': float(diff),
                            'pct_diff': float(pct_diff),
                            'better': better
                        }
            
            # Compare sport metrics
            comparison['sport_metrics'] = {}
            
            for sport in set(metrics_a.get('sport_metrics', {}).keys()) | set(metrics_b.get('sport_metrics', {}).keys()):
                sport_metrics_a = metrics_a.get('sport_metrics', {}).get(sport, {})
                sport_metrics_b = metrics_b.get('sport_metrics', {}).get(sport, {})
                
                if not sport_metrics_a or not sport_metrics_b:
                    continue
                
                comparison['sport_metrics'][sport] = {}
                
                for metric in ['mae', 'rmse', 'over_under_accuracy']:
                    if metric in sport_metrics_a and metric in sport_metrics_b:
                        value_a = sport_metrics_a[metric]
                        value_b = sport_metrics_b[metric]
                        
                        if metric in ['mae', 'rmse']:
                            # Lower is better
                            diff = value_a - value_b
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        else:
                            # Higher is better
                            diff = value_b - value_a
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        
                        comparison['sport_metrics'][sport][metric] = {
                            'value_a': value_a,
                            'value_b': value_b,
                            'diff': float(diff),
                            'pct_diff': float(pct_diff),
                            'better': better
                        }
            
            # Compare stat type metrics
            comparison['stat_type_metrics'] = {}
            
            for stat_type in set(metrics_a.get('stat_type_metrics', {}).keys()) | set(metrics_b.get('stat_type_metrics', {}).keys()):
                stat_type_metrics_a = metrics_a.get('stat_type_metrics', {}).get(stat_type, {})
                stat_type_metrics_b = metrics_b.get('stat_type_metrics', {}).get(stat_type, {})
                
                if not stat_type_metrics_a or not stat_type_metrics_b:
                    continue
                
                comparison['stat_type_metrics'][stat_type] = {}
                
                for metric in ['mae', 'rmse', 'over_under_accuracy']:
                    if metric in stat_type_metrics_a and metric in stat_type_metrics_b:
                        value_a = stat_type_metrics_a[metric]
                        value_b = stat_type_metrics_b[metric]
                        
                        if metric in ['mae', 'rmse']:
                            # Lower is better
                            diff = value_a - value_b
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        else:
                            # Higher is better
                            diff = value_b - value_a
                            pct_diff = diff / value_a * 100 if value_a != 0 else 0
                            better = 'B' if diff > 0 else 'A'
                        
                        comparison['stat_type_metrics'][stat_type][metric] = {
                            'value_a': value_a,
                            'value_b': value_b,
                            'diff': float(diff),
                            'pct_diff': float(pct_diff),
                            'better': better
                        }
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing metrics: {e}")
            return {}
    
    def _determine_winner(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the winner between two models.
        
        Args:
            comparison: Dictionary of metric comparisons
            
        Returns:
            Dictionary with winner information
        """
        try:
            # Count wins for each model
            wins_a = 0
            wins_b = 0
            
            # Overall metrics
            for metric, comp in comparison.items():
                if isinstance(comp, dict) and 'better' in comp:
                    if comp['better'] == 'A':
                        wins_a += 1
                    elif comp['better'] == 'B':
                        wins_b += 1
            
            # Confidence metrics
            for bucket, metrics in comparison.get('confidence_metrics', {}).items():
                for metric, comp in metrics.items():
                    if 'better' in comp:
                        if comp['better'] == 'A':
                            wins_a += 0.5  # Weight less than overall metrics
                        elif comp['better'] == 'B':
                            wins_b += 0.5
            
            # Sport metrics
            for sport, metrics in comparison.get('sport_metrics', {}).items():
                for metric, comp in metrics.items():
                    if 'better' in comp:
                        if comp['better'] == 'A':
                            wins_a += 0.5
                        elif comp['better'] == 'B':
                            wins_b += 0.5
            
            # Stat type metrics
            for stat_type, metrics in comparison.get('stat_type_metrics', {}).items():
                for metric, comp in metrics.items():
                    if 'better' in comp:
                        if comp['better'] == 'A':
                            wins_a += 0.5
                        elif comp['better'] == 'B':
                            wins_b += 0.5
            
            # Determine winner
            if wins_a > wins_b:
                winner = 'A'
                win_margin = wins_a - wins_b
            elif wins_b > wins_a:
                winner = 'B'
                win_margin = wins_b - wins_a
            else:
                winner = 'Tie'
                win_margin = 0
            
            return {
                'winner': winner,
                'wins_a': float(wins_a),
                'wins_b': float(wins_b),
                'win_margin': float(win_margin)
            }
        except Exception as e:
            logger.error(f"Error determining winner: {e}")
            return {'winner': 'Unknown', 'error': str(e)}
    
    def _generate_optimization_plots(self, optimization_results: Dict[str, Any], 
                                   sport: Optional[str] = None, model_version: Optional[str] = None) -> None:
        """
        Generate plots for optimization results.
        
        Args:
            optimization_results: Dictionary of optimization results
            sport: Sport to filter by (optional)
            model_version: Model version to filter by (optional)
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot episode rewards
            episode_rewards = optimization_results.get('episode_rewards', [])
            
            if episode_rewards:
                plt.figure(figsize=(14, 8))
                plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
                plt.title(f"Episode Rewards{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.grid(True)
                plt.tight_layout()
                
                rewards_file = os.path.join(
                    plots_dir, 
                    f"episode_rewards{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(rewards_file)
                plt.close()
            
            # Plot action distribution
            episode_actions = optimization_results.get('episode_actions', [])
            
            if episode_actions:
                plt.figure(figsize=(14, 8))
                
                skip_actions = [actions[0] for actions in episode_actions]
                over_actions = [actions[1] for actions in episode_actions]
                under_actions = [actions[2] for actions in episode_actions]
                
                plt.plot(range(1, len(episode_actions) + 1), skip_actions, label='Skip')
                plt.plot(range(1, len(episode_actions) + 1), over_actions, label='Over')
                plt.plot(range(1, len(episode_actions) + 1), under_actions, label='Under')
                
                plt.title(f"Action Distribution by Episode{' for ' + sport.upper() if sport else ''}")
                plt.xlabel('Episode')
                plt.ylabel('Action Count')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                actions_file = os.path.join(
                    plots_dir, 
                    f"action_distribution{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
                )
                plt.savefig(actions_file)
                plt.close()
            
            # Plot final action distribution
            skipped_predictions = optimization_results.get('skipped_predictions', 0)
            over_predictions = optimization_results.get('over_predictions', 0)
            under_predictions = optimization_results.get('under_predictions', 0)
            
            plt.figure(figsize=(10, 8))
            plt.bar(['Skip', 'Over', 'Under'], [skipped_predictions, over_predictions, under_predictions])
            plt.title(f"Final Action Distribution{' for ' + sport.upper() if sport else ''}")
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.grid(True)
            plt.tight_layout()
            
            final_actions_file = os.path.join(
                plots_dir, 
                f"final_action_distribution{'_' + sport if sport else ''}{'_' + model_version if model_version else ''}.png"
            )
            plt.savefig(final_actions_file)
            plt.close()
            
            logger.info(f"Generated optimization plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating optimization plots: {e}")
    
    def _generate_ab_testing_plots(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], 
                                 comparison: Dict[str, Any], ab_testing_id: str) -> None:
        """
        Generate plots for A/B testing results.
        
        Args:
            metrics_a: Metrics for model A
            metrics_b: Metrics for model B
            comparison: Dictionary of metric comparisons
            ab_testing_id: A/B testing ID
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.reports_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot overall metrics comparison
            overall_metrics = ['mae', 'rmse', 'over_under_accuracy']
            
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(overall_metrics))
            width = 0.35
            
            values_a = [metrics_a.get(metric, 0) for metric in overall_metrics]
            values_b = [metrics_b.get(metric, 0) for metric in overall_metrics]
            
            plt.bar(x - width/2, values_a, width, label='Model A')
            plt.bar(x + width/2, values_b, width, label='Model B')
            
            plt.title(f"Overall Metrics Comparison (A/B Testing ID: {ab_testing_id})")
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.xticks(x, overall_metrics)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            overall_file = os.path.join(
                plots_dir, 
                f"ab_testing_overall_metrics_{ab_testing_id}.png"
            )
            plt.savefig(overall_file)
            plt.close()
            
            # Plot confidence metrics comparison
            confidence_buckets = sorted(set(metrics_a.get('confidence_metrics', {}).keys()) & set(metrics_b.get('confidence_metrics', {}).keys()))
            
            if confidence_buckets:
                for metric in ['mae', 'over_under_accuracy']:
                    plt.figure(figsize=(14, 8))
                    
                    values_a = [metrics_a.get('confidence_metrics', {}).get(bucket, {}).get(metric, 0) for bucket in confidence_buckets]
                    values_b = [metrics_b.get('confidence_metrics', {}).get(bucket, {}).get(metric, 0) for bucket in confidence_buckets]
                    
                    x = np.arange(len(confidence_buckets))
                    width = 0.35
                    
                    plt.bar(x - width/2, values_a, width, label='Model A')
                    plt.bar(x + width/2, values_b, width, label='Model B')
                    
                    plt.title(f"{metric.upper()} by Confidence Level (A/B Testing ID: {ab_testing_id})")
                    plt.xlabel('Confidence Level')
                    plt.ylabel(metric.upper())
                    plt.xticks(x, confidence_buckets)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    confidence_file = os.path.join(
                        plots_dir, 
                        f"ab_testing_confidence_{metric}_{ab_testing_id}.png"
                    )
                    plt.savefig(confidence_file)
                    plt.close()
            
            # Plot sport metrics comparison
            sports = sorted(set(metrics_a.get('sport_metrics', {}).keys()) & set(metrics_b.get('sport_metrics', {}).keys()))
            
            if sports:
                for metric in ['mae', 'over_under_accuracy']:
                    plt.figure(figsize=(14, 8))
                    
                    values_a = [metrics_a.get('sport_metrics', {}).get(sport, {}).get(metric, 0) for sport in sports]
                    values_b = [metrics_b.get('sport_metrics', {}).get(sport, {}).get(metric, 0) for sport in sports]
                    
                    x = np.arange(len(sports))
                    width = 0.35
                    
                    plt.bar(x - width/2, values_a, width, label='Model A')
                    plt.bar(x + width/2, values_b, width, label='Model B')
                    
                    plt.title(f"{metric.upper()} by Sport (A/B Testing ID: {ab_testing_id})")
                    plt.xlabel('Sport')
                    plt.ylabel(metric.upper())
                    plt.xticks(x, sports)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    sport_file = os.path.join(
                        plots_dir, 
                        f"ab_testing_sport_{metric}_{ab_testing_id}.png"
                    )
                    plt.savefig(sport_file)
                    plt.close()
            
            logger.info(f"Generated A/B testing plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating A/B testing plots: {e}")


if __name__ == "__main__":
    # Run continuous improvement
    improvement = ContinuousImprovement()
    optimization_results = improvement.optimize_prediction_strategy()
    
    # Log results
    logger.info(f"Continuous improvement completed")
