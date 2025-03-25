"""
Multi-Task Learning Implementation for Sports Prediction System

This module implements multi-task learning capabilities for the sports prediction system,
enabling simultaneous prediction of multiple related statistics with shared layers
and specialized output heads.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Concatenate, 
    BatchNormalization, Attention, MultiHeadAttention,
    LayerNormalization
)
import numpy as np

class MultiTaskSportsPredictionModel:
    """
    Multi-task learning model for sports predictions that can predict multiple
    related statistics simultaneously using shared layers with specialized output heads.
    """
    
    def __init__(self, config):
        """
        Initialize the multi-task learning model with configuration parameters.
        
        Args:
            config (dict): Configuration parameters including:
                - input_shape: Shape of the input features
                - shared_layers: List of units in shared layers
                - task_specific_layers: List of units in task-specific layers
                - tasks: List of tasks to predict (e.g., ['points', 'rebounds', 'assists'])
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Learning rate for optimizer
                - uncertainty_estimation: Whether to estimate prediction uncertainty
        """
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the multi-task learning model architecture.
        
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        # Input layer
        input_layer = Input(shape=self.config['input_shape'], name='input')
        
        # Shared layers
        x = input_layer
        for i, units in enumerate(self.config['shared_layers']):
            x = Dense(units, activation='relu', name=f'shared_dense_{i}')(x)
            x = BatchNormalization(name=f'shared_bn_{i}')(x)
            x = Dropout(self.config['dropout_rate'], name=f'shared_dropout_{i}')(x)
        
        # Task-specific layers and outputs
        outputs = {}
        losses = {}
        metrics = {}
        
        for task in self.config['tasks']:
            # Task-specific layers
            task_x = x
            for i, units in enumerate(self.config['task_specific_layers']):
                task_x = Dense(units, activation='relu', name=f'{task}_dense_{i}')(task_x)
                task_x = BatchNormalization(name=f'{task}_bn_{i}')(task_x)
                task_x = Dropout(self.config['dropout_rate'], name=f'{task}_dropout_{i}')(task_x)
            
            # Determine if this is a regression or classification task
            is_classification = task.endswith('_prob') or task.endswith('_binary')
            
            if self.config.get('uncertainty_estimation', False):
                # For uncertainty estimation, output mean and variance
                task_mean = Dense(1, name=f'{task}_mean')(task_x)
                task_var = Dense(1, activation='softplus', name=f'{task}_variance')(task_x)
                
                # Create a lambda layer to sample from the distribution during training
                def negative_log_likelihood(y_true, y_pred):
                    # Unpack mean and variance
                    mean, var = y_pred[:, 0:1], y_pred[:, 1:2]
                    # Compute negative log likelihood
                    return 0.5 * tf.reduce_mean(
                        tf.math.log(var) + tf.square(y_true - mean) / var
                    )
                
                # Concatenate mean and variance for output
                task_output = Concatenate(name=f'{task}_output')([task_mean, task_var])
                
                # Set loss and metrics
                losses[f'{task}_output'] = negative_log_likelihood
                metrics[f'{task}_output'] = ['mae', 'mse']
                
            else:
                # Standard output without uncertainty estimation
                if is_classification:
                    task_output = Dense(1, activation='sigmoid', name=f'{task}_output')(task_x)
                    losses[f'{task}_output'] = 'binary_crossentropy'
                    metrics[f'{task}_output'] = ['accuracy', 'AUC']
                else:
                    task_output = Dense(1, name=f'{task}_output')(task_x)
                    losses[f'{task}_output'] = 'mse'
                    metrics[f'{task}_output'] = ['mae', 'mse']
            
            outputs[f'{task}_output'] = task_output
        
        # Create model with multiple outputs
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Compile model with appropriate loss functions and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=losses,
            metrics=metrics
        )
        
        return model
    
    def fit(self, X, y, **kwargs):
        """
        Train the model with the provided data.
        
        Args:
            X (np.ndarray): Input features
            y (dict): Dictionary mapping task names to target values
            **kwargs: Additional arguments to pass to model.fit()
                
        Returns:
            History object from model training
        """
        return self.model.fit(X, y, **kwargs)
    
    def predict(self, X):
        """
        Generate predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
                
        Returns:
            dict: Dictionary mapping task names to predictions
        """
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X, num_samples=100):
        """
        Generate predictions with uncertainty estimates using Monte Carlo sampling.
        Only applicable if uncertainty_estimation is enabled.
        
        Args:
            X (np.ndarray): Input features
            num_samples (int): Number of Monte Carlo samples
                
        Returns:
            dict: Dictionary mapping task names to (mean, std) tuples
        """
        if not self.config.get('uncertainty_estimation', False):
            raise ValueError("Uncertainty estimation is not enabled for this model")
        
        # Enable dropout at inference time
        def enable_dropout():
            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    layer.trainable = True
        
        enable_dropout()
        
        # Perform multiple forward passes
        predictions = []
        for _ in range(num_samples):
            preds = self.model.predict(X)
            predictions.append(preds)
        
        # Compute mean and standard deviation across samples
        results = {}
        for task in self.config['tasks']:
            task_preds = np.array([p[f'{task}_output'] for p in predictions])
            
            if self.config.get('uncertainty_estimation', False):
                # Extract mean and variance from model outputs
                means = task_preds[:, :, 0]
                variances = task_preds[:, :, 1]
                
                # Compute predictive mean and variance
                predictive_mean = np.mean(means, axis=0)
                predictive_variance = np.mean(variances + np.square(means), axis=0) - np.square(predictive_mean)
                predictive_std = np.sqrt(predictive_variance)
                
                results[task] = (predictive_mean, predictive_std)
            else:
                # Standard MC dropout uncertainty
                mean = np.mean(task_preds, axis=0)
                std = np.std(task_preds, axis=0)
                results[task] = (mean, std)
        
        return results
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath, custom_objects=None):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            custom_objects (dict): Dictionary mapping names to custom classes or functions
                
        Returns:
            MultiTaskSportsPredictionModel: Loaded model
        """
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        # Create a dummy config to initialize the class
        dummy_config = {}
        instance = cls(dummy_config)
        instance.model = model
        return instance


class UncertaintyEstimation:
    """
    Utility class for uncertainty estimation in sports predictions.
    """
    
    @staticmethod
    def gaussian_nll_loss(y_true, y_pred):
        """
        Gaussian negative log likelihood loss for uncertainty estimation.
        
        Args:
            y_true (tf.Tensor): True values
            y_pred (tf.Tensor): Predicted values (mean and log variance)
            
        Returns:
            tf.Tensor: Loss value
        """
        # Unpack mean and log variance
        mean, log_var = tf.split(y_pred, 2, axis=-1)
        
        # Compute precision
        precision = tf.exp(-log_var)
        
        # Compute loss
        loss = 0.5 * tf.reduce_mean(
            log_var + tf.square(y_true - mean) * precision
        )
        
        return loss
    
    @staticmethod
    def quantile_loss(y_true, y_pred, quantile):
        """
        Quantile loss for quantile regression.
        
        Args:
            y_true (tf.Tensor): True values
            y_pred (tf.Tensor): Predicted values
            quantile (float): Quantile value (0 to 1)
            
        Returns:
            tf.Tensor: Loss value
        """
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(quantile * error, (quantile - 1) * error)
        )
    
    @staticmethod
    def prediction_intervals(mean, std, confidence=0.95):
        """
        Compute prediction intervals based on mean and standard deviation.
        
        Args:
            mean (np.ndarray): Predicted mean values
            std (np.ndarray): Predicted standard deviation values
            confidence (float): Confidence level (0 to 1)
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        # Compute z-score for the given confidence level
        z_score = {
            0.50: 0.674,
            0.68: 1.000,
            0.80: 1.282,
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }.get(confidence, 1.960)
        
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        return lower_bound, upper_bound


def create_multitask_config(tasks, input_shape):
    """
    Create default configuration for the multi-task model.
    
    Args:
        tasks (list): List of tasks to predict
        input_shape (tuple): Shape of the input features
        
    Returns:
        dict: Default configuration
    """
    return {
        'input_shape': input_shape,
        'shared_layers': [256, 128, 64],
        'task_specific_layers': [32, 16],
        'tasks': tasks,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'uncertainty_estimation': True
    }
