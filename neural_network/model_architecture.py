"""
Enhanced Neural Network Architecture for Sports Prediction System

This module implements a hybrid neural network architecture for sports predictions,
combining LSTM/Transformer networks for sequential data with feedforward networks
for static features, and incorporating attention mechanisms and multi-task learning.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Concatenate, 
    BatchNormalization, Attention, MultiHeadAttention,
    LayerNormalization, Embedding, Flatten
)
import numpy as np

class HybridSportsPredictionModel:
    """
    Hybrid neural network model for sports predictions that combines:
    1. LSTM/Transformer networks for sequential player performance data
    2. Feedforward networks for static features (player attributes, team stats)
    3. Attention mechanisms to focus on relevant historical games
    4. Multi-task learning to predict both exact values and over/under probabilities
    """
    
    def __init__(self, config):
        """
        Initialize the hybrid model with configuration parameters.
        
        Args:
            config (dict): Configuration parameters including:
                - sequence_length: Number of historical games to consider
                - num_features: Number of features per game
                - num_static_features: Number of static features
                - embedding_dim: Dimension for player and team embeddings
                - num_players: Total number of players for embedding
                - num_teams: Total number of teams for embedding
                - lstm_units: Number of units in LSTM layers
                - dense_units: List of units in dense layers
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Learning rate for optimizer
                - use_transformer: Whether to use Transformer instead of LSTM
                - num_heads: Number of attention heads if using Transformer
        """
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the hybrid neural network model architecture.
        
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        # Input layers
        # 1. Sequential game data input (shape: [batch_size, sequence_length, num_features])
        seq_input = Input(shape=(self.config['sequence_length'], self.config['num_features']), 
                          name='sequential_input')
        
        # 2. Static player and team features input
        static_input = Input(shape=(self.config['num_static_features'],), 
                             name='static_input')
        
        # 3. Player ID input for embedding
        player_id_input = Input(shape=(1,), name='player_id_input')
        
        # 4. Team ID input for embedding
        team_id_input = Input(shape=(1,), name='team_id_input')
        
        # 5. Opponent team ID input for embedding
        opponent_id_input = Input(shape=(1,), name='opponent_id_input')
        
        # Player and team embeddings
        player_embedding = Embedding(
            input_dim=self.config['num_players'],
            output_dim=self.config['embedding_dim'],
            name='player_embedding'
        )(player_id_input)
        player_embedding = Flatten()(player_embedding)
        
        team_embedding = Embedding(
            input_dim=self.config['num_teams'],
            output_dim=self.config['embedding_dim'],
            name='team_embedding'
        )(team_id_input)
        team_embedding = Flatten()(team_embedding)
        
        opponent_embedding = Embedding(
            input_dim=self.config['num_teams'],
            output_dim=self.config['embedding_dim'],
            name='opponent_embedding'
        )(opponent_id_input)
        opponent_embedding = Flatten()(opponent_embedding)
        
        # Process sequential data with either LSTM or Transformer
        if self.config.get('use_transformer', False):
            # Transformer approach with multi-head attention
            x_seq = seq_input
            for i in range(2):  # 2 transformer blocks
                # Multi-head attention
                attention_output = MultiHeadAttention(
                    num_heads=self.config['num_heads'],
                    key_dim=self.config['num_features'] // self.config['num_heads'],
                    name=f'transformer_block_{i}_attention'
                )(x_seq, x_seq)
                
                # Add & Normalize
                x_seq = LayerNormalization(name=f'transformer_block_{i}_norm1')(x_seq + attention_output)
                
                # Feed Forward
                ff_output = Dense(self.config['num_features'] * 4, activation='relu',
                                 name=f'transformer_block_{i}_ff1')(x_seq)
                ff_output = Dense(self.config['num_features'], name=f'transformer_block_{i}_ff2')(ff_output)
                
                # Add & Normalize
                x_seq = LayerNormalization(name=f'transformer_block_{i}_norm2')(x_seq + ff_output)
            
            # Global average pooling to get fixed-size representation
            seq_features = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pooling')(x_seq)
            
        else:
            # LSTM approach
            x_seq = LSTM(self.config['lstm_units'], return_sequences=True, 
                         name='lstm_1')(seq_input)
            x_seq = Dropout(self.config['dropout_rate'])(x_seq)
            
            # Second LSTM layer with attention
            x_seq = LSTM(self.config['lstm_units'], return_sequences=True, 
                         name='lstm_2')(x_seq)
            
            # Self-attention mechanism
            attention = Attention(name='attention_layer')([x_seq, x_seq])
            
            # Combine LSTM output with attention
            x_seq = Concatenate(axis=-1)([x_seq, attention])
            
            # Final LSTM layer to get fixed-size representation
            seq_features = LSTM(self.config['lstm_units'], name='lstm_3')(x_seq)
        
        # Combine all features
        combined_features = Concatenate(axis=-1)([
            seq_features,
            static_input,
            player_embedding,
            team_embedding,
            opponent_embedding
        ])
        
        # Shared layers for both tasks
        x = Dense(self.config['dense_units'][0], activation='relu', name='shared_dense_1')(combined_features)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        x = Dense(self.config['dense_units'][1], activation='relu', name='shared_dense_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        # Task-specific layers
        # 1. Regression task (predict exact value)
        regression_output = Dense(1, name='regression_output')(x)
        
        # 2. Classification task (predict over/under probability)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(x)
        
        # 3. Confidence score output
        confidence_output = Dense(1, activation='sigmoid', name='confidence_output')(x)
        
        # Create model with multiple outputs
        model = Model(
            inputs=[seq_input, static_input, player_id_input, team_id_input, opponent_id_input],
            outputs=[regression_output, classification_output, confidence_output]
        )
        
        # Compile model with appropriate loss functions and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss={
                'regression_output': 'mse',
                'classification_output': 'binary_crossentropy',
                'confidence_output': 'mse'
            },
            metrics={
                'regression_output': ['mae', 'mse'],
                'classification_output': ['accuracy', 'AUC'],
                'confidence_output': ['mae']
            },
            loss_weights={
                'regression_output': 1.0,
                'classification_output': 1.0,
                'confidence_output': 0.5
            }
        )
        
        return model
    
    def fit(self, X, y, **kwargs):
        """
        Train the model with the provided data.
        
        Args:
            X (dict): Dictionary containing input data:
                - 'sequential_input': Sequential game data
                - 'static_input': Static player and team features
                - 'player_id_input': Player IDs
                - 'team_id_input': Team IDs
                - 'opponent_id_input': Opponent team IDs
            y (dict): Dictionary containing target data:
                - 'regression_output': Exact values to predict
                - 'classification_output': Over/under binary labels
                - 'confidence_output': Confidence scores
            **kwargs: Additional arguments to pass to model.fit()
                
        Returns:
            History object from model training
        """
        return self.model.fit(X, y, **kwargs)
    
    def predict(self, X):
        """
        Generate predictions using the trained model.
        
        Args:
            X (dict): Dictionary containing input data
                
        Returns:
            tuple: (predicted_values, over_probabilities, confidence_scores)
        """
        predictions = self.model.predict(X)
        return predictions
    
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
            HybridSportsPredictionModel: Loaded model
        """
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        # Create a dummy config to initialize the class
        dummy_config = {}
        instance = cls(dummy_config)
        instance.model = model
        return instance


class PlayerEmbeddingModel:
    """
    Model to generate player embeddings that capture playing style and skill level.
    These embeddings can be pre-trained and then used in the main prediction model.
    """
    
    def __init__(self, num_players, embedding_dim, hidden_dims=[128, 64]):
        """
        Initialize the player embedding model.
        
        Args:
            num_players (int): Total number of players
            embedding_dim (int): Dimension of the final embedding
            hidden_dims (list): Dimensions of hidden layers
        """
        self.num_players = num_players
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the player embedding model.
        
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        # Input: player stats for a single game
        input_layer = Input(shape=(None,), name='player_game_stats')
        
        # Hidden layers
        x = input_layer
        for i, dim in enumerate(self.hidden_dims):
            x = Dense(dim, activation='relu', name=f'dense_{i}')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # Final embedding layer
        embedding = Dense(self.embedding_dim, name='player_embedding')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=embedding)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def get_embeddings(self, player_stats):
        """
        Generate embeddings for players based on their stats.
        
        Args:
            player_stats (np.ndarray): Array of player statistics
                
        Returns:
            np.ndarray: Player embeddings
        """
        return self.model.predict(player_stats)


class TeamChemistryModel:
    """
    Model to generate team chemistry metrics based on lineup combinations.
    """
    
    def __init__(self, num_players, embedding_dim, output_dim=1):
        """
        Initialize the team chemistry model.
        
        Args:
            num_players (int): Total number of players
            embedding_dim (int): Dimension of player embeddings
            output_dim (int): Dimension of chemistry output (typically 1)
        """
        self.num_players = num_players
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the team chemistry model.
        
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        # Input: embeddings for 5 players in a lineup
        player_inputs = [Input(shape=(self.embedding_dim,), name=f'player_{i}_embedding') 
                         for i in range(5)]
        
        # Combine player embeddings
        combined = Concatenate()(player_inputs)
        
        # Process combined embeddings
        x = Dense(128, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output chemistry score
        chemistry = Dense(self.output_dim, name='chemistry_score')(x)
        
        # Create model
        model = Model(inputs=player_inputs, outputs=chemistry)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def predict_chemistry(self, player_embeddings):
        """
        Predict chemistry score for a lineup.
        
        Args:
            player_embeddings (list): List of 5 player embedding arrays
                
        Returns:
            float: Chemistry score
        """
        return self.model.predict(player_embeddings)


def create_default_config():
    """
    Create default configuration for the hybrid model.
    
    Returns:
        dict: Default configuration
    """
    return {
        'sequence_length': 10,  # Consider last 10 games
        'num_features': 20,     # Features per game
        'num_static_features': 15,  # Static player and team features
        'embedding_dim': 32,    # Dimension for player and team embeddings
        'num_players': 5000,    # Total number of players for embedding
        'num_teams': 150,       # Total number of teams for embedding
        'lstm_units': 64,       # Units in LSTM layers
        'dense_units': [128, 64],  # Units in dense layers
        'dropout_rate': 0.3,    # Dropout rate
        'learning_rate': 0.001, # Learning rate
        'use_transformer': True,  # Use Transformer instead of LSTM
        'num_heads': 4          # Number of attention heads
    }
