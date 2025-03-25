"""
Attention Mechanisms for Sports Prediction System

This module implements various attention mechanisms for the sports prediction system,
focusing on identifying and highlighting the most relevant historical games and features.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class TemporalAttention(Layer):
    """
    Temporal attention mechanism that focuses on the most relevant historical games
    in a player's performance sequence.
    """
    
    def __init__(self, units):
        """
        Initialize the temporal attention layer.
        
        Args:
            units (int): Number of attention units
        """
        super(TemporalAttention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape (tuple): Shape of the input tensor
        """
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='u'
        )
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of the attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, seq_length, features]
            
        Returns:
            tuple: (context_vector, attention_weights)
        """
        # inputs shape: (batch_size, seq_length, features)
        
        # Calculate attention scores
        # uit = tanh(W * hit + b)
        uit = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        
        # ait = softmax(uit * u)
        ait = tf.matmul(uit, self.u)
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Calculate context vector
        # ci = sum(ait * hit)
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class FeatureAttention(Layer):
    """
    Feature attention mechanism that focuses on the most relevant features
    for a specific prediction task.
    """
    
    def __init__(self, units):
        """
        Initialize the feature attention layer.
        
        Args:
            units (int): Number of attention units
        """
        super(FeatureAttention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape (tuple): Shape of the input tensor
        """
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='u'
        )
        super(FeatureAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of the attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, features]
            
        Returns:
            tuple: (weighted_features, attention_weights)
        """
        # inputs shape: (batch_size, features)
        
        # Expand dimensions for attention calculation
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
        
        # Calculate attention scores
        uit = tf.tanh(tf.matmul(expanded_inputs, self.W) + self.b)  # (batch_size, 1, units)
        ait = tf.matmul(uit, self.u)  # (batch_size, 1, 1)
        attention_weights = tf.nn.softmax(ait, axis=2)  # (batch_size, 1, 1)
        
        # Reshape attention weights to match features
        attention_weights = tf.reshape(attention_weights, [-1, 1, inputs.shape[-1]])  # (batch_size, 1, features)
        
        # Apply attention weights to features
        weighted_features = inputs * tf.squeeze(attention_weights, axis=1)  # (batch_size, features)
        
        return weighted_features, tf.squeeze(attention_weights, axis=1)


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self-attention mechanism for capturing complex relationships
    between different time steps in sequential data.
    """
    
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head self-attention layer.
        
        Args:
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth).
        
        Args:
            x (tf.Tensor): Input tensor
            batch_size (int): Batch size
            
        Returns:
            tf.Tensor: Tensor with shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        """
        Forward pass of the multi-head self-attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            tuple: (output, attention_weights)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        q = self.wq(inputs)  # (batch_size, seq_len, d_model)
        k = self.wk(inputs)  # (batch_size, seq_len, d_model)
        v = self.wv(inputs)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)
        
        # Scaled dot-product attention
        # Transpose k for matrix multiplication
        k_transposed = tf.transpose(k, perm=[0, 1, 3, 2])  # (batch_size, num_heads, depth, seq_len)
        
        # Calculate attention scores
        matmul_qk = tf.matmul(q, k_transposed)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Scale attention scores
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)
        
        # Reshape output
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.dense(output)  # (batch_size, seq_len, d_model)
        
        return output, attention_weights


class ContextualAttention(Layer):
    """
    Contextual attention mechanism that incorporates external context
    (like opponent strength, venue, etc.) into the attention calculation.
    """
    
    def __init__(self, units):
        """
        Initialize the contextual attention layer.
        
        Args:
            units (int): Number of attention units
        """
        super(ContextualAttention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape (list): List of shapes for [sequence_input, context_input]
        """
        seq_shape, ctx_shape = input_shape
        
        self.W_seq = self.add_weight(
            shape=(seq_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W_seq'
        )
        
        self.W_ctx = self.add_weight(
            shape=(ctx_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W_ctx'
        )
        
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='u'
        )
        
        super(ContextualAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of the contextual attention layer.
        
        Args:
            inputs (list): List of [sequence_input, context_input]
                sequence_input: Tensor of shape [batch_size, seq_length, seq_features]
                context_input: Tensor of shape [batch_size, ctx_features]
            
        Returns:
            tuple: (context_vector, attention_weights)
        """
        sequence_input, context_input = inputs
        
        # Expand context dimensions to match sequence length
        batch_size = tf.shape(sequence_input)[0]
        seq_length = tf.shape(sequence_input)[1]
        
        # Reshape context to [batch_size, 1, ctx_features]
        context_expanded = tf.expand_dims(context_input, axis=1)
        
        # Repeat context for each time step
        context_tiled = tf.tile(context_expanded, [1, seq_length, 1])
        
        # Calculate attention scores using both sequence and context
        # uit = tanh(W_seq * hit + W_ctx * ctx + b)
        seq_transform = tf.matmul(sequence_input, self.W_seq)
        ctx_transform = tf.matmul(context_tiled, self.W_ctx)
        
        uit = tf.tanh(seq_transform + ctx_transform + self.b)
        
        # ait = softmax(uit * u)
        ait = tf.matmul(uit, self.u)
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Calculate context vector
        # ci = sum(ait * hit)
        context_vector = sequence_input * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class HierarchicalAttention(Layer):
    """
    Hierarchical attention mechanism that first applies attention at the feature level
    and then at the temporal level, capturing both feature importance and temporal relevance.
    """
    
    def __init__(self, feature_units, temporal_units):
        """
        Initialize the hierarchical attention layer.
        
        Args:
            feature_units (int): Number of feature attention units
            temporal_units (int): Number of temporal attention units
        """
        super(HierarchicalAttention, self).__init__()
        self.feature_attention = FeatureAttention(feature_units)
        self.temporal_attention = TemporalAttention(temporal_units)
        
    def call(self, inputs):
        """
        Forward pass of the hierarchical attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, seq_length, features]
            
        Returns:
            tuple: (context_vector, (feature_weights, temporal_weights))
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        # Apply feature attention to each time step
        feature_weighted_seq = tf.TensorArray(tf.float32, size=seq_length)
        feature_weights_seq = tf.TensorArray(tf.float32, size=seq_length)
        
        for t in range(seq_length):
            # Get features at time step t
            features_t = inputs[:, t, :]
            
            # Apply feature attention
            weighted_features, feature_weights = self.feature_attention(features_t)
            
            # Store results
            feature_weighted_seq = feature_weighted_seq.write(t, weighted_features)
            feature_weights_seq = feature_weights_seq.write(t, feature_weights)
        
        # Stack results back into sequence
        feature_weighted_inputs = feature_weighted_seq.stack()
        feature_weights = feature_weights_seq.stack()
        
        # Transpose to get [batch_size, seq_length, features]
        feature_weighted_inputs = tf.transpose(feature_weighted_inputs, [1, 0, 2])
        feature_weights = tf.transpose(feature_weights, [1, 0, 2])
        
        # Apply temporal attention
        context_vector, temporal_weights = self.temporal_attention(feature_weighted_inputs)
        
        return context_vector, (feature_weights, temporal_weights)


def create_attention_layer(attention_type, config):
    """
    Factory function to create attention layers based on type.
    
    Args:
        attention_type (str): Type of attention ('temporal', 'feature', 'multihead', 'contextual', 'hierarchical')
        config (dict): Configuration parameters
        
    Returns:
        Layer: Attention layer
    """
    if attention_type == 'temporal':
        return TemporalAttention(config.get('units', 64))
    elif attention_type == 'feature':
        return FeatureAttention(config.get('units', 64))
    elif attention_type == 'multihead':
        return MultiHeadSelfAttention(
            d_model=config.get('d_model', 128),
            num_heads=config.get('num_heads', 4)
        )
    elif attention_type == 'contextual':
        return ContextualAttention(config.get('units', 64))
    elif attention_type == 'hierarchical':
        return HierarchicalAttention(
            feature_units=config.get('feature_units', 32),
            temporal_units=config.get('temporal_units', 64)
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
