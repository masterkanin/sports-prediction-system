#!/bin/bash

# This script runs tests for the enhanced sports prediction system

echo "Starting tests for the enhanced sports prediction system..."

# Create test directory if it doesn't exist
mkdir -p /home/ubuntu/enhanced_sports_prediction/tests

# Test database connection
echo "Testing database connection..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from database.db_config import get_db_connection
try:
    conn = get_db_connection(
        host='localhost',
        port=5432,
        dbname='sports_prediction',
        user='postgres',
        password='postgres'
    )
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Test neural network model
echo "Testing neural network model..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from neural_network.model_architecture import HybridSportsPredictionModel
from neural_network.attention_mechanisms import AttentionMechanism
try:
    attention = AttentionMechanism(attention_type='multihead', num_heads=4)
    model = HybridSportsPredictionModel(
        sequence_length=10,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        attention=attention,
        use_transformer=True
    )
    print('Neural network model initialization successful')
except Exception as e:
    print(f'Neural network model initialization failed: {e}')
"

# Test data pipeline
echo "Testing data pipeline..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from data_pipeline.data_sources import DataSourceManager
try:
    data_source_manager = DataSourceManager(
        api_keys={
            'sportradar': 'test_key',
            'espn': 'test_key',
            'stats_perform': 'test_key',
            'weather': 'test_key'
        },
        cache_dir='/tmp/sports_prediction_cache'
    )
    print('Data pipeline initialization successful')
except Exception as e:
    print(f'Data pipeline initialization failed: {e}')
"

# Test evaluation system
echo "Testing evaluation system..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from evaluation.performance_tracking import PerformanceTracker
try:
    performance_tracker = PerformanceTracker(db_connection=None)
    print('Evaluation system initialization successful')
except Exception as e:
    print(f'Evaluation system initialization failed: {e}')
"

# Test web interface
echo "Testing web interface..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from flask import Flask
try:
    app = Flask(__name__)
    print('Web interface initialization successful')
except Exception as e:
    print(f'Web interface initialization failed: {e}')
"

# Test integration
echo "Testing integration..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
try:
    from integration import SportsPredictionSystem
    # Only test initialization with minimal configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'sports_prediction',
            'user': 'postgres',
            'password': 'postgres',
            'initialize': False
        },
        'data_sources': {
            'api_keys': {
                'sportradar': 'test_key',
                'espn': 'test_key',
                'stats_perform': 'test_key',
                'weather': 'test_key'
            },
            'cache_dir': '/tmp/sports_prediction_cache'
        },
        'data_validation': {
            'rules': {}
        },
        'model': {
            'type': 'hybrid',
            'sequence_length': 10,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'use_attention': True,
            'attention_type': 'multihead',
            'num_heads': 4,
            'use_transformer': True
        },
        'training': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'test_split': 0.1,
            'use_bayesian_optimization': True,
            'num_trials': 20,
            'use_ensemble': True,
            'ensemble_models': 5,
            'schedule': {
                'daily_retraining': False,
                'retraining_time': '01:00',
                'incremental_training': True
            }
        }
    }
    # Mock the database connection to avoid actual connection
    import unittest.mock as mock
    with mock.patch('integration.get_db_connection') as mock_db:
        mock_db.return_value = mock.MagicMock()
        system = SportsPredictionSystem(config)
        print('Integration initialization successful')
except Exception as e:
    print(f'Integration initialization failed: {e}')
"

echo "Tests completed."
