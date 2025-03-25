# Enhanced Sports Prediction System - README

## Project Overview

This repository contains an enhanced sports prediction system designed to provide accurate predictions for player performance across multiple sports (NBA, NFL, MLB, NHL, soccer). The system uses advanced neural network architectures, optimized database structures, and improved data pipelines to generate predictions with confidence intervals and over/under probabilities.

## Key Features

- **Hybrid Neural Network Architecture**: Combines LSTM/Transformer networks for sequential data with feedforward networks for static features
- **Multi-task Learning**: Predicts both exact statistical values and over/under probabilities
- **Attention Mechanisms**: Focuses on the most relevant historical games
- **PostgreSQL Database**: Optimized schema design with JSON/JSONB support
- **Multi-source Data Collection**: Integrates data from Sportradar, ESPN, Stats Perform, and more
- **Automated Model Training**: Daily retraining with Bayesian hyperparameter optimization
- **Comprehensive Evaluation**: Tracks prediction accuracy and analyzes errors
- **User-friendly Web Interface**: Displays predictions with filtering and visualization capabilities

## Directory Structure

```
enhanced_sports_prediction/
├── neural_network/               # Neural network architecture
│   ├── model_architecture.py     # Hybrid model implementation
│   ├── feature_engineering.py    # Feature engineering components
│   ├── multi_task_learning.py    # Multi-task learning implementation
│   └── attention_mechanisms.py   # Attention mechanisms
├── database/                     # Database structure
│   ├── schema.sql                # SQL schema definition
│   ├── db_config.py              # Database configuration
│   ├── migration.py              # Migration scripts
│   └── versioning.py             # Data versioning implementation
├── data_pipeline/                # Data pipeline
│   ├── data_sources.py           # Multi-source data collection
│   ├── feature_computation.py    # Feature computation pipeline
│   ├── data_validation.py        # Data validation and cleaning
│   └── automated_training.py     # Automated model training
├── evaluation/                   # Evaluation system
│   ├── performance_tracking.py   # Performance tracking
│   ├── error_analysis.py         # Error analysis
│   ├── feature_importance.py     # Feature importance analysis
│   └── continuous_improvement.py # Continuous improvement mechanisms
├── web_interface/                # Web interface
│   ├── app.py                    # Flask application
│   ├── templates/                # HTML templates
│   └── static/                   # Static files (CSS, JS, images)
├── integration.py                # System integration
├── run_tests.sh                  # Test script
├── run_system.sh                 # System startup script
└── documentation.md              # Comprehensive documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Node.js 14+ (for web interface development)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-organization/enhanced-sports-prediction.git
cd enhanced-sports-prediction
```

2. Run the setup script:
```bash
bash run_system.sh
```

This will:
- Install required packages
- Set up PostgreSQL database
- Initialize the database schema
- Run tests
- Start the web interface

### Manual Setup

1. Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow torch flask flask-wtf psycopg2-binary plotly sqlalchemy pytest
```

2. Set up PostgreSQL database:
```bash
sudo -u postgres psql -c "CREATE DATABASE sports_prediction"
sudo -u postgres psql -c "CREATE USER sports_prediction WITH PASSWORD 'sports_prediction'"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sports_prediction TO sports_prediction"
```

3. Initialize the database schema:
```bash
python -c "from database.db_config import initialize_database, get_db_connection; initialize_database(get_db_connection())"
```

4. Set up environment variables:
```bash
export SPORTRADAR_API_KEY=your_sportradar_api_key
export ESPN_API_KEY=your_espn_api_key
export STATS_PERFORM_API_KEY=your_stats_perform_api_key
export WEATHER_API_KEY=your_weather_api_key
```

5. Run tests:
```bash
bash run_tests.sh
```

6. Start the web interface:
```bash
cd web_interface
python app.py
```

## Usage

### Web Interface

The web interface is available at `http://localhost:5000` when the system is running. It provides:

- Dashboard with summary statistics and visualizations
- Prediction filtering by sport, date, confidence level, etc.
- Performance analysis with interactive charts
- Feature importance visualization
- Anomaly detection and analysis

### API Usage

To use the system programmatically:

```python
from integration import SportsPredictionSystem

# Initialize the system
system = SportsPredictionSystem()

# Run the complete pipeline
results = system.run_pipeline(
    sports=['nba', 'nfl', 'mlb', 'nhl', 'soccer'],
    start_date='2025-03-17',
    end_date='2025-03-24'
)

# Generate predictions for upcoming games
from data_pipeline.data_sources import DataSourceManager
data_source_manager = DataSourceManager()
upcoming_games = data_source_manager.get_upcoming_games(['nba'])
predictions = system.generate_predictions(upcoming_games)

# Close the system when done
system.close()
```

## Documentation

For detailed documentation, see [documentation.md](documentation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sportradar API for providing sports data
- ESPN API for additional player and team information
- Stats Perform for advanced metrics
- The open-source community for various libraries and tools used in this project
