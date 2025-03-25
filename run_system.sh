#!/bin/bash

# This script sets up and runs the enhanced sports prediction system

echo "Setting up the enhanced sports prediction system..."

# Create required directories
mkdir -p /tmp/sports_prediction_cache
mkdir -p /home/ubuntu/enhanced_sports_prediction/logs

# Install required packages
echo "Installing required packages..."
pip install -q numpy pandas scikit-learn tensorflow torch flask flask-wtf psycopg2-binary plotly sqlalchemy pytest

# Set up PostgreSQL (if not already installed)
if ! command -v psql &> /dev/null; then
    echo "Installing PostgreSQL..."
    sudo apt-get update
    sudo apt-get install -y postgresql postgresql-contrib
fi

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
sudo service postgresql start

# Create database and user (if they don't exist)
echo "Setting up database..."
sudo -u postgres psql -c "SELECT 1 FROM pg_database WHERE datname = 'sports_prediction'" | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE sports_prediction"
sudo -u postgres psql -c "SELECT 1 FROM pg_roles WHERE rolname = 'sports_prediction'" | grep -q 1 || sudo -u postgres psql -c "CREATE USER sports_prediction WITH PASSWORD 'sports_prediction'"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sports_prediction TO sports_prediction"

# Run database initialization script
echo "Initializing database schema..."
python3 -c "
import sys
sys.path.append('/home/ubuntu/enhanced_sports_prediction')
from database.db_config import initialize_database, get_db_connection
conn = get_db_connection(
    host='localhost',
    port=5432,
    dbname='sports_prediction',
    user='postgres',
    password='postgres'
)
initialize_database(conn)
conn.close()
"

# Run tests
echo "Running tests..."
bash /home/ubuntu/enhanced_sports_prediction/run_tests.sh

# Start the web interface
echo "Starting web interface..."
cd /home/ubuntu/enhanced_sports_prediction/web_interface
python3 app.py &
WEB_PID=$!

echo "Sports prediction system is now running!"
echo "Web interface available at: http://localhost:5000"
echo "Press Ctrl+C to stop the system"

# Wait for user to press Ctrl+C
trap "kill $WEB_PID; echo 'Stopping sports prediction system...'; exit 0" INT
wait $WEB_PID
