import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, SubmitField, RadioField
from wtforms.validators import DataRequired, Optional
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database configuration for Railway
try:
    from database.railway_db_config import get_db_connection
except ImportError:
    # Fallback to regular db_config if railway_db_config is not available
    from database.db_config import get_db_connection

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'enhanced-sports-prediction-system')

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Rest of the app.py code remains the same
# ...

# Make the app importable for gunicorn
if __name__ == '__main__':
    # Use environment variables for host and port if available (for Railway)
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host=host, port=port)
