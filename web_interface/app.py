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

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'enhanced-sports-prediction-system')

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Define forms
class PredictionFilterForm(FlaskForm):
    sport = SelectField('Sport', choices=[
        ('all', 'All Sports'),
        ('nba', 'NBA Basketball'),
        ('nfl', 'NFL Football'),
        ('mlb', 'MLB Baseball'),
        ('nhl', 'NHL Hockey'),
        ('soccer', 'Soccer')
    ])
    date = DateField('Date', validators=[Optional()], format='%Y-%m-%d')
    confidence = SelectField('Confidence Level', choices=[
        ('all', 'All Confidence Levels'),
        ('high', 'High (>80%)'),
        ('medium', 'Medium (50-80%)'),
        ('low', 'Low (<50%)')
    ])
    submit = SubmitField('Filter')

# Mock data for demonstration
def get_mock_predictions():
    sports = ['nba', 'nfl', 'mlb', 'nhl', 'soccer']
    players = [
        'LeBron James', 'Kevin Durant', 'Stephen Curry', 'Giannis Antetokounmpo',
        'Patrick Mahomes', 'Tom Brady', 'Aaron Rodgers', 'Travis Kelce',
        'Mike Trout', 'Shohei Ohtani', 'Aaron Judge', 'Mookie Betts',
        'Connor McDavid', 'Sidney Crosby', 'Alex Ovechkin', 'Auston Matthews',
        'Lionel Messi', 'Cristiano Ronaldo', 'Kylian Mbappé', 'Erling Haaland'
    ]
    stats = ['points', 'rebounds', 'assists', 'touchdowns', 'passing_yards', 'rushing_yards',
             'hits', 'home_runs', 'RBIs', 'goals', 'assists', 'shots']
    
    predictions = []
    for _ in range(50):
        sport = np.random.choice(sports)
        player = np.random.choice(players)
        stat = np.random.choice(stats)
        predicted_value = np.random.randint(5, 40)
        confidence = np.random.uniform(0.3, 0.95)
        over_probability = np.random.uniform(0.1, 0.9)
        line_value = predicted_value * (1 + np.random.uniform(-0.2, 0.2))
        
        predictions.append({
            'sport': sport,
            'player': player,
            'stat': stat,
            'predicted_value': predicted_value,
            'confidence': confidence,
            'prediction_range_low': predicted_value * (1 - (1 - confidence) / 2),
            'prediction_range_high': predicted_value * (1 + (1 - confidence) / 2),
            'over_probability': over_probability,
            'line_value': line_value,
            'date': (datetime.now() + timedelta(days=np.random.randint(-3, 4))).strftime('%Y-%m-%d')
        })
    
    return predictions

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    form = PredictionFilterForm()
    all_predictions = get_mock_predictions()
    
    if form.validate_on_submit():
        sport = form.sport.data
        date = form.date.data
        confidence = form.confidence.data
        
        filtered_predictions = all_predictions
        
        if sport != 'all':
            filtered_predictions = [p for p in filtered_predictions if p['sport'] == sport]
        
        if date:
            date_str = date.strftime('%Y-%m-%d')
            filtered_predictions = [p for p in filtered_predictions if p['date'] == date_str]
        
        if confidence != 'all':
            if confidence == 'high':
                filtered_predictions = [p for p in filtered_predictions if p['confidence'] > 0.8]
            elif confidence == 'medium':
                filtered_predictions = [p for p in filtered_predictions if 0.5 <= p['confidence'] <= 0.8]
            elif confidence == 'low':
                filtered_predictions = [p for p in filtered_predictions if p['confidence'] < 0.5]
    else:
        filtered_predictions = all_predictions
    
    return render_template('predictions.html', predictions=filtered_predictions, form=form)

@app.route('/performance')
def performance():
    # Create mock performance data
    dates = pd.date_range(start='2025-01-01', end='2025-03-25')
    accuracy = [0.65 + 0.01 * i + np.random.uniform(-0.05, 0.05) for i in range(len(dates))]
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='Prediction Accuracy'))
    fig.update_layout(
        title='Prediction Accuracy Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0.5, 1.0])
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('performance.html', graphJSON=graphJSON)

@app.route('/analysis')
def analysis():
    # Create mock data for analysis
    sports = ['NBA', 'NFL', 'MLB', 'NHL', 'Soccer']
    accuracy = [0.78, 0.72, 0.75, 0.71, 0.68]
    
    # Create plotly figure
    fig = go.Figure(data=[go.Bar(x=sports, y=accuracy)])
    fig.update_layout(
        title='Prediction Accuracy by Sport',
        xaxis_title='Sport',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0.5, 1.0])
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('analysis.html', graphJSON=graphJSON)

@app.route('/features')
def features():
    # Create mock data for feature importance
    features = ['Recent Performance', 'Season Average', 'Matchup History', 
                'Rest Days', 'Home/Away', 'Team Quality', 'Weather', 'Injuries']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    
    # Create plotly figure
    fig = go.Figure(data=[go.Bar(x=importance, y=features, orientation='h')])
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('features.html', graphJSON=graphJSON)

@app.route('/anomalies')
def anomalies():
    # Create mock data for anomalies
    players = ['LeBron James', 'Kevin Durant', 'Patrick Mahomes', 'Mike Trout', 'Connor McDavid']
    expected = [25, 28, 300, 0.320, 1.5]
    actual = [42, 12, 450, 0.180, 3.0]
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=players, y=expected, name='Expected'))
    fig.add_trace(go.Bar(x=players, y=actual, name='Actual'))
    fig.update_layout(
        title='Anomaly Detection: Expected vs Actual Performance',
        xaxis_title='Player',
        yaxis_title='Performance Metric',
        barmode='group'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('anomalies.html', graphJSON=graphJSON)

@app.route('/optimization')
def optimization():
    # Create mock data for optimization
    iterations = list(range(1, 21))
    accuracy = [0.65 + 0.01 * i - 0.005 * i**2 + 0.0001 * i**3 + np.random.uniform(-0.01, 0.01) for i in iterations]
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=accuracy, mode='lines+markers', name='Optimization Progress'))
    fig.update_layout(
        title='Model Optimization Progress',
        xaxis_title='Iteration',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0.6, 0.8])
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('optimization.html', graphJSON=graphJSON)

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """API endpoint to get predictions in JSON format"""
    sport = request.args.get('sport', 'all')
    date = request.args.get('date', None)
    
    all_predictions = get_mock_predictions()
    filtered_predictions = all_predictions
    
    if sport != 'all':
        filtered_predictions = [p for p in filtered_predictions if p['sport'] == sport]
    
    if date:
        filtered_predictions = [p for p in filtered_predictions if p['date'] == date]
    
    return jsonify(predictions=filtered_predictions)

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify(status="healthy", timestamp=datetime.now().isoformat())

# Initialize database if needed - REMOVED before_first_request decorator which is not supported in Flask 2.3+
# Instead, we'll use a simple function that can be called manually if needed
def initialize_db():
    try:
        # Import here to avoid circular imports
        from database.railway_db_config import get_db_connection, initialize_database
        conn = get_db_connection()
        initialize_database(conn)
        conn.close()
        app.logger.info("Database initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize database: {e}")
        # Continue without database since we're using mock data

# Make the app importable for gunicorn
if __name__ == '__main__':
    # Use environment variables for host and port if available (for Railway)
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host=host, port=port)
