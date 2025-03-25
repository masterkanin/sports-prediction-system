import os

def get_db_connection(host=None, port=None, dbname=None, user=None, password=None):
    """
    Get a database connection using either provided parameters or environment variables.
    This is configured to work with Railway's PostgreSQL deployment.
    
    Returns:
        connection: PostgreSQL database connection
    """
    import psycopg2
    
    # Check if we're running on Railway
    if 'RAILWAY_ENVIRONMENT' in os.environ or 'PGHOST' in os.environ:
        # Use Railway's environment variables
        connection = psycopg2.connect(
            host=os.environ.get('PGHOST', 'localhost'),
            port=os.environ.get('PGPORT', '5432'),
            dbname=os.environ.get('PGDATABASE', 'railway'),
            user=os.environ.get('PGUSER', 'postgres'),
            password=os.environ.get('PGPASSWORD', '')
        )
    else:
        # Use provided parameters or defaults
        connection = psycopg2.connect(
            host=host or 'localhost',
            port=port or '5432',
            dbname=dbname or 'sports_prediction',
            user=user or 'postgres',
            password=password or 'postgres'
        )
    
    return connection

def initialize_database(connection):
    """
    Initialize the database schema.
    
    Args:
        connection: PostgreSQL database connection
    """
    # Read schema from file
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
    else:
        # Fallback to hardcoded schema if file doesn't exist
        schema_sql = """
        -- Players table
        CREATE TABLE IF NOT EXISTS players (
            player_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            position VARCHAR(20),
            team_id VARCHAR(50),
            attributes JSONB
        );

        -- Teams table
        CREATE TABLE IF NOT EXISTS teams (
            team_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            location VARCHAR(100),
            conference VARCHAR(50),
            division VARCHAR(50),
            team_stats JSONB
        );

        -- Games table
        CREATE TABLE IF NOT EXISTS games (
            game_id VARCHAR(50) PRIMARY KEY,
            home_team_id VARCHAR(50),
            away_team_id VARCHAR(50),
            date TIMESTAMP NOT NULL,
            venue VARCHAR(100),
            weather_conditions JSONB,
            game_stats JSONB,
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
        );

        -- PlayerGameStats table
        CREATE TABLE IF NOT EXISTS player_game_stats (
            stat_id SERIAL PRIMARY KEY,
            player_id VARCHAR(50),
            game_id VARCHAR(50),
            minutes_played INTEGER,
            points INTEGER,
            rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            three_pointers_made INTEGER,
            three_pointers_attempted INTEGER,
            free_throws_made INTEGER,
            free_throws_attempted INTEGER,
            other_stats JSONB,
            UNIQUE (player_id, game_id),
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );

        -- Predictions table
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id SERIAL PRIMARY KEY,
            player_id VARCHAR(50),
            game_id VARCHAR(50),
            stat_type VARCHAR(50) NOT NULL,
            predicted_value FLOAT NOT NULL,
            confidence_score FLOAT NOT NULL,
            prediction_range_low FLOAT NOT NULL,
            prediction_range_high FLOAT NOT NULL,
            over_probability FLOAT NOT NULL,
            line_value FLOAT,
            actual_value FLOAT,
            over_under_result BOOLEAN,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP,
            model_version VARCHAR(50) NOT NULL,
            UNIQUE (player_id, game_id, stat_type, model_version),
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );

        -- ModelVersions table
        CREATE TABLE IF NOT EXISTS model_versions (
            version VARCHAR(50) PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            accuracy FLOAT,
            mae FLOAT,
            rmse FLOAT,
            over_under_accuracy FLOAT,
            parameters JSONB
        );
        """
    
    # Execute schema
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)
    
    connection.commit()
