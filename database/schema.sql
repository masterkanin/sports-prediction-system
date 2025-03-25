-- Enhanced Sports Prediction System Database Schema
-- PostgreSQL optimized schema with normalized design

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for versioning
CREATE SCHEMA IF NOT EXISTS versioning;

-- Players table
CREATE TABLE players (
    player_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(100) UNIQUE,  -- ID from external data source
    name VARCHAR(100) NOT NULL,
    position VARCHAR(50),
    birth_date DATE,
    height NUMERIC(5,2),  -- in cm
    weight NUMERIC(5,2),  -- in kg
    team_id UUID,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    attributes JSONB,  -- Flexible storage for additional attributes
    CONSTRAINT fk_team FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE SET NULL
);

-- Create index on player name for faster searches
CREATE INDEX idx_player_name ON players(name);
CREATE INDEX idx_player_team ON players(team_id);
CREATE INDEX idx_player_attributes ON players USING GIN (attributes);

-- Teams table
CREATE TABLE teams (
    team_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(100) UNIQUE,  -- ID from external data source
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10),
    location VARCHAR(100),
    conference VARCHAR(50),
    division VARCHAR(50),
    venue VARCHAR(100),
    founded_year INTEGER,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    team_stats JSONB,  -- Flexible storage for team statistics
    logo_url VARCHAR(255)
);

-- Create index on team name for faster searches
CREATE INDEX idx_team_name ON teams(name);
CREATE INDEX idx_team_conference ON teams(conference);
CREATE INDEX idx_team_division ON teams(division);
CREATE INDEX idx_team_stats ON teams USING GIN (team_stats);

-- Games table
CREATE TABLE games (
    game_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(100) UNIQUE,  -- ID from external data source
    sport VARCHAR(50) NOT NULL,
    season VARCHAR(20) NOT NULL,
    season_type VARCHAR(20) NOT NULL,  -- regular, playoff, etc.
    home_team_id UUID NOT NULL,
    away_team_id UUID NOT NULL,
    game_date DATE NOT NULL,
    scheduled_time TIMESTAMP WITH TIME ZONE,
    venue VARCHAR(100),
    status VARCHAR(20) DEFAULT 'scheduled',  -- scheduled, in_progress, completed, cancelled
    home_score INTEGER,
    away_score INTEGER,
    weather_conditions JSONB,  -- For outdoor sports
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    game_stats JSONB,  -- Flexible storage for additional game stats
    CONSTRAINT fk_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    CONSTRAINT fk_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

-- Create indexes for game queries
CREATE INDEX idx_game_date ON games(game_date);
CREATE INDEX idx_game_sport ON games(sport);
CREATE INDEX idx_game_teams ON games(home_team_id, away_team_id);
CREATE INDEX idx_game_season ON games(season, season_type);
CREATE INDEX idx_game_stats ON games USING GIN (game_stats);

-- Player game stats table
CREATE TABLE player_game_stats (
    stat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID NOT NULL,
    game_id UUID NOT NULL,
    team_id UUID NOT NULL,
    minutes_played INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Common stats across sports
    points INTEGER,
    assists INTEGER,
    rebounds INTEGER,
    
    -- Sport-specific stats stored in JSONB
    stats JSONB,
    
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,
    CONSTRAINT fk_team FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    CONSTRAINT unique_player_game UNIQUE (player_id, game_id)
);

-- Create indexes for player game stats
CREATE INDEX idx_player_game_stats_player ON player_game_stats(player_id);
CREATE INDEX idx_player_game_stats_game ON player_game_stats(game_id);
CREATE INDEX idx_player_game_stats_team ON player_game_stats(team_id);
CREATE INDEX idx_player_game_stats_stats ON player_game_stats USING GIN (stats);

-- Predictions table
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID NOT NULL,
    game_id UUID NOT NULL,
    stat_type VARCHAR(50) NOT NULL,  -- points, rebounds, assists, etc.
    predicted_value NUMERIC(10,2) NOT NULL,
    confidence_score NUMERIC(5,2) NOT NULL,  -- 0-100 scale
    prediction_range_low NUMERIC(10,2),
    prediction_range_high NUMERIC(10,2),
    over_probability NUMERIC(5,4) NOT NULL,  -- 0-1 scale
    line_value NUMERIC(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    top_factors JSONB,  -- Key factors that influenced the prediction
    feature_importance JSONB,  -- Importance of each feature in the prediction
    
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- Create indexes for predictions
CREATE INDEX idx_predictions_player ON predictions(player_id);
CREATE INDEX idx_predictions_game ON predictions(game_id);
CREATE INDEX idx_predictions_stat ON predictions(stat_type);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score);
CREATE INDEX idx_predictions_model ON predictions(model_version);
CREATE INDEX idx_predictions_created ON predictions(created_at);
CREATE INDEX idx_predictions_factors ON predictions USING GIN (top_factors);

-- Actual results table
CREATE TABLE actual_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID NOT NULL,
    game_id UUID NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    actual_value NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,
    CONSTRAINT unique_player_game_stat UNIQUE (player_id, game_id, stat_type)
);

-- Create indexes for actual results
CREATE INDEX idx_results_player ON actual_results(player_id);
CREATE INDEX idx_results_game ON actual_results(game_id);
CREATE INDEX idx_results_stat ON actual_results(stat_type);

-- Model versions table
CREATE TABLE model_versions (
    version_id VARCHAR(50) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- hybrid, lstm, transformer, etc.
    description TEXT,
    hyperparameters JSONB NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    training_data_range JSONB,  -- Start and end dates of training data
    validation_metrics JSONB,  -- Accuracy, MSE, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),
    active BOOLEAN DEFAULT FALSE
);

-- Create index on model versions
CREATE INDEX idx_model_versions_type ON model_versions(model_type);
CREATE INDEX idx_model_versions_active ON model_versions(active);

-- Data sources table
CREATE TABLE data_sources (
    source_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    api_endpoint VARCHAR(255),
    credentials JSONB,
    last_sync TIMESTAMP WITH TIME ZONE,
    sync_frequency VARCHAR(50),  -- daily, hourly, etc.
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on data sources
CREATE INDEX idx_data_sources_name ON data_sources(name);
CREATE INDEX idx_data_sources_active ON data_sources(active);

-- Player embeddings table
CREATE TABLE player_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    embedding_vector FLOAT[] NOT NULL,  -- Store embedding as array
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    CONSTRAINT fk_model_version FOREIGN KEY (model_version) REFERENCES model_versions(version_id) ON DELETE CASCADE,
    CONSTRAINT unique_player_model UNIQUE (player_id, model_version)
);

-- Create index on player embeddings
CREATE INDEX idx_player_embeddings_player ON player_embeddings(player_id);
CREATE INDEX idx_player_embeddings_model ON player_embeddings(model_version);

-- Team chemistry table
CREATE TABLE team_chemistry (
    chemistry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID NOT NULL,
    lineup JSONB NOT NULL,  -- Array of player_ids in the lineup
    chemistry_score NUMERIC(5,2) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_team FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    CONSTRAINT fk_model_version FOREIGN KEY (model_version) REFERENCES model_versions(version_id) ON DELETE CASCADE
);

-- Create index on team chemistry
CREATE INDEX idx_team_chemistry_team ON team_chemistry(team_id);
CREATE INDEX idx_team_chemistry_lineup ON team_chemistry USING GIN (lineup);
CREATE INDEX idx_team_chemistry_model ON team_chemistry(model_version);

-- Prediction performance table
CREATE TABLE prediction_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    sport VARCHAR(50) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    date_range JSONB NOT NULL,  -- Start and end dates
    total_predictions INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    accuracy NUMERIC(5,4) NOT NULL,
    mean_absolute_error NUMERIC(10,4),
    root_mean_squared_error NUMERIC(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_model_version FOREIGN KEY (model_version) REFERENCES model_versions(version_id) ON DELETE CASCADE
);

-- Create index on prediction performance
CREATE INDEX idx_prediction_performance_model ON prediction_performance(model_version);
CREATE INDEX idx_prediction_performance_sport ON prediction_performance(sport);
CREATE INDEX idx_prediction_performance_stat ON prediction_performance(stat_type);

-- A/B testing table
CREATE TABLE ab_testing (
    test_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    model_a VARCHAR(50) NOT NULL,
    model_b VARCHAR(50) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',  -- active, completed, cancelled
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_model_a FOREIGN KEY (model_a) REFERENCES model_versions(version_id) ON DELETE CASCADE,
    CONSTRAINT fk_model_b FOREIGN KEY (model_b) REFERENCES model_versions(version_id) ON DELETE CASCADE
);

-- Create index on A/B testing
CREATE INDEX idx_ab_testing_name ON ab_testing(name);
CREATE INDEX idx_ab_testing_status ON ab_testing(status);
CREATE INDEX idx_ab_testing_models ON ab_testing(model_a, model_b);

-- Audit log table for tracking changes
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(50) NOT NULL,
    record_id UUID NOT NULL,
    action VARCHAR(10) NOT NULL,  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on audit log
CREATE INDEX idx_audit_log_table ON audit_log(table_name);
CREATE INDEX idx_audit_log_record ON audit_log(record_id);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_changed_at ON audit_log(changed_at);

-- Versioning tables for historical data
CREATE TABLE versioning.players AS TABLE players WITH NO DATA;
CREATE TABLE versioning.teams AS TABLE teams WITH NO DATA;
CREATE TABLE versioning.predictions AS TABLE predictions WITH NO DATA;
CREATE TABLE versioning.player_game_stats AS TABLE player_game_stats WITH NO DATA;

-- Add version columns to versioning tables
ALTER TABLE versioning.players ADD COLUMN version_id SERIAL;
ALTER TABLE versioning.players ADD COLUMN valid_from TIMESTAMP WITH TIME ZONE;
ALTER TABLE versioning.players ADD COLUMN valid_to TIMESTAMP WITH TIME ZONE;

ALTER TABLE versioning.teams ADD COLUMN version_id SERIAL;
ALTER TABLE versioning.teams ADD COLUMN valid_from TIMESTAMP WITH TIME ZONE;
ALTER TABLE versioning.teams ADD COLUMN valid_to TIMESTAMP WITH TIME ZONE;

ALTER TABLE versioning.predictions ADD COLUMN version_id SERIAL;
ALTER TABLE versioning.predictions ADD COLUMN valid_from TIMESTAMP WITH TIME ZONE;
ALTER TABLE versioning.predictions ADD COLUMN valid_to TIMESTAMP WITH TIME ZONE;

ALTER TABLE versioning.player_game_stats ADD COLUMN version_id SERIAL;
ALTER TABLE versioning.player_game_stats ADD COLUMN valid_from TIMESTAMP WITH TIME ZONE;
ALTER TABLE versioning.player_game_stats ADD COLUMN valid_to TIMESTAMP WITH TIME ZONE;

-- Create indexes on versioning tables
CREATE INDEX idx_versioning_players_id ON versioning.players(player_id);
CREATE INDEX idx_versioning_players_valid ON versioning.players(valid_from, valid_to);

CREATE INDEX idx_versioning_teams_id ON versioning.teams(team_id);
CREATE INDEX idx_versioning_teams_valid ON versioning.teams(valid_from, valid_to);

CREATE INDEX idx_versioning_predictions_id ON versioning.predictions(prediction_id);
CREATE INDEX idx_versioning_predictions_valid ON versioning.predictions(valid_from, valid_to);

CREATE INDEX idx_versioning_player_game_stats_id ON versioning.player_game_stats(stat_id);
CREATE INDEX idx_versioning_player_game_stats_valid ON versioning.player_game_stats(valid_from, valid_to);

-- Create functions for versioning
CREATE OR REPLACE FUNCTION versioning.create_player_version()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'UPDATE') THEN
        INSERT INTO versioning.players 
        SELECT old.*, nextval('versioning.players_version_id_seq'), old.created_at, now()
        FROM old;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION versioning.create_team_version()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'UPDATE') THEN
        INSERT INTO versioning.teams 
        SELECT old.*, nextval('versioning.teams_version_id_seq'), old.created_at, now()
        FROM old;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION versioning.create_prediction_version()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'UPDATE') THEN
        INSERT INTO versioning.predictions 
        SELECT old.*, nextval('versioning.predictions_version_id_seq'), old.created_at, now()
        FROM old;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION versioning.create_player_game_stats_version()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'UPDATE') THEN
        INSERT INTO versioning.player_game_stats 
        SELECT old.*, nextval('versioning.player_game_stats_version_id_seq'), old.created_at, now()
        FROM old;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for versioning
CREATE TRIGGER player_versioning
BEFORE UPDATE ON players
FOR EACH ROW EXECUTE FUNCTION versioning.create_player_version();

CREATE TRIGGER team_versioning
BEFORE UPDATE ON teams
FOR EACH ROW EXECUTE FUNCTION versioning.create_team_version();

CREATE TRIGGER prediction_versioning
BEFORE UPDATE ON predictions
FOR EACH ROW EXECUTE FUNCTION versioning.create_prediction_version();

CREATE TRIGGER player_game_stats_versioning
BEFORE UPDATE ON player_game_stats
FOR EACH ROW EXECUTE FUNCTION versioning.create_player_game_stats_version();

-- Create function for audit logging
CREATE OR REPLACE FUNCTION log_audit()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, changed_by)
        VALUES (TG_TABLE_NAME, OLD.player_id, 'DELETE', row_to_json(OLD), current_user);
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.player_id, 'UPDATE', row_to_json(OLD), row_to_json(NEW), current_user);
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO audit_log (table_name, record_id, action, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.player_id, 'INSERT', row_to_json(NEW), current_user);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for main tables
CREATE TRIGGER players_audit
AFTER INSERT OR UPDATE OR DELETE ON players
FOR EACH ROW EXECUTE FUNCTION log_audit();

CREATE TRIGGER teams_audit
AFTER INSERT OR UPDATE OR DELETE ON teams
FOR EACH ROW EXECUTE FUNCTION log_audit();

CREATE TRIGGER predictions_audit
AFTER INSERT OR UPDATE OR DELETE ON predictions
FOR EACH ROW EXECUTE FUNCTION log_audit();

CREATE TRIGGER player_game_stats_audit
AFTER INSERT OR UPDATE OR DELETE ON player_game_stats
FOR EACH ROW EXECUTE FUNCTION log_audit();

-- Create views for common queries
CREATE VIEW upcoming_games_view AS
SELECT 
    g.game_id,
    g.sport,
    g.game_date,
    g.scheduled_time,
    g.venue,
    ht.name AS home_team,
    at.name AS away_team,
    g.weather_conditions
FROM 
    games g
JOIN 
    teams ht ON g.home_team_id = ht.team_id
JOIN 
    teams at ON g.away_team_id = at.team_id
WHERE 
    g.game_date >= CURRENT_DATE
    AND g.status = 'scheduled'
ORDER BY 
    g.scheduled_time;

CREATE VIEW high_confidence_predictions_view AS
SELECT 
    p.prediction_id,
    pl.name AS player_name,
    t.name AS team_name,
    g.game_date,
    g.scheduled_time,
    p.stat_type,
    p.predicted_value,
    p.line_value,
    p.over_probability,
    p.confidence_score,
    p.prediction_range_low,
    p.prediction_range_high,
    p.top_factors,
    ht.name AS home_team,
    at.name AS away_team
FROM 
    predictions p
JOIN 
    players pl ON p.player_id = pl.player_id
JOIN 
    teams t ON pl.team_id = t.team_id
JOIN 
    games g ON p.game_id = g.game_id
JOIN 
    teams ht ON g.home_team_id = ht.team_id
JOIN 
    teams at ON g.away_team_id = at.team_id
WHERE 
    p.confidence_score >= 75
    AND g.game_date >= CURRENT_DATE
ORDER BY 
    p.confidence_score DESC, g.scheduled_time;

CREATE VIEW prediction_accuracy_view AS
SELECT 
    p.model_version,
    p.stat_type,
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN (p.over_probability > 0.5 AND ar.actual_value > p.line_value) OR 
               (p.over_probability <= 0.5 AND ar.actual_value <= p.line_value) 
        THEN 1 ELSE 0 END) AS correct_predictions,
    ROUND(SUM(CASE WHEN (p.over_probability > 0.5 AND ar.actual_value > p.line_value) OR 
                    (p.over_probability <= 0.5 AND ar.actual_value <= p.line_value) 
             THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) AS accuracy_percentage,
    ROUND(AVG(ABS(p.predicted_value - ar.actual_value)), 2) AS mean_absolute_error,
    ROUND(SQRT(AVG(POWER(p.predicted_value - ar.actual_value, 2))), 2) AS root_mean_squared_error
FROM 
    predictions p
JOIN 
    actual_results ar ON p.player_id = ar.player_id AND p.game_id = ar.game_id AND p.stat_type = ar.stat_type
JOIN 
    games g ON p.game_id = g.game_id
WHERE 
    g.status = 'completed'
GROUP BY 
    p.model_version, p.stat_type
ORDER BY 
    p.model_version, p.stat_type;

-- Create materialized view for player performance trends
CREATE MATERIALIZED VIEW player_performance_trends AS
SELECT 
    pgs.player_id,
    p.name AS player_name,
    t.name AS team_name,
    pgs.stat_type,
    g.game_date,
    pgs.actual_value,
    AVG(pgs.actual_value) OVER (
        PARTITION BY pgs.player_id, pgs.stat_type 
        ORDER BY g.game_date 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_5_games,
    AVG(pgs.actual_value) OVER (
        PARTITION BY pgs.player_id, pgs.stat_type 
        ORDER BY g.game_date 
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_10_games,
    AVG(pgs.actual_value) OVER (
        PARTITION BY pgs.player_id, pgs.stat_type
    ) AS season_avg
FROM 
    actual_results pgs
JOIN 
    players p ON pgs.player_id = p.player_id
JOIN 
    teams t ON p.team_id = t.team_id
JOIN 
    games g ON pgs.game_id = g.game_id
ORDER BY 
    pgs.player_id, pgs.stat_type, g.game_date;

-- Create index on materialized view
CREATE INDEX idx_player_performance_trends_player ON player_performance_trends(player_id);
CREATE INDEX idx_player_performance_trends_stat ON player_performance_trends(stat_type);
CREATE INDEX idx_player_performance_trends_date ON player_performance_trends(game_date);

-- Create refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_player_performance_trends()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY player_performance_trends;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to refresh materialized view
CREATE TRIGGER refresh_player_performance_trends_trigger
AFTER INSERT OR UPDATE ON actual_results
FOR EACH STATEMENT EXECUTE FUNCTION refresh_player_performance_trends();
