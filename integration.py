import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from other modules
from neural_network.model_architecture import HybridSportsPredictionModel
from neural_network.feature_engineering import FeatureEngineering
from neural_network.multi_task_learning import MultiTaskLearningModel
from neural_network.attention_mechanisms import AttentionMechanism
from data_pipeline.data_sources import DataSourceManager
from data_pipeline.feature_computation import FeatureComputation
from data_pipeline.data_validation import DataValidator
from data_pipeline.automated_training import AutomatedTrainingPipeline
from evaluation.performance_tracking import PerformanceTracker
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.feature_importance import FeatureImportanceAnalyzer
from evaluation.continuous_improvement import ContinuousImprovementSystem
from database.db_config import get_db_connection, initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SportsPredictionSystem:
    """
    Main class that integrates all components of the enhanced sports prediction system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the sports prediction system with all its components.
        
        Args:
            config (dict, optional): Configuration parameters for the system.
        """
        logger.info("Initializing Sports Prediction System")
        
        self.config = config or self._get_default_config()
        self.start_time = datetime.now()
        
        # Initialize database
        logger.info("Initializing database connection")
        self.db_connection = get_db_connection(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            dbname=self.config['database']['dbname'],
            user=self.config['database']['user'],
            password=self.config['database']['password']
        )
        
        # Initialize database schema if needed
        if self.config['database']['initialize']:
            logger.info("Initializing database schema")
            initialize_database(self.db_connection)
        
        # Initialize components
        logger.info("Initializing feature engineering")
        self.feature_engineering = FeatureEngineering()
        
        logger.info("Initializing data source manager")
        self.data_source_manager = DataSourceManager(
            api_keys=self.config['data_sources']['api_keys'],
            cache_dir=self.config['data_sources']['cache_dir']
        )
        
        logger.info("Initializing feature computation")
        self.feature_computation = FeatureComputation(
            db_connection=self.db_connection,
            feature_engineering=self.feature_engineering
        )
        
        logger.info("Initializing data validator")
        self.data_validator = DataValidator(
            validation_rules=self.config['data_validation']['rules']
        )
        
        logger.info("Initializing neural network model")
        self.model = self._initialize_model()
        
        logger.info("Initializing performance tracker")
        self.performance_tracker = PerformanceTracker(
            db_connection=self.db_connection
        )
        
        logger.info("Initializing error analyzer")
        self.error_analyzer = ErrorAnalyzer(
            db_connection=self.db_connection
        )
        
        logger.info("Initializing feature importance analyzer")
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(
            db_connection=self.db_connection
        )
        
        logger.info("Initializing continuous improvement system")
        self.continuous_improvement = ContinuousImprovementSystem(
            db_connection=self.db_connection,
            model=self.model,
            performance_tracker=self.performance_tracker
        )
        
        logger.info("Initializing automated training pipeline")
        self.training_pipeline = AutomatedTrainingPipeline(
            db_connection=self.db_connection,
            data_source_manager=self.data_source_manager,
            feature_computation=self.feature_computation,
            data_validator=self.data_validator,
            model=self.model,
            performance_tracker=self.performance_tracker,
            config=self.config['training']
        )
        
        logger.info("Sports Prediction System initialized successfully")
    
    def _get_default_config(self):
        """
        Get default configuration for the system.
        
        Returns:
            dict: Default configuration parameters.
        """
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'dbname': 'sports_prediction',
                'user': 'postgres',
                'password': 'postgres',
                'initialize': True
            },
            'data_sources': {
                'api_keys': {
                    'sportradar': os.environ.get('SPORTRADAR_API_KEY', ''),
                    'espn': os.environ.get('ESPN_API_KEY', ''),
                    'stats_perform': os.environ.get('STATS_PERFORM_API_KEY', ''),
                    'weather': os.environ.get('WEATHER_API_KEY', '')
                },
                'cache_dir': '/tmp/sports_prediction_cache'
            },
            'data_validation': {
                'rules': {
                    'player_stats': {
                        'min_values': {'minutes_played': 0, 'points': 0, 'rebounds': 0, 'assists': 0},
                        'max_values': {'minutes_played': 60, 'points': 100, 'rebounds': 40, 'assists': 30}
                    }
                }
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
                    'daily_retraining': True,
                    'retraining_time': '01:00',  # 1 AM
                    'incremental_training': True
                }
            }
        }
    
    def _initialize_model(self):
        """
        Initialize the neural network model based on configuration.
        
        Returns:
            HybridSportsPredictionModel: The initialized model.
        """
        model_config = self.config['model']
        
        # Create attention mechanism if specified
        attention = None
        if model_config['use_attention']:
            attention = AttentionMechanism(
                attention_type=model_config['attention_type'],
                num_heads=model_config['num_heads']
            )
        
        # Create base model
        base_model = HybridSportsPredictionModel(
            sequence_length=model_config['sequence_length'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            attention=attention,
            use_transformer=model_config['use_transformer']
        )
        
        # Wrap with multi-task learning if needed
        model = MultiTaskLearningModel(
            base_model=base_model,
            tasks=['regression', 'classification'],
            uncertainty_estimation=True
        )
        
        return model
    
    def collect_data(self, sports=None, start_date=None, end_date=None):
        """
        Collect data from all configured data sources.
        
        Args:
            sports (list, optional): List of sports to collect data for.
            start_date (str, optional): Start date for data collection (YYYY-MM-DD).
            end_date (str, optional): End date for data collection (YYYY-MM-DD).
            
        Returns:
            dict: Collected data organized by sport and data type.
        """
        logger.info(f"Collecting data for sports: {sports}, from {start_date} to {end_date}")
        
        sports = sports or ['nba', 'nfl', 'mlb', 'nhl', 'soccer']
        
        collected_data = {}
        for sport in sports:
            logger.info(f"Collecting data for {sport}")
            
            # Collect player data
            player_data = self.data_source_manager.get_players(sport)
            
            # Collect team data
            team_data = self.data_source_manager.get_teams(sport)
            
            # Collect game data
            game_data = self.data_source_manager.get_games(
                sport, start_date=start_date, end_date=end_date
            )
            
            # Collect player game stats
            player_game_stats = self.data_source_manager.get_player_game_stats(
                sport, start_date=start_date, end_date=end_date
            )
            
            # Collect additional data (weather, injuries, etc.)
            additional_data = self.data_source_manager.get_additional_data(
                sport, start_date=start_date, end_date=end_date
            )
            
            collected_data[sport] = {
                'players': player_data,
                'teams': team_data,
                'games': game_data,
                'player_game_stats': player_game_stats,
                'additional_data': additional_data
            }
            
            logger.info(f"Collected {len(player_data)} players, {len(team_data)} teams, "
                       f"{len(game_data)} games, and {len(player_game_stats)} player game stats for {sport}")
        
        return collected_data
    
    def process_data(self, collected_data):
        """
        Process and validate the collected data.
        
        Args:
            collected_data (dict): Data collected from data sources.
            
        Returns:
            dict: Processed and validated data.
        """
        logger.info("Processing and validating collected data")
        
        processed_data = {}
        for sport, sport_data in collected_data.items():
            logger.info(f"Processing data for {sport}")
            
            # Validate data
            validated_data = {}
            for data_type, data in sport_data.items():
                validated_data[data_type] = self.data_validator.validate(data, sport, data_type)
                
                invalid_count = len(data) - len(validated_data[data_type])
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} invalid records in {sport} {data_type}")
            
            # Compute features
            features = self.feature_computation.compute_features(validated_data, sport)
            
            processed_data[sport] = {
                'validated_data': validated_data,
                'features': features
            }
            
            logger.info(f"Processed and validated data for {sport}")
        
        return processed_data
    
    def train_model(self, processed_data, force_retrain=False):
        """
        Train the prediction model using processed data.
        
        Args:
            processed_data (dict): Processed and validated data.
            force_retrain (bool, optional): Force retraining even if not scheduled.
            
        Returns:
            dict: Training results and metrics.
        """
        logger.info("Training prediction model")
        
        # Check if training is needed
        if not force_retrain and not self._should_retrain():
            logger.info("Skipping training as it's not scheduled and not forced")
            return {'status': 'skipped', 'reason': 'not scheduled'}
        
        # Prepare training data
        training_data = self.training_pipeline.prepare_training_data(processed_data)
        
        # Train the model
        training_results = self.training_pipeline.train(
            training_data,
            batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            num_epochs=self.config['training']['num_epochs'],
            early_stopping_patience=self.config['training']['early_stopping_patience'],
            validation_split=self.config['training']['validation_split'],
            test_split=self.config['training']['test_split']
        )
        
        # Evaluate the model
        evaluation_results = self.training_pipeline.evaluate(training_data['test'])
        
        # Save the model
        model_version = self.training_pipeline.save_model()
        
        # Update model version in database
        self._update_model_version(model_version, evaluation_results)
        
        logger.info(f"Model training completed. New model version: {model_version}")
        
        return {
            'status': 'success',
            'model_version': model_version,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
    
    def _should_retrain(self):
        """
        Check if model retraining is scheduled.
        
        Returns:
            bool: True if retraining is scheduled, False otherwise.
        """
        if not self.config['training']['schedule']['daily_retraining']:
            return False
        
        now = datetime.now()
        retraining_time = datetime.strptime(
            self.config['training']['schedule']['retraining_time'], 
            '%H:%M'
        ).time()
        
        # Check if current time is within 5 minutes of scheduled retraining time
        current_time = now.time()
        retraining_minutes = retraining_time.hour * 60 + retraining_time.minute
        current_minutes = current_time.hour * 60 + current_time.minute
        
        return abs(current_minutes - retraining_minutes) <= 5
    
    def _update_model_version(self, model_version, evaluation_results):
        """
        Update model version information in the database.
        
        Args:
            model_version (str): Model version identifier.
            evaluation_results (dict): Model evaluation results.
        """
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_versions (
                    version, 
                    created_at, 
                    accuracy, 
                    mae, 
                    rmse, 
                    over_under_accuracy,
                    parameters
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_version,
                    datetime.now(),
                    evaluation_results['accuracy'],
                    evaluation_results['mae'],
                    evaluation_results['rmse'],
                    evaluation_results['over_under_accuracy'],
                    self.config['model']
                )
            )
            self.db_connection.commit()
    
    def generate_predictions(self, upcoming_games, model_version=None):
        """
        Generate predictions for upcoming games.
        
        Args:
            upcoming_games (dict): Information about upcoming games.
            model_version (str, optional): Model version to use for predictions.
            
        Returns:
            dict: Predictions for upcoming games.
        """
        logger.info(f"Generating predictions for {len(upcoming_games)} upcoming games")
        
        # Get the latest model version if not specified
        if model_version is None:
            model_version = self._get_latest_model_version()
            logger.info(f"Using latest model version: {model_version}")
        
        # Load the specified model version
        self.training_pipeline.load_model(model_version)
        
        # Prepare features for prediction
        prediction_features = self.feature_computation.compute_prediction_features(upcoming_games)
        
        # Generate predictions
        predictions = self.model.predict(prediction_features)
        
        # Format predictions
        formatted_predictions = self._format_predictions(predictions, upcoming_games)
        
        # Save predictions to database
        self._save_predictions(formatted_predictions, model_version)
        
        logger.info(f"Generated {len(formatted_predictions)} predictions")
        
        return formatted_predictions
    
    def _get_latest_model_version(self):
        """
        Get the latest model version from the database.
        
        Returns:
            str: Latest model version.
        """
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT version FROM model_versions
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            else:
                return 'v1.0.0'  # Default version if no versions exist
    
    def _format_predictions(self, predictions, upcoming_games):
        """
        Format raw predictions into a structured format.
        
        Args:
            predictions (dict): Raw predictions from the model.
            upcoming_games (dict): Information about upcoming games.
            
        Returns:
            list: Formatted predictions.
        """
        formatted_predictions = []
        
        for i, (game_id, game_info) in enumerate(upcoming_games.items()):
            for player_id, player_info in game_info['players'].items():
                for stat_type in player_info['stat_types']:
                    prediction_index = (game_id, player_id, stat_type)
                    
                    if prediction_index in predictions:
                        pred = predictions[prediction_index]
                        
                        formatted_predictions.append({
                            'game_id': game_id,
                            'player_id': player_id,
                            'player_name': player_info['name'],
                            'team_id': player_info['team_id'],
                            'team_name': game_info['teams'][player_info['team_id']]['name'],
                            'sport': game_info['sport'],
                            'game_date': game_info['date'],
                            'home_team_id': game_info['home_team_id'],
                            'away_team_id': game_info['away_team_id'],
                            'home_team_name': game_info['teams'][game_info['home_team_id']]['name'],
                            'away_team_name': game_info['teams'][game_info['away_team_id']]['name'],
                            'stat_type': stat_type,
                            'predicted_value': pred['value'],
                            'prediction_range_low': pred['range_low'],
                            'prediction_range_high': pred['range_high'],
                            'over_probability': pred['over_probability'],
                            'confidence_score': pred['confidence'],
                            'line_value': player_info['lines'].get(stat_type, pred['value']),
                            'created_at': datetime.now()
                        })
        
        return formatted_predictions
    
    def _save_predictions(self, predictions, model_version):
        """
        Save predictions to the database.
        
        Args:
            predictions (list): Formatted predictions.
            model_version (str): Model version used for predictions.
        """
        with self.db_connection.cursor() as cursor:
            for pred in predictions:
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        game_id,
                        player_id,
                        stat_type,
                        predicted_value,
                        prediction_range_low,
                        prediction_range_high,
                        over_probability,
                        confidence_score,
                        line_value,
                        created_at,
                        model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        pred['game_id'],
                        pred['player_id'],
                        pred['stat_type'],
                        pred['predicted_value'],
                        pred['prediction_range_low'],
                        pred['prediction_range_high'],
                        pred['over_probability'],
                        pred['confidence_score'],
                        pred['line_value'],
                        pred['created_at'],
                        model_version
                    )
                )
            
            self.db_connection.commit()
    
    def update_actual_results(self, completed_games):
        """
        Update predictions with actual results.
        
        Args:
            completed_games (dict): Information about completed games with actual results.
            
        Returns:
            dict: Updated prediction performance metrics.
        """
        logger.info(f"Updating actual results for {len(completed_games)} completed games")
        
        # Extract actual results
        actual_results = []
        for game_id, game_info in completed_games.items():
            for player_id, player_info in game_info['players'].items():
                for stat_type, actual_value in player_info['actual_stats'].items():
                    actual_results.append({
                        'game_id': game_id,
                        'player_id': player_id,
                        'stat_type': stat_type,
                        'actual_value': actual_value
                    })
        
        # Update predictions with actual results
        updated_count = 0
        with self.db_connection.cursor() as cursor:
            for result in actual_results:
                cursor.execute(
                    """
                    UPDATE predictions
                    SET actual_value = %s,
                        over_under_result = CASE
                            WHEN %s > line_value THEN true
                            ELSE false
                        END,
                        updated_at = %s
                    WHERE game_id = %s
                    AND player_id = %s
                    AND stat_type = %s
                    AND actual_value IS NULL
                    """,
                    (
                        result['actual_value'],
                        result['actual_value'],
                        datetime.now(),
                        result['game_id'],
                        result['player_id'],
                        result['stat_type']
                    )
                )
                updated_count += cursor.rowcount
            
            self.db_connection.commit()
        
        logger.info(f"Updated {updated_count} predictions with actual results")
        
        # Calculate updated performance metrics
        performance_metrics = self.performance_tracker.calculate_metrics()
        
        # Analyze errors
        error_analysis = self.error_analyzer.analyze()
        
        # Analyze feature importance
        feature_importance = self.feature_importance_analyzer.analyze()
        
        # Run continuous improvement
        improvement_suggestions = self.continuous_improvement.analyze()
        
        return {
            'updated_count': updated_count,
            'performance_metrics': performance_metrics,
            'error_analysis': error_analysis,
            'feature_importance': feature_importance,
            'improvement_suggestions': improvement_suggestions
        }
    
    def run_pipeline(self, sports=None, start_date=None, end_date=None, force_retrain=False):
        """
        Run the complete prediction pipeline.
        
        Args:
            sports (list, optional): List of sports to process.
            start_date (str, optional): Start date for data collection (YYYY-MM-DD).
            end_date (str, optional): End date for data collection (YYYY-MM-DD).
            force_retrain (bool, optional): Force model retraining.
            
        Returns:
            dict: Pipeline execution results.
        """
        logger.info(f"Running complete prediction pipeline for sports: {sports}")
        
        # Collect data
        collected_data = self.collect_data(sports, start_date, end_date)
        
        # Process data
        processed_data = self.process_data(collected_data)
        
        # Train model if needed
        training_results = self.train_model(processed_data, force_retrain)
        
        # Get upcoming games
        upcoming_games = self.data_source_manager.get_upcoming_games(sports)
        
        # Generate predictions
        predictions = self.generate_predictions(
            upcoming_games, 
            model_version=training_results.get('model_version')
        )
        
        # Get completed games with actual results
        completed_games = self.data_source_manager.get_completed_games(sports, start_date, end_date)
        
        # Update actual results
        update_results = self.update_actual_results(completed_games)
        
        logger.info("Prediction pipeline completed successfully")
        
        return {
            'status': 'success',
            'execution_time': (datetime.now() - self.start_time).total_seconds(),
            'data_collection': {
                'sports': list(collected_data.keys()),
                'count': {sport: len(data['games']) for sport, data in collected_data.items()}
            },
            'training': training_results,
            'predictions': {
                'count': len(predictions),
                'sports': list(set(pred['sport'] for pred in predictions))
            },
            'actual_results': {
                'updated_count': update_results['updated_count'],
                'performance_metrics': update_results['performance_metrics']
            }
        }
    
    def close(self):
        """
        Close all connections and resources.
        """
        logger.info("Closing Sports Prediction System")
        
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")
        
        logger.info("Sports Prediction System closed successfully")


def main():
    """
    Main function to run the sports prediction system.
    """
    # Create and run the sports prediction system
    system = SportsPredictionSystem()
    
    try:
        # Run the complete pipeline
        results = system.run_pipeline(
            sports=['nba', 'nfl', 'mlb', 'nhl', 'soccer'],
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        logger.info(f"Pipeline execution results: {results}")
    except Exception as e:
        logger.error(f"Error running sports prediction system: {e}", exc_info=True)
    finally:
        # Close the system
        system.close()


if __name__ == '__main__':
    main()
