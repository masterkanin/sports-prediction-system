"""
Data Validation and Cleaning for Sports Prediction System

This module implements data validation and cleaning procedures for the sports prediction system,
ensuring data quality and consistency across multiple sources.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_validation')

class DataValidator:
    """
    Class for validating and cleaning data for the sports prediction system.
    """
    
    def __init__(self):
        """
        Initialize the data validator.
        """
        # Define validation rules
        self.validation_rules = {
            'player': {
                'name': {'type': str, 'required': True, 'min_length': 2},
                'position': {'type': str, 'required': True, 'allowed_values': self._get_allowed_positions()},
                'team_id': {'type': str, 'required': True},
                'active': {'type': bool, 'required': False, 'default': True}
            },
            'team': {
                'name': {'type': str, 'required': True, 'min_length': 2},
                'abbreviation': {'type': str, 'required': True, 'min_length': 2, 'max_length': 5},
                'location': {'type': str, 'required': True},
                'conference': {'type': str, 'required': False},
                'division': {'type': str, 'required': False}
            },
            'game': {
                'sport': {'type': str, 'required': True, 'allowed_values': ['nba', 'nfl', 'mlb', 'nhl', 'soccer']},
                'season': {'type': str, 'required': True},
                'season_type': {'type': str, 'required': True, 'allowed_values': ['regular', 'playoff', 'preseason']},
                'home_team_id': {'type': str, 'required': True},
                'away_team_id': {'type': str, 'required': True},
                'game_date': {'type': str, 'required': True, 'format': 'date'},
                'scheduled_time': {'type': str, 'required': False, 'format': 'datetime'},
                'status': {'type': str, 'required': False, 'allowed_values': ['scheduled', 'in_progress', 'completed', 'cancelled']}
            },
            'player_game_stats': {
                'player_id': {'type': str, 'required': True},
                'game_id': {'type': str, 'required': True},
                'team_id': {'type': str, 'required': True},
                'minutes_played': {'type': int, 'required': False, 'min_value': 0, 'max_value': 60}
            },
            'prediction': {
                'player_id': {'type': str, 'required': True},
                'game_id': {'type': str, 'required': True},
                'stat_type': {'type': str, 'required': True},
                'predicted_value': {'type': float, 'required': True},
                'confidence_score': {'type': float, 'required': True, 'min_value': 0, 'max_value': 100},
                'over_probability': {'type': float, 'required': True, 'min_value': 0, 'max_value': 1},
                'line_value': {'type': float, 'required': True}
            }
        }
        
        # Define data cleaning rules
        self.cleaning_rules = {
            'player': {
                'name': [self._clean_name],
                'position': [self._standardize_position]
            },
            'team': {
                'name': [self._clean_name],
                'abbreviation': [self._clean_abbreviation]
            },
            'game': {
                'sport': [self._clean_sport_code],
                'game_date': [self._format_date],
                'scheduled_time': [self._format_datetime]
            }
        }
        
        # Define data reconciliation rules
        self.reconciliation_rules = {
            'player_stats': {
                'priority_sources': ['sportradar', 'statsperform', 'espn'],
                'conflict_resolution': 'weighted_average',
                'weights': {'sportradar': 0.5, 'statsperform': 0.3, 'espn': 0.2}
            },
            'game_info': {
                'priority_sources': ['sportradar', 'espn', 'statsperform'],
                'conflict_resolution': 'priority_source'
            },
            'weather': {
                'priority_sources': ['weather'],
                'conflict_resolution': 'priority_source'
            },
            'news': {
                'priority_sources': ['news'],
                'conflict_resolution': 'priority_source'
            }
        }
    
    def validate_data(self, data: Dict[str, Any], data_type: str) -> Tuple[bool, List[str]]:
        """
        Validate data against defined rules.
        
        Args:
            data: Data to validate
            data_type: Type of data (player, team, game, etc.)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if data_type not in self.validation_rules:
            return False, [f"Unknown data type: {data_type}"]
        
        rules = self.validation_rules[data_type]
        errors = []
        
        # Check required fields and types
        for field, rule in rules.items():
            # Check if required field is present
            if rule.get('required', False) and (field not in data or data[field] is None):
                errors.append(f"Missing required field: {field}")
                continue
            
            # Skip validation if field is not present and not required
            if field not in data or data[field] is None:
                if 'default' in rule:
                    data[field] = rule['default']
                continue
            
            # Check field type
            expected_type = rule.get('type')
            if expected_type and not isinstance(data[field], expected_type):
                errors.append(f"Field {field} should be of type {expected_type.__name__}")
            
            # Check string length
            if expected_type == str:
                min_length = rule.get('min_length')
                max_length = rule.get('max_length')
                
                if min_length and len(data[field]) < min_length:
                    errors.append(f"Field {field} should have minimum length of {min_length}")
                
                if max_length and len(data[field]) > max_length:
                    errors.append(f"Field {field} should have maximum length of {max_length}")
            
            # Check numeric range
            if expected_type in (int, float):
                min_value = rule.get('min_value')
                max_value = rule.get('max_value')
                
                if min_value is not None and data[field] < min_value:
                    errors.append(f"Field {field} should be at least {min_value}")
                
                if max_value is not None and data[field] > max_value:
                    errors.append(f"Field {field} should be at most {max_value}")
            
            # Check allowed values
            allowed_values = rule.get('allowed_values')
            if allowed_values and data[field] not in allowed_values:
                errors.append(f"Field {field} should be one of: {', '.join(str(v) for v in allowed_values)}")
            
            # Check date/datetime format
            format_type = rule.get('format')
            if format_type == 'date' and not self._is_valid_date(data[field]):
                errors.append(f"Field {field} should be a valid date (YYYY-MM-DD)")
            
            if format_type == 'datetime' and not self._is_valid_datetime(data[field]):
                errors.append(f"Field {field} should be a valid datetime (YYYY-MM-DD HH:MM:SS)")
        
        return len(errors) == 0, errors
    
    def clean_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Clean data according to defined rules.
        
        Args:
            data: Data to clean
            data_type: Type of data (player, team, game, etc.)
            
        Returns:
            Cleaned data
        """
        if data_type not in self.cleaning_rules:
            return data
        
        rules = self.cleaning_rules[data_type]
        cleaned_data = data.copy()
        
        for field, cleaning_functions in rules.items():
            if field in cleaned_data and cleaned_data[field] is not None:
                for clean_func in cleaning_functions:
                    cleaned_data[field] = clean_func(cleaned_data[field])
        
        return cleaned_data
    
    def reconcile_data_sources(self, data_sources: Dict[str, Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """
        Reconcile data from multiple sources.
        
        Args:
            data_sources: Dictionary mapping source names to data
            data_type: Type of data (player_stats, game_info, etc.)
            
        Returns:
            Reconciled data
        """
        if data_type not in self.reconciliation_rules:
            # Default to first source if no specific rules
            for source_name, source_data in data_sources.items():
                return source_data
        
        rules = self.reconciliation_rules[data_type]
        priority_sources = rules.get('priority_sources', [])
        conflict_resolution = rules.get('conflict_resolution', 'priority_source')
        
        # Filter out sources with errors
        valid_sources = {}
        for source_name, source_data in data_sources.items():
            if not isinstance(source_data, dict) or 'error' not in source_data:
                valid_sources[source_name] = source_data
        
        if not valid_sources:
            logger.warning(f"No valid data sources for {data_type}")
            return {}
        
        # If only one source, use it
        if len(valid_sources) == 1:
            return next(iter(valid_sources.values()))
        
        # Use priority source if available
        for source_name in priority_sources:
            if source_name in valid_sources:
                if conflict_resolution == 'priority_source':
                    return valid_sources[source_name]
                break
        
        # Weighted average for numeric fields
        if conflict_resolution == 'weighted_average':
            weights = rules.get('weights', {})
            reconciled_data = {}
            
            # Get all fields from all sources
            all_fields = set()
            for source_data in valid_sources.values():
                all_fields.update(source_data.keys())
            
            for field in all_fields:
                field_values = []
                field_weights = []
                
                for source_name, source_data in valid_sources.items():
                    if field in source_data and source_data[field] is not None:
                        if isinstance(source_data[field], (int, float)):
                            field_values.append(source_data[field])
                            field_weights.append(weights.get(source_name, 1.0))
                
                if field_values:
                    if len(field_values) == 1:
                        reconciled_data[field] = field_values[0]
                    else:
                        # Calculate weighted average
                        total_weight = sum(field_weights)
                        weighted_sum = sum(value * weight for value, weight in zip(field_values, field_weights))
                        reconciled_data[field] = weighted_sum / total_weight if total_weight > 0 else sum(field_values) / len(field_values)
                else:
                    # For non-numeric fields, use the first available value
                    for source_name in priority_sources:
                        if source_name in valid_sources and field in valid_sources[source_name]:
                            reconciled_data[field] = valid_sources[source_name][field]
                            break
            
            return reconciled_data
        
        # Default to first priority source
        for source_name in priority_sources:
            if source_name in valid_sources:
                return valid_sources[source_name]
        
        # If no priority source is available, use the first source
        return next(iter(valid_sources.values()))
    
    def detect_outliers(self, data: List[Dict[str, Any]], field: str, method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """
        Detect outliers in a dataset.
        
        Args:
            data: List of data points
            field: Field to check for outliers
            method: Method to use ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            List of indices of outliers
        """
        values = [d[field] for d in data if field in d and d[field] is not None]
        
        if not values:
            return []
        
        if method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return []
            
            zscores = [(value - mean) / std for value in values]
            outliers = [i for i, zscore in enumerate(zscores) if abs(zscore) > threshold]
            
            return outliers
        
        elif method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = [i for i, value in enumerate(values) if value < lower_bound or value > upper_bound]
            
            return outliers
        
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return []
    
    def handle_missing_values(self, data: Dict[str, Any], strategy: str = 'mean') -> Dict[str, Any]:
        """
        Handle missing values in data.
        
        Args:
            data: Data with missing values
            strategy: Strategy to use ('mean', 'median', 'mode', 'zero', 'none')
            
        Returns:
            Data with missing values handled
        """
        if strategy == 'none':
            return data
        
        result = data.copy()
        
        # Get all numeric fields
        numeric_fields = [field for field, value in data.items() 
                         if isinstance(value, (int, float)) or value is None]
        
        # Calculate replacement values
        replacement_values = {}
        
        if strategy in ('mean', 'median', 'mode'):
            # Get historical values for each field
            for field in numeric_fields:
                if field in data and data[field] is None:
                    historical_values = self._get_historical_values(field)
                    
                    if historical_values:
                        if strategy == 'mean':
                            replacement_values[field] = np.mean(historical_values)
                        elif strategy == 'median':
                            replacement_values[field] = np.median(historical_values)
                        elif strategy == 'mode':
                            # Get most common value
                            unique, counts = np.unique(historical_values, return_counts=True)
                            replacement_values[field] = unique[np.argmax(counts)]
        
        # Replace missing values
        for field in numeric_fields:
            if field in data and data[field] is None:
                if field in replacement_values:
                    result[field] = replacement_values[field]
                elif strategy == 'zero':
                    result[field] = 0
        
        return result
    
    def validate_consistency(self, data: Dict[str, Any], data_type: str) -> Tuple[bool, List[str]]:
        """
        Validate data consistency across related entities.
        
        Args:
            data: Data to validate
            data_type: Type of data (player, team, game, etc.)
            
        Returns:
            Tuple of (is_consistent, error_messages)
        """
        errors = []
        
        if data_type == 'player':
            # Check if team exists
            if 'team_id' in data and data['team_id']:
                team_exists = self._check_entity_exists('teams', 'team_id', data['team_id'])
                if not team_exists:
                    errors.append(f"Team with ID {data['team_id']} does not exist")
        
        elif data_type == 'game':
            # Check if teams exist
            if 'home_team_id' in data and data['home_team_id']:
                team_exists = self._check_entity_exists('teams', 'team_id', data['home_team_id'])
                if not team_exists:
                    errors.append(f"Home team with ID {data['home_team_id']} does not exist")
            
            if 'away_team_id' in data and data['away_team_id']:
                team_exists = self._check_entity_exists('teams', 'team_id', data['away_team_id'])
                if not team_exists:
                    errors.append(f"Away team with ID {data['away_team_id']} does not exist")
            
            # Check that home and away teams are different
            if ('home_team_id' in data and 'away_team_id' in data and 
                data['home_team_id'] and data['away_team_id'] and 
                data['home_team_id'] == data['away_team_id']):
                errors.append("Home team and away team cannot be the same")
        
        elif data_type == 'player_game_stats':
            # Check if player exists
            if 'player_id' in data and data['player_id']:
                player_exists = self._check_entity_exists('players', 'player_id', data['player_id'])
                if not player_exists:
                    errors.append(f"Player with ID {data['player_id']} does not exist")
            
            # Check if game exists
            if 'game_id' in data and data['game_id']:
                game_exists = self._check_entity_exists('games', 'game_id', data['game_id'])
                if not game_exists:
                    errors.append(f"Game with ID {data['game_id']} does not exist")
            
            # Check if team exists
            if 'team_id' in data and data['team_id']:
                team_exists = self._check_entity_exists('teams', 'team_id', data['team_id'])
                if not team_exists:
                    errors.append(f"Team with ID {data['team_id']} does not exist")
            
            # Check if player belongs to team
            if ('player_id' in data and 'team_id' in data and 
                data['player_id'] and data['team_id']):
                player_on_team = self._check_player_on_team(data['player_id'], data['team_id'])
                if not player_on_team:
                    errors.append(f"Player with ID {data['player_id']} is not on team with ID {data['team_id']}")
            
            # Check if team is playing in game
            if ('game_id' in data and 'team_id' in data and 
                data['game_id'] and data['team_id']):
                team_in_game = self._check_team_in_game(data['team_id'], data['game_id'])
                if not team_in_game:
                    errors.append(f"Team with ID {data['team_id']} is not playing in game with ID {data['game_id']}")
        
        elif data_type == 'prediction':
            # Check if player exists
            if 'player_id' in data and data['player_id']:
                player_exists = self._check_entity_exists('players', 'player_id', data['player_id'])
                if not player_exists:
                    errors.append(f"Player with ID {data['player_id']} does not exist")
            
            # Check if game exists
            if 'game_id' in data and data['game_id']:
                game_exists = self._check_entity_exists('games', 'game_id', data['game_id'])
                if not game_exists:
                    errors.append(f"Game with ID {data['game_id']} does not exist")
            
            # Check if game is in the future
            if 'game_id' in data and data['game_id']:
                game_in_future = self._check_game_in_future(data['game_id'])
                if not game_in_future:
                    errors.append(f"Game with ID {data['game_id']} is not in the future")
        
        return len(errors) == 0, errors
    
    def log_validation_errors(self, errors: List[str], data_type: str, source: str) -> None:
        """
        Log validation errors.
        
        Args:
            errors: List of error messages
            data_type: Type of data
            source: Source of the data
        """
        for error in errors:
            logger.error(f"Validation error in {data_type} from {source}: {error}")
    
    def _get_allowed_positions(self) -> List[str]:
        """
        Get allowed positions for players.
        
        Returns:
            List of allowed positions
        """
        # Basketball positions
        basketball = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F']
        
        # Football positions
        football = ['QB', 'RB', 'FB', 'WR', 'TE', 'OT', 'OG', 'C', 'DT', 'DE', 'LB', 'CB', 'S', 'K', 'P']
        
        # Baseball positions
        baseball = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']
        
        # Hockey positions
        hockey = ['C', 'LW', 'RW', 'D', 'G']
        
        # Soccer positions
        soccer = ['GK', 'DF', 'MF', 'FW']
        
        return basketball + football + baseball + hockey + soccer
    
    def _clean_name(self, name: str) -> str:
        """
        Clean a name string.
        
        Args:
            name: Name to clean
            
        Returns:
            Cleaned name
        """
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def _clean_abbreviation(self, abbr: str) -> str:
        """
        Clean a team abbreviation.
        
        Args:
            abbr: Abbreviation to clean
            
        Returns:
            Cleaned abbreviation
        """
        # Convert to uppercase
        return abbr.upper()
    
    def _standardize_position(self, position: str) -> str:
        """
        Standardize a player position.
        
        Args:
            position: Position to standardize
            
        Returns:
            Standardized position
        """
        # Convert to uppercase
        position = position.upper()
        
        # Map common variations
        position_map = {
            'POINT GUARD': 'PG',
            'SHOOTING GUARD': 'SG',
            'SMALL FORWARD': 'SF',
            'POWER FORWARD': 'PF',
            'CENTER': 'C',
            'GUARD': 'G',
            'FORWARD': 'F',
            'QUARTERBACK': 'QB',
            'RUNNING BACK': 'RB',
            'FULLBACK': 'FB',
            'WIDE RECEIVER': 'WR',
            'TIGHT END': 'TE',
            'OFFENSIVE TACKLE': 'OT',
            'OFFENSIVE GUARD': 'OG',
            'DEFENSIVE TACKLE': 'DT',
            'DEFENSIVE END': 'DE',
            'LINEBACKER': 'LB',
            'CORNERBACK': 'CB',
            'SAFETY': 'S',
            'KICKER': 'K',
            'PUNTER': 'P',
            'PITCHER': 'P',
            'CATCHER': 'C',
            'FIRST BASE': '1B',
            'SECOND BASE': '2B',
            'THIRD BASE': '3B',
            'SHORTSTOP': 'SS',
            'LEFT FIELD': 'LF',
            'CENTER FIELD': 'CF',
            'RIGHT FIELD': 'RF',
            'DESIGNATED HITTER': 'DH',
            'LEFT WING': 'LW',
            'RIGHT WING': 'RW',
            'DEFENSE': 'D',
            'GOALIE': 'G',
            'GOALKEEPER': 'GK',
            'DEFENDER': 'DF',
            'MIDFIELDER': 'MF',
            'FORWARD': 'FW'
        }
        
        return position_map.get(position, position)
    
    def _clean_sport_code(self, sport: str) -> str:
        """
        Clean a sport code.
        
        Args:
            sport: Sport code to clean
            
        Returns:
            Cleaned sport code
        """
        # Convert to lowercase
        sport = sport.lower()
        
        # Map common variations
        sport_map = {
            'basketball': 'nba',
            'football': 'nfl',
            'baseball': 'mlb',
            'hockey': 'nhl'
        }
        
        return sport_map.get(sport, sport)
    
    def _format_date(self, date_str: str) -> str:
        """
        Format a date string to YYYY-MM-DD.
        
        Args:
            date_str: Date string to format
            
        Returns:
            Formatted date string
        """
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format matches, return original string
        return date_str
    
    def _format_datetime(self, datetime_str: str) -> str:
        """
        Format a datetime string to YYYY-MM-DD HH:MM:SS.
        
        Args:
            datetime_str: Datetime string to format
            
        Returns:
            Formatted datetime string
        """
        # Try different datetime formats
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M'
        ]
        
        for fmt in datetime_formats:
            try:
                datetime_obj = datetime.strptime(datetime_str, fmt)
                return datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        # If no format matches, return original string
        return datetime_str
    
    def _is_valid_date(self, date_str: str) -> bool:
        """
        Check if a string is a valid date.
        
        Args:
            date_str: Date string to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _is_valid_datetime(self, datetime_str: str) -> bool:
        """
        Check if a string is a valid datetime.
        
        Args:
            datetime_str: Datetime string to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            return True
        except ValueError:
            return False
    
    def _get_historical_values(self, field: str) -> List[float]:
        """
        Get historical values for a field.
        
        Args:
            field: Field name
            
        Returns:
            List of historical values
        """
        # This is a placeholder - in a real implementation, this would query
        # the database for historical values of the field
        return []
    
    def _check_entity_exists(self, table: str, id_field: str, entity_id: str) -> bool:
        """
        Check if an entity exists in the database.
        
        Args:
            table: Table name
            id_field: ID field name
            entity_id: Entity ID
            
        Returns:
            True if exists, False otherwise
        """
        query = f"SELECT COUNT(*) as count FROM {table} WHERE {id_field} = %s"
        params = (entity_id,)
        results = execute_query(query, params)
        
        return results[0]['count'] > 0 if results else False
    
    def _check_player_on_team(self, player_id: str, team_id: str) -> bool:
        """
        Check if a player is on a team.
        
        Args:
            player_id: Player ID
            team_id: Team ID
            
        Returns:
            True if player is on team, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM players WHERE player_id = %s AND team_id = %s"
        params = (player_id, team_id)
        results = execute_query(query, params)
        
        return results[0]['count'] > 0 if results else False
    
    def _check_team_in_game(self, team_id: str, game_id: str) -> bool:
        """
        Check if a team is playing in a game.
        
        Args:
            team_id: Team ID
            game_id: Game ID
            
        Returns:
            True if team is in game, False otherwise
        """
        query = """
        SELECT COUNT(*) as count FROM games 
        WHERE game_id = %s AND (home_team_id = %s OR away_team_id = %s)
        """
        params = (game_id, team_id, team_id)
        results = execute_query(query, params)
        
        return results[0]['count'] > 0 if results else False
    
    def _check_game_in_future(self, game_id: str) -> bool:
        """
        Check if a game is in the future.
        
        Args:
            game_id: Game ID
            
        Returns:
            True if game is in the future, False otherwise
        """
        query = "SELECT game_date FROM games WHERE game_id = %s"
        params = (game_id,)
        results = execute_query(query, params)
        
        if not results:
            return False
        
        game_date = datetime.strptime(results[0]['game_date'], '%Y-%m-%d')
        return game_date > datetime.now()


class DataReconciliation:
    """
    Class for reconciling data from multiple sources.
    """
    
    def __init__(self, validator: DataValidator):
        """
        Initialize the data reconciliation system.
        
        Args:
            validator: DataValidator instance
        """
        self.validator = validator
    
    def reconcile_player_data(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile player data from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to player data
            
        Returns:
            Reconciled player data
        """
        return self.validator.reconcile_data_sources(sources, 'player')
    
    def reconcile_team_data(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile team data from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to team data
            
        Returns:
            Reconciled team data
        """
        return self.validator.reconcile_data_sources(sources, 'team')
    
    def reconcile_game_data(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile game data from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to game data
            
        Returns:
            Reconciled game data
        """
        return self.validator.reconcile_data_sources(sources, 'game')
    
    def reconcile_player_stats(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile player statistics from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to player statistics
            
        Returns:
            Reconciled player statistics
        """
        return self.validator.reconcile_data_sources(sources, 'player_stats')
    
    def reconcile_weather_data(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile weather data from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to weather data
            
        Returns:
            Reconciled weather data
        """
        return self.validator.reconcile_data_sources(sources, 'weather')
    
    def reconcile_news_data(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile news data from multiple sources.
        
        Args:
            sources: Dictionary mapping source names to news data
            
        Returns:
            Reconciled news data
        """
        return self.validator.reconcile_data_sources(sources, 'news')
    
    def log_reconciliation_results(self, sources: Dict[str, Dict[str, Any]], reconciled: Dict[str, Any], data_type: str) -> None:
        """
        Log reconciliation results.
        
        Args:
            sources: Dictionary mapping source names to data
            reconciled: Reconciled data
            data_type: Type of data
        """
        source_names = ', '.join(sources.keys())
        logger.info(f"Reconciled {data_type} data from sources: {source_names}")
        
        # Log conflicts
        for field in reconciled:
            values = {}
            for source_name, source_data in sources.items():
                if field in source_data:
                    values[source_name] = source_data[field]
            
            if len(set(values.values())) > 1:
                logger.info(f"Conflict in field {field}: {values}, reconciled value: {reconciled[field]}")


class DataQualityMonitor:
    """
    Class for monitoring data quality over time.
    """
    
    def __init__(self):
        """
        Initialize the data quality monitor.
        """
        pass
    
    def log_data_quality_metrics(self, data_type: str, metrics: Dict[str, Any]) -> None:
        """
        Log data quality metrics.
        
        Args:
            data_type: Type of data
            metrics: Dictionary of metrics
        """
        query = """
        INSERT INTO data_quality_metrics (
            data_type, metrics, created_at
        ) VALUES (
            %s, %s, %s
        )
        """
        
        params = (
            data_type,
            json.dumps(metrics),
            datetime.now()
        )
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            cursor.close()
        
        logger.info(f"Logged data quality metrics for {data_type}")
    
    def calculate_data_quality_metrics(self, data: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """
        Calculate data quality metrics.
        
        Args:
            data: List of data points
            data_type: Type of data
            
        Returns:
            Dictionary of metrics
        """
        if not data:
            return {
                'count': 0,
                'completeness': 0,
                'validity': 0,
                'consistency': 0,
                'timeliness': 0
            }
        
        # Count
        count = len(data)
        
        # Completeness
        required_fields = self._get_required_fields(data_type)
        completeness_scores = []
        
        for item in data:
            if not required_fields:
                completeness_scores.append(1.0)
            else:
                present_fields = sum(1 for field in required_fields if field in item and item[field] is not None)
                completeness_scores.append(present_fields / len(required_fields))
        
        completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Validity
        validator = DataValidator()
        validity_scores = []
        
        for item in data:
            is_valid, _ = validator.validate_data(item, data_type)
            validity_scores.append(1.0 if is_valid else 0.0)
        
        validity = sum(validity_scores) / len(validity_scores) if validity_scores else 0
        
        # Consistency
        consistency_scores = []
        
        for item in data:
            is_consistent, _ = validator.validate_consistency(item, data_type)
            consistency_scores.append(1.0 if is_consistent else 0.0)
        
        consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # Timeliness
        timeliness = self._calculate_timeliness(data, data_type)
        
        return {
            'count': count,
            'completeness': completeness,
            'validity': validity,
            'consistency': consistency,
            'timeliness': timeliness
        }
    
    def _get_required_fields(self, data_type: str) -> List[str]:
        """
        Get required fields for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            List of required fields
        """
        validator = DataValidator()
        
        if data_type not in validator.validation_rules:
            return []
        
        rules = validator.validation_rules[data_type]
        return [field for field, rule in rules.items() if rule.get('required', False)]
    
    def _calculate_timeliness(self, data: List[Dict[str, Any]], data_type: str) -> float:
        """
        Calculate timeliness metric.
        
        Args:
            data: List of data points
            data_type: Type of data
            
        Returns:
            Timeliness score (0-1)
        """
        # This is a placeholder - in a real implementation, this would calculate
        # how up-to-date the data is based on timestamps
        return 1.0
