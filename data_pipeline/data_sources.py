"""
Multi-Source Data Collection for Sports Prediction System

This module implements data collection from multiple sources beyond Sportradar,
including ESPN, Stats Perform, and other sports data providers.
"""

import os
import sys
import logging
import json
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_sources.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_sources')

class DataSource(ABC):
    """
    Abstract base class for data sources.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the data source.
        
        Args:
            api_key: API key for the data source
            base_url: Base URL for API requests
        """
        self.api_key = api_key or os.environ.get(f"{self.__class__.__name__.upper()}_API_KEY")
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
    
    @abstractmethod
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming games for a sport.
        
        Args:
            sport: Sport code (e.g., 'nba', 'nfl')
            days: Number of days to look ahead
            
        Returns:
            List of upcoming games
        """
        pass
    
    @abstractmethod
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a player.
        
        Args:
            player_id: Player ID
            season: Season code (optional)
            
        Returns:
            Player statistics
        """
        pass
    
    @abstractmethod
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a team.
        
        Args:
            team_id: Team ID
            season: Season code (optional)
            
        Returns:
            Team statistics
        """
        pass
    
    @abstractmethod
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Get stats for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game statistics
        """
        pass
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        # Check rate limits
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 1:
            wait_time = max(0, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time + 1)  # Add 1 second buffer
        
        try:
            response = self.session.get(url, headers=headers, params=params)
            
            # Update rate limit info if available
            if 'X-RateLimit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in response.headers:
                self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def register_in_database(self) -> bool:
        """
        Register this data source in the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            INSERT INTO data_sources (
                name, description, api_endpoint, credentials, 
                last_sync, sync_frequency, active, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                api_endpoint = EXCLUDED.api_endpoint,
                credentials = EXCLUDED.credentials,
                sync_frequency = EXCLUDED.sync_frequency,
                active = EXCLUDED.active,
                updated_at = EXCLUDED.updated_at
            """
            
            params = (
                self.__class__.__name__,
                self.__doc__.split('\n')[0] if self.__doc__ else None,
                self.base_url,
                json.dumps({"api_key": "********"}),  # Don't store actual API key
                None,
                "daily",
                True,
                datetime.now(),
                datetime.now()
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Registered data source: {self.__class__.__name__}")
            return True
        except Exception as e:
            logger.error(f"Error registering data source: {e}")
            return False
    
    def update_last_sync(self) -> bool:
        """
        Update the last sync timestamp for this data source.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            UPDATE data_sources
            SET last_sync = %s, updated_at = %s
            WHERE name = %s
            """
            
            params = (
                datetime.now(),
                datetime.now(),
                self.__class__.__name__
            )
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                cursor.close()
            
            logger.info(f"Updated last sync for data source: {self.__class__.__name__}")
            return True
        except Exception as e:
            logger.error(f"Error updating last sync: {e}")
            return False


class SportradarAPI(DataSource):
    """
    Sportradar API data source for sports data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Sportradar API client.
        
        Args:
            api_key: Sportradar API key
        """
        super().__init__(api_key, "https://api.sportradar.com/v1")
        self.sport_endpoints = {
            'nba': 'basketball/nba',
            'nfl': 'football/nfl',
            'mlb': 'baseball/mlb',
            'nhl': 'hockey/nhl',
            'soccer': 'soccer/international'
        }
    
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming games for a sport.
        
        Args:
            sport: Sport code (e.g., 'nba', 'nfl')
            days: Number of days to look ahead
            
        Returns:
            List of upcoming games
        """
        if sport.lower() not in self.sport_endpoints:
            raise ValueError(f"Unsupported sport: {sport}")
        
        endpoint = f"{self.sport_endpoints[sport.lower()]}/schedule"
        params = {
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'end_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        }
        
        response = self._make_request(endpoint, params)
        
        # Extract games from response
        games = []
        if 'games' in response:
            games = response['games']
        elif 'schedules' in response:
            for day in response['schedules']:
                if 'games' in day:
                    games.extend(day['games'])
        
        return games
    
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a player.
        
        Args:
            player_id: Player ID
            season: Season code (optional)
            
        Returns:
            Player statistics
        """
        endpoint = f"players/{player_id}/profile"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a team.
        
        Args:
            team_id: Team ID
            season: Season code (optional)
            
        Returns:
            Team statistics
        """
        endpoint = f"teams/{team_id}/profile"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Get stats for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game statistics
        """
        endpoint = f"games/{game_id}/summary"
        return self._make_request(endpoint)


class ESPNAPI(DataSource):
    """
    ESPN API data source for sports data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ESPN API client.
        
        Args:
            api_key: ESPN API key
        """
        super().__init__(api_key, "https://site.api.espn.com/apis/site/v2/sports")
        self.sport_endpoints = {
            'nba': 'basketball/nba',
            'nfl': 'football/nfl',
            'mlb': 'baseball/mlb',
            'nhl': 'hockey/nhl',
            'soccer': 'soccer/usa.1'  # MLS
        }
    
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming games for a sport.
        
        Args:
            sport: Sport code (e.g., 'nba', 'nfl')
            days: Number of days to look ahead
            
        Returns:
            List of upcoming games
        """
        if sport.lower() not in self.sport_endpoints:
            raise ValueError(f"Unsupported sport: {sport}")
        
        endpoint = f"{self.sport_endpoints[sport.lower()]}/scoreboard"
        params = {
            'limit': 100,
            'dates': ','.join([
                (datetime.now() + timedelta(days=i)).strftime('%Y%m%d')
                for i in range(days)
            ])
        }
        
        response = self._make_request(endpoint, params)
        
        # Extract games from response
        games = []
        if 'events' in response:
            for event in response['events']:
                game = {
                    'id': event.get('id'),
                    'scheduled': event.get('date'),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                    'home': {
                        'id': event.get('competitions', [{}])[0].get('competitors', [{}])[0].get('id'),
                        'name': event.get('competitions', [{}])[0].get('competitors', [{}])[0].get('team', {}).get('name')
                    },
                    'away': {
                        'id': event.get('competitions', [{}])[0].get('competitors', [{}])[1].get('id'),
                        'name': event.get('competitions', [{}])[0].get('competitors', [{}])[1].get('team', {}).get('name')
                    },
                    'venue': {
                        'name': event.get('competitions', [{}])[0].get('venue', {}).get('fullName')
                    }
                }
                games.append(game)
        
        return games
    
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a player.
        
        Args:
            player_id: Player ID
            season: Season code (optional)
            
        Returns:
            Player statistics
        """
        endpoint = f"athletes/{player_id}"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a team.
        
        Args:
            team_id: Team ID
            season: Season code (optional)
            
        Returns:
            Team statistics
        """
        endpoint = f"teams/{team_id}"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Get stats for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game statistics
        """
        endpoint = f"summary"
        params = {'event': game_id}
        return self._make_request(endpoint, params)


class StatsPerformAPI(DataSource):
    """
    Stats Perform API data source for advanced sports analytics.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Stats Perform API client.
        
        Args:
            api_key: Stats Perform API key
        """
        super().__init__(api_key, "https://api.statsperform.com/v1")
        self.sport_endpoints = {
            'nba': 'basketball',
            'nfl': 'football',
            'mlb': 'baseball',
            'nhl': 'hockey',
            'soccer': 'soccer'
        }
    
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming games for a sport.
        
        Args:
            sport: Sport code (e.g., 'nba', 'nfl')
            days: Number of days to look ahead
            
        Returns:
            List of upcoming games
        """
        if sport.lower() not in self.sport_endpoints:
            raise ValueError(f"Unsupported sport: {sport}")
        
        endpoint = f"{self.sport_endpoints[sport.lower()]}/schedule"
        params = {
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'end_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        }
        
        response = self._make_request(endpoint, params)
        
        # Extract games from response (format may vary)
        games = []
        if 'fixtures' in response:
            games = response['fixtures']
        
        return games
    
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a player.
        
        Args:
            player_id: Player ID
            season: Season code (optional)
            
        Returns:
            Player statistics
        """
        endpoint = f"players/{player_id}/stats"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stats for a team.
        
        Args:
            team_id: Team ID
            season: Season code (optional)
            
        Returns:
            Team statistics
        """
        endpoint = f"teams/{team_id}/stats"
        params = {}
        if season:
            params['season'] = season
        
        return self._make_request(endpoint, params)
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Get stats for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game statistics
        """
        endpoint = f"fixtures/{game_id}/stats"
        return self._make_request(endpoint)
    
    def get_advanced_metrics(self, entity_type: str, entity_id: str, metric_type: str) -> Dict[str, Any]:
        """
        Get advanced metrics for a player or team.
        
        Args:
            entity_type: 'player' or 'team'
            entity_id: Player or team ID
            metric_type: Type of advanced metric
            
        Returns:
            Advanced metrics
        """
        endpoint = f"{entity_type}s/{entity_id}/advanced/{metric_type}"
        return self._make_request(endpoint)


class WeatherAPI(DataSource):
    """
    Weather API data source for game weather conditions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Weather API client.
        
        Args:
            api_key: Weather API key
        """
        super().__init__(api_key, "https://api.weatherapi.com/v1")
    
    def get_forecast(self, location: str, days: int = 7) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Location name or coordinates
            days: Number of days to forecast
            
        Returns:
            Weather forecast
        """
        endpoint = "forecast.json"
        params = {
            'q': location,
            'days': days,
            'key': self.api_key
        }
        
        return self._make_request(endpoint, params)
    
    def get_historical(self, location: str, date: str) -> Dict[str, Any]:
        """
        Get historical weather for a location.
        
        Args:
            location: Location name or coordinates
            date: Date in YYYY-MM-DD format
            
        Returns:
            Historical weather
        """
        endpoint = "history.json"
        params = {
            'q': location,
            'dt': date,
            'key': self.api_key
        }
        
        return self._make_request(endpoint, params)
    
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Not applicable for weather API.
        """
        raise NotImplementedError("Weather API does not provide game data")
    
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Not applicable for weather API.
        """
        raise NotImplementedError("Weather API does not provide player data")
    
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Not applicable for weather API.
        """
        raise NotImplementedError("Weather API does not provide team data")
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Not applicable for weather API.
        """
        raise NotImplementedError("Weather API does not provide game data")


class NewsAPI(DataSource):
    """
    News API data source for player and team news.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News API client.
        
        Args:
            api_key: News API key
        """
        super().__init__(api_key, "https://newsapi.org/v2")
    
    def get_player_news(self, player_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get news articles about a player.
        
        Args:
            player_name: Player name
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        endpoint = "everything"
        params = {
            'q': player_name,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key
        }
        
        response = self._make_request(endpoint, params)
        
        if 'articles' in response:
            return response['articles']
        return []
    
    def get_team_news(self, team_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get news articles about a team.
        
        Args:
            team_name: Team name
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        endpoint = "everything"
        params = {
            'q': team_name,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key
        }
        
        response = self._make_request(endpoint, params)
        
        if 'articles' in response:
            return response['articles']
        return []
    
    def get_upcoming_games(self, sport: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Not applicable for news API.
        """
        raise NotImplementedError("News API does not provide game data")
    
    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Not applicable for news API.
        """
        raise NotImplementedError("News API does not provide player stats")
    
    def get_team_stats(self, team_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Not applicable for news API.
        """
        raise NotImplementedError("News API does not provide team stats")
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        Not applicable for news API.
        """
        raise NotImplementedError("News API does not provide game stats")


class DataSourceManager:
    """
    Manager class for handling multiple data sources.
    """
    
    def __init__(self):
        """
        Initialize the data source manager.
        """
        self.sources = {}
    
    def register_source(self, name: str, source: DataSource) -> None:
        """
        Register a data source.
        
        Args:
            name: Name of the data source
            source: Data source instance
        """
        self.sources[name] = source
        source.register_in_database()
        logger.info(f"Registered data source: {name}")
    
    def get_source(self, name: str) -> Optional[DataSource]:
        """
        Get a data source by name.
        
        Args:
            name: Name of the data source
            
        Returns:
            Data source instance or None if not found
        """
        return self.sources.get(name)
    
    def get_all_sources(self) -> Dict[str, DataSource]:
        """
        Get all registered data sources.
        
        Returns:
            Dictionary of data sources
        """
        return self.sources
    
    def fetch_from_all_sources(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Fetch data from all sources using a specified method.
        
        Args:
            method_name: Name of the method to call on each source
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Dictionary mapping source names to results
        """
        results = {}
        
        for name, source in self.sources.items():
            try:
                method = getattr(source, method_name, None)
                if method and callable(method):
                    results[name] = method(*args, **kwargs)
                    source.update_last_sync()
                else:
                    logger.warning(f"Method {method_name} not found in source {name}")
            except Exception as e:
                logger.error(f"Error fetching from source {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def reconcile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile data from multiple sources to resolve conflicts.
        
        Args:
            data: Dictionary mapping source names to data
            
        Returns:
            Reconciled data
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated conflict resolution
        
        # Remove sources with errors
        valid_data = {k: v for k, v in data.items() if not isinstance(v, dict) or 'error' not in v}
        
        if not valid_data:
            logger.warning("No valid data to reconcile")
            return {}
        
        # For now, just use the first source's data as the base
        base_source = list(valid_data.keys())[0]
        reconciled = valid_data[base_source]
        
        # Log the reconciliation
        logger.info(f"Reconciled data using {base_source} as base source")
        
        return reconciled


# Initialize data sources
def initialize_data_sources() -> DataSourceManager:
    """
    Initialize all data sources.
    
    Returns:
        DataSourceManager instance
    """
    manager = DataSourceManager()
    
    # Register data sources
    manager.register_source('sportradar', SportradarAPI())
    manager.register_source('espn', ESPNAPI())
    manager.register_source('statsperform', StatsPerformAPI())
    manager.register_source('weather', WeatherAPI())
    manager.register_source('news', NewsAPI())
    
    return manager
