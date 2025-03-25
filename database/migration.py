"""
Database Migration Scripts for Sports Prediction System

This module provides utilities for migrating data from the existing SQLite database
to the new PostgreSQL database with the enhanced schema.
"""

import os
import sys
import logging
import sqlite3
import json
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_config import get_db_connection, get_db_cursor, execute_query, execute_script

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('migration')

class DatabaseMigration:
    """
    Class to handle migration from SQLite to PostgreSQL.
    """
    
    def __init__(self, sqlite_path, schema_path):
        """
        Initialize the migration with source and target database information.
        
        Args:
            sqlite_path (str): Path to SQLite database file
            schema_path (str): Path to PostgreSQL schema SQL file
        """
        self.sqlite_path = sqlite_path
        self.schema_path = schema_path
        self.sqlite_conn = None
        
    def connect_sqlite(self):
        """
        Connect to the SQLite database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite database: {self.sqlite_path}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            return False
    
    def create_postgres_schema(self):
        """
        Create the PostgreSQL schema using the provided SQL file.
        
        Returns:
            bool: True if schema creation successful, False otherwise
        """
        try:
            # Execute the schema SQL file
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                conn.commit()
                cursor.close()
            
            logger.info("PostgreSQL schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating PostgreSQL schema: {e}")
            return False
    
    def get_sqlite_tables(self):
        """
        Get a list of tables in the SQLite database.
        
        Returns:
            list: List of table names
        """
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    
    def get_sqlite_table_schema(self, table_name):
        """
        Get the schema of a table in the SQLite database.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            list: List of column information
        """
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        cursor.close()
        return columns
    
    def get_sqlite_data(self, table_name):
        """
        Get all data from a table in the SQLite database.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            list: List of rows as dictionaries
        """
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]
    
    def map_sqlite_to_postgres(self):
        """
        Create a mapping between SQLite tables/columns and PostgreSQL tables/columns.
        
        Returns:
            dict: Mapping of tables and columns
        """
        # This is a simplified mapping - in a real implementation, this would be more comprehensive
        mapping = {
            'players': {
                'table': 'players',
                'columns': {
                    'id': 'player_id',
                    'name': 'name',
                    'position': 'position',
                    'team_id': 'team_id',
                    # Add more column mappings as needed
                }
            },
            'teams': {
                'table': 'teams',
                'columns': {
                    'id': 'team_id',
                    'name': 'name',
                    'location': 'location',
                    'conference': 'conference',
                    'division': 'division',
                    # Add more column mappings as needed
                }
            },
            'games': {
                'table': 'games',
                'columns': {
                    'id': 'game_id',
                    'home_team_id': 'home_team_id',
                    'away_team_id': 'away_team_id',
                    'date': 'game_date',
                    'venue': 'venue',
                    'status': 'status',
                    # Add more column mappings as needed
                }
            },
            'player_game_stats': {
                'table': 'player_game_stats',
                'columns': {
                    'id': 'stat_id',
                    'player_id': 'player_id',
                    'game_id': 'game_id',
                    'team_id': 'team_id',
                    'minutes_played': 'minutes_played',
                    # Add more column mappings as needed
                }
            },
            'predictions': {
                'table': 'predictions',
                'columns': {
                    'id': 'prediction_id',
                    'player_id': 'player_id',
                    'game_id': 'game_id',
                    'stat_type': 'stat_type',
                    'predicted_value': 'predicted_value',
                    'confidence': 'confidence_score',
                    'over_probability': 'over_probability',
                    'line': 'line_value',
                    'created_at': 'created_at',
                    # Add more column mappings as needed
                }
            }
        }
        return mapping
    
    def transform_data(self, table_name, data, mapping):
        """
        Transform data from SQLite format to PostgreSQL format.
        
        Args:
            table_name (str): Name of the table
            data (list): List of rows as dictionaries
            mapping (dict): Mapping of tables and columns
            
        Returns:
            list: Transformed data
        """
        if table_name not in mapping:
            logger.warning(f"No mapping found for table: {table_name}")
            return []
        
        table_mapping = mapping[table_name]
        transformed_data = []
        
        for row in data:
            transformed_row = {}
            
            # Map columns according to the mapping
            for sqlite_col, postgres_col in table_mapping['columns'].items():
                if sqlite_col in row:
                    transformed_row[postgres_col] = row[sqlite_col]
            
            # Add UUID for primary keys if needed
            if 'id' in table_mapping['columns'] and table_mapping['columns']['id'].endswith('_id'):
                transformed_row[table_mapping['columns']['id']] = str(uuid.uuid4())
            
            # Add timestamps if needed
            if 'created_at' not in transformed_row and 'created_at' in table_mapping['columns'].values():
                transformed_row['created_at'] = datetime.now()
            
            if 'updated_at' not in transformed_row and 'updated_at' in table_mapping['columns'].values():
                transformed_row['updated_at'] = datetime.now()
            
            # Handle JSON fields
            for col in transformed_row:
                if col.endswith('_json') or col in ['attributes', 'team_stats', 'game_stats', 'weather_conditions', 'top_factors']:
                    if isinstance(transformed_row[col], str):
                        try:
                            transformed_row[col] = json.loads(transformed_row[col])
                        except:
                            pass
            
            transformed_data.append(transformed_row)
        
        return transformed_data
    
    def insert_data_to_postgres(self, table_name, data, mapping):
        """
        Insert data into PostgreSQL table.
        
        Args:
            table_name (str): Name of the table
            data (list): List of rows as dictionaries
            mapping (dict): Mapping of tables and columns
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        if not data:
            logger.warning(f"No data to insert for table: {table_name}")
            return True
        
        if table_name not in mapping:
            logger.warning(f"No mapping found for table: {table_name}")
            return False
        
        postgres_table = mapping[table_name]['table']
        
        try:
            # Get column names from the first row
            columns = list(data[0].keys())
            
            # Prepare values
            values = [[row[col] for col in columns] for row in data]
            
            # Create SQL query
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO {postgres_table} ({columns_str}) VALUES ({placeholders})"
            
            # Execute query
            with get_db_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, query, values)
                conn.commit()
                cursor.close()
            
            logger.info(f"Inserted {len(data)} rows into {postgres_table}")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into {postgres_table}: {e}")
            return False
    
    def migrate_table(self, table_name, mapping):
        """
        Migrate a single table from SQLite to PostgreSQL.
        
        Args:
            table_name (str): Name of the table
            mapping (dict): Mapping of tables and columns
            
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            # Get data from SQLite
            sqlite_data = self.get_sqlite_data(table_name)
            logger.info(f"Retrieved {len(sqlite_data)} rows from SQLite table: {table_name}")
            
            # Transform data
            postgres_data = self.transform_data(table_name, sqlite_data, mapping)
            logger.info(f"Transformed {len(postgres_data)} rows for PostgreSQL table: {mapping[table_name]['table']}")
            
            # Insert data into PostgreSQL
            success = self.insert_data_to_postgres(table_name, postgres_data, mapping)
            
            return success
        except Exception as e:
            logger.error(f"Error migrating table {table_name}: {e}")
            return False
    
    def migrate_all_tables(self):
        """
        Migrate all tables from SQLite to PostgreSQL.
        
        Returns:
            bool: True if all migrations successful, False otherwise
        """
        # Get mapping
        mapping = self.map_sqlite_to_postgres()
        
        # Get tables
        tables = self.get_sqlite_tables()
        
        # Migrate each table
        success = True
        for table in tables:
            if table in mapping:
                logger.info(f"Migrating table: {table}")
                table_success = self.migrate_table(table, mapping)
                if not table_success:
                    success = False
            else:
                logger.warning(f"Skipping table not in mapping: {table}")
        
        return success
    
    def run_migration(self):
        """
        Run the complete migration process.
        
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            # Connect to SQLite
            if not self.connect_sqlite():
                return False
            
            # Create PostgreSQL schema
            if not self.create_postgres_schema():
                return False
            
            # Migrate all tables
            if not self.migrate_all_tables():
                return False
            
            logger.info("Migration completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False
        finally:
            # Close SQLite connection
            if self.sqlite_conn:
                self.sqlite_conn.close()

def main():
    """
    Main function to run the migration.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Migrate data from SQLite to PostgreSQL')
    parser.add_argument('--sqlite', required=True, help='Path to SQLite database file')
    parser.add_argument('--schema', required=True, help='Path to PostgreSQL schema SQL file')
    args = parser.parse_args()
    
    # Run migration
    migration = DatabaseMigration(args.sqlite, args.schema)
    success = migration.run_migration()
    
    if success:
        logger.info("Migration completed successfully")
        return 0
    else:
        logger.error("Migration failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
