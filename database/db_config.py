"""
PostgreSQL Configuration for Sports Prediction System

This module provides configuration and setup for the PostgreSQL database,
including connection management, pooling, and migration utilities.
"""

import os
import sys
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('database')

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'sports_prediction'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'min_connections': int(os.environ.get('DB_MIN_CONNECTIONS', '1')),
    'max_connections': int(os.environ.get('DB_MAX_CONNECTIONS', '10'))
}

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL, pool_size=DB_CONFIG['min_connections'], max_overflow=DB_CONFIG['max_connections']-DB_CONFIG['min_connections'])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Connection pool
connection_pool = None

def init_db():
    """
    Initialize the database connection pool and create tables if they don't exist.
    """
    global connection_pool
    
    try:
        # Create connection pool
        connection_pool = pool.ThreadedConnectionPool(
            DB_CONFIG['min_connections'],
            DB_CONFIG['max_connections'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        logger.info(f"Connection pool created with {DB_CONFIG['min_connections']} to {DB_CONFIG['max_connections']} connections")
        
        # Create tables using SQLAlchemy
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

@contextmanager
def get_db_connection():
    """
    Get a database connection from the pool.
    
    Yields:
        connection: Database connection
    """
    connection = None
    try:
        connection = connection_pool.getconn()
        yield connection
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        raise
    finally:
        if connection:
            connection_pool.putconn(connection)

@contextmanager
def get_db_cursor(commit=False):
    """
    Get a database cursor from a connection in the pool.
    
    Args:
        commit (bool): Whether to commit the transaction
        
    Yields:
        cursor: Database cursor
    """
    with get_db_connection() as connection:
        cursor = None
        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            yield cursor
            if commit:
                connection.commit()
        except Exception as e:
            connection.rollback()
            logger.error(f"Error in database transaction: {e}")
            raise
        finally:
            if cursor:
                cursor.close()

def get_db():
    """
    Get a database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_query(query, params=None, fetch=True, commit=False):
    """
    Execute a SQL query.
    
    Args:
        query (str): SQL query
        params (tuple): Query parameters
        fetch (bool): Whether to fetch results
        commit (bool): Whether to commit the transaction
        
    Returns:
        list: Query results if fetch is True, else None
    """
    with get_db_cursor(commit=commit) as cursor:
        cursor.execute(query, params or ())
        if fetch:
            return cursor.fetchall()
        return None

def execute_script(script_path):
    """
    Execute a SQL script file.
    
    Args:
        script_path (str): Path to SQL script
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(script_path, 'r') as f:
            script = f.read()
        
        with get_db_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(script)
            connection.commit()
            cursor.close()
        
        logger.info(f"Executed SQL script: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error executing SQL script {script_path}: {e}")
        return False

def check_connection():
    """
    Check if the database connection is working.
    
    Returns:
        bool: True if connection is working, False otherwise
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

def close_db():
    """
    Close the database connection pool.
    """
    if connection_pool:
        connection_pool.closeall()
        logger.info("Database connection pool closed")

# Initialize database when module is imported
if __name__ != "__main__":
    init_db()
