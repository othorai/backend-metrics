import psycopg2
import psycopg2.extras
from app.connectors.base import BaseConnector
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLConnector(BaseConnector):
    def __init__(self, host, username, password, database, port=5432):
        super().__init__()  # Call parent constructor
        self.source_type = 'postgresql'  # Set the source type
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.port = int(port) if port else 5432
        self.connection = None
        self.schema = 'public'
        self.sslmode = os.getenv('POSTGRES_SSLMODE', 'prefer')

    def connect(self):
        """Establish connection to PostgreSQL with error handling."""
        try:
            logger.info(f"Attempting to connect to PostgreSQL database {self.database} as user {self.username}")
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                dbname=self.database,
                port=self.port,
                sslmode=self.sslmode,
                connect_timeout=10
            )
            # Test connection immediately
            with self.connection.cursor() as cursor:
                cursor.execute('SELECT 1')
                cursor.fetchone()
            logger.info("Successfully connected to PostgreSQL")
            
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL connection error: {str(e)}")
            # Clean up the connection if it was created
            if self.connection:
                self.connection.close()
                self.connection = None
            if "authentication failed" in str(e):
                raise ValueError(f"Authentication failed for user '{self.username}'. Please verify credentials.")
            elif "could not connect to server" in str(e):
                raise ValueError(f"Could not connect to database server at {self.host}:{self.port}. Please verify connection details.")
            else:
                raise ValueError(f"Database connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL connection: {str(e)}")
            if self.connection:
                self.connection.close()
                self.connection = None
            raise ValueError(f"Failed to establish database connection: {str(e)}")

    def disconnect(self):
        """Safely close the connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info("PostgreSQL connection closed")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {str(e)}")

    def query(self, query_string, params=None):
        """Execute query with error handling and automatic reconnection."""
        if not self.connection:
            self.connect()
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query_string, params)
                return cursor.fetchall()
                
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL operational error: {str(e)}")
            # Try to reconnect once
            self.connect()
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query_string, params)
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query_string}")
            logger.error(f"Params: {params}")
            raise ValueError(f"Query execution failed: {str(e)}")

    def insert(self, table, data):
        """Insert data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                columns = ', '.join(data.keys())
                values = ', '.join(['%s'] * len(data))
                query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
                cursor.execute(query, tuple(data.values()))
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Insert operation failed: {str(e)}")
            raise ValueError(f"Failed to insert data: {str(e)}")

    def update(self, table, data, condition):
        """Update data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
                query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
                cursor.execute(query, tuple(data.values()))
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Update operation failed: {str(e)}")
            raise ValueError(f"Failed to update data: {str(e)}")

    def delete(self, table, condition):
        """Delete data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                query = f"DELETE FROM {table} WHERE {condition}"
                cursor.execute(query)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Delete operation failed: {str(e)}")
            raise ValueError(f"Failed to delete data: {str(e)}")