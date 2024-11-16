import snowflake.connector
from app.connectors.base import BaseConnector
import logging

logger = logging.getLogger(__name__)

class SnowflakeConnector(BaseConnector):
    def __init__(self, account, username, password, warehouse, database, schema):
        super().__init__()
        self.source_type = 'snowflake'
        self.account = account
        self.username = username
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.connection = None

    def connect(self):
        """Establish connection to Snowflake with better error handling."""
        try:
            if not self.username:
                raise ValueError("Snowflake username is not provided")
            
            if not self.database:
                raise ValueError("Snowflake database is not provided")
                
            if not self.schema:
                raise ValueError("Snowflake schema is not provided")
            
            logger.info(f"Attempting to connect to Snowflake with user: {self.username}")
            
            connection_params = {
                'account': self.account,
                'user': self.username,
                'password': self.password,
                'warehouse': self.warehouse,
                'database': self.database,
                'schema': self.schema
            }
            
            # Log connection attempt (without password)
            log_params = {k: v for k, v in connection_params.items() if k != 'password'}
            logger.info(f"Connecting to Snowflake with parameters: {log_params}")
            
            self.connection = snowflake.connector.connect(**connection_params)
            
            # Test connection and get session info
            with self.connection.cursor(snowflake.connector.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        CURRENT_WAREHOUSE() as warehouse,
                        CURRENT_DATABASE() as database,
                        CURRENT_SCHEMA() as schema,
                        CURRENT_SESSION() as session
                """)
                info = cursor.fetchone()
                logger.info(
                    f"Successfully connected to Snowflake:"
                    f" Warehouse={info['WAREHOUSE']},"
                    f" Database={info['DATABASE']},"
                    f" Schema={info['SCHEMA']},"
                    f" Session={info['SESSION']}"
                )
            
        except snowflake.connector.errors.ProgrammingError as e:
            error_msg = str(e)
            logger.error(f"Snowflake programming error: {error_msg}")
            
            if "Object does not exist" in error_msg:
                raise ValueError(f"Database object not found: {error_msg}")
            elif "Invalid credentials" in error_msg or "Authentication failed" in error_msg:
                raise ValueError("Invalid Snowflake credentials")
            elif "Account must be specified" in error_msg:
                raise ValueError("Snowflake account must be specified")
            else:
                raise ValueError(f"Snowflake connection error: {error_msg}")
                
        except snowflake.connector.errors.DatabaseError as e:
            error_msg = str(e)
            logger.error(f"Snowflake database error: {error_msg}")
            raise ValueError(f"Database error: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error during Snowflake connection: {error_msg}")
            raise ValueError(f"Connection error: {error_msg}")

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def query(self, query_string, params=None):
        if not self.connection or self.connection.is_closed():
            logger.info("Reconnecting to Snowflake as connection was closed")
            self.connect()
            
        cursor = self.connection.cursor(snowflake.connector.DictCursor)
        try:
            logger.info(f"Executing Snowflake query: {query_string}")
            if params:
                logger.info(f"With parameters: {params}")
                cursor.execute(query_string, params)
            else:
                cursor.execute(query_string)
            
            result = cursor.fetchall()
            logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query string: {query_string}")
            logger.error(f"Parameters: {params}")
            raise
        finally:
            cursor.close()

    def execute_with_result(self, query_string, params=None):
        """Execute a query and ensure we get a result."""
        result = self.query(query_string, params)
        if not result:
            return [{'row_count': 0}]
        return result

    def verify_table_exists(self, table_name: str) -> bool:
        """Verify that a table exists in the current schema."""
        try:
            query = f"""
                SELECT COUNT(*) as table_exists 
                FROM {self.database}.INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{self.schema}' 
                AND TABLE_NAME = '{table_name.upper()}'
            """
            result = self.query(query)
            return result[0]['TABLE_EXISTS'] > 0
        except Exception as e:
            logger.error(f"Error verifying table existence: {str(e)}")
            return False

    def get_column_names(self, table_name: str) -> list:
        """Get column names for a table."""
        try:
            query = f"""
                SELECT COLUMN_NAME 
                FROM {self.database}.INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.schema}' 
                AND TABLE_NAME = '{table_name.upper()}'
                ORDER BY ORDINAL_POSITION
            """
            result = self.query(query)
            return [row['COLUMN_NAME'] for row in result]
        except Exception as e:
            logger.error(f"Error getting column names: {str(e)}")
            return []

    def insert(self, table, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join([f':{i+1}' for i in range(len(data))])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.connection.cursor()
        try:
            params_dict = {str(i+1): val for i, val in enumerate(data.values())}
            cursor.execute(query, params_dict)
            self.connection.commit()
        finally:
            cursor.close()

    def update(self, table, data, condition):
        set_clause = ', '.join([f"{k} = :{i+1}" for i, k in enumerate(data.keys())])
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        cursor = self.connection.cursor()
        try:
            params_dict = {str(i+1): val for i, val in enumerate(data.values())}
            cursor.execute(query, params_dict)
            self.connection.commit()
        finally:
            cursor.close()

    def delete(self, table, condition):
        query = f"DELETE FROM {table} WHERE {condition}"
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
        finally:
            cursor.close()