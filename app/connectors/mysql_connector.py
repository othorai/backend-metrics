import mysql.connector
from app.connectors.base import BaseConnector
import logging

logger = logging.getLogger(__name__)

class MySQLConnector(BaseConnector):
    def __init__(self, host, user, password, database, port=3306):
        super().__init__()
        self.source_type = 'mysql'
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            logger.info(f"Successfully connected to MySQL database: {self.database}")
        except mysql.connector.Error as e:
            logger.error(f"MySQL connection error: {str(e)}")
            raise

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def query(self, query_string, params=None):
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query_string, params)
            else:
                cursor.execute(query_string)
            result = cursor.fetchall()
            return result
        except mysql.connector.Error as e:
            logger.error(f"MySQL query error: {str(e)}")
            logger.error(f"Query: {query_string}")
            logger.error(f"Params: {params}")
            raise
        finally:
            if cursor:
                cursor.close()

    def verify_table_exists(self, table_name: str) -> bool:
        """Verify that a table exists in the current database."""
        try:
            query = """
                SELECT COUNT(*) as table_exists 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = %s
            """
            result = self.query(query, (self.database, table_name))
            return result[0]['table_exists'] > 0
        except Exception as e:
            logger.error(f"Error verifying table existence: {str(e)}")
            return False

    def get_column_names(self, table_name: str) -> list:
        """Get column names for a table."""
        try:
            query = """
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s 
                ORDER BY ORDINAL_POSITION
            """
            result = self.query(query, (self.database, table_name))
            return [row['COLUMN_NAME'] for row in result]
        except Exception as e:
            logger.error(f"Error getting column names: {str(e)}")
            return []

    def insert(self, table, data):
        cursor = None
        try:
            cursor = self.connection.cursor()
            columns = ', '.join(data.keys())
            values = ', '.join(['%s'] * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
            cursor.execute(query, tuple(data.values()))
            self.connection.commit()
        except mysql.connector.Error as e:
            logger.error(f"MySQL insert error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()

    def update(self, table, data, condition):
        cursor = None
        try:
            cursor = self.connection.cursor()
            set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
            cursor.execute(query, tuple(data.values()))
            self.connection.commit()
        except mysql.connector.Error as e:
            logger.error(f"MySQL update error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()

    def delete(self, table, condition):
        cursor = None
        try:
            cursor = self.connection.cursor()
            query = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(query)
            self.connection.commit()
        except mysql.connector.Error as e:
            logger.error(f"MySQL delete error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()