from app.connectors.mysql_connector import MySQLConnector
from app.connectors.postgresql_connector import PostgreSQLConnector
from app.connectors.google_sheets_connector import GoogleSheetsConnector
from app.connectors.salesforce_connector import SalesforceConnector
from app.connectors.snowflake_connector import SnowflakeConnector
from app.connectors.mongodb_connector import MongoDBConnector

class ConnectorFactory:
    @staticmethod
    def get_connector(connector_type, **kwargs):
        if connector_type == 'mysql':
            return MySQLConnector(
                host=kwargs.get('host'),
                user=kwargs.get('username') or kwargs.get('user'),
                password=kwargs.get('password'),
                database=kwargs.get('database')
            )
        elif connector_type == 'postgresql':
            return PostgreSQLConnector(
                host=kwargs.get('host'),
                username=kwargs.get('username') or kwargs.get('user'),
                password=kwargs.get('password'),
                database=kwargs.get('database'),
                port=kwargs.get('port', 5432)
            )
        elif connector_type == 'mongodb':
            return MongoDBConnector(
                host=kwargs.get('host'),
                username=kwargs.get('username') or kwargs.get('user'),
                password=kwargs.get('password'),
                database=kwargs.get('database'),
                port=kwargs.get('port', 27017)
            )
        elif connector_type == 'google_sheets':
            return GoogleSheetsConnector(
                credentials_file=kwargs.get('credentials_file'),
                spreadsheet_id=kwargs.get('spreadsheet_id')
            )
        elif connector_type == 'salesforce':
            return SalesforceConnector(
                username=kwargs.get('user'),
                password=kwargs.get('password'),
                security_token=kwargs.get('security_token'),
                domain=kwargs.get('domain', 'login')
            )
        elif connector_type == 'snowflake':
            return SnowflakeConnector(
                account=kwargs.get('account'),
                username=kwargs.get('username') or kwargs.get('user'),
                password=kwargs.get('password'),
                warehouse=kwargs.get('warehouse'),
                database=kwargs.get('database'),
                schema=kwargs.get('schema')
            )
        else:
            raise ValueError(f"Unsupported connector type: {connector_type}")