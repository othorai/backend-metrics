# services/metric_discovery.py

import logging
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import json
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from app.models.models import MetricDefinition, DataSourceConnection
from app.connectors.connector_factory import ConnectorFactory
from app.connectors.mysql_connector import MySQLConnector
from app.connectors.postgresql_connector import PostgreSQLConnector
from app.connectors.snowflake_connector import SnowflakeConnector
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# services/metric_discovery.py

class MetricDiscoveryService:
    def __init__(self, client: OpenAI):
        self.client = client

    def analyze_data_structure(self, sample_data: List[Dict], table_schema: Dict, table_name: str) -> str:
        """Generate prompt for metric discovery."""
        schema_description = json.dumps(table_schema, indent=2)
        sample_data_str = json.dumps(sample_data[:5], indent=2, cls=CustomJSONEncoder)
        
        system_message = """You are a data analyst expert in discovering meaningful business metrics from data structures. 
        Return ONLY a JSON array of metric definitions without any additional text, explanation, or markdown formatting."""
        
        prompt = f"""Based on this database structure, generate appropriate metrics:

    Table Structure:
    {schema_description}

    Sample Data:
    {sample_data_str}

    Generate SQL metrics that:
    1. Use appropriate aggregations (SUM, AVG, COUNT, etc.)
    2. Return a single numeric value
    3. Handle NULL values appropriately
    4. Focus on meaningful business insights

    Return ONLY a JSON array containing 8-10 metrics in this exact format:
    [
    {{
        "name": "metric_name",
        "category": "category_name",
        "calculation": "SQL_calculation",
        "required_columns": ["column1", "column2"],
        "aggregation_period": "period",
        "visualization_type": "chart_type",
        "business_context": "explanation",
        "confidence_score": score
    }}
    ]

    Focus on metrics related to:
    1. Employee Performance
    2. Training and Development
    3. Satisfaction and Retention
    4. Workload and Efficiency
    5. Compensation and Growth

    IMPORTANT: Return only the JSON array with no additional text or formatting."""
        
        return system_message, prompt

    async def discover_metrics(self, connection_id: int, db: Session) -> List[MetricDefinition]:
        """Discover metrics from any data source."""
        connector = None
        try:
            # Get connection details and create connector
            connection = db.query(DataSourceConnection).filter_by(id=connection_id).first()
            if not connection:
                raise ValueError(f"Connection {connection_id} not found")

            connector = ConnectorFactory.get_connector(
                connection.source_type,
                **connection.connection_params
            )
            
            # Get sample data and generate prompt
            connector.connect()
            sample_data, table_schema = await self.fetch_sample_data(connector, connection.table_name)
            system_message, prompt = self.analyze_data_structure(sample_data, table_schema, connection.table_name)
            
            # Get metrics from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse and validate metrics
            metrics_data = self.parse_openai_response(response.choices[0].message.content)
            logger.info(f"Successfully parsed {len(metrics_data)} metrics")
            
            # Create and validate metric definitions
            metric_definitions = []
            for metric_data in metrics_data:
                try:
                    # Validate query
                    test_query = f"""
                        SELECT 
                            {metric_data['calculation']} as metric_value
                        FROM {connection.table_name}
                        LIMIT 1
                    """
                    
                    logger.info(f"Testing query for {metric_data['name']}: {test_query}")
                    connector.query(test_query)
                    logger.info(f"Query validation successful for {metric_data['name']}")
                    
                    # Create metric definition
                    metric = MetricDefinition(
                        connection_id=connection_id,
                        name=metric_data["name"],
                        category=metric_data["category"],
                        calculation=metric_data["calculation"],
                        data_dependencies=metric_data["required_columns"],
                        aggregation_period=metric_data["aggregation_period"],
                        visualization_type=metric_data["visualization_type"],
                        business_context=metric_data.get("business_context", ""),
                        confidence_score=float(metric_data["confidence_score"])
                    )
                    
                    db.add(metric)
                    metric_definitions.append(metric)
                    
                except Exception as e:
                    logger.error(f"Error processing metric {metric_data['name']}: {str(e)}")
                    continue
            
            if not metric_definitions:
                raise ValueError("No valid metrics could be generated")
            
            db.commit()
            return metric_definitions
            
        except Exception as e:
            logger.error(f"Error discovering metrics: {str(e)}")
            db.rollback()
            raise
        finally:
            if connector:
                try:
                    connector.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting: {str(e)}")

    async def fetch_sample_data(self, connector: Any, table_name: str) -> Tuple[List[Dict], Dict]:
        """Fetch sample data and schema information from the data source."""
        try:
            # Different schema queries for different databases
            if isinstance(connector, MySQLConnector):
                schema_query = """
                    SELECT 
                        COLUMN_NAME as column_name, 
                        DATA_TYPE as data_type, 
                        IS_NULLABLE as is_nullable
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_NAME = %s 
                    AND TABLE_SCHEMA = DATABASE()
                """
                params = (table_name,)
                schema_data = connector.query(schema_query, params)
            elif isinstance(connector, PostgreSQLConnector):
                schema_query = """
                    SELECT 
                        LOWER(column_name) as column_name,
                        LOWER(data_type) as data_type,
                        is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND table_schema = current_schema()
                """
                params = (table_name,)
                schema_data = connector.query(schema_query, params)

            elif isinstance(connector, SnowflakeConnector):
                # Snowflake specific query
                schema_query = f"""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM {connector.database}.INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = %s 
                    AND TABLE_SCHEMA = '{connector.schema}'
                """
                schema_data = connector.query(schema_query, (table_name,))

                if not schema_data:
                    raise ValueError(f"No schema information found for table {table_name}")
                    
                # Standardize Snowflake schema data specifically
                table_schema = {}
                for row in schema_data:
                    # Handle uppercase column names from Snowflake
                    column_name = str(row.get('COLUMN_NAME', '')).lower()
                    data_type = str(row.get('DATA_TYPE', '')).lower()
                    is_nullable = str(row.get('IS_NULLABLE', '')).upper() == 'YES'
                    
                    table_schema[column_name] = {
                        'data_type': data_type,
                        'nullable': is_nullable
                    }

                logger.info(f"Retrieved schema for table {table_name}: {table_schema}")

                # Get sample data with fully qualified path
                sample_query = f"SELECT * FROM {connector.database}.{connector.schema}.{table_name} LIMIT 5"
                sample_data = connector.query(sample_query)
                
                # Standardize Snowflake sample data specifically
                standardized_sample_data = []
                for row in sample_data:
                    standardized_row = {}
                    for key, value in row.items():
                        # Handle uppercase keys from Snowflake
                        standardized_row[str(key).lower()] = value
                    standardized_sample_data.append(standardized_row)

                return standardized_sample_data, table_schema
            else:
                # Generic fallback for other databases
                schema_query = """
                    SELECT 
                        LOWER(column_name) as column_name,
                        LOWER(data_type) as data_type,
                        is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """
                params = (table_name,)
                schema_data = connector.query(schema_query, params)

            logger.info(f"Executing schema query for {type(connector).__name__}")
            
            if not schema_data:
                raise ValueError(f"No schema information found for table {table_name}")

            # Standardize schema data
            table_schema = {
                row['column_name'].lower(): {
                    'data_type': row['data_type'].lower(),
                    'nullable': row['is_nullable'] == 'YES'
                }
                for row in schema_data
            }

            logger.info(f"Retrieved schema for table {table_name}: {table_schema}")

            # Get sample data with simple query
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            sample_data = connector.query(sample_query)
            
            # Standardize case in sample data
            standardized_sample_data = [
                {k.lower(): v for k, v in row.items()}
                for row in sample_data
            ]

            return standardized_sample_data, table_schema

        except Exception as e:
            logger.error(f"Error fetching sample data and schema: {str(e)}")
            logger.error(f"Connector type: {type(connector).__name__}")
            logger.error(f"Table name: {table_name}")
            raise

    def parse_openai_response(self, response_content: str) -> List[Dict]:
        """Parse OpenAI response and extract metric definitions."""
        try:
            # Clean up the response content
            content = response_content.strip()
            
            # Extract content between ```json and ``` if present
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            else:
                # Try to find a JSON array anywhere in the content
                array_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
                if array_match:
                    content = array_match.group(0)
                
            # Remove any trailing text after the JSON array
            content = re.sub(r'\]\s*[^]\s}]*$', ']', content)
            
            logger.info(f"Cleaned JSON content: {content}")
            
            # Parse the JSON
            metrics_data = json.loads(content)
            
            if not isinstance(metrics_data, list):
                metrics_data = [metrics_data]
                
            # Validate each metric
            valid_metrics = []
            required_fields = ["name", "category", "calculation", "required_columns", 
                            "aggregation_period", "visualization_type", "confidence_score"]
            
            for metric in metrics_data:
                if all(field in metric for field in required_fields):
                    valid_metrics.append(metric)
                else:
                    logger.warning(f"Skipping invalid metric: {metric}")
                    
            if not valid_metrics:
                raise ValueError("No valid metrics found in response")
                
            logger.info(f"Successfully parsed {len(valid_metrics)} metrics")
            return valid_metrics
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}")
            logger.error(f"Raw response content: {response_content}")
            raise ValueError(f"Invalid metrics data format from OpenAI: {str(e)}")

    def _categorize_columns(self, schema: Dict) -> Dict[str, List[str]]:
        """Categorize columns by data type."""
        categories = {
            'numeric': [],
            'temporal': [],
            'categorical': [],
            'boolean': []
        }
        
        numeric_types = ['integer', 'numeric', 'decimal', 'double', 'float', 'real']
        temporal_types = ['date', 'timestamp', 'time']
        boolean_types = ['boolean', 'bool']
        
        for column, info in schema.items():
            data_type = info['data_type'].lower()
            if any(t in data_type for t in numeric_types):
                categories['numeric'].append(column)
            elif any(t in data_type for t in temporal_types):
                categories['temporal'].append(column)
            elif any(t in data_type for t in boolean_types):
                categories['boolean'].append(column)
            else:
                categories['categorical'].append(column)
        
        return categories