from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime
from openai import OpenAI
import json

logger = logging.getLogger(__name__)

class DateColumnDetection:
    """Service to detect and validate the most suitable date column for time series analysis using OpenAI."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    async def detect_date_column(self, connector: Any, table_name: str) -> Optional[str]:
        """
        Detect the most suitable date column from a table using OpenAI.
        
        Args:
            connector: Database connector instance
            table_name: Name of the table to analyze
            
        Returns:
            str: Name of the most suitable date column, or None if not found
        """
        try:
            # Fetch schema and sample data using the working approach
            schema_data, table_schema = await self._fetch_schema(connector, table_name)
            if not schema_data:
                logger.warning(f"No schema found for table {table_name}")
                return None

            # Get sample data
            sample_data = await self._fetch_sample_records(connector, table_name)

            # Identify date columns
            date_columns = await self._identify_date_columns(table_schema)
            if not date_columns:
                logger.warning(f"No date columns found in table {table_name}")
                return None

            # Select the best date column
            date_column = await self._select_date_column(
                date_columns,
                table_schema,
                sample_data,
                connector.source_type
            )

            if date_column and await self._validate_date_column(connector, table_name, date_column):
                logger.info(f"Selected and validated date column: {date_column}")
                return date_column
            return None

        except Exception as e:
            logger.error(f"Error detecting date column: {str(e)}")
            return None

    async def _fetch_schema(self, connector: Any, table_name: str) -> Tuple[List[Dict], Dict]:
        """Fetch schema information based on connector type."""
        try:
            if connector.source_type == 'snowflake':
                schema_query = f"""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM {connector.database}.INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table_name.upper()}'
                    AND TABLE_SCHEMA = '{connector.schema}'
                """
                params = None

            elif connector.source_type == 'mysql':
                schema_query = """
                    SELECT 
                        COLUMN_NAME as column_name,
                        DATA_TYPE as data_type,
                        IS_NULLABLE as is_nullable
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = %s
                    AND TABLE_SCHEMA = %s
                """
                params = (table_name, connector.database)

            else:  # postgresql and others
                schema_query = """
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    AND table_schema = current_schema()
                """
                params = (table_name,)

            # Execute query with appropriate parameters
            schema_data = connector.query(schema_query, params)
            if not schema_data:
                raise ValueError(f"No schema information found for table {table_name}")

            # Create standardized schema dictionary
            table_schema = {}
            for row in schema_data:
                # Handle different case conventions
                column_name = row.get('COLUMN_NAME') or row.get('column_name')
                data_type = row.get('DATA_TYPE') or row.get('data_type')
                is_nullable = str(row.get('IS_NULLABLE') or row.get('is_nullable')).upper() == 'YES'

                # Normalize based on database type
                if connector.source_type == 'snowflake':
                    # Keep Snowflake columns in uppercase
                    pass
                else:
                    # Normalize to lowercase for other databases
                    column_name = column_name.lower()
                    data_type = data_type.lower()

                table_schema[column_name] = {
                    'data_type': data_type,
                    'nullable': is_nullable
                }

            return schema_data, table_schema

        except Exception as e:
            logger.error(f"Error fetching schema: {str(e)}")
            raise

    async def _fetch_sample_records(self, connector: Any, table_name: str) -> List[Dict]:
        """Fetch sample records from the table."""
        try:
            if connector.source_type == 'snowflake':
                query = f"SELECT * FROM {connector.database}.{connector.schema}.{table_name} LIMIT 5"
                params = None
            elif connector.source_type == 'mysql':
                query = f"SELECT * FROM {connector.database}.{table_name} LIMIT 5"
                params = None
            else:
                query = f"SELECT * FROM {table_name} LIMIT 5"
                params = None

            sample_data = connector.query(query, params)

            # Standardize case based on database type
            if connector.source_type == 'snowflake':
                # Keep Snowflake data as is
                return sample_data
            else:
                # Normalize to lowercase for other databases
                return [{k.lower(): v for k, v in row.items()} for row in sample_data]

        except Exception as e:
            logger.error(f"Error fetching sample records: {str(e)}")
            raise

    async def _identify_date_columns(self, table_schema: Dict) -> List[str]:
        """Identify all date-type columns in the schema."""
        date_columns = []
        date_types = {
            'date', 'timestamp', 'datetime', 'timestamptz', 
            'timestamp without time zone', 'timestamp with time zone',
            'timestamp_ntz', 'timestamp_tz', 'timestamp_ltz'  # Added Snowflake specific types
        }
        
        for column_name, info in table_schema.items():
            data_type = info['data_type'].lower()
            if any(date_type in data_type for date_type in date_types):
                date_columns.append(column_name)
                
        return date_columns

    async def _select_date_column(
        self, 
        date_columns: List[str], 
        table_schema: Dict, 
        sample_data: List[Dict],
        source_type: str
    ) -> str:
        """Select the most appropriate date column using GPT."""
        try:
            # Prepare context about the columns
            column_info = []
            for col in date_columns:
                # Get sample values
                sample_values = [str(row.get(col)) for row in sample_data if row.get(col)]
                sample_str = ', '.join(sample_values[:3])
                
                column_info.append({
                    'name': col,
                    'type': table_schema[col]['data_type'],
                    'sample_values': sample_str
                })

            prompt = f"""Analyze these date columns and select the most appropriate for time series analysis:

Columns:
{json.dumps(column_info, indent=2)}

Consider:
1. Primary event dates (created_at, event_date, transaction_date)
2. Avoid auxiliary dates (updated_at, modified_date, deleted_at)
3. Data completeness and time patterns
4. Standard conventions for {source_type}

Return ONLY the column name."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a database expert. Respond only with the column name."},
                    {"role": "user", "content": prompt}
                ]
            )

            suggested_column = response.choices[0].message.content.strip()
            
            # Verify the suggested column exists
            if suggested_column in date_columns:
                return suggested_column
            
            # Fallback to first date column
            logger.warning(f"Suggested column {suggested_column} not found, using {date_columns[0]}")
            return date_columns[0]

        except Exception as e:
            logger.error(f"Error selecting date column: {str(e)}")
            return date_columns[0] if date_columns else None

    async def _validate_date_column(self, connector: Any, table_name: str, column_name: str) -> bool:
        """Validate if a column is suitable for time series analysis."""
        try:
            # Adjust query based on database type
            if connector.source_type == 'snowflake':
                validation_query = f"""
                    SELECT 
                        COUNT(*) as TOTAL_ROWS,
                        COUNT("{column_name}") as NON_NULL_ROWS,
                        MIN("{column_name}") as MIN_DATE,
                        MAX("{column_name}") as MAX_DATE
                    FROM {connector.database}.{connector.schema}.{table_name}
                """
            else:
                validation_query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT({column_name}) as non_null_rows,
                        MIN({column_name}) as min_date,
                        MAX({column_name}) as max_date
                    FROM {table_name}
                """
            
            result = connector.query(validation_query)
            if not result:
                return False
            
            stats = {k.lower(): v for k, v in result[0].items()}
            
            total_rows = int(stats['total_rows'])
            non_null_rows = int(stats['non_null_rows'])
            
            if total_rows == 0 or non_null_rows / total_rows < 0.9:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating date column {column_name}: {str(e)}")
            return False