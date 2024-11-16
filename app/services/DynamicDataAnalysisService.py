from typing import Dict, List, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import logging
from app.models.models import DataSourceConnection, MetricDefinition
import numpy as np
import math
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)

class DynamicAnalysisService:
    def __init__(self):
        self.cached_schemas = {}
        self.cache_duration = timedelta(minutes=15)
        self.cache_timestamp = None

    async def analyze_data(
        self,
        db: Session,
        connection: DataSourceConnection,
        question: str
    ) -> Dict[str, Any]:
        """
        Dynamically analyze data based on the question and available metrics.
        """
        try:
            # Get metrics for this connection
            metrics = db.query(MetricDefinition).filter(
                MetricDefinition.connection_id == connection.id,
                MetricDefinition.is_active == True
            ).all()

            if not metrics:
                return {"error": "No metrics defined for this data source"}

            # Get table schema if not cached
            schema = await self._get_table_schema(connection)
            if not schema:
                return {"error": "Could not retrieve table schema"}

            # Analyze question to determine required metrics
            required_metrics = self._identify_relevant_metrics(question, metrics)
            if not required_metrics:
                return {"error": "No relevant metrics found for this question"}

            # Build and execute dynamic query
            query = self._build_dynamic_query(
                connection.table_name,
                connection.date_column,
                required_metrics,
                schema
            )

            # Execute query and get results
            results = await self._execute_query(connection, query)
            
            # Format results based on question type
            formatted_results = self._format_results(
                results,
                required_metrics,
                question
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            return {"error": str(e)}

    async def _get_table_schema(self, connection: DataSourceConnection) -> Dict[str, str]:
        """Dynamically fetch and cache table schema."""
        cache_key = f"{connection.id}_{connection.table_name}"
        
        if self._is_cache_valid(cache_key):
            return self.cached_schemas[cache_key]

        try:
            if connection.source_type == 'postgresql':
                schema_query = """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """
                params = (connection.table_name,)
            elif connection.source_type == 'mysql':
                schema_query = """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """
                params = (connection.table_name,)
            elif connection.source_type == 'snowflake':
                schema_query = f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = '{connection.table_name.upper()}'
                """
                params = None
            else:
                raise ValueError(f"Unsupported source type: {connection.source_type}")

            connector = self._get_connector(connection)
            schema_data = connector.query(schema_query, params)

            # Process and cache schema
            schema = {
                row['column_name'].lower(): {
                    'type': row['data_type'].lower(),
                    'nullable': row['is_nullable'].lower() == 'yes'
                }
                for row in schema_data
            }

            self.cached_schemas[cache_key] = schema
            self.cache_timestamp = datetime.utcnow()

            return schema

        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            return {}

    def _identify_relevant_metrics(
        self,
        question: str,
        metrics: List[MetricDefinition]
    ) -> List[MetricDefinition]:
        """Identify metrics relevant to the question."""
        question_lower = question.lower()
        relevant_metrics = []

        # Map common terms to metric categories
        category_mappings = {
            'revenue': ['revenue', 'sales', 'income'],
            'performance': ['performance', 'metrics', 'kpi'],
            'customer': ['customer', 'satisfaction', 'nps'],
            'cost': ['cost', 'expense', 'spending'],
            'growth': ['growth', 'increase', 'trend']
        }

        # Find metrics matching question context
        for metric in metrics:
            if any(term in question_lower for term in category_mappings.get(metric.category, [])):
                relevant_metrics.append(metric)
            elif metric.name.lower() in question_lower:
                relevant_metrics.append(metric)
            elif any(dep.lower() in question_lower for dep in metric.data_dependencies):
                relevant_metrics.append(metric)

        return relevant_metrics or metrics[:5]  # Return top 5 metrics if no specific matches

    def _build_dynamic_query(
        self,
        table_name: str,
        date_column: str,
        metrics: List[MetricDefinition],
        schema: Dict[str, Dict]
    ) -> str:
        """Build dynamic SQL query based on metrics and schema."""
        try:
            # Prepare metric calculations
            metric_calculations = []
            for metric in metrics:
                calculation = self._sanitize_calculation(metric.calculation, schema)
                metric_calculations.append(f"{calculation} as {metric.name}")

            # Identify dimension columns (excluding date column)
            dimensions = self._identify_dimensions(schema)
            dimensions = [d for d in dimensions if d.lower() != date_column.lower()]
            
            # Create dimension clause for GROUP BY
            dimension_clause = ', '.join(dimensions) if dimensions else ''
            
            # For Snowflake, handle case sensitivity
            if table_name.isupper():
                # Snowflake query
                query = f"""
                    WITH metric_data AS (
                        SELECT 
                            DATE_TRUNC('day', "{date_column}") as grouped_date,
                            {', '.join([f'"{d}"' for d in dimensions]) + ',' if dimensions else ''}
                            {', '.join(metric_calculations)}
                        FROM "{table_name}"
                        WHERE "{date_column}" >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY grouped_date {', ' + ', '.join([f'"{d}"' for d in dimensions]) if dimensions else ''}
                    )
                    SELECT *
                    FROM metric_data
                    ORDER BY grouped_date DESC
                """
            else:
                # PostgreSQL/MySQL query
                query = f"""
                    WITH metric_data AS (
                        SELECT 
                            DATE_TRUNC('day', {date_column}) as grouped_date,
                            {dimension_clause + ',' if dimension_clause else ''}
                            {', '.join(metric_calculations)}
                        FROM {table_name}
                        WHERE {date_column} >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY grouped_date {', ' + dimension_clause if dimension_clause else ''}
                    )
                    SELECT *
                    FROM metric_data
                    ORDER BY grouped_date DESC
                """

            logger.debug(f"Generated query: {query}")
            return query

        except Exception as e:
            logger.error(f"Error building query: {str(e)}")
            raise
    async def _execute_query(
        self,
        connection: DataSourceConnection,
        query: str
    ) -> List[Dict[str, Any]]:
        """Execute query using appropriate connector."""
        try:
            connector = self._get_connector(connection)
            results = connector.query(query)
            return results
        finally:
            if connector:
                connector.disconnect()

    def _format_results(
        self,
        results: List[Dict[str, Any]],
        metrics: List[MetricDefinition],
        question: str
    ) -> Dict[str, Any]:
        """Format results based on question context and metrics."""
        try:
            df = pd.DataFrame(results)
            formatted_data = {
                "metrics": {},
                "trends": {},
                "dimensions": {},
                "summary": ""
            }

            # Calculate metric summaries
            for metric in metrics:
                metric_data = df[metric.name].agg(['sum', 'mean', 'min', 'max']).to_dict()
                formatted_data["metrics"][metric.name] = {
                    "total": metric_data['sum'],
                    "average": metric_data['mean'],
                    "range": {
                        "min": metric_data['min'],
                        "max": metric_data['max']
                    }
                }

            # Add dimensional breakdowns if available
            dimension_cols = [col for col in df.columns if col not in [metric.name for metric in metrics]]
            for dim in dimension_cols:
                if dim != 'date':
                    dim_summary = df.groupby(dim)[metrics[0].name].sum().sort_values(ascending=False)
                    formatted_data["dimensions"][dim] = dim_summary.to_dict()

            # Generate overall summary
            highest_metric = max(formatted_data["metrics"].items(), key=lambda x: x[1]["total"])
            formatted_data["summary"] = (
                f"Analysis shows {highest_metric[0]} with a total of {highest_metric[1]['total']:,.2f}. "
            )

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return {"error": str(e)}

    def _sanitize_calculation(self, calculation: str, schema: Dict[str, Dict]) -> str:
        """Sanitize calculation based on schema."""
        try:
            # Remove any potential SQL injection attempts
            forbidden_keywords = ['delete', 'drop', 'truncate', 'insert', 'update']
            calculation_lower = calculation.lower()
            
            for keyword in forbidden_keywords:
                if keyword in calculation_lower:
                    raise ValueError(f"Invalid calculation containing forbidden keyword: {keyword}")

            # Handle aggregate functions properly
            agg_functions = ['sum', 'avg', 'min', 'max', 'count']
            for func in agg_functions:
                if func in calculation_lower:
                    # Already has aggregation, return as is
                    return calculation

            # If no aggregation found, wrap in AVG by default
            return f"AVG({calculation})"

        except Exception as e:
            logger.error(f"Error sanitizing calculation: {str(e)}")
            raise

    def _identify_dimensions(self, schema: Dict[str, Dict]) -> List[str]:
        """Identify dimensional columns from schema."""
        dimensions = []
        categorical_types = [
            'character varying', 'varchar', 'text', 'char',
            'nvarchar', 'nchar', 'string'
        ]
        
        for column, info in schema.items():
            col_type = info['type'].lower()
            # Skip common metric and date columns
            if (col_type in categorical_types and 
                not any(metric_word in column.lower() 
                    for metric_word in ['date', 'time', 'amount', 'value', 'total'])):
                dimensions.append(column)
                
        return dimensions

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached schema is still valid."""
        return (
            cache_key in self.cached_schemas
            and self.cache_timestamp
            and datetime.utcnow() - self.cache_timestamp < self.cache_duration
        )

    def _get_connector(self, connection: DataSourceConnection):
        """Get appropriate database connector."""
        from connectors.connector_factory import ConnectorFactory
        return ConnectorFactory.get_connector(
            connection.source_type,
            **connection.connection_params
        )
    
    async def analyze_metrics(
        self,
        db: Session,
        org_id: int,
        scope: str = "this_year",
        resolution: str = "monthly",
        forecast: bool = False
    ) -> Dict[str, Any]:
        """Analyze metrics across all data sources with optional forecasting."""
        try:
            # Get all data sources
            connections = db.query(DataSourceConnection).filter(
                DataSourceConnection.organization_id == org_id
            ).all()

            if not connections:
                return self._format_empty_response(scope, resolution)

            aggregated_metrics = {}

            for connection in connections:
                try:
                    # Get metrics for this connection
                    metrics = db.query(MetricDefinition).filter(
                        MetricDefinition.connection_id == connection.id,
                        MetricDefinition.is_active == True
                    ).all()

                    if not metrics:
                        continue

                    # Build and execute query
                    results = await self._fetch_metric_data(
                        connection=connection,
                        metrics=metrics,
                        scope=scope,
                        resolution=resolution
                    )
                    
                    if results:
                        source_metrics = self._process_source_metrics(
                            results=results,
                            metrics=metrics,
                            source_name=connection.name
                        )
                        self._merge_metrics(aggregated_metrics, source_metrics)

                except Exception as e:
                    logger.error(f"Error processing metrics from {connection.name}: {str(e)}")
                    continue

            # Format and return response
            return self._format_metrics_response(
                metrics=aggregated_metrics,
                scope=scope,
                resolution=resolution,
                has_forecast=forecast
            )

        except Exception as e:
            logger.error(f"Error analyzing metrics: {str(e)}")
            return self._format_empty_response(scope, resolution)

    def _get_date_trunc_unit(self, resolution: str, database_type: str) -> str:
        """
        Get appropriate date truncation unit based on resolution and database type.
        
        Args:
            resolution: Desired time resolution (daily, weekly, monthly, quarterly)
            database_type: Type of database (postgresql, snowflake, mysql)
            
        Returns:
            Correct date truncation unit for the specific database
        """
        # PostgreSQL date trunc units
        postgres_units = {
            'daily': 'day',
            'weekly': 'week',
            'monthly': 'month',
            'quarterly': 'quarter',
            'yearly': 'year'
        }

        # Snowflake date trunc units
        snowflake_units = {
            'daily': 'DAY',
            'weekly': 'WEEK',
            'monthly': 'MONTH',
            'quarterly': 'QUARTER',
            'yearly': 'YEAR'
        }

        # MySQL date trunc units (using different syntax)
        mysql_units = {
            'daily': '%Y-%m-%d',
            'weekly': '%Y-%U',
            'monthly': '%Y-%m',
            'quarterly': '%Y-%m',  # Will need special handling
            'yearly': '%Y'
        }

        database_type = database_type.lower()
        resolution = resolution.lower()

        if database_type == 'postgresql':
            return postgres_units.get(resolution, 'month')
        elif database_type == 'snowflake':
            return snowflake_units.get(resolution, 'MONTH')
        elif database_type == 'mysql':
            return mysql_units.get(resolution, '%Y-%m')
        else:
            return postgres_units.get(resolution, 'month')

    def _build_date_trunc_expression(
        self,
        date_column: str,
        resolution: str,
        database_type: str
    ) -> str:
        """
        Build appropriate date truncation SQL expression for different databases.
        
        Args:
            date_column: Name of the date column
            resolution: Desired time resolution
            database_type: Type of database
            
        Returns:
            SQL expression for date truncation
        """
        database_type = database_type.lower()
        trunc_unit = self._get_date_trunc_unit(resolution, database_type)

        if database_type == 'mysql':
            if resolution == 'quarterly':
                return f"""
                    DATE_FORMAT(
                        DATE_SUB({date_column}, 
                        INTERVAL (MONTH({date_column}) - 1) %% 3 MONTH),
                        '%Y-%m-01'
                    )
                """
            else:
                return f"DATE_FORMAT({date_column}, '{trunc_unit}')"
        elif database_type == 'snowflake':
            return f"DATE_TRUNC('{trunc_unit}', {date_column})"
        else:  # PostgreSQL and others
            return f"DATE_TRUNC('{trunc_unit}', {date_column})"

    async def _fetch_metric_data(
        self,
        connection: DataSourceConnection,
        metrics: List[MetricDefinition],
        scope: str,
        resolution: str
    ) -> List[Dict[str, Any]]:
        """Fetch metric data from a data source."""
        try:
            # Get date range
            start_date, end_date = self._get_date_range(scope)
            
            # Build metric calculations
            metric_calculations = []
            for metric in metrics:
                calc = self._sanitize_calculation(metric.calculation, {})
                metric_calculations.append(f"{calc} as {metric.name}")

            # Build date truncation expression
            period_expression = self._build_date_trunc_expression(
                connection.date_column,
                resolution,
                connection.source_type
            )

            # Build the complete query
            query = f"""
            WITH metric_data AS (
                SELECT 
                    {period_expression} as period,
                    {', '.join(metric_calculations)}
                FROM {connection.table_name}
                WHERE {connection.date_column} BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY period
            )
            SELECT * FROM metric_data
            ORDER BY period DESC
            """

            # Execute query
            connector = self._get_connector(connection)
            results = connector.query(query)
            
            return results

        except Exception as e:
            logger.error(f"Error fetching metric data: {str(e)}")
            raise
    
    def _process_query_results(
        self,
        results: List[Dict[str, Any]],
        resolution: str
    ) -> List[Dict[str, Any]]:
        """
        Process query results and standardize date formats.
        
        Args:
            results: Raw query results
            resolution: Time resolution used
            
        Returns:
            Processed results with standardized dates
        """
        try:
            processed_results = []
            for row in results:
                processed_row = {}
                for key, value in row.items():
                    if key == 'period':
                        # Convert period to standard ISO format
                        if isinstance(value, (datetime, pd.Timestamp)):
                            processed_row[key] = value.isoformat()
                        else:
                            # Try to parse string date
                            try:
                                date_obj = pd.to_datetime(value)
                                processed_row[key] = date_obj.isoformat()
                            except:
                                processed_row[key] = value
                    else:
                        # Handle numeric values
                        if isinstance(value, (int, float, Decimal)):
                            processed_row[key] = float(value)
                        else:
                            processed_row[key] = value
                processed_results.append(processed_row)

            return processed_results

        except Exception as e:
            logger.error(f"Error processing query results: {str(e)}")
            return results

    def _process_source_metrics(
        self,
        results: List[Dict[str, Any]],
        metrics: List[MetricDefinition],
        source_name: str
    ) -> Dict[str, Any]:
        """Process raw metrics results into structured format."""
        try:
            processed_metrics = {}
            df = pd.DataFrame(results)

            for metric in metrics:
                try:
                    if metric.name not in df.columns:
                        continue

                    current_value = float(df[metric.name].iloc[0]) if not df.empty else 0
                    previous_value = float(df[metric.name].iloc[1]) if len(df) > 1 else 0

                    processed_metrics[metric.name] = {
                        "current": current_value,
                        "previous": previous_value,
                        "source": source_name,
                        "category": metric.category,
                        "visualization_type": metric.visualization_type,
                        "trend_data": self._get_trend_data(df, metric.name),
                        "dimensions": self._get_dimensional_data(df, metric.name)
                    }

                except Exception as e:
                    logger.error(f"Error processing metric {metric.name}: {str(e)}")
                    continue

            return processed_metrics

        except Exception as e:
            logger.error(f"Error processing source metrics: {str(e)}")
            return {}



    def _build_metrics_query(
        self,
        table_name: str,
        date_column: str,
        metrics: List[MetricDefinition],
        schema: Dict[str, Dict],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> str:
        """Build SQL query for metrics analysis."""
        try:
            # Get date truncation based on resolution
            date_trunc = self._get_date_trunc(resolution, date_column)
            
            # Process metrics calculations
            metric_calculations = []
            for metric in metrics:
                calc = self._sanitize_calculation(metric.calculation, schema)
                metric_calculations.append(f"{calc} as {metric.name}")

            # Get dimensions
            dimensions = self._identify_dimensions(schema)
            dimension_clause = ', '.join(dimensions) if dimensions else ''

            # Build query
            query = f"""
                WITH base_data AS (
                    SELECT 
                        {date_trunc} as period,
                        {dimension_clause + ',' if dimension_clause else ''}
                        {', '.join(metric_calculations)}
                    FROM {table_name}
                    WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
                    GROUP BY period {', ' + dimension_clause if dimension_clause else ''}
                )
                SELECT * FROM base_data
                ORDER BY period DESC
            """

            return query

        except Exception as e:
            logger.error(f"Error building metrics query: {str(e)}")
            raise

    async def _generate_forecasts(
        self,
        results: List[Dict],
        metrics: List[MetricDefinition],
        resolution: str
    ) -> Dict[str, Any]:
        """Generate forecasts for all applicable metrics."""
        forecasts = {}
        
        for metric in metrics:
            if metric.visualization_type in ['line', 'bar', 'area']:  # Metrics suitable for forecasting
                try:
                    df = pd.DataFrame(results)
                    df = df.set_index('period')[[metric.name]]

                    forecast_result = await self._forecast_metric(
                        data=df,
                        metric_name=metric.name,
                        resolution=resolution
                    )
                    
                    if forecast_result:
                        forecasts[metric.name] = forecast_result

                except Exception as e:
                    logger.error(f"Error forecasting metric {metric.name}: {str(e)}")
                    continue

        return forecasts

    def _get_forecast_period(self, duration: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get start and end dates for forecast based on duration.
        
        Args:
            duration: Forecast duration (next_week, next_month, next_quarter, next_year)
            
        Returns:
            Tuple of (start_date, end_date)
        """
        current_date = pd.Timestamp.now().normalize()
        
        if duration == 'next_week':
            # Start from next Monday
            next_monday = current_date + pd.Timedelta(days=(7 - current_date.weekday()))
            return next_monday, next_monday + pd.Timedelta(days=6)
        
        elif duration == 'next_month':
            # Start from 1st of next month
            if current_date.month == 12:
                start_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
            else:
                start_date = pd.Timestamp(year=current_date.year, month=current_date.month + 1, day=1)
            end_date = start_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            return start_date, end_date
        
        elif duration == 'next_quarter':
            # Start from beginning of next quarter
            current_quarter = (current_date.month - 1) // 3
            next_quarter_start_month = 3 * (current_quarter + 1) + 1
            
            if next_quarter_start_month > 12:
                start_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
            else:
                start_date = pd.Timestamp(year=current_date.year, month=next_quarter_start_month, day=1)
            
            end_date = start_date + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            return start_date, end_date
        
        elif duration == 'next_year':
            # Start from January 1st of next year
            start_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
            end_date = pd.Timestamp(year=current_date.year + 2, month=1, day=1) - pd.Timedelta(days=1)
            return start_date, end_date
        
        else:  # default to next month
            if current_date.month == 12:
                start_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
            else:
                start_date = pd.Timestamp(year=current_date.year, month=current_date.month + 1, day=1)
            end_date = start_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            return start_date, end_date

    def generate_forecast(
        self,
        db: Session,
        org_id: int,
        metric: MetricDefinition,
        duration: str,
        resolution: str
    ) -> Dict[str, Any]:
        """Generate forecast for a specific metric."""
        try:
            # Get historical data and prepare DataFrame
            historical_data = self._get_metric_history(
                db=db,
                org_id=org_id,
                metric=metric,
                lookback_days=365
            )

            if not historical_data:
                raise ValueError("No historical data available for forecasting")

            df = pd.DataFrame(historical_data, columns=['period', 'value'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna()
            df = df.sort_values('period').reset_index(drop=True)
            df['ds'] = pd.to_datetime(df['period']).dt.tz_localize(None)
            df['y'] = df['value']

            # Get forecast period
            forecast_start, forecast_end = self._get_forecast_period(duration)
            
            # Generate date range based on resolution
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'MS',  # Month Start
                'quarterly': 'QS'  # Quarter Start
            }
            freq = freq_map.get(resolution, 'D')
            
            forecast_dates = pd.date_range(
                start=forecast_start,
                end=forecast_end,
                freq=freq
            )
            
            forecast_horizon = len(forecast_dates)

            # Generate forecasts using multiple models
            forecasts = []
            metrics = {}

            # Prophet forecast
            try:
                prophet_values, prophet_metrics = self._prophet_forecast(df.copy(), forecast_horizon)
                if prophet_values is not None:
                    # Resample prophet values to match desired frequency
                    prophet_df = pd.DataFrame({
                        'ds': forecast_dates,
                        'y': prophet_values[:forecast_horizon]
                    })
                    forecasts.append(prophet_df['y'].values)
                    metrics['prophet'] = prophet_metrics
            except Exception as e:
                logger.error(f"Prophet forecast failed: {str(e)}")

            # Add other models similarly...

            if not forecasts:
                raise ValueError("All forecasting methods failed")

            # Calculate ensemble forecast
            ensemble_forecast = np.mean(forecasts, axis=0)

            # Format forecast data
            forecast_data = {
                "metric_name": metric.name,
                "forecast_points": [
                    {
                        "date": date.isoformat(),
                        "value": float(value),
                        "confidence_interval": {
                            "lower": float(value * 0.9),
                            "upper": float(value * 1.1)
                        }
                    }
                    for date, value in zip(forecast_dates, ensemble_forecast)
                    if not (math.isnan(value) or math.isinf(value))
                ],
                "metadata": {
                    "start_date": forecast_start.isoformat(),
                    "end_date": forecast_end.isoformat(),
                    "duration": duration,
                    "resolution": resolution,
                    "source": metric.connection.name,
                    "model_metrics": metrics,
                    "data_points_used": len(df),
                    "forecast_points": len(forecast_dates)
                }
            }

            return forecast_data

        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
            raise

    def _prophet_forecast(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate forecast using Prophet."""
        try:
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
            model.fit(df)
            
            # Create future dates starting from tomorrow
            future_dates = pd.date_range(
                start=pd.Timestamp.now().normalize() + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make prediction
            forecast = model.predict(future_df)
            
            # Get forecast values and uncertainty intervals
            forecast_values = forecast['yhat'].values
            lower_bound = forecast['yhat_lower'].values
            upper_bound = forecast['yhat_upper'].values
            
            # Calculate metrics using the last portion of historical data
            test_size = min(forecast_horizon, len(df))
            historical_values = df['y'].tail(test_size).values
            predicted_historical = model.predict(df.tail(test_size))['yhat'].values
            
            metrics = {
                'mae': float(mean_absolute_error(historical_values, predicted_historical)),
                'mse': float(mean_squared_error(historical_values, predicted_historical)),
                'rmse': float(np.sqrt(mean_squared_error(historical_values, predicted_historical))),
                'mape': float(mean_absolute_percentage_error(historical_values, predicted_historical) * 100),
                'uncertainty_intervals': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist()
                }
            }
            
            return forecast_values, metrics
        except Exception as e:
            logger.error(f"Prophet forecast failed: {str(e)}", exc_info=True)
            return None, None

    def _sarima_forecast(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate forecast using SARIMA."""
        try:
            model = SARIMAX(
                df['y'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            )
            results = model.fit(disp=False)
            forecast_values = results.forecast(steps=forecast_horizon)
            
            # Calculate metrics
            test_size = min(forecast_horizon, len(df))
            historical_values = df['y'].tail(test_size).values
            predicted_historical = results.get_prediction(
                start=-test_size
            ).predicted_mean.values
            
            metrics = {
                'mae': float(mean_absolute_error(historical_values, predicted_historical)),
                'mse': float(mean_squared_error(historical_values, predicted_historical)),
                'rmse': float(np.sqrt(mean_squared_error(historical_values, predicted_historical))),
                'mape': float(mean_absolute_percentage_error(historical_values, predicted_historical) * 100)
            }
            
            return forecast_values, metrics
        except Exception as e:
            logger.error(f"SARIMA forecast failed: {str(e)}")
            return None, None

    def _exp_smoothing_forecast(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate forecast using Exponential Smoothing."""
        try:
            model = ExponentialSmoothing(
                df['y'],
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            )
            results = model.fit()
            forecast_values = results.forecast(forecast_horizon)
            
            # Calculate metrics
            test_size = min(forecast_horizon, len(df))
            historical_values = df['y'].tail(test_size).values
            predicted_historical = results.fittedvalues[-test_size:]
            
            metrics = {
                'mae': float(mean_absolute_error(historical_values, predicted_historical)),
                'mse': float(mean_squared_error(historical_values, predicted_historical)),
                'rmse': float(np.sqrt(mean_squared_error(historical_values, predicted_historical))),
                'mape': float(mean_absolute_percentage_error(historical_values, predicted_historical) * 100)
            }
            
            return forecast_values, metrics
        except Exception as e:
            logger.error(f"Exponential Smoothing forecast failed: {str(e)}")
            return None, None

    def _get_forecast_horizon(self, duration: str) -> int:
        """Get number of days to forecast based on duration."""
        current_date = pd.Timestamp.now().normalize()
        
        if duration == 'next_week':
            return 7
        elif duration == 'next_month':
            next_month = current_date + pd.Timedelta(days=30)
            return (next_month - current_date).days
        elif duration == 'next_quarter':
            next_quarter = current_date + pd.Timedelta(days=90)
            return (next_quarter - current_date).days
        elif duration == 'next_year':
            next_year = current_date + pd.Timedelta(days=365)
            return (next_year - current_date).days
        else:
            return 30 
        
    def _format_forecast_response(
        self, 
        forecast_dates: pd.DatetimeIndex, 
        forecast_values: np.ndarray, 
        metrics: Dict,
        metric_name: str,
        duration: str,
        resolution: str,
        source_name: str
    ) -> Dict[str, Any]:
        """Format forecast response with proper date handling."""
        return {
            "metric_name": metric_name,
            "forecast_points": [
                {
                    "date": date.isoformat(),
                    "value": float(value),
                    "confidence_interval": {
                        "lower": float(value * 0.9),
                        "upper": float(value * 1.1)
                    }
                }
                for date, value in zip(forecast_dates, forecast_values)
                if not (math.isnan(value) or math.isinf(value))
            ],
            "metadata": {
                "start_date": forecast_dates[0].isoformat(),
                "end_date": forecast_dates[-1].isoformat(),
                "duration": duration,
                "resolution": resolution,
                "source": source_name,
                "model_metrics": metrics,
                "forecast_length": len(forecast_dates)
            }
        }
    
    def _process_metrics_results(
        self,
        results: List[Dict],
        metrics: List[MetricDefinition],
        source_name: str
    ) -> Dict[str, Any]:
        """Process raw metrics results into structured format."""
        processed = {}
        df = pd.DataFrame(results)

        for metric in metrics:
            try:
                current_value = df[metric.name].iloc[0] if not df.empty else 0
                previous_value = df[metric.name].iloc[1] if len(df) > 1 else 0
                
                change = current_value - previous_value
                change_percentage = (change / previous_value * 100) if previous_value != 0 else 0

                processed[metric.name] = {
                    "current": current_value,
                    "previous": previous_value,
                    "change": change,
                    "change_percentage": change_percentage,
                    "source": source_name,
                    "category": metric.category,
                    "visualization_type": metric.visualization_type,
                    "trend_data": self._get_trend_data(df, metric.name),
                    "dimensions": self._get_dimensional_data(df, metric.name)
                }

            except Exception as e:
                logger.error(f"Error processing metric {metric.name}: {str(e)}")
                continue

        return processed

    def _get_date_trunc(self, resolution: str, date_column: str) -> str:
        """Get appropriate date truncation SQL."""
        if resolution == 'daily':
            return f"DATE_TRUNC('day', {date_column})"
        elif resolution == 'weekly':
            return f"DATE_TRUNC('week', {date_column})"
        elif resolution == 'monthly':
            return f"DATE_TRUNC('month', {date_column})"
        elif resolution == 'quarterly':
            return f"DATE_TRUNC('quarter', {date_column})"
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

    def _merge_trend_data(self, existing_trends: List[Dict], new_trends: List[Dict]) -> List[Dict]:
        """
        Merge trend data from different sources.
        
        Args:
            existing_trends: Existing trend data
            new_trends: New trend data to merge
            
        Returns:
            Merged trend data
        """
        try:
            # Create a dictionary of existing trends by date
            trend_dict = {trend["date"]: trend for trend in existing_trends}
            
            # Merge new trends
            for new_trend in new_trends:
                date = new_trend["date"]
                if date in trend_dict:
                    # Add values for existing date
                    trend_dict[date]["value"] += self._sanitize_metric_value(new_trend.get("value", 0))
                    
                    # Merge moving averages if they exist
                    if "ma3" in new_trend and "ma3" in trend_dict[date]:
                        trend_dict[date]["ma3"] += self._sanitize_metric_value(new_trend["ma3"])
                    if "ma7" in new_trend and "ma7" in trend_dict[date]:
                        trend_dict[date]["ma7"] += self._sanitize_metric_value(new_trend["ma7"])
                else:
                    # Add new date point
                    trend_dict[date] = new_trend

            # Convert back to list and sort by date
            merged_trends = list(trend_dict.values())
            merged_trends.sort(key=lambda x: x["date"])
            
            return merged_trends

        except Exception as e:
            logger.error(f"Error merging trend data: {str(e)}")
            return existing_trends

    def _merge_dimensional_data(
        self,
        existing_dims: Dict[str, Dict],
        new_dims: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Merge dimensional data from different sources.
        
        Args:
            existing_dims: Existing dimensional data
            new_dims: New dimensional data to merge
            
        Returns:
            Merged dimensional data
        """
        try:
            merged_dims = existing_dims.copy()
            
            for dim_name, dim_data in new_dims.items():
                if dim_name not in merged_dims:
                    merged_dims[dim_name] = {}
                    
                for category, values in dim_data.items():
                    if category not in merged_dims[dim_name]:
                        merged_dims[dim_name][category] = {
                            "total": 0,
                            "average": 0,
                            "count": 0
                        }
                    
                    current_data = merged_dims[dim_name][category]
                    
                    # Update totals and counts
                    current_data["total"] += self._sanitize_metric_value(values.get("total", 0))
                    current_data["count"] += int(values.get("count", 0))
                    
                    # Recalculate average
                    if current_data["count"] > 0:
                        current_data["average"] = current_data["total"] / current_data["count"]
                    
                    # Update min/max if present
                    if "min" in values:
                        current_data["min"] = min(
                            current_data.get("min", float('inf')),
                            self._sanitize_metric_value(values["min"])
                        )
                    if "max" in values:
                        current_data["max"] = max(
                            current_data.get("max", float('-inf')),
                            self._sanitize_metric_value(values["max"])
                        )

            return merged_dims

        except Exception as e:
            logger.error(f"Error merging dimensional data: {str(e)}")
            return existing_dims
    
    def _sanitize_metric_value(self, value: Any) -> float:
        """
        Sanitize metric value to ensure it's a valid number.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized float value
        """
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            return 0.0
        except Exception:
            return 0.0

    def _merge_metrics(self, target: Dict[str, Any], source_metrics: Dict[str, Any]) -> None:
        """
        Merge metrics from different sources into target dictionary.
        
        Args:
            target: Target dictionary to merge metrics into
            source_metrics: Source metrics to merge
        """
        try:
            for metric_name, metric_data in source_metrics.items():
                # Initialize metric in target if it doesn't exist
                if metric_name not in target:
                    target[metric_name] = {
                        "current": 0,
                        "previous": 0,
                        "change": {
                            "absolute": 0,
                            "percentage": 0
                        },
                        "sources": [],
                        "trend_data": [],
                        "dimensions": {}
                    }
                
                # Update aggregate values
                if isinstance(metric_data.get("current"), (int, float)):
                    target[metric_name]["current"] += metric_data["current"]
                if isinstance(metric_data.get("previous"), (int, float)):
                    target[metric_name]["previous"] += metric_data["previous"]
                
                # Add source information
                source_info = {
                    "name": metric_data.get("source", "Unknown"),
                    "current": metric_data.get("current", 0),
                    "previous": metric_data.get("previous", 0),
                    "change": {
                        "absolute": metric_data.get("change", 0),
                        "percentage": metric_data.get("change_percentage", 0)
                    }
                }
                target[metric_name]["sources"].append(source_info)
                
                # Merge trend data
                if metric_data.get("trend_data"):
                    target[metric_name]["trend_data"].extend(metric_data["trend_data"])
                
                # Merge dimensional data
                if metric_data.get("dimensions"):
                    for dim_name, dim_data in metric_data["dimensions"].items():
                        if dim_name not in target[metric_name]["dimensions"]:
                            target[metric_name]["dimensions"][dim_name] = {}
                        
                        # Merge dimension values
                        for key, value in dim_data.items():
                            if key not in target[metric_name]["dimensions"][dim_name]:
                                target[metric_name]["dimensions"][dim_name][key] = 0
                            if isinstance(value, (int, float)):
                                target[metric_name]["dimensions"][dim_name][key] += value

            # Calculate aggregated changes
            for metric_name, metric_data in target.items():
                if metric_data["previous"] != 0:
                    metric_data["change"]["absolute"] = metric_data["current"] - metric_data["previous"]
                    metric_data["change"]["percentage"] = (
                        (metric_data["change"]["absolute"] / metric_data["previous"]) * 100
                    )
                else:
                    metric_data["change"]["percentage"] = 100 if metric_data["current"] > 0 else 0

                # Sort trend data by date
                if metric_data["trend_data"]:
                    metric_data["trend_data"].sort(key=lambda x: x["date"])

        except Exception as e:
            logger.error(f"Error merging metrics: {str(e)}")
            raise

    def _get_date_range(self, scope: str) -> tuple:
        """
        Get date range based on scope.
        Args:
            scope: one of ["this_week", "this_month", "this_quarter", "this_year"]
        Returns:
            tuple of (start_date, end_date)
        """
        today = datetime.now().date()
        
        if scope == "this_week":
            # Start from Monday of current week
            start_date = today - timedelta(days=today.weekday())
            end_date = today
            
        elif scope == "this_month":
            # Start from first day of current month
            start_date = today.replace(day=1)
            end_date = today
            
        elif scope == "this_quarter":
            # Start from first day of current quarter
            quarter = (today.month - 1) // 3
            start_date = today.replace(
                month=3 * quarter + 1,
                day=1
            )
            end_date = today
            
        elif scope == "last_month":
            # Last month's date range
            if today.month == 1:
                start_date = today.replace(year=today.year-1, month=12, day=1)
            else:
                start_date = today.replace(month=today.month-1, day=1)
            end_date = today.replace(day=1) - timedelta(days=1)
            
        elif scope == "last_quarter":
            # Last quarter's date range
            quarter = (today.month - 1) // 3
            if quarter == 0:
                start_date = today.replace(year=today.year-1, month=10, day=1)
                end_date = today.replace(year=today.year-1, month=12, day=31)
            else:
                start_date = today.replace(month=3 * (quarter - 1) + 1, day=1)
                end_date = today.replace(month=3 * quarter, day=1) - timedelta(days=1)
                
        elif scope == "last_year":
            # Last year's date range
            start_date = today.replace(year=today.year-1, month=1, day=1)
            end_date = today.replace(year=today.year-1, month=12, day=31)
            
        elif scope == "ytd" or scope == "this_year":
            # Year to date
            start_date = today.replace(month=1, day=1)
            end_date = today
            
        else:
            # Default to last 30 days if scope not recognized
            start_date = today - timedelta(days=30)
            end_date = today

        return start_date, end_date

    def _get_comparison_date_range(self, scope: str, start_date: datetime.date) -> tuple:
        """
        Get comparison date range for the given scope and start date.
        Args:
            scope: time period scope
            start_date: start date of current period
        Returns:
            tuple of (comparison_start_date, comparison_end_date)
        """
        if scope == "this_week":
            # Previous week
            comp_start = start_date - timedelta(days=7)
            comp_end = start_date - timedelta(days=1)
            
        elif scope == "this_month":
            # Previous month
            if start_date.month == 1:
                comp_start = start_date.replace(year=start_date.year-1, month=12, day=1)
                comp_end = start_date.replace(year=start_date.year-1, month=12, day=31)
            else:
                comp_start = start_date.replace(month=start_date.month-1, day=1)
                comp_end = start_date - timedelta(days=1)
                
        elif scope == "this_quarter":
            # Previous quarter
            quarter = (start_date.month - 1) // 3
            if quarter == 0:
                comp_start = start_date.replace(year=start_date.year-1, month=10, day=1)
                comp_end = start_date.replace(year=start_date.year-1, month=12, day=31)
            else:
                comp_start = start_date.replace(month=3 * (quarter - 1) + 1, day=1)
                comp_end = start_date - timedelta(days=1)
                
        elif scope == "this_year":
            # Previous year
            comp_start = start_date.replace(year=start_date.year-1)
            comp_end = start_date.replace(year=start_date.year-1, month=12, day=31)
            
        else:
            # Default to previous 30 days
            days_diff = (datetime.now().date() - start_date).days
            comp_start = start_date - timedelta(days=days_diff)
            comp_end = start_date - timedelta(days=1)

        return comp_start, comp_end

    def _get_forecast_days(self, forecast_duration: str) -> int:
        """
        Get number of days to forecast based on duration.
        Args:
            forecast_duration: one of ["next_week", "next_month", "next_quarter", "next_year"]
        Returns:
            number of days to forecast
        """
        if forecast_duration == "next_week":
            return 7
        elif forecast_duration == "next_month":
            return 30
        elif forecast_duration == "next_quarter":
            return 90
        elif forecast_duration == "next_year":
            return 365
        else:
            return 30  # Default to one month

    def _get_resolution_days(self, resolution: str) -> int:
        """
        Get number of days for each data point based on resolution.
        Args:
            resolution: one of ["daily", "weekly", "monthly", "quarterly"]
        Returns:
            number of days per data point
        """
        if resolution == "daily":
            return 1
        elif resolution == "weekly":
            return 7
        elif resolution == "monthly":
            return 30
        elif resolution == "quarterly":
            return 90
        else:
            return 1  # Default to daily
    


    def _format_empty_response(self, scope: str, resolution: str) -> Dict[str, Any]:
        """Format empty response when no metrics are available."""
        start_date, end_date = self._get_date_range(scope)
        return {
            "metadata": {
                "scope": scope,
                "resolution": resolution,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "has_forecast": False,
                "generated_at": datetime.utcnow().isoformat(),
                "source_count": 0
            },
            "metrics": {}
        }

    def _format_metrics_response(
        self,
        metrics: Dict[str, Any],
        scope: str,
        resolution: str,
        has_forecast: bool
    ) -> Dict[str, Any]:
        """Format metrics data into standardized response structure."""
        try:
            if not metrics:
                return self._format_empty_response(scope, resolution)

            start_date, end_date = self._get_date_range(scope)
            
            response = {
                "metadata": {
                    "scope": scope,
                    "resolution": resolution,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "has_forecast": has_forecast,
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_count": len(set(m.get("source", "") for m in metrics.values()))
                },
                "metrics": {}
            }

            for metric_name, metric_data in metrics.items():
                response["metrics"][metric_name] = {
                    "current_value": metric_data.get("current", 0),
                    "previous_value": metric_data.get("previous", 0),
                    "change": {
                        "absolute": metric_data.get("current", 0) - metric_data.get("previous", 0),
                        "percentage": self._calculate_percentage_change(
                            metric_data.get("current", 0),
                            metric_data.get("previous", 0)
                        )
                    },
                    "source": metric_data.get("source", "Unknown"),
                    "category": metric_data.get("category", "Unknown"),
                    "visualization_type": metric_data.get("visualization_type", "line"),
                    "trend_data": metric_data.get("trend_data", []),
                    "dimensions": metric_data.get("dimensions", {})
                }

            return response

        except Exception as e:
            logger.error(f"Error formatting metrics response: {str(e)}")
            return self._format_empty_response(scope, resolution)

    def _calculate_percentage_change(self, current: float, previous: float) -> float:
        """Calculate percentage change between two values."""
        try:
            if previous == 0:
                return 100.0 if current > 0 else 0.0
            return round((current - previous) / previous * 100, 2)
        except Exception:
            return 0.0

    def _determine_trend(self, metric_data: Dict[str, Any]) -> str:
        """Determine trend direction from metric data."""
        if not metric_data.get("trend_data"):
            change = metric_data["current"] - metric_data["previous"]
            return "up" if change > 0 else "down" if change < 0 else "stable"
        
        # Use trend data if available
        values = [point["value"] for point in metric_data["trend_data"]]
        if len(values) < 2:
            return "stable"
        
        # Calculate trend using last few points
        recent_change = values[-1] - values[0]
        return "up" if recent_change > 0 else "down" if recent_change < 0 else "stable"

    def _format_source_data(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source-specific metric data."""
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                "name": source["name"],
                "value": source["value"],
                "change": {
                    "absolute": source["change"],
                    "percentage": source["change_percentage"]
                }
            })
        return formatted_sources

    def _format_dimensional_data(self, dimensions: Dict[str, Dict]) -> Dict[str, List[Dict[str, Any]]]:
        """Format dimensional breakdowns of metrics."""
        formatted_dimensions = {}
        for dim_name, dim_data in dimensions.items():
            # Sort dimensional data by value
            sorted_data = sorted(
                dim_data.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Format each dimension's data
            formatted_dimensions[dim_name] = [
                {
                    "category": category,
                    "value": value,
                    "percentage": self._calculate_percentage_of_total(value, dim_data.values())
                }
                for category, value in sorted_data
            ]
        
        return formatted_dimensions

    def _format_time_series(
        self,
        trend_data: List[Dict[str, Any]],
        resolution: str
    ) -> List[Dict[str, Any]]:
        """Format time series data based on resolution."""
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(trend_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Resample based on resolution
            if resolution == 'weekly':
                df = df.resample('W', on='date').mean().reset_index()
            elif resolution == 'monthly':
                df = df.resample('M', on='date').mean().reset_index()
            elif resolution == 'quarterly':
                df = df.resample('Q', on='date').mean().reset_index()
                
            # Format back to list of dicts
            return [
                {
                    "date": row['date'].isoformat(),
                    "value": float(row['value'])
                }
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error formatting time series: {str(e)}")
            return trend_data

    def _format_forecast_data(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Format forecast data and metrics."""
        return {
            "values": [
                {
                    "date": point["date"].isoformat(),
                    "value": float(point["value"]),
                    "confidence_interval": {
                        "lower": float(point.get("lower", point["value"] * 0.9)),
                        "upper": float(point.get("upper", point["value"] * 1.1))
                    }
                }
                for point in forecast["predictions"]
            ],
            "metrics": forecast.get("metrics", {}),
            "model_info": forecast.get("model_info", {})
        }

    def _generate_insights(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from metrics data."""
        insights = []
        
        # Top performing metrics
        top_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1]["change"]["percentage"],
            reverse=True
        )[:3]
        
        for metric_name, metric_data in top_metrics:
            if metric_data["change"]["percentage"] > 0:
                insights.append({
                    "type": "improvement",
                    "metric": metric_name,
                    "change": metric_data["change"]["percentage"],
                    "message": f"{metric_name} showed strong growth of {metric_data['change']['percentage']}%"
                })
        
        # Metrics needing attention
        attention_metrics = [
            (name, data) for name, data in metrics.items()
            if data["change"]["percentage"] < -10
        ]
        
        for metric_name, metric_data in attention_metrics:
            insights.append({
                "type": "attention",
                "metric": metric_name,
                "change": metric_data["change"]["percentage"],
                "message": f"{metric_name} declined by {abs(metric_data['change']['percentage'])}%"
            })
        
        return insights

    def _calculate_percentage_of_total(self, value: float, all_values: List[float]) -> float:
        """Calculate percentage of total for dimensional breakdowns."""
        total = sum(all_values)
        if total == 0:
            return 0.0
        return round((value / total) * 100, 2)
    
    def _get_trend_data(self, df: pd.DataFrame, metric_name: str) -> List[Dict[str, Any]]:
        """
        Generate trend data for a metric over time.

        Args:
            df: DataFrame containing the metric data
            metric_name: Name of the metric to analyze

        Returns:
            List of data points with dates and values
        """
        try:
            # Ensure DataFrame has required columns
            if 'period' not in df.columns or metric_name not in df.columns:
                return []

            # Sort by date and get trend points
            df_sorted = df.sort_values('period')
            
            trend_data = [
                {
                    "date": row['period'].isoformat() if isinstance(row['period'], (datetime, pd.Timestamp)) 
                            else row['period'],
                    "value": float(row[metric_name])
                }
                for _, row in df_sorted.iterrows()
                if pd.notnull(row[metric_name])  # Filter out null values
            ]

            # Add moving averages if enough data points
            if len(trend_data) >= 3:
                values = [point["value"] for point in trend_data]
                ma_3 = self._calculate_moving_average(values, 3)
                ma_7 = self._calculate_moving_average(values, 7) if len(values) >= 7 else None

                for i, point in enumerate(trend_data):
                    point["ma3"] = ma_3[i] if i < len(ma_3) else None
                    point["ma7"] = ma_7[i] if ma_7 and i < len(ma_7) else None

            # Add trend indicators
            if len(trend_data) >= 2:
                self._add_trend_indicators(trend_data)

            return trend_data

        except Exception as e:
            logger.error(f"Error generating trend data for {metric_name}: {str(e)}")
            return []

    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average for a list of values."""
        ma = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i + window]
            ma.append(sum(window_values) / window)
        return ma

    def _add_trend_indicators(self, trend_data: List[Dict[str, Any]]) -> None:
        """Add trend direction indicators to data points."""
        for i in range(len(trend_data)):
            if i == 0:
                trend_data[i]["trend"] = "stable"
                continue

            current_value = trend_data[i]["value"]
            previous_value = trend_data[i-1]["value"]
            
            if current_value > previous_value:
                trend_data[i]["trend"] = "up"
            elif current_value < previous_value:
                trend_data[i]["trend"] = "down"
            else:
                trend_data[i]["trend"] = "stable"

    def _analyze_trend_strength(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the strength and consistency of a trend."""
        if not trend_data or len(trend_data) < 2:
            return {
                "strength": "insufficient_data",
                "consistency": 0,
                "volatility": 0
            }

        values = [point["value"] for point in trend_data]
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        
        # Calculate trend consistency
        direction_changes = sum(1 for i in range(1, len(changes)) 
                            if (changes[i] > 0) != (changes[i-1] > 0))
        consistency = 1 - (direction_changes / (len(changes) - 1)) if len(changes) > 1 else 1
        
        # Calculate volatility
        mean_value = sum(values) / len(values)
        volatility = sum(abs(v - mean_value) for v in values) / (len(values) * mean_value) if mean_value != 0 else 0
        
        # Determine trend strength
        if consistency > 0.8 and volatility < 0.1:
            strength = "strong"
        elif consistency > 0.6 and volatility < 0.2:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "strength": strength,
            "consistency": round(consistency * 100, 2),
            "volatility": round(volatility * 100, 2)
        }

    def _get_seasonality_info(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect and analyze seasonality in trend data."""
        try:
            if len(trend_data) < 14:  # Need at least 2 weeks of data
                return {"has_seasonality": False}

            values = np.array([point["value"] for point in trend_data])
            
            # Check weekly seasonality
            weekly_pattern = self._check_seasonality(values, 7)
            
            # Check monthly seasonality if enough data
            monthly_pattern = self._check_seasonality(values, 30) if len(values) >= 60 else False
            
            return {
                "has_seasonality": weekly_pattern or monthly_pattern,
                "patterns": {
                    "weekly": weekly_pattern,
                    "monthly": monthly_pattern if len(values) >= 60 else None
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return {"has_seasonality": False}

    def _check_seasonality(self, values: np.ndarray, period: int) -> bool:
        """Check for seasonality with a specific period."""
        if len(values) < period * 2:
            return False
            
        # Calculate autocorrelation
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Check if there's a significant correlation at the period
        threshold = 0.3  # Correlation threshold for seasonality
        if period < len(autocorr) and autocorr[period] > threshold * autocorr[0]:
            return True
            
        return False
    
    def _get_dimensional_data(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Dict[str, float]]:
        """
        Generate dimensional breakdowns for a metric.
        
        Args:
            df: DataFrame containing the metric data
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary of dimensional breakdowns with their values
        """
        try:
            if df.empty or metric_name not in df.columns:
                return {}

            # Identify dimension columns (excluding metric and date columns)
            dimension_columns = [
                col for col in df.columns 
                if col not in [metric_name, 'period'] 
                and df[col].dtype == 'object'
            ]

            dimensional_data = {}
            
            for dimension in dimension_columns:
                try:
                    # Calculate aggregates for each dimension value
                    dimension_breakdown = df.groupby(dimension)[metric_name].agg([
                        ('total', 'sum'),
                        ('average', 'mean'),
                        ('min', 'min'),
                        ('max', 'max'),
                        ('count', 'count')
                    ]).to_dict('index')

                    # Calculate percentages of total
                    total_metric = df[metric_name].sum()
                    
                    # Format the breakdown data
                    formatted_breakdown = {}
                    for value, metrics in dimension_breakdown.items():
                        formatted_breakdown[str(value)] = {
                            'total': float(metrics['total']),
                            'average': float(metrics['average']),
                            'min': float(metrics['min']),
                            'max': float(metrics['max']),
                            'count': int(metrics['count']),
                            'percentage': float(round((metrics['total'] / total_metric * 100), 2)) if total_metric != 0 else 0
                        }

                    # Sort by total value and get top values
                    sorted_breakdown = dict(
                        sorted(
                            formatted_breakdown.items(),
                            key=lambda x: x[1]['total'],
                            reverse=True
                        )
                    )

                    dimensional_data[dimension] = sorted_breakdown

                except Exception as e:
                    logger.error(f"Error processing dimension {dimension}: {str(e)}")
                    continue

            # Add time-based dimensions if date column exists
            if 'period' in df.columns:
                time_dimensions = self._get_time_based_dimensions(df, metric_name)
                dimensional_data.update(time_dimensions)

            return dimensional_data

        except Exception as e:
            logger.error(f"Error getting dimensional data for {metric_name}: {str(e)}")
            return {}

    def _get_time_based_dimensions(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Dict[str, float]]:
        """
        Generate time-based dimensional breakdowns.
        
        Args:
            df: DataFrame containing the metric data
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary of time-based dimensional breakdowns
        """
        try:
            time_dimensions = {}
            df = df.copy()
            df['period'] = pd.to_datetime(df['period'])

            # Monthly breakdown
            monthly_data = df.set_index('period').resample('M')[metric_name].agg([
                ('total', 'sum'),
                ('average', 'mean'),
                ('min', 'min'),
                ('max', 'max'),
                ('count', 'count')
            ]).to_dict('index')

            # Format monthly data
            monthly_breakdown = {}
            total_metric = df[metric_name].sum()

            for date, metrics in monthly_data.items():
                month_key = date.strftime('%Y-%m')
                monthly_breakdown[month_key] = {
                    'total': float(metrics['total']),
                    'average': float(metrics['average']),
                    'min': float(metrics['min']),
                    'max': float(metrics['max']),
                    'count': int(metrics['count']),
                    'percentage': float(round((metrics['total'] / total_metric * 100), 2)) if total_metric != 0 else 0
                }

            time_dimensions['monthly'] = monthly_breakdown

            # Quarterly breakdown
            quarterly_data = df.set_index('period').resample('Q')[metric_name].agg([
                ('total', 'sum'),
                ('average', 'mean'),
                ('min', 'min'),
                ('max', 'max'),
                ('count', 'count')
            ]).to_dict('index')

            # Format quarterly data
            quarterly_breakdown = {}
            for date, metrics in quarterly_data.items():
                quarter_key = f"{date.year}-Q{date.quarter}"
                quarterly_breakdown[quarter_key] = {
                    'total': float(metrics['total']),
                    'average': float(metrics['average']),
                    'min': float(metrics['min']),
                    'max': float(metrics['max']),
                    'count': int(metrics['count']),
                    'percentage': float(round((metrics['total'] / total_metric * 100), 2)) if total_metric != 0 else 0
                }

            time_dimensions['quarterly'] = quarterly_breakdown

            return time_dimensions

        except Exception as e:
            logger.error(f"Error getting time-based dimensions: {str(e)}")
            return {}

    def _get_correlation_analysis(self, df: pd.DataFrame, metric_name: str) -> Dict[str, float]:
        """
        Analyze correlations between the metric and other numeric columns.
        
        Args:
            df: DataFrame containing the metric data
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary of correlation coefficients
        """
        try:
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            # Calculate correlations
            correlations = {}
            for col in numeric_columns:
                if col != metric_name:
                    correlation = df[metric_name].corr(df[col])
                    if not pd.isna(correlation):
                        correlations[col] = float(round(correlation, 3))

            return correlations

        except Exception as e:
            logger.error(f"Error calculating correlations for {metric_name}: {str(e)}")
            return {}

    def _get_dimension_statistics(self, df: pd.DataFrame, dimension: str, metric_name: str) -> Dict[str, Any]:
        """
        Calculate detailed statistics for a specific dimension.
        
        Args:
            df: DataFrame containing the metric data
            dimension: Dimension to analyze
            metric_name: Name of the metric
            
        Returns:
            Dictionary of dimensional statistics
        """
        try:
            stats = {}
            
            # Basic statistics by dimension value
            dimension_stats = df.groupby(dimension)[metric_name].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75)
            ]).round(2)
            
            dimension_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'q1', 'q3']
            
            # Convert to dictionary
            stats['value_distribution'] = dimension_stats.to_dict('index')
            
            # Calculate dimension value frequencies
            value_counts = df[dimension].value_counts().to_dict()
            stats['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
            
            # Add additional metrics
            stats['unique_values'] = len(value_counts)
            stats['most_common'] = max(value_counts.items(), key=lambda x: x[1])[0]
            
            return stats

        except Exception as e:
            logger.error(f"Error calculating dimension statistics: {str(e)}")
            return {}
        
    def _get_metric_history(
        self,
        db: Session,
        org_id: int,
        metric: MetricDefinition,
        lookback_days: int = 365
    ) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric."""
        try:
            # Get connection details
            connection = db.query(DataSourceConnection).get(metric.connection_id)
            if not connection:
                raise ValueError(f"Connection not found for metric {metric.name}")

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            # Build query with metric calculation
            period_expression = self._build_date_trunc_expression(
                connection.date_column,
                'daily',
                connection.source_type
            )

            # Modify the query to output lowercase column names
            query = f"""
            WITH metric_data AS (
                SELECT 
                    {period_expression}::date as period,
                    CAST({metric.calculation} AS FLOAT) as value
                FROM {connection.table_name}
                WHERE {connection.date_column} BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY period
            )
            SELECT period, value 
            FROM metric_data
            WHERE value IS NOT NULL
            ORDER BY period ASC
            """

            # Execute query
            connector = self._get_connector(connection)
            try:
                results = connector.query(query)
                logger.info(f"Query returned {len(results)} rows")
                if results:
                    sample = results[0]
                    logger.info(f"Sample result: {sample}")

                processed_results = []
                for row in results:
                    # Convert dictionary keys to lowercase
                    processed_row = {
                        'period': row.get('PERIOD', row.get('period')),
                        'value': row.get('VALUE', row.get('value'))
                    }
                    processed_results.append(processed_row)

                return processed_results

            finally:
                connector.disconnect()

        except Exception as e:
            logger.error(f"Error getting metric history for {metric.name}: {str(e)}")
            return []