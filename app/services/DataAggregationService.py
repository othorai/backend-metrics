from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException
import logging
import pandas as pd
from app.models.models import DataSourceConnection, Organization, MetricDefinition
from app.connectors.connector_factory import ConnectorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicDataAggregationService:
    def __init__(self):
        self.cached_data = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=15)

    async def get_aggregated_data(
        self,
        db: Session,
        org_id: int,
        time_range: Optional[str] = "all"
    ) -> Dict[str, Any]:
        """
        Fetch and aggregate data from all data sources using dynamically discovered metrics.
        """
        cache_key = f"{org_id}_{time_range}"
        if self._is_cache_valid(cache_key):
            return self.cached_data[cache_key]

        try:
            # Get all data source connections for the organization
            connections = db.query(DataSourceConnection).filter(
                DataSourceConnection.organization_id == org_id
            ).all()

            if not connections:
                raise HTTPException(status_code=404, detail="No data sources found")

            aggregated_data = {
                "metrics": {},
                "trends": {},
                "summaries": {},
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "data_sources": len(connections)
                }
            }

            for connection in connections:
                source_data = await self._fetch_source_metrics(db, connection, time_range)
                self._merge_source_data(aggregated_data, source_data, connection.name)

            # Calculate global insights
            self._add_global_insights(aggregated_data)
            
            # Cache the results
            self.cached_data[cache_key] = aggregated_data
            self.cache_timestamp = datetime.utcnow()

            return aggregated_data

        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data aggregation failed: {str(e)}")

    async def _fetch_source_metrics(
        self,
        db: Session,
        connection: DataSourceConnection,
        time_range: str
    ) -> Dict[str, Any]:
        """
        Fetch metrics from a single data source using its defined metrics.
        """
        try:
            # Get metrics defined for this connection
            metrics = db.query(MetricDefinition).filter(
                MetricDefinition.connection_id == connection.id,
                MetricDefinition.is_active == True
            ).all()

            if not metrics:
                logger.warning(f"No metrics defined for connection {connection.name}")
                return {}

            connector = ConnectorFactory.get_connector(
                connection.source_type,
                **connection.connection_params
            )
            
            connector.connect()
            
            source_data = {
                "metrics": {},
                "trends": {},
                "metadata": {
                    "source_name": connection.name,
                    "source_type": connection.source_type,
                    "metric_count": len(metrics)
                }
            }

            # Build date ranges
            date_ranges = self._get_date_ranges(time_range)
            
            for metric in metrics:
                try:
                    metric_data = await self._calculate_metric(
                        connector,
                        connection.table_name,
                        metric,
                        connection.date_column,
                        date_ranges
                    )
                    
                    if metric_data:
                        source_data["metrics"][metric.name] = {
                            "current": metric_data.get("current_value"),
                            "previous": metric_data.get("previous_value"),
                            "change": metric_data.get("change"),
                            "change_percentage": metric_data.get("change_percentage"),
                            "category": metric.category,
                            "visualization_type": metric.visualization_type,
                            "confidence_score": metric.confidence_score,
                            "business_context": metric.business_context
                        }
                        
                        if metric_data.get("trend"):
                            source_data["trends"][metric.name] = metric_data["trend"]

                except Exception as e:
                    logger.error(f"Error calculating metric {metric.name}: {str(e)}")
                    continue

            connector.disconnect()
            return source_data

        except Exception as e:
            logger.error(f"Error fetching from source {connection.name}: {str(e)}")
            return {}

    async def _calculate_metric(
        self,
        connector: Any,
        table_name: str,
        metric: MetricDefinition,
        date_column: str,
        date_ranges: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Calculate a single metric with its current and previous values.
        """
        try:
            current_query = f"""
                WITH metric_calculation AS (
                    SELECT {metric.calculation} as value
                    FROM {table_name}
                    WHERE {date_column} BETWEEN '{date_ranges['current']['start']}' 
                        AND '{date_ranges['current']['end']}'
                )
                SELECT COALESCE(value, 0) as current_value
                FROM metric_calculation
            """

            previous_query = f"""
                WITH metric_calculation AS (
                    SELECT {metric.calculation} as value
                    FROM {table_name}
                    WHERE {date_column} BETWEEN '{date_ranges['previous']['start']}' 
                        AND '{date_ranges['previous']['end']}'
                )
                SELECT COALESCE(value, 0) as previous_value
                FROM metric_calculation
            """

            # Execute queries
            current_result = connector.query(current_query)
            previous_result = connector.query(previous_query)

            # Get trend data if needed
            trend_data = []
            if metric.visualization_type in ['line', 'bar', 'area']:
                trend_query = f"""
                    SELECT 
                        {date_column} as date,
                        {metric.calculation} as value
                    FROM {table_name}
                    WHERE {date_column} BETWEEN '{date_ranges['current']['start']}' 
                        AND '{date_ranges['current']['end']}'
                    GROUP BY {date_column}
                    ORDER BY {date_column}
                """
                trend_data = connector.query(trend_query)

            # Process results
            current_value = float(current_result[0]['current_value']) if current_result else 0
            previous_value = float(previous_result[0]['previous_value']) if previous_result else 0
            
            change = current_value - previous_value
            change_percentage = (change / previous_value * 100) if previous_value != 0 else 0

            return {
                "current_value": current_value,
                "previous_value": previous_value,
                "change": change,
                "change_percentage": change_percentage,
                "trend": trend_data
            }

        except Exception as e:
            logger.error(f"Error calculating metric: {str(e)}")
            return None

    def _merge_source_data(
        self,
        aggregated_data: Dict[str, Any],
        source_data: Dict[str, Any],
        source_name: str
    ) -> None:
        """
        Merge data from a single source into the aggregated results.
        """
        if not source_data:
            return

        # Merge metrics
        for metric_name, metric_data in source_data.get("metrics", {}).items():
            if metric_name not in aggregated_data["metrics"]:
                aggregated_data["metrics"][metric_name] = {
                    "current": 0,
                    "previous": 0,
                    "sources": {},
                    "category": metric_data["category"],
                    "visualization_type": metric_data["visualization_type"]
                }

            metric = aggregated_data["metrics"][metric_name]
            metric["current"] += metric_data["current"]
            metric["previous"] += metric_data["previous"]
            metric["sources"][source_name] = {
                "current": metric_data["current"],
                "previous": metric_data["previous"],
                "change": metric_data["change"],
                "change_percentage": metric_data["change_percentage"]
            }

        # Merge trends
        for metric_name, trend_data in source_data.get("trends", {}).items():
            if metric_name not in aggregated_data["trends"]:
                aggregated_data["trends"][metric_name] = {}
            
            for entry in trend_data:
                date = entry["date"].strftime("%Y-%m-%d")
                if date not in aggregated_data["trends"][metric_name]:
                    aggregated_data["trends"][metric_name][date] = 0
                aggregated_data["trends"][metric_name][date] += entry["value"]

    def _add_global_insights(self, data: Dict[str, Any]) -> None:
        """
        Add global insights and analysis to the aggregated data.
        """
        insights = {
            "top_metrics": [],
            "concerning_metrics": [],
            "stable_metrics": []
        }

        for metric_name, metric_data in data["metrics"].items():
            change_percentage = (
                (metric_data["current"] - metric_data["previous"]) 
                / metric_data["previous"] * 100 if metric_data["previous"] != 0 else 0
            )

            if change_percentage > 10:
                insights["top_metrics"].append({
                    "name": metric_name,
                    "change": change_percentage
                })
            elif change_percentage < -10:
                insights["concerning_metrics"].append({
                    "name": metric_name,
                    "change": change_percentage
                })
            else:
                insights["stable_metrics"].append({
                    "name": metric_name,
                    "change": change_percentage
                })

        data["insights"] = insights

    def _get_date_ranges(self, time_range: str) -> Dict[str, Dict[str, str]]:
        """
        Get date ranges for current and previous periods.
        """
        now = datetime.utcnow()
        
        if time_range == "month":
            current_end = now
            current_start = current_end - timedelta(days=30)
            previous_end = current_start
            previous_start = previous_end - timedelta(days=30)
        elif time_range == "quarter":
            current_end = now
            current_start = current_end - timedelta(days=90)
            previous_end = current_start
            previous_start = previous_end - timedelta(days=90)
        else:  # Default to year
            current_end = now
            current_start = current_end - timedelta(days=365)
            previous_end = current_start
            previous_start = previous_end - timedelta(days=365)

        return {
            "current": {
                "start": current_start.strftime("%Y-%m-%d"),
                "end": current_end.strftime("%Y-%m-%d")
            },
            "previous": {
                "start": previous_start.strftime("%Y-%m-%d"),
                "end": previous_end.strftime("%Y-%m-%d")
            }
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if (
            cache_key in self.cached_data
            and self.cache_timestamp
            and datetime.utcnow() - self.cache_timestamp < self.cache_duration
        ):
            return True
        return False