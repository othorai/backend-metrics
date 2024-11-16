# services/query_generation.py

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI
import json
from sqlalchemy.orm import Session
from app.models.models import MetricDefinition, AnalyticsConfiguration, DataSourceConnection
from app.connectors.connector_factory import ConnectorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryGenerationService:
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_metric_query(
        self, 
        metric: MetricDefinition,
        time_range: str,
        dimensions: List[str],
        db: Session
    ) -> str:
        """Generate SQL query for a specific metric."""
        try:
            # Get the data source connection
            connection = db.query(DataSourceConnection).filter_by(
                id=metric.connection_id
            ).first()
            
            if not connection:
                raise ValueError(f"Connection {metric.connection_id} not found")

            # Prepare context for OpenAI
            context = {
                "metric_name": metric.name,
                "calculation": metric.calculation,
                "dependencies": metric.data_dependencies,
                "table_name": connection.table_name,
                "time_range": time_range,
                "dimensions": dimensions,
                "aggregation_period": metric.aggregation_period
            }

            prompt = self._create_query_prompt(context)
            
            # Get query from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            query = response.choices[0].message.content.strip()
            
            # Validate the generated query
            self._validate_query(query, connection)
            
            return query
            
        except Exception as e:
            logger.error(f"Error generating query: {str(e)}")
            raise

    def _create_query_prompt(self, context: Dict) -> str:
        """Create prompt for OpenAI query generation."""
        return f"""Generate an optimized SQL query for the following metric:

Metric Name: {context['metric_name']}
Base Calculation: {context['calculation']}
Required Columns: {', '.join(context['dependencies'])}
Table Name: {context['table_name']}
Time Range: {context['time_range']}
Dimensions: {', '.join(context['dimensions'])}
Aggregation Period: {context['aggregation_period']}

Requirements:
1. Include proper time range filters
2. Group by the specified dimensions
3. Include comparison with previous period
4. Handle null values appropriately
5. Optimize for performance
6. Use appropriate aggregation functions

Return only the SQL query without any explanation."""

    def _validate_query(self, query: str, connection: DataSourceConnection) -> None:
        """Validate the generated query."""
        try:
            connector = ConnectorFactory.get_connector(
                connection.source_type,
                **connection.connection_params
            )
            connector.connect()
            # Try executing with EXPLAIN
            explain_query = f"EXPLAIN {query}"
            connector.query(explain_query)
            connector.disconnect()
        except Exception as e:
            raise ValueError(f"Invalid query generated: {str(e)}")

class AnalyticsGenerationService:
    def __init__(self, query_service: QueryGenerationService):
        self.query_service = query_service

    async def generate_analytics(
        self,
        config: AnalyticsConfiguration,
        end_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate analytics based on configuration."""
        try:
            results = {}
            
            # Get metrics information
            metrics = db.query(MetricDefinition).filter(
                MetricDefinition.id.in_(config.metrics)
            ).all()
            
            # Get connection information
            connection = db.query(DataSourceConnection).filter_by(
                id=config.connection_id
            ).first()
            
            if not connection:
                raise ValueError(f"Connection {config.connection_id} not found")
            
            connector = ConnectorFactory.get_connector(
                connection.source_type,
                **connection.connection_params
            )
            
            # Generate and execute queries for each metric
            for metric in metrics:
                metric_results = {}
                
                for time_range in config.time_ranges:
                    # Generate query
                    query = await self.query_service.generate_metric_query(
                        metric=metric,
                        time_range=time_range,
                        dimensions=config.dimensions,
                        db=db
                    )
                    
                    # Execute query
                    connector.connect()
                    try:
                        data = connector.query(query)
                        metric_results[time_range] = self._process_results(
                            data,
                            metric,
                            time_range
                        )
                    finally:
                        connector.disconnect()
                
                results[metric.name] = metric_results
            
            return self._format_analytics_response(results)
            
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            raise

    def _process_results(
        self,
        data: List[Dict],
        metric: MetricDefinition,
        time_range: str
    ) -> Dict[str, Any]:
        """Process raw query results into analytics."""
        try:
            processed = {
                "current_value": None,
                "previous_value": None,
                "change": None,
                "change_percentage": None,
                "trend": [],
                "breakdown": {}
            }
            
            if not data:
                return processed
                
            # Extract main metric value
            processed["current_value"] = data[0].get("current_value")
            processed["previous_value"] = data[0].get("previous_value")
            
            # Calculate changes
            if processed["current_value"] and processed["previous_value"]:
                processed["change"] = processed["current_value"] - processed["previous_value"]
                if processed["previous_value"] != 0:
                    processed["change_percentage"] = (
                        processed["change"] / processed["previous_value"]
                    ) * 100
            
            # Process trend data if available
            if "trend_data" in data[0]:
                processed["trend"] = json.loads(data[0]["trend_data"])
            
            # Process dimensional breakdown
            for row in data:
                for key, value in row.items():
                    if key.startswith("by_"):
                        dimension = key[3:]
                        if dimension not in processed["breakdown"]:
                            processed["breakdown"][dimension] = []
                        processed["breakdown"][dimension].append({
                            "category": value,
                            "value": row.get(f"{dimension}_value")
                        })
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise

    def _format_analytics_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final analytics response."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": results,
            "summary": self._generate_summary(results)
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analytics results."""
        summary = {
            "total_metrics": len(results),
            "improving_metrics": 0,
            "declining_metrics": 0,
            "stable_metrics": 0,
            "notable_changes": []
        }
        
        for metric_name, metric_data in results.items():
            for time_range, data in metric_data.items():
                if data["change_percentage"]:
                    if data["change_percentage"] > 5:
                        summary["improving_metrics"] += 1
                        if data["change_percentage"] > 20:
                            summary["notable_changes"].append({
                                "metric": metric_name,
                                "time_range": time_range,
                                "change": data["change_percentage"],
                                "type": "improvement"
                            })
                    elif data["change_percentage"] < -5:
                        summary["declining_metrics"] += 1
                        if data["change_percentage"] < -20:
                            summary["notable_changes"].append({
                                "metric": metric_name,
                                "time_range": time_range,
                                "change": data["change_percentage"],
                                "type": "decline"
                            })
                    else:
                        summary["stable_metrics"] += 1
        
        return summary