#metrics.py (metrics)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.models.models import WayneEnterprise
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union, Any
from sqlalchemy import func
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
from app.models.models import MetricDefinition, DataSourceConnection 
from app.utils.auth import get_current_user
from app.services.DynamicDataAnalysisService import DynamicAnalysisService

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Calculate the date range based on the given duration
def get_date_range(scope: str):
    today = datetime.now().date()
    year_start = datetime(today.year, 1, 1).date()
    
    if scope == 'this_week':
        start_date = today - timedelta(days=today.weekday())
        return start_date, today
    elif scope == 'this_month':
        start_date = today.replace(day=1)
        return start_date, today
    elif scope == 'this_quarter':
        quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * quarter - 2, 1).date()
        return start_date, today
    else:  # this_year (default)
        return year_start, today
    
def get_forecast_end_date(forecast_duration: str, start_date: datetime.date):
    if forecast_duration == 'next_week':
        return start_date + timedelta(days=7)
    elif forecast_duration == 'next_month':
        next_month = start_date.replace(day=28) + timedelta(days=4)
        return next_month - timedelta(days=next_month.day)
    elif forecast_duration == 'next_quarter':
        days_in_quarter = 91  # Approximately 3 months
        return start_date + timedelta(days=days_in_quarter)
    else:  # this_year (rest of the year)
        return datetime(start_date.year, 12, 31).date()

# Fetch data from the database for the given date range
def fetch_data(db: Session, start_date: datetime, end_date: datetime):
    data = db.query(WayneEnterprise).filter(WayneEnterprise.date.between(start_date, end_date)).all()
    if not data:
        raise HTTPException(status_code=404, detail="No data found for the given criteria")
    return pd.DataFrame([vars(item) for item in data])

# Calculate percentage change between two values
def calculate_percentage_change(start_value, end_value):
    if start_value == 0:
        return 100 if end_value > 0 else 0
    return float(round(((end_value - start_value) / start_value) * 100, 2))

# Convert numpy types to JSON serializable types
def to_json_serializable(value):
    if isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32, float)):
        return sanitize_float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    elif isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {key: to_json_serializable(val) for key, val in value.items()}
    else:
        return value
    
# Generate forecast using Prophet
def calculate_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    return {
        'MAE': float(mean_absolute_error(actual, forecast)),
        'MSE': float(mean_squared_error(actual, forecast)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, forecast))),
        'MAPE': float(mean_absolute_percentage_error(actual, forecast) * 100)
    }

def prophet_forecast(df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
    try:
        train_df = df[:-forecast_horizon]
        test_df = df[-forecast_horizon:]
        
        model = Prophet(daily_seasonality=False, yearly_seasonality=False)
        model.fit(train_df)
        future_dates = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future_dates)
        forecast_values = forecast['yhat'].tail(forecast_horizon).values
        
        metrics = calculate_metrics(test_df['y'].values, forecast_values)
        return forecast_values, metrics
    except Exception as e:
        logger.error(f"Prophet forecast failed: {str(e)}")
        return None, None

def sarima_forecast(df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
    try:
        train_df = df[:-forecast_horizon]
        test_df = df[-forecast_horizon:]
        
        model = SARIMAX(train_df['y'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        results = model.fit(disp=False)
        forecast_values = results.forecast(steps=forecast_horizon)
        
        metrics = calculate_metrics(test_df['y'].values, forecast_values)
        return forecast_values, metrics
    except Exception as e:
        logger.error(f"SARIMA forecast failed: {str(e)}")
        return None, None

def exp_smoothing_forecast(df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
    try:
        train_df = df[:-forecast_horizon]
        test_df = df[-forecast_horizon:]
        
        model = ExponentialSmoothing(train_df['y'], seasonal_periods=1, trend=None, seasonal=None)
        results = model.fit()
        forecast_values = results.forecast(forecast_horizon)
        
        metrics = calculate_metrics(test_df['y'].values, forecast_values)
        return forecast_values, metrics
    except Exception as e:
        logger.error(f"Exponential Smoothing forecast failed: {str(e)}")
        return None, None

def parallel_forecast(df: pd.DataFrame, forecast_horizon: int) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(prophet_forecast, df, forecast_horizon),
            executor.submit(sarima_forecast, df, forecast_horizon),
            executor.submit(exp_smoothing_forecast, df, forecast_horizon)
        ]
        
        forecasts = []
        all_metrics = {}
        for future, method in zip(as_completed(futures), ['prophet', 'sarima', 'exp_smoothing']):
            result, metrics = future.result()
            if result is not None and metrics is not None:
                forecasts.append(result)
                all_metrics[method] = metrics
    
    if not forecasts:
        raise ValueError("All forecasting methods failed")
    
    ensemble_forecast = np.mean(forecasts, axis=0)
    ensemble_metrics = calculate_metrics(df['y'].values[-forecast_horizon:], ensemble_forecast)
    all_metrics['ensemble'] = ensemble_metrics
    
    return ensemble_forecast, all_metrics

def sanitize_float(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None  # or return 0 if that's preferred
        return value
    return value

def get_graph_data(df: pd.DataFrame, metric: str, resolution: str) -> List[Dict[str, Union[str, float]]]:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if resolution == 'daily':
        df_resampled = df.resample('D')[metric].sum().reset_index()
    elif resolution == 'weekly':
        df_resampled = df.resample('W-MON')[metric].sum().reset_index()
    elif resolution == 'monthly':
        df_resampled = df.resample('M')[metric].sum().reset_index()
    elif resolution == 'quarterly':
        df_resampled = df.resample('Q')[metric].sum().reset_index()
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    
    return [{"date": row['date'].isoformat(), "value": float(row[metric])} for _, row in df_resampled.iterrows()]


# # API endpoint for leading indicators
# @router.get("/leading_indicators")
# def get_leading_indicators(
#     db: Session = Depends(get_db),
#     duration: str = Query("1w", description="Duration for calculations (1w, 1m, 1q, 1y)")
# ):
#     start_date, end_date = get_date_range(duration)
#     df = fetch_data(db, start_date, end_date)

#     indicators = {}
#     for metric in ['customer_satisfaction', 'marketing_spend', 'website_visits', 'new_customers', 'units_sold']:
#         start_value = df[df['date'] == start_date][metric].mean()
#         end_value = df[df['date'] == end_date][metric].mean()
#         percentage_change = calculate_percentage_change(start_value, end_value)
        
#         indicators[metric] = {
#             "percentage_change": percentage_change,
#             "start_date": start_date.isoformat(),
#             "end_date": end_date.isoformat(),
#             "start_amount": float(round(start_value, 2)),
#             "end_amount": float(round(end_value, 2)),
#             "trend": "up" if percentage_change > 0 else "down"
#         }

#     return to_json_serializable({
#         "duration": duration,
#         "indicators": indicators
#     })

# # API endpoint for lagging indicators
# @router.get("/lagging_indicators")
# def get_lagging_indicators(
#     db: Session = Depends(get_db),
#     duration: str = Query("1w", description="Duration for calculations (1w, 1m, 1q, 1y)")
# ):
#     start_date, end_date = get_date_range(duration)
#     df = fetch_data(db, start_date, end_date)

#     indicators = {}
#     lagging_metrics = ['revenue', 'units_sold', 'costs', 'customer_satisfaction', 'repeat_customers', 'new_customers', 'website_visits']
    
#     for metric in lagging_metrics:
#         start_value = df[df['date'] == start_date][metric].sum() if metric != 'customer_satisfaction' else df[df['date'] == start_date][metric].mean()
#         end_value = df[df['date'] == end_date][metric].sum() if metric != 'customer_satisfaction' else df[df['date'] == end_date][metric].mean()
#         percentage_change = calculate_percentage_change(start_value, end_value)
        
#         indicators[metric] = {
#             "percentage_change": percentage_change,
#             "start_date": start_date.isoformat(),
#             "end_date": end_date.isoformat(),
#             "start_amount": float(round(start_value, 2)),
#             "end_amount": float(round(end_value, 2)),
#             "trend": "up" if percentage_change > 0 else "down"
#         }

#     return to_json_serializable({
#         "duration": duration,
#         "indicators": indicators
#     })

# API endpoint for forecast
@router.get("/forecast")
def get_forecast(
    db: Session = Depends(get_db),
    duration: str = Query("1w", description="Duration for forecast (1w, 1m, 1q, 1y)")
):
    today = datetime.now().date()
    durations = {
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "1q": timedelta(days=90),
        "1y": timedelta(days=365)
    }
    forecast_duration = durations.get(duration, timedelta(weeks=1))
    
    # Calculate the end date based on the duration
    end_date = today + forecast_duration
    
    # Use a fixed amount of historical data for training, e.g., 3 times the forecast duration
    start_date = today - (forecast_duration * 3)

    df = fetch_data(db, start_date, today)

    forecast_metrics = ['revenue', 'costs', 'units_sold', 'repeat_customers']
    forecast_results = {}

    for metric in forecast_metrics:
        df_metric = df[['date', metric]].rename(columns={'date': 'ds', metric: 'y'})
        
        try:
            ensemble_pred, all_metrics = parallel_forecast(df_metric, forecast_duration.days)
            
            start_value = df_metric['y'].iloc[-1]  # Last known value (today)
            end_value = ensemble_pred[-1]  # Last forecasted value
            percentage_change = ((end_value - start_value) / start_value) * 100
            
            forecast_results[metric] = {
                "ensemble_forecast": {
                    "percentage_change": float(round(percentage_change, 2)),
                    "start_date": today.isoformat(),
                    "end_date": end_date.isoformat(),
                    "start_amount": float(round(start_value, 2)),
                    "end_amount": float(round(end_value, 2)),
                    "trend": "up" if percentage_change > 0 else "down"
                },
                "evaluation_metrics": all_metrics
            }
        except Exception as e:
            logger.error(f"Error forecasting {metric}: {str(e)}")
            forecast_results[metric] = {"error": str(e)}

    return {
        "duration": duration,
        "forecast": forecast_results
    }

@router.get("/metric_cards")
async def get_metrics(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    scope: str = Query("this_year", description="Data scope"),
    resolution: str = Query("monthly", description="Data resolution"),
    forecast: bool = Query(False, description="Include forecasts")
):
    analysis_service = DynamicAnalysisService()
    
    metrics = await analysis_service.analyze_metrics(
        db=db,
        org_id=current_user["current_org_id"],
        scope=scope,
        resolution=resolution,
        forecast=forecast
    )
    
    return metrics

@router.get("/metric_names")
async def get_available_metrics(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all available metrics for the organization."""
    try:
        # Get all active metrics from all data sources for this organization
        metrics = (
            db.query(MetricDefinition)
            .join(DataSourceConnection)
            .filter(
                DataSourceConnection.organization_id == current_user["current_org_id"],
                MetricDefinition.is_active == True
            )
            .all()
        )

        # Group metrics by category
        categorized_metrics = {}
        for metric in metrics:
            if metric.category not in categorized_metrics:
                categorized_metrics[metric.category] = []
                
            categorized_metrics[metric.category].append({
                "name": metric.name,
                "visualization_type": metric.visualization_type,
                "business_context": metric.business_context,
                "source": metric.connection.name
            })

        return {
            "categories": list(categorized_metrics.keys()),
            "metrics": categorized_metrics
        }

    except Exception as e:
        logger.error(f"Error getting available metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )

@router.get("/metric_forecast")
def get_metric_forecast(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    metric_id: int = Query(..., description="ID of the metric to forecast"),
    forecast_duration: str = Query("next_month", description="Forecast duration"),
    resolution: str = Query("monthly", description="Data resolution")
):
    """Get forecast for a specific metric."""
    try:
        # Get the metric definition
        metric = (
            db.query(MetricDefinition)
            .join(DataSourceConnection)
            .filter(
                MetricDefinition.id == metric_id,
                MetricDefinition.is_active == True,
                DataSourceConnection.organization_id == current_user["current_org_id"]
            )
            .first()
        )

        if not metric:
            raise HTTPException(
                status_code=404,
                detail="Metric not found or not accessible"
            )

        analysis_service = DynamicAnalysisService()
        forecast_data = analysis_service.generate_forecast(
            db=db,
            org_id=current_user["current_org_id"],
            metric=metric,
            duration=forecast_duration,
            resolution=resolution
        )

        return forecast_data

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )

@router.get("/bulk_forecast")
async def get_bulk_forecast(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    metric_ids: List[int] = Query(..., description="List of metric IDs to forecast"),
    forecast_duration: str = Query("next_month", description="Forecast duration"),
    resolution: str = Query("monthly", description="Data resolution")
):
    """Get forecasts for multiple metrics."""
    try:
        # Get all requested metrics that are active and accessible to the user
        metrics = (
            db.query(MetricDefinition)
            .join(DataSourceConnection)
            .filter(
                MetricDefinition.id.in_(metric_ids),
                MetricDefinition.is_active == True,
                DataSourceConnection.organization_id == current_user["current_org_id"]
            )
            .all()
        )

        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="No valid metrics found"
            )

        analysis_service = DynamicAnalysisService()
        forecasts = {}

        for metric in metrics:
            try:
                forecast_data = await analysis_service.generate_forecast(
                    db=db,
                    org_id=current_user["current_org_id"],
                    metric=metric,
                    duration=forecast_duration,
                    resolution=resolution
                )
                forecasts[metric.name] = {
                    "data": forecast_data,
                    "metadata": {
                        "category": metric.category,
                        "visualization_type": metric.visualization_type,
                        "business_context": metric.business_context,
                        "source": metric.connection.name
                    }
                }
            except Exception as e:
                logger.error(f"Error forecasting {metric.name}: {str(e)}")
                forecasts[metric.name] = {"error": str(e)}

        return {
            "forecast_duration": forecast_duration,
            "resolution": resolution,
            "forecasts": forecasts
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating bulk forecast: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating bulk forecast: {str(e)}"
        )

@router.get("/available_forecast_metrics")
def get_forecastable_metrics(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all metrics that are suitable for forecasting."""
    try:
        # Get metrics that have appropriate types for forecasting
        metrics = (
            db.query(MetricDefinition)
            .join(DataSourceConnection)
            .filter(
                DataSourceConnection.organization_id == current_user["current_org_id"],
                MetricDefinition.is_active == True,
                MetricDefinition.visualization_type.in_([
                    'line_chart', 'line', 'bar_chart', 'bar', 'area_chart', 'area',
                    'Line Chart', 'Bar Chart', 'Area Chart'
                ])
            )
            .all()
        )

        # Further filter metrics based on calculation and data dependencies
        forecastable_metrics = []
        analysis_service = DynamicAnalysisService()
        
        for metric in metrics:
            # Check if metric has sufficient historical data for forecasting
            historical_data = analysis_service._get_metric_history(
                db=db,
                org_id=current_user["current_org_id"],
                metric=metric,
                lookback_days=90  # Minimum data points needed for forecasting
            )

            if historical_data and len(historical_data) >= 30:  # Minimum number of data points
                # Verify the metric calculation involves numeric operations
                calculation = metric.calculation.lower()
                numeric_indicators = ['sum', 'avg', 'count', 'min', 'max', 'mean', 'median']
                
                # Check if the calculation involves numeric operations
                if any(indicator in calculation for indicator in numeric_indicators):
                    confidence_score = metric.confidence_score or 0.5  # Default confidence if None
                    
                    # Add additional confidence based on data quality and quantity
                    if len(historical_data) > 180:  # More historical data increases confidence
                        confidence_score += 0.2
                    if metric.aggregation_period.lower() in ['daily', 'weekly', 'monthly']:
                        confidence_score += 0.1
                        
                    metric.confidence_score = min(confidence_score, 1.0)  # Cap at 1.0
                    forecastable_metrics.append(metric)

        # Organize metrics by category
        categorized_metrics = {}
        for metric in forecastable_metrics:
            if metric.category not in categorized_metrics:
                categorized_metrics[metric.category] = []

            categorized_metrics[metric.category].append({
                "id": metric.id,
                "name": metric.name,
                "visualization_type": metric.visualization_type,
                "business_context": metric.business_context,
                "source": metric.connection.name,
                "confidence_score": metric.confidence_score,
                "calculation": metric.calculation,
                "aggregation_period": metric.aggregation_period,
                "forecast_settings": {
                    "min_historical_days": 30,
                    "recommended_forecast_period": metric.aggregation_period,
                    "max_forecast_horizon": 90  # days
                }
            })

        # Only include categories that have metrics
        filtered_categories = [cat for cat in categorized_metrics.keys() if categorized_metrics[cat]]

        return {
            "categories": filtered_categories,
            "metrics": categorized_metrics
        }

    except Exception as e:
        logger.error(f"Error getting forecastable metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )


# Add the forecast generation method to DynamicAnalysisService
async def generate_forecast(
    self,
    db: Session,
    org_id: int,
    metric: MetricDefinition,
    duration: str,
    resolution: str
) -> Dict[str, Any]:
    """Generate forecast for a specific metric."""
    try:
        # Get historical data for the metric
        historical_data = await self._get_metric_history(
            db=db,
            org_id=org_id,
            metric=metric,
            lookback_days=365  # Use 1 year of historical data
        )

        if not historical_data:
            raise ValueError("No historical data available for forecasting")

        # Prepare data for forecasting
        df = pd.DataFrame(historical_data)
        df['ds'] = pd.to_datetime(df['period'])
        df['y'] = df[metric.name]

        # Calculate forecast horizon
        forecast_days = self._get_forecast_days(duration)

        # Generate forecasts using multiple models
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._prophet_forecast, df.copy(), forecast_days),
                executor.submit(self._sarima_forecast, df.copy(), forecast_days),
                executor.submit(self._exp_smoothing_forecast, df.copy(), forecast_days)
            ]

            forecasts = []
            metrics = {}
            for future, method in zip(as_completed(futures), ['prophet', 'sarima', 'exp_smoothing']):
                try:
                    forecast, forecast_metrics = future.result()
                    if forecast is not None:
                        forecasts.append(forecast)
                        metrics[method] = forecast_metrics
                except Exception as e:
                    logger.error(f"Error in {method} forecast: {str(e)}")

        if not forecasts:
            raise ValueError("All forecasting methods failed")

        # Calculate ensemble forecast
        ensemble_forecast = np.mean(forecasts, axis=0)
        
        # Generate forecast dates
        last_date = pd.to_datetime(df['ds'].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Format results
        forecast_results = {
            "metric_name": metric.name,
            "forecast_points": [
                {
                    "date": date.isoformat(),
                    "value": float(value),
                    "confidence_interval": {
                        "lower": float(value * 0.9),  # Simple confidence interval
                        "upper": float(value * 1.1)
                    }
                }
                for date, value in zip(forecast_dates, ensemble_forecast)
            ],
            "metadata": {
                "start_date": last_date.isoformat(),
                "end_date": forecast_dates[-1].isoformat(),
                "duration": duration,
                "resolution": resolution,
                "model_metrics": metrics
            }
        }

        return forecast_results

    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise

@router.get("/single_metric_card")
async def get_single_metric_card(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    metric: str = Query(..., description="Metric to display"),
    scope: str = Query("this_year", description="Data scope (this_week, this_month, this_quarter, this_year)"),
    resolution: str = Query("monthly", description="Graph data resolution (daily, weekly, monthly, quarterly)")
):
    try:
        # First try to get metric definition from MetricDefinition
        metric_def = db.query(MetricDefinition)\
            .join(DataSourceConnection)\
            .filter(
                MetricDefinition.name == metric,
                DataSourceConnection.organization_id == current_user["current_org_id"],
                MetricDefinition.is_active == True
            ).first()

        # Get date range
        start_date, end_date = get_date_range(scope)
        
        if metric_def:
            # Use DynamicAnalysisService for metrics from MetricDefinition
            analysis_service = DynamicAnalysisService()
            all_metrics = await analysis_service.analyze_metrics(
                db=db,
                org_id=current_user["current_org_id"],
                scope=scope,
                resolution=resolution,
                forecast=False
            )
            
            # Extract the specific metric we want
            if metric in all_metrics.get("metrics", {}):
                metric_data = {
                    "percentage_change": all_metrics["metrics"][metric]["change"]["percentage"],
                    "start_date": all_metrics["metadata"]["start_date"],
                    "end_date": all_metrics["metadata"]["end_date"],
                    "start_amount": all_metrics["metrics"][metric]["previous_value"],
                    "end_amount": all_metrics["metrics"][metric]["current_value"],
                    "trend": all_metrics["metrics"][metric].get("trend", "up" if all_metrics["metrics"][metric]["change"]["percentage"] > 0 else "down"),
                    "graph_data": all_metrics["metrics"][metric].get("trend_data", [])
                }
                
                return to_json_serializable({
                    "scope": scope,
                    "resolution": resolution,
                    "metric": metric,
                    "metric_card": metric_data
                })
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No data found for metric: {metric}"
                )
            
        else:
            # Fallback to WayneEnterprise model for built-in metrics
            if not hasattr(WayneEnterprise, metric):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metric name: {metric}"
                )

            # Query data with dynamic metric selection
            df = fetch_data(db, start_date, end_date)
            
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Fetched data shape: {df.shape}")
            logger.info(f"Data date range: {df['date'].min()} to {df['date'].max()}")

            df_start = df[df['date'] == df['date'].min()]
            df_end = df[df['date'] == df['date'].max()]
            
            # Determine aggregation method based on the metric
            should_use_mean = metric in ['customer_satisfaction_rating', 'average_performance_rating']
            
            # Calculate start and end values using appropriate aggregation
            start_value = df_start[metric].mean() if should_use_mean else df_start[metric].sum()
            end_value = df_end[metric].mean() if should_use_mean else df_end[metric].sum()
            
            logger.info(f"Metric: {metric}, Start value: {start_value}, End value: {end_value}")
            
            percentage_change = calculate_percentage_change(start_value, end_value)
            
            try:
                graph_data = get_graph_data(df, metric, resolution)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            metric_card = {
                "percentage_change": percentage_change,
                "start_date": df['date'].min().isoformat(),
                "end_date": df['date'].max().isoformat(),
                "start_amount": float(round(start_value, 2)),
                "end_amount": float(round(end_value, 2)),
                "trend": "up" if percentage_change > 0 else "down",
                "graph_data": graph_data
            }

            return to_json_serializable({
                "scope": scope,
                "resolution": resolution,
                "metric": metric,
                "metric_card": metric_card
            })

    except Exception as e:
        logger.error(f"Error getting single metric card: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metric data: {str(e)}"
        )