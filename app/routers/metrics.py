#metrics.py (metrics)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.models.models import WayneEnterprise
import pandas as pd
import numpy as np
import calendar
import asyncio
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
    """Calculate date range based on rolling periods from today."""
    today = datetime.now().date()
    
    if scope == 'past_7_days':
        start_date = today - timedelta(days=7)
    elif scope == 'past_30_days':
        start_date = today - timedelta(days=30)
    elif scope == 'past_4_months':
        # Calculate start date as 4 months ago from today
        year = today.year
        month = today.month - 4  # Go back 4 months
        
        # Handle year boundary
        if month <= 0:
            month = 12 + month
            year -= 1
            
        start_date = datetime(year, month, today.day).date()
    elif scope == 'past_12_months':
        # Calculate start date as 12 months ago from today
        start_date = today.replace(year=today.year - 1)
    else:  # Default to past 30 days
        start_date = today - timedelta(days=30)
    
    return start_date, today

def get_forecast_end_date(forecast_duration: str, start_date: datetime.date):
    """Calculate forecast end date based on duration."""
    if forecast_duration == 'next_7_days':
        return start_date + timedelta(days=7)
    elif forecast_duration == 'next_30_days':
        return start_date + timedelta(days=30)
    elif forecast_duration == 'next_4_months':
        end_date = start_date
        for _ in range(4):
            next_month = end_date.replace(day=28) + timedelta(days=4)
            end_date = next_month - timedelta(days=next_month.day - 1)
        return end_date
    elif forecast_duration == 'next_12_months':
        end_date = start_date
        for _ in range(12):
            next_month = end_date.replace(day=28) + timedelta(days=4)
            end_date = next_month - timedelta(days=next_month.day - 1)
        return end_date
    else:  # Default to next 30 days
        return start_date + timedelta(days=30)
    
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
    scope: str = Query("past_30_days", description="Data scope"),
    resolution: str = Query("monthly", description="Data resolution"),
    forecast: bool = Query(False, description="Include forecasts")
):
    try:
        # Get date range based on scope
        start_date, end_date = get_date_range(scope)
        
        logger.info(f"""
        Fetching metrics:
        Organization ID: {current_user['current_org_id']}
        Scope: {scope}
        Resolution: {resolution}
        Date Range: {start_date} to {end_date}
        """)

        # Get metrics with calculated date range
        analysis_service = DynamicAnalysisService()
        metrics_result = await analysis_service.analyze_metrics(
            db=db,
            org_id=current_user["current_org_id"],
            scope=scope,
            resolution=resolution,
            forecast=forecast
        )
        
        # Include date range in response metadata
        if "metadata" not in metrics_result:
            metrics_result["metadata"] = {}
        
        metrics_result["metadata"].update({
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "scope": scope,
            "resolution": resolution
        })
        
        # Update the metrics data with the correct date range
        if "metrics" in metrics_result:
            for metric_name, metric_data in metrics_result["metrics"].items():
                if "trend_data" in metric_data:
                    # Filter trend data to only include points within our date range
                    filtered_trend_data = [
                        point for point in metric_data["trend_data"]
                        if start_date.isoformat() <= point["date"] <= end_date.isoformat()
                    ]
                    metric_data["trend_data"] = filtered_trend_data
                
                # Update metric start and end dates
                if metric_data.get("trend_data"):
                    metric_data["start_date"] = start_date.isoformat()
                    metric_data["end_date"] = end_date.isoformat()
        
        return metrics_result

    except Exception as e:
        logger.error(f"Error in get_metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )

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
async def get_metric_forecast(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    metric_id: int = Query(..., description="ID of the metric to forecast"),
    forecast_duration: str = Query("next_month", description="Forecast duration"),
    resolution: str = Query("monthly", description="Data resolution")
):
    """Get forecast for a specific metric."""
    try:
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
        forecast_data = await analysis_service.generate_forecast(  # Properly await the async function
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
async def get_forecastable_metrics(
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
        analysis_service = DynamicAnalysisService()
        
        forecastable_metrics = []
        
        # Use asyncio to run historical data checks concurrently
        async def check_metric(metric):
            # Check historical data length
            historical_data_length = await analysis_service._get_metric_history(
                db=db,
                org_id=current_user["current_org_id"],
                metric=metric,
                lookback_days=365
            )

            if len(historical_data_length) >= 30:  # Minimum number of data points
                # Verify the metric calculation involves numeric operations
                calculation = metric.calculation.lower()
                numeric_indicators = ['sum', 'avg', 'count', 'min', 'max', 'mean', 'median']
                
                # Check if the calculation involves numeric operations
                if any(indicator in calculation for indicator in numeric_indicators):
                    confidence_score = metric.confidence_score or 0.5  # Default confidence if None
                    
                    # Add additional confidence based on data quality and quantity
                    if len(historical_data_length) > 180:  # More historical data increases confidence
                        confidence_score += 0.2
                    if metric.aggregation_period.lower() in ['daily', 'weekly', 'monthly']:
                        confidence_score += 0.1
                        
                    metric.confidence_score = min(confidence_score, 1.0)  # Cap at 1.0
                    return metric

            return None

        # Run checks concurrently
        tasks = [check_metric(metric) for metric in metrics]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (metrics that didn't pass checks)
        forecastable_metrics = [metric for metric in results if metric is not None]

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


def _get_frequency_by_resolution(resolution: str) -> str:
    """Get pandas frequency string based on resolution."""
    resolution_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q'
    }
    return resolution_map.get(resolution, 'D')

def _get_forecast_points_by_resolution(resolution: str, duration: str) -> int:
    """Calculate number of forecast points based on resolution and duration."""
    if duration == 'next_7_days':
        return 7 if resolution == 'daily' else 1
    elif duration == 'next_30_days':
        if resolution == 'daily':
            return 30
        elif resolution == 'weekly':
            return 4
        return 1  # monthly
    elif duration == 'next_4_months':
        if resolution == 'daily':
            return 120
        elif resolution == 'weekly':
            return 16
        elif resolution == 'monthly':
            return 4
        return 2  # quarterly
    elif duration == 'next_12_months':
        if resolution == 'daily':
            return 365
        elif resolution == 'weekly':
            return 52
        elif resolution == 'monthly':
            return 12
        return 4  # quarterly
    return 30  # default

async def generate_forecast(
    self,
    db: Session,
    org_id: int,
    metric: MetricDefinition,
    duration: str,
    resolution: str
) -> Dict[str, Any]:
    """Generate forecast for a specific metric with exact date handling."""
    try:
        today = datetime.now().date()
        
        # Get historical data
        historical_data = await self._get_metric_history(
            db=db,
            org_id=org_id,
            metric=metric,
            lookback_days=365
        )

        if not historical_data:
            raise ValueError("No historical data available for forecasting")

        # Calculate exact end date based on duration
        if duration == 'next_7_days':
            end_date = today + timedelta(days=7)
        elif duration == 'next_30_days':
            end_date = today + timedelta(days=30)
        elif duration == 'next_4_months':
            # Calculate exact days for 4 months from today
            days_in_4_months = sum(
                calendar.monthrange(
                    today.year + ((today.month + i) // 12),
                    ((today.month + i - 1) % 12) + 1
                )[1] for i in range(4)
            )
            end_date = today + timedelta(days=days_in_4_months)
        elif duration == 'next_12_months':
            # Calculate exact days for 12 months
            days_in_12_months = sum(
                calendar.monthrange(
                    today.year + ((today.month + i) // 12),
                    ((today.month + i - 1) % 12) + 1
                )[1] for i in range(12)
            )
            end_date = today + timedelta(days=days_in_12_months)
        else:
            end_date = today + timedelta(days=30)

        # Generate date range based on resolution
        if resolution == 'daily':
            dates = pd.date_range(start=today, end=end_date, freq='D')
        elif resolution == 'weekly':
            # Generate weekly dates but preserve the start date
            dates = pd.date_range(start=today, end=end_date + timedelta(days=7), freq='W')
            dates = dates[dates >= pd.Timestamp(today)]
            dates = dates[dates <= pd.Timestamp(end_date)]
        elif resolution == 'monthly':
            # Generate monthly dates but preserve the start date
            dates = pd.date_range(start=today, end=end_date + timedelta(days=31), freq='M')
            dates = dates[dates >= pd.Timestamp(today)]
            dates = dates[dates <= pd.Timestamp(end_date)]
        else:  # quarterly
            # Generate quarterly dates but preserve the start date
            dates = pd.date_range(start=today, end=end_date + timedelta(days=92), freq='Q')
            dates = dates[dates >= pd.Timestamp(today)]
            dates = dates[dates <= pd.Timestamp(end_date)]

        # Prepare data for forecasting
        df = pd.DataFrame(historical_data)
        df['ds'] = pd.to_datetime(df['period'])
        df['y'] = df[metric.name]
        
        # Generate forecasts using multiple models
        forecast_points = len(dates)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._prophet_forecast, df.copy(), forecast_points),
                executor.submit(self._sarima_forecast, df.copy(), forecast_points),
                executor.submit(self._exp_smoothing_forecast, df.copy(), forecast_points)
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

        # Format results
        forecast_results = {
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
                for date, value in zip(dates, ensemble_forecast)
            ],
            "metadata": {
                "start_date": today.isoformat(),
                "end_date": end_date.isoformat(),
                "duration": duration,
                "resolution": resolution,
                "source": metric.connection.name,
                "model_metrics": metrics,
                "data_points_used": len(df),
                "forecast_points": forecast_points
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
    scope: str = Query("past_30_days", description="Data scope"),
    resolution: str = Query("monthly", description="Data resolution")
):
    try:
        # Get date range based on scope
        start_date, end_date = get_date_range(scope)
        
        logger.info(f"""
        Fetching single metric:
        Metric: {metric}
        Date Range: {start_date} to {end_date}
        Resolution: {resolution}
        """)

        metric_def = db.query(MetricDefinition)\
            .join(DataSourceConnection)\
            .filter(
                MetricDefinition.name == metric,
                DataSourceConnection.organization_id == current_user["current_org_id"],
                MetricDefinition.is_active == True
            ).first()

        if metric_def:
            analysis_service = DynamicAnalysisService()
            result = await analysis_service.analyze_metrics(
                db=db,
                org_id=current_user["current_org_id"],
                scope=scope,
                resolution=resolution,
                forecast=False
            )

            if metric not in result.get("metrics", {}):
                raise HTTPException(status_code=404, detail=f"No data found for metric: {metric}")

            metric_data = result["metrics"][metric]
            if "trend_data" in metric_data:
                # Filter trend data to only include points within our date range
                filtered_trend_data = [
                    point for point in metric_data["trend_data"]
                    if start_date.isoformat() <= point["date"] <= end_date.isoformat()
                ]
                metric_data["trend_data"] = filtered_trend_data

            response_data = {
                "scope": scope,
                "resolution": resolution,
                "metric": metric,
                "metric_card": {
                    "percentage_change": metric_data["change"]["percentage"],
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "start_amount": metric_data["previous_value"],
                    "end_amount": metric_data["current_value"],
                    "trend": metric_data.get("trend", "up" if metric_data["change"]["percentage"] > 0 else "down"),
                    "graph_data": metric_data.get("trend_data", [])
                }
            }

            return response_data
            
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