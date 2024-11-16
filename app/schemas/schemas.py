#schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

class OrganizationBase(BaseModel):
    name: str
    is_demo: bool = False
    data_source_connected: bool = False

class OrganizationCreate(BaseModel):
    name: str
    is_demo: bool = False
    data_source_connected: bool = False

class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    is_demo: Optional[bool] = None
    data_source_connected: Optional[bool] = None

class Organization(OrganizationBase):
    id: int
    created_at: datetime
    created_by: Optional[int] = None  # Make it optional

    class Config:
        from_attributes = True

class OrganizationResponse(BaseModel):
    message: str
    organization: Optional[Organization] = None

class OrganizationMembershipResponse(BaseModel):
    message: str
    success: bool

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    data_access: Optional[str] = None
    organization_name: str

class UserRole(BaseModel):
    organization_id: int
    role: str

class UserInOrg(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True

class User(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    data_access: str
    is_active: bool
    is_admin: bool
    organizations: List[Organization]

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    suggested_questions: List[str]
    session_id: str

class ChatHistoryResponse(BaseModel):
    question: str
    answer: str
    timestamp: datetime
    documents: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class LikedPostResponse(BaseModel):
    message: str
    liked: bool

class EmailRequest(BaseModel):
    email: EmailStr

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    is_active: bool

    class Config:
        from_attributes = True

class DataSourceConnection(BaseModel):
    source_type: str
    name: str
    host: Optional[str] = None
    user: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    port: Optional[int] = None
    table_name: str
    date_column: Optional[str] = None  # Added date_column field
    credentials_file: Optional[str] = None
    spreadsheet_id: Optional[str] = None
    security_token: Optional[str] = None
    domain: Optional[str] = 'login'
    account: Optional[str] = None
    warehouse: Optional[str] = None
    schema: Optional[str] = None

class DataSourceConnectionResponse(BaseModel):
    id: str
    organization_id: int
    name: str
    source_type: str
    table_name: str
    date_column: Optional[str] = None  # Added date_column field
    connection_params: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MetricDefinitionCreate(BaseModel):
    name: str
    category: str
    calculation: str
    data_dependencies: List[str]
    aggregation_period: str
    visualization_type: str
    business_context: Optional[str]
    confidence_score: float

class MetricDefinitionResponse(BaseModel):
    id: int
    name: str
    category: str
    calculation: str
    aggregation_period: str
    visualization_type: str
    confidence_score: float
    business_context: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class AnalyticsConfigurationCreate(BaseModel):
    metrics: List[int]
    time_ranges: List[str]
    dimensions: List[str]
    refresh_schedule: Optional[str]

class AnalyticsConfigurationResponse(BaseModel):
    id: int
    connection_id: int
    metrics: List[int]
    time_ranges: List[str]
    dimensions: List[str]
    refresh_schedule: Optional[str]
    priority_score: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Visualization(BaseModel):
    """Schema for chart visualization properties."""
    type: str
    axis_label: str
    value_format: Dict[str, Any]
    show_points: bool
    stack_type: Optional[str]
    show_labels: bool

    class Config:
        from_attributes = True

class GraphData(BaseModel):
    """Schema for metric data with visualization."""
    current: float
    previous: float
    change: float
    change_percentage: float
    visualization: Optional[Visualization] = None

    class Config:
        from_attributes = True

class SourceInfo(BaseModel):
    """Schema for source information."""
    id: str
    name: str
    type: str

class MetricSourceInfo(BaseModel):
    """Schema for metrics by source."""
    metrics: List[str]
    values: Dict[str, Any]

class ArticleSourceInfo(BaseModel):
    """Schema for article source information."""
    sources: List[SourceInfo]
    metrics_by_source: Dict[str, MetricSourceInfo]

class NewsArticle(BaseModel):
    """Schema for news articles with visualizations."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    category: str
    time_period: str
    graph_data: Dict[str, GraphData]
    source_info: Optional[ArticleSourceInfo] = None  

    class Config:
        from_attributes = True

class NewsFeed(BaseModel):
    """Schema for collection of news articles."""
    articles: List[NewsArticle]

    class Config:
        from_attributes = True