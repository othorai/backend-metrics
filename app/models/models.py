#models.py
from sqlalchemy import Column, Integer, String, Boolean, Float, Date, ForeignKey, DateTime, Text, JSON, LargeBinary, UniqueConstraint,Table

from app.utils.database import Base
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

user_organizations = Table(
    'user_organizations', 
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('organization_id', Integer, ForeignKey('organizations.id', ondelete='CASCADE'), primary_key=True),
    Column('role', String(50), default="member"),  # This will store 'admin' or 'member'
    Column('joined_at', DateTime, default=datetime.utcnow)
)

class DataSourceConnection(Base):
    __tablename__ = "data_source_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'))
    name = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    connection_params = Column(JSON, nullable=False)
    table_name = Column(String, nullable=False)
    date_column = Column(String, nullable=True)  # Added date_column field
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="data_source_connections")
    metrics = relationship("MetricDefinition", back_populates="connection", cascade="all, delete-orphan")
    analytics_configs = relationship("AnalyticsConfiguration", back_populates="connection", cascade="all, delete-orphan")

    def to_dict(self) -> Dict:
        """Convert connection to dictionary format."""
        return {
            'id': str(self.id),
            'name': self.name,
            'source_type': self.source_type,
            'params': self.connection_params,
            'table_name': self.table_name,
            'date_column': self.date_column,  # Include date_column in the dict
            'connection_id': str(self.id)
        }

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    is_demo = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    data_source_connected = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey('users.id'))

    users = relationship(
        "User",
        secondary=user_organizations,
        back_populates="organizations",
        cascade="all, delete",
        passive_deletes=True
    )
    creator = relationship(
        "User",
        foreign_keys=[created_by],
        backref="created_organizations"
    )
    data_source_connections = relationship("DataSourceConnection", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    role = Column(String(50))
    data_access = Column(Text)
    
    # Add cascade delete for organization associations
    organizations = relationship(
        "Organization", 
        secondary=user_organizations, 
        back_populates="users",
        cascade="all, delete"
    )
    interactions = relationship("InteractionHistory", back_populates="user")
    liked_posts = relationship("LikedPost", back_populates="user")
    
    def is_org_admin(self, org_id: int, db: Session) -> bool:
        """Check if user is admin in a specific organization"""
        result = db.query(user_organizations).filter(
            user_organizations.c.user_id == self.id,
            user_organizations.c.organization_id == org_id,
            user_organizations.c.role == 'admin'
        ).first()
        return result is not None

    def get_org_role(self, org_id: int, db: Session) -> Optional[str]:
        """Get user's role in a specific organization"""
        result = db.query(user_organizations).filter(
            user_organizations.c.user_id == self.id,
            user_organizations.c.organization_id == org_id
        ).first()
        return result.role if result else None


class InteractionHistory(Base):
    __tablename__ = "interaction_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    question = Column(Text)
    answer = Column(Text)
    documents = Column(Text)
    original_document = Column(LargeBinary)  # New column to store the original document
    document_filename = Column(String)  # New column to store the original filename
    document_type = Column(String)  # New column to store the document type (e.g., 'pdf', 'csv')
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")

class SuggestedQuestion(Base):
    __tablename__ = "suggested_questions"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50))
    question = Column(String)

class WayneEnterprise(Base):
    __tablename__ = "wayne_enterprise"

    date = Column(Date, primary_key=True)
    department = Column(String, primary_key=True)
    product = Column(String, primary_key=True)
    location = Column(String, primary_key=True)
    revenue = Column(Float)
    costs = Column(Float)
    units_sold = Column(Integer)
    customer_satisfaction = Column(Float)
    marketing_spend = Column(Float)
    new_customers = Column(Integer)
    repeat_customers = Column(Integer)
    website_visits = Column(Integer)
    

    def __repr__(self):
        return f"<WayneEnterprise(date={self.date}, department={self.department}, product={self.product}, location={self.location})>"
    
class Article(Base):
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    time_period = Column(String, nullable=False)
    graph_data = Column(JSON, nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    likes = relationship("LikedPost", back_populates="article")

    def __repr__(self):
        return f"<Article(id={self.id}, date={self.date}, title={self.title})>"

class LikedPost(Base):
    __tablename__ = "liked_posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False)
    liked_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="liked_posts")
    article = relationship("Article", back_populates="likes")

    __table_args__ = (UniqueConstraint('user_id', 'article_id', name='uq_user_article_like'),)

class MetricDefinition(Base):
    __tablename__ = "metric_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey('data_source_connections.id', ondelete='CASCADE'))
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    calculation = Column(String, nullable=False)
    data_dependencies = Column(JSON, nullable=False)
    aggregation_period = Column(String, nullable=False)
    visualization_type = Column(String, nullable=False)
    business_context = Column(String)
    confidence_score = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    connection = relationship("DataSourceConnection", back_populates="metrics")

class AnalyticsConfiguration(Base):
    __tablename__ = "analytics_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey('data_source_connections.id', ondelete='CASCADE'))
    metrics = Column(JSON, nullable=False)
    time_ranges = Column(JSON, nullable=False)
    dimensions = Column(JSON, nullable=False)
    refresh_schedule = Column(String)
    priority_score = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    connection = relationship("DataSourceConnection", back_populates="analytics_configs")