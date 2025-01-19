from pymongo import MongoClient
from app.connectors.base import BaseConnector
import logging
from bson import ObjectId
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class MongoDBConnector(BaseConnector):
    def __init__(self, host: str, username: str = None, password: str = None, database: str = None, port: int = 27017):
        super().__init__()
        self.source_type = 'mongodb'
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB with error handling."""
        try:
            # Create the MongoDB connection URI
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
            else:
                uri = f"mongodb://{self.host}:{self.port}"
            
            logger.info(f"Attempting to connect to MongoDB database {self.database}")
            self.connection = MongoClient(uri)
            self.db = self.connection[self.database]
            
            # Test connection by executing a simple command
            self.db.command('ping')
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            if self.connection:
                self.connection.close()
                self.connection = None
            raise ValueError(f"Failed to establish MongoDB connection: {str(e)}")

    def disconnect(self):
        """Safely close the MongoDB connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.db = None
                logger.info("MongoDB connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")
            raise ValueError(f"Failed to close MongoDB connection: {str(e)}")

    def _process_metric_calculation(self, calculation: Any) -> Dict:
        """Process metric calculation into MongoDB operator."""
        try:
            if isinstance(calculation, dict):
                # Already a MongoDB operator
                return calculation
                
            if isinstance(calculation, str):
                # Try to parse as JSON first
                try:
                    parsed = json.loads(calculation)
                    if isinstance(parsed, dict):
                        # Ensure MongoDB operators are properly formatted
                        return self._ensure_mongo_operators(parsed)
                except json.JSONDecodeError:
                    pass
                
                # Common SQL aggregations to MongoDB mappings
                sql_to_mongo = {
                    'sum': '$sum',
                    'avg': '$avg',
                    'min': '$min',
                    'max': '$max',
                    'count': '$sum',
                }
                
                # Try to identify SQL aggregation pattern
                sql_pattern = r'(\w+)\((.*?)\)'
                match = re.match(sql_pattern, calculation, re.IGNORECASE)
                if match:
                    agg_type, field = match.groups()
                    mongo_op = sql_to_mongo.get(agg_type.lower())
                    if mongo_op:
                        # Handle special cases
                        if agg_type.lower() == 'count':
                            if field.strip() == '*':
                                return {mongo_op: 1}
                            return {mongo_op: {
                                "$cond": [
                                    {"$ne": [f"${field.strip()}", None]},
                                    1,
                                    0
                                ]
                            }}
                        return {mongo_op: f"${field.strip()}"}
                
                # Check if it's a condition
                if 'CASE' in calculation.upper():
                    return self._convert_case_to_mongo(calculation)
                
                # Check if it's a simple field reference
                if not any(op in calculation for op in ['(', ')', '+', '-', '*', '/']):
                    return {"$first": f"${calculation.strip()}"}
                
                # Handle arithmetic expressions
                return self._convert_arithmetic_to_mongo(calculation)
                
            return {"$first": calculation}  # Default fallback
            
        except Exception as e:
            logger.error(f"Error processing metric calculation: {str(e)}")
            return {"$first": str(calculation)}

    def _ensure_mongo_operators(self, obj: Any) -> Any:
        """Ensure all MongoDB operators have proper $ prefix."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Add $ prefix to operator keys if missing
                new_key = k if k.startswith('$') else f'${k}'
                result[new_key] = self._ensure_mongo_operators(v)
            return result
        elif isinstance(obj, list):
            return [self._ensure_mongo_operators(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('$'):
            # Handle field references
            return obj
        elif isinstance(obj, str) and not obj.startswith('$') and not obj.startswith('{'):
            # Add $ prefix to field references
            return f"${obj}"
        return obj

    def _convert_case_to_mongo(self, case_stmt: str) -> Dict:
        """Convert SQL CASE statement to MongoDB $cond operator."""
        try:
            # Very basic CASE conversion
            # CASE WHEN condition THEN value [ELSE else_value] END
            pattern = r'CASE\s+WHEN\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END'
            match = re.match(pattern, case_stmt, re.IGNORECASE | re.DOTALL)
            
            if match:
                condition, then_value, else_value = match.groups()
                return {
                    "$cond": [
                        self._convert_condition_to_mongo(condition),
                        self._convert_value_to_mongo(then_value),
                        self._convert_value_to_mongo(else_value or None)
                    ]
                }
            return {"$first": case_stmt}
            
        except Exception as e:
            logger.error(f"Error converting CASE to MongoDB: {str(e)}")
            return {"$first": case_stmt}

    def _convert_condition_to_mongo(self, condition: str) -> Dict:
        """Convert SQL condition to MongoDB operator."""
        try:
            # Handle basic comparisons
            comp_map = {
                '=': '$eq',
                '!=': '$ne',
                '>': '$gt',
                '>=': '$gte',
                '<': '$lt',
                '<=': '$lte'
            }
            
            for op, mongo_op in comp_map.items():
                if op in condition:
                    left, right = condition.split(op)
                    return {
                        mongo_op: [
                            f"${left.strip()}",
                            self._convert_value_to_mongo(right.strip())
                        ]
                    }
            
            return {"$expr": condition}  # Default fallback
            
        except Exception as e:
            logger.error(f"Error converting condition to MongoDB: {str(e)}")
            return {"$expr": condition}

    def _convert_value_to_mongo(self, value: str) -> Any:
        """Convert SQL value to MongoDB value."""
        if value is None:
            return None
            
        value = value.strip()
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        # Handle string literals
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
            
        # Handle field references
        return f"${value}"

    def _convert_arithmetic_to_mongo(self, expr: str) -> Dict:
        """Convert arithmetic expression to MongoDB operator."""
        try:
            # Very basic arithmetic conversion
            # Note: This is a simplified version and doesn't handle complex expressions
            ops = {
                '+': '$add',
                '-': '$subtract',
                '*': '$multiply',
                '/': '$divide'
            }
            
            for op, mongo_op in ops.items():
                if op in expr:
                    left, right = expr.split(op)
                    return {
                        mongo_op: [
                            self._convert_value_to_mongo(left.strip()),
                            self._convert_value_to_mongo(right.strip())
                        ]
                    }
            
            return {"$first": expr}  # Default fallback
            
        except Exception as e:
            logger.error(f"Error converting arithmetic to MongoDB: {str(e)}")
            return {"$first": expr}

    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        """Execute an aggregation pipeline on a MongoDB collection."""
        try:
            if self.connection is None:
                self.connect()

            if self.db is None:
                raise ValueError("Database connection not established")

            collection_obj = self.db.get_collection(collection)
            if collection_obj is None:
                raise ValueError(f"Collection {collection} not found")

            # Process pipeline stages
            processed_pipeline = []
            for stage in pipeline:
                if not isinstance(stage, dict):
                    continue
                    
                processed_stage = {}
                for operator, spec in stage.items():
                    # Ensure operator has $ prefix
                    clean_op = operator.lstrip('$')
                    if not clean_op.startswith('$'):
                        clean_op = f"${clean_op}"
                    
                    # Process specification
                    if isinstance(spec, dict):
                        processed_stage[clean_op] = self._process_operator_spec(spec)
                    else:
                        processed_stage[clean_op] = spec
                        
                processed_pipeline.append(processed_stage)
                
            # Log the final pipeline
            logger.info(f"Executing aggregation pipeline on {collection}: {json.dumps(processed_pipeline, default=str)}")
            
            # Execute pipeline
            results = []
            cursor = collection_obj.aggregate(processed_pipeline)
            
            # Process results
            for doc in cursor:
                processed_doc = {}
                for key, value in doc.items():
                    if isinstance(value, (datetime, ObjectId)):
                        processed_doc[key] = str(value)
                    else:
                        processed_doc[key] = value
                results.append(processed_doc)
                
            logger.info(f"Aggregation returned {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            logger.error(f"Collection: {collection}")
            logger.error(f"Pipeline: {json.dumps(pipeline, default=str)}")
            raise ValueError(f"Failed to execute aggregation: {str(e)}")

    def _process_operator_spec(self, spec: Dict) -> Dict:
        """Process MongoDB operator specification."""
        if not isinstance(spec, dict):
            return spec
            
        result = {}
        for key, value in spec.items():
            # Clean the key
            clean_key = key.lstrip('$')
            
            # Process values
            if isinstance(value, dict):
                result[clean_key] = self._process_operator_spec(value)
            elif isinstance(value, list):
                result[clean_key] = [
                    self._process_operator_spec(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str) and value.startswith('$'):
                # Handle field references
                if value.startswith('$$'):  # System variables
                    result[clean_key] = value
                else:
                    result[clean_key] = value.replace('$$', '$')
            else:
                result[clean_key] = value
                
        return result

    def query(self, collection: str, query: Dict = None, projection: Dict = None) -> List[Dict]:
        """Execute a MongoDB query."""
        if not self.connection:
            self.connect()
        
        try:
            collection_obj = self.db[collection]
            if isinstance(query, str):
                # Try to parse as aggregation pipeline
                try:
                    pipeline = json.loads(query)
                    return self.aggregate(collection, pipeline)
                except json.JSONDecodeError:
                    # Try to convert SQL-like query to MongoDB
                    query = self._process_metric_calculation(query)
            
            cursor = collection_obj.find(query or {}, projection or {})
            
            results = []
            for doc in cursor:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                if any(isinstance(value, datetime) for value in doc.values()):
                    # Found at least one date field, document is good for analysis
                    results.append(doc)
                else:
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Collection: {collection}")
            logger.error(f"Query: {query}")
            raise ValueError(f"Query execution failed: {str(e)}")

    def get_schema_info(self, collection_name: str):
        """Get schema information for MongoDB collection by sampling documents"""
        try:
            collection = self.db[collection_name]
            
            # Sample some documents directly
            sample_docs = list(collection.find().limit(100))
            
            # Analyze field types across documents
            field_types = {}
            date_fields = set()
            
            for doc in sample_docs:
                for field, value in doc.items():
                    if isinstance(value, datetime):
                        date_fields.add(field)
                    current_type = type(value).__name__
                    if field not in field_types:
                        field_types[field] = set()
                    field_types[field].add(current_type)
            
            # Format results similar to SQL schema
            schema_info = []
            for field, types in field_types.items():
                type_str = '/'.join(sorted(types))
                schema_info.append({
                    'column_name': field,
                    'data_type': type_str,
                    'is_date': field in date_fields
                })
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting MongoDB schema: {str(e)}")
            raise ValueError(f"Failed to get MongoDB schema: {str(e)}")

    def detect_date_column(self, collection_name: str) -> Optional[str]:
        """
        Detect the primary date column in a MongoDB collection
        """
        try:
            schema_info = self.get_schema_info(collection_name)
            date_columns = [
                col['column_name'] for col in schema_info 
                if col['is_date']
            ]
            
            # Prioritize common date field names
            priority_names = ['dateOfJoining', 'date', 'timestamp', 'created_at', 'createdAt', 'hire_date']
            for name in priority_names:
                if name in date_columns:
                    return name
                    
            # If no priority match, return the first date column found
            return date_columns[0] if date_columns else None
            
        except Exception as e:
            logger.error(f"Error detecting date column: {str(e)}")
            return None

    def insert(self, collection: str, data: Dict):
        """Insert a document into MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Insert operation failed: {str(e)}")
            raise ValueError(f"Failed to insert data: {str(e)}")

    def update(self, collection: str, query: Dict, data: Dict, upsert: bool = False):
        """Update documents in MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.update_many(
                query,
                {'$set': data},
                upsert=upsert
            )
            return {
                'matched_count': result.matched_count,
                'modified_count': result.modified_count,
                'upserted_id': str(result.upserted_id) if result.upserted_id else None
            }
        except Exception as e:
            logger.error(f"Update operation failed: {str(e)}")
            raise ValueError(f"Failed to update data: {str(e)}")

    def delete(self, collection: str, query: Dict):
        """Delete documents from MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.delete_many(query)
            return {
                'deleted_count': result.deleted_count
            }
        except Exception as e:
            logger.error(f"Delete operation failed: {str(e)}")
            raise ValueError(f"Failed to delete data: {str(e)}")

    def _process_pipeline_dates(self, pipeline: List[Dict]) -> List[Dict]:
        """Process pipeline to ensure dates are datetime objects."""
        def convert_dates(obj):
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif isinstance(obj, str):
                try:
                    # Try to parse as date if it looks like a date string
                    if len(obj) == 10 and obj.count('-') == 2:  # YYYY-MM-DD format
                        return datetime.strptime(obj, '%Y-%m-%d')
                    elif len(obj) > 10 and 'T' in obj:  # ISO format
                        return datetime.fromisoformat(obj.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pass
            return obj

        return convert_dates(pipeline)

    def verify_collection_exists(self, collection: str) -> bool:
        """Verify that a collection exists in the current database."""
        try:
            return collection in self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Error verifying collection existence: {str(e)}")
            return False

    def get_collection_stats(self, collection: str) -> Dict:
        """Get statistics about a collection."""
        try:
            return self.db.command('collStats', collection)
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}