#config.py
from pydantic_settings import BaseSettings
from typing import List
from urllib.parse import quote_plus
from pathlib import Path

class Settings(BaseSettings):
    # Database settings
    DB_PASSWORD: str
    DB_NAME: str
    DB_HOST: str
    DB_PORT: str
    DB_USER: str

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 

    # Other settings
    DEBUG: bool = True 
    ALLOWED_HOSTS: str = "*"

    # OpenAI settings
    OPENAI_API_KEY: str

    # AWS settings - Match exactly with .env file
    AWS_REGION: str
    AWS_ACCOUNT_ID: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    APP_NAME: str
    CLUSTER_NAME: str
    SERVICE_NAME: str

    # ECS Configuration
    ECS_CPU: str
    ECS_MEMORY: str
    ECS_CONTAINER_PORT: str

    # VPC Configuration
    VPC_ID: str
    VPC_SUBNET_1: str
    VPC_SUBNET_2: str
    SECURITY_GROUP: str

    @property
    def DATABASE_URL(self):
        url = f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"
        print(f"Constructed DATABASE_URL: {url}")
        return url

    @property
    def ALLOWED_HOSTS_LIST(self) -> List[str]:
        return [host.strip() for host in self.ALLOWED_HOSTS.split(',')]

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True 
        extra = "ignore"  

settings = Settings()

MYSQL_HOST = 'your_mysql_host'
MYSQL_USER = 'your_mysql_user'
MYSQL_PASSWORD = 'your_mysql_password'
MYSQL_DATABASE = 'your_mysql_database'

PG_HOST = 'your_postgresql_host'
PG_USER = 'your_postgresql_user'
PG_PASSWORD = 'your_postgresql_password'
PG_DATABASE = 'your_postgresql_database'

GOOGLE_SHEETS_CREDENTIALS_FILE = 'path/to/your/credentials.json'
GOOGLE_SHEETS_SPREADSHEET_ID = 'your_spreadsheet_id'

SALESFORCE_USERNAME = 'your_salesforce_username'
SALESFORCE_PASSWORD = 'your_salesforce_password'
SALESFORCE_SECURITY_TOKEN = 'your_salesforce_security_token'