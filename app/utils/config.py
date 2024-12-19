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


    DB_SSLMODE: str = "disable"
    @property
    def DATABASE_URL(self):
        url = f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSLMODE}"
        return url

    @property
    def ALLOWED_HOSTS_LIST(self) -> List[str]:
        return [host.strip() for host in self.ALLOWED_HOSTS.split(',')]

    class Config:
        env_file = "../../../../.env"
        env_file_encoding = "utf-8"
        case_sensitive = True 
        extra = "ignore"  

settings = Settings()
