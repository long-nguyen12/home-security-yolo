import secrets

from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/"
    PROJECT_NAME: str = "Home Security API"
    SECRET_KEY: str = "home-security-secret-key"
    MONGO_DATABASE_URI:str = "mongodb://homesecurity:Homesecurity208@cuongit.ddns.net:27017/homesecurity?serverSelectionTimeoutMS=5000&connectTimeoutMS=10000&authSource=homesecurity&authMechanism=SCRAM-SHA-256"
    MONGO_COLLECTION:str = "homesecurity"
    JWT_ALGORITHM = "HS256"
    RESOURCES = "public/files/uploads/images"
    HOST = '0.0.0.0'
    PORT = 5055
    DOMAIN = "http://127.0.0.1"
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()