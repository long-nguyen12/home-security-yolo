import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.base import PyObjectId
from bson import ObjectId
from datetime import datetime


class UserSchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    full_name: str = Field(...)
    username: str = Field(...)
    password: str = Field(...)
    created_at: datetime = None
    updated_at: datetime = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "username": "JohnDoe",
                "full_name": "jdoe@x.edu.ng",
                "password": "Water resources engineering",
            }
        }


class UserUpdateSchema(BaseModel):
    password: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "bio": "update your bio",
            }
        }


class UserLoginSchema(BaseModel):
    username: str = Field(...)
    password: str = Field(...)

    class Config:
        schema_extra = {"example": {"username": "can@gmail.com", "password": "123456"}}
