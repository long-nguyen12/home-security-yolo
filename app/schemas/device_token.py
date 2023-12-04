import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.base import PyObjectId
from bson import ObjectId


class DeviceTokenSchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    device_token: str = Field(...)
    user_id: str = Field(...)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {"example": {}}


class DeviceTokenUpdateSchema(BaseModel):
    device_token: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {"example": {}}


class DeviceTokenUserSchema(BaseModel):
    user_id: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {"example": {}}
