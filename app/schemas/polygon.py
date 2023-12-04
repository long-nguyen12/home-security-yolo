import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.base import PyObjectId
from bson import ObjectId


class PolygonSchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    video: str = Field(...)
    point_1: dict = Field(...)
    point_2: dict = Field(...)
    point_3: dict = Field(...)
    point_4: dict = Field(...)
    width: float = Field(...)
    height: float = Field(...)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {"example": {}}
