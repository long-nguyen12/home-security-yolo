import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.base import PyObjectId
from bson import ObjectId
from datetime import datetime


class HistorySchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    result_path: str = Field(...)
    user_id: str = Field(...)
    status: bool = Field(...)
    created_at: Optional[datetime] = Field(default=datetime.now(), alias="created_at")
    updated_at: Optional[datetime] = Field(default=datetime.now(), alias="updated_at")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {"example": {}}
