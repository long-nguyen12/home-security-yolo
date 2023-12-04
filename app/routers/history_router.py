from fastapi import APIRouter, Body, Depends, HTTPException, status, UploadFile, Query
from pathlib import Path
import aiofiles
import os
from typing import Union
import time
from constants import Constants
import cv2
import os
import requests
from requests.exceptions import ConnectionError, HTTPError
from app.schemas import history
from fastapi.encoders import jsonable_encoder
from app.services.user_service import UserService
from app.services.history_service import HistoryService
from datetime import datetime
from app.middlewares.auth_bearer import JwtBearer

router = APIRouter()
user_service = UserService()
history_service = HistoryService()

SUB_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
PARENT_PATH = SUB_PATH.parent.absolute()


@router.post("/api/history/", tags=["Histories"])
async def create_history(history: history.HistorySchema = Body(...)):
    notification_dict = jsonable_encoder(history)
    notification_dict["created_at"] = datetime.now()
    notification_dict["updated_at"] = datetime.now()

    return await history_service.create(notification_dict)


@router.get("/api/history/", dependencies=[Depends(JwtBearer())], tags=["Histories"])
async def get_histories(
    username: str = Depends(JwtBearer()),
    skip: int = Query(0, alias="skip", ge=0),
    limit: int = Query(10, alias="limit", le=100),
):
    current_user = await user_service.get(search_by="username", search_value=username)

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    skip = skip * limit
    return await history_service.get_all(
        skip, limit, history.HistorySchema, {"user_id": str(current_user.id)}
    )
