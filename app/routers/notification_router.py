from fastapi import APIRouter, Body, Depends, HTTPException, status, UploadFile, Query
from pathlib import Path
import aiofiles
import os
from typing import Union
import time
from constants import Constants
import cv2
from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
import os
import requests
from requests.exceptions import ConnectionError, HTTPError
from app.schemas import notification
from fastapi.encoders import jsonable_encoder
from app.services.notification_service import NotificationService
from app.services.user_service import UserService
from app.services.device_token_service import DeviceTokenService
from datetime import datetime
from app.middlewares.auth_bearer import JwtBearer

router = APIRouter()
notification_service = NotificationService()
user_service = UserService()
device_token_service = DeviceTokenService()

SUB_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
PARENT_PATH = SUB_PATH.parent.absolute()

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Bearer uoL08PbLeofLRDBCoqdojSWVr2E_tTAPt6KZjBBa",
        "accept": "application/json",
        "accept-encoding": "gzip, deflate",
        "content-type": "application/json",
    }
)


def send_push_message(token, message, extra=None):
    try:
        response = PushClient(session=session).publish(
            PushMessage(to=token, body=message, data=extra)
        )
        print(response)
    except PushServerError as exc:
        print(exc)
        raise
    except (ConnectionError, HTTPError) as exc:
        print(exc)

    try:
        # We got a response back, but we don't know whether it's an error yet.
        # This call raises errors so we can handle them with normal exception
        # flows.
        response.validate_response()
    except DeviceNotRegisteredError as e:
        # Mark the push token as inactive
        print(e)


@router.post("/api/notification/", tags=["Notifications"])
async def create_polygon(notification: notification.NotificationSchema = Body(...)):
    notification_dict = jsonable_encoder(notification)
    notification_dict["created_at"] = datetime.now()
    notification_dict["updated_at"] = datetime.now()
    device_token = await device_token_service.get(
        "user_id", notification_dict["user_id"]
    )
    send_push_message(
        device_token.device_token, "Phát hiện đối tượng trong vùng theo dõi"
    )

    return await notification_service.create(notification_dict)


@router.get(
    "/api/notification/", dependencies=[Depends(JwtBearer())], tags=["Notifications"]
)
async def get_notifications(
    username: str = Depends(JwtBearer()),
    skip: int = Query(0, alias="skip", ge=0),
    limit: int = Query(10, alias="limit", le=100),
):
    current_user = await user_service.get(search_by="username", search_value=username)

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    skip = skip * limit
    return await notification_service.get_all(
        skip, limit, notification.NotificationSchema, {"user_id": str(current_user.id)}
    )


@router.get("/api/notification/id", tags=["Notifications"])
async def get_notifications(notification: notification.NotificationSchema = Body(...)):
    notification_dict = jsonable_encoder(notification)
    notification_data = await notification_service.create(notification_dict)
    device_token = await device_token_service.get(
        "user_id", notification_dict["user_id"]
    )
    send_push_message(
        device_token.device_token, "Phát hiện đối tượng trong vùng theo dõi"
    )
    return notification_data
