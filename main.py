import datetime
import json
import os
import pickle
import subprocess
import threading
import uuid

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from pymongo import MongoClient
from requests_toolbelt.multipart.encoder import MultipartEncoder

from constants import Constants
from app.utils.env_service import env_service
from app.config.mongo_service import db_mongo
from app.routers.user_router import router as user_router
from app.routers.upload_router import router as upload_router
from app.routers.polygon_router import router as polygon_router
from app.routers.notification_router import router as notification_router
from app.routers.history_router import router as history_router
from app.utils.env_service import env_service

PARENT_PATH = os.getcwd()

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI()

if not os.path.exists(Constants.PUBLIC_FOLDER):
    os.makedirs(Constants.PUBLIC_FOLDER)
if not os.path.exists(Constants.DETECTION_FOLDER):
    os.makedirs(Constants.DETECTION_FOLDER)
if not os.path.exists(Constants.THUMBNAIL_FOLDER):
    os.makedirs(Constants.THUMBNAIL_FOLDER)
app.mount("/static", StaticFiles(directory=Constants.PUBLIC_FOLDER), name="static")
app.mount(
    "/detection", StaticFiles(directory=Constants.DETECTION_FOLDER), name="detection"
)
app.mount(
    "/thumbnail", StaticFiles(directory=Constants.THUMBNAIL_FOLDER), name="thumbnail"
)


@app.get("/")
def root():
    return {"message": ""}


def configure() -> None:
    app.include_router(user_router)
    app.include_router(upload_router)
    app.include_router(polygon_router)
    app.include_router(notification_router)
    app.include_router(history_router)


@app.on_event("startup")
async def startup_process():
    configure()
    env_service.load_env("")
    await db_mongo.connect_to_mongo()


@app.on_event("shutdown")
def shutdown_db_client():
    db_mongo.close_mongo_connection


# if __name__ == "__main__":
#     connection_string = "rtsp://admin:Admin12345@tronghau7.kbvision.tv:37779/cam/realmonitor?channel=1&subtype=0"

if __name__ == "__main__":
    uvicorn.run("main:app", host=env_service.get_env_var("BASE_ADDRESS"), port=8008, reload=True)
