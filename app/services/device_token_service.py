from app.config.mongo_service import db_mongo
from .base import BaseService
from app.schemas import device_token
from fastapi.responses import JSONResponse
from fastapi import status
from app.schemas.device_token import DeviceTokenSchema


class DeviceTokenService(BaseService):
    def __init__(self):
        super().__init__("devicetokens", device_token.DeviceTokenSchema)

    async def create(self, data):
        result = await db_mongo.create(self.collection_name, data)
        if result:
            return JSONResponse(
                status_code=status.HTTP_201_CREATED, content={"data": result}
            )

        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, content={"data": result}
        )