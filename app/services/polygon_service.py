from app.config.mongo_service import db_mongo
from .base import BaseService
from app.schemas import polygon
from fastapi.responses import JSONResponse
from fastapi import status


class PolygonService(BaseService):
    def __init__(self):
        super().__init__("polygons", polygon.PolygonSchema)

    async def create(self, data):
        result = await db_mongo.create(self.collection_name, data)
        if result:
            return JSONResponse(
                status_code=status.HTTP_201_CREATED, content={"data": result}
            )

        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, content={"data": result}
        )
