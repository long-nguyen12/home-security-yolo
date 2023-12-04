from app.config.mongo_service import db_mongo
from .base import BaseService
from app.schemas import history
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder


class HistoryService(BaseService):
    def __init__(self):
        super().__init__("histories", history.HistorySchema)

    async def create(self, data):
        result = await db_mongo.create(self.collection_name, data)
        result = jsonable_encoder(result)
        if result:
            return JSONResponse(
                status_code=status.HTTP_201_CREATED, content={"data": result}
            )

        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, content={"data": result}
        )

    async def get_all(self, skip, limit, data, query):
        result = await db_mongo.get_all(
            collection_name=self.collection_name,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            model_cls=data,
            query=query,
        )

        result = jsonable_encoder(result)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"data": result})
