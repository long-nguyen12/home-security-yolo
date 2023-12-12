from typing import List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from ..utils.env_service import env_service


class MongoModel(BaseModel):
    id: Optional[str] = str(ObjectId())


class MongoService:
    def __init__(self):
        self.mongo_client = None
        self.db_name = None

    async def connect_to_mongo(self):
        self.mongo_client = AsyncIOMotorClient(
            "{}".format(env_service.get_env_var("DB_URL"))
        )
        self.db_name = env_service.get_env_var("DB_NAME")
        return self.mongo_client

    async def close_mongo_connection(self):
        self.mongo_client.close()

    async def is_exists(
        self, collection_name: str, search_by: str, search_value: str
    ) -> bool:
        result = await self.mongo_client[self.db_name][collection_name].find_one(
            {f"{search_by}": search_value}
        )
        return bool(result)

    async def create(self, collection_name: str, obj: MongoModel) -> MongoModel:
        # if await self.is_exists(collection_name, "username", obj["username"]):
        #     return None
        result = await self.mongo_client[self.db_name][collection_name].insert_one(obj)
        obj["id"] = str(result.inserted_id)
        return obj

    async def get(
        self,
        collection_name: str,
        search_by: str,
        search_value: str,
        model_cls: MongoModel,
    ) -> Optional[MongoModel]:
        result = await self.mongo_client[self.db_name][collection_name].find_one(
            {f"{search_by}": search_value}
        )
        if result:
            return model_cls(**result)
        else:
            raise HTTPException(status_code=404, detail="Not found!")
        
    async def get_user(
        self,
        collection_name: str,
        search_by: str,
        search_value: str,
        model_cls: MongoModel,
    ) -> Optional[MongoModel]:
        result = await self.mongo_client[self.db_name][collection_name].find_one(
            {f"{search_by}": search_value}
        )
        if result:
            return model_cls(**result)
        else:
            return None

    async def update(
        self,
        collection_name: str,
        update_search_by: str,
        update_search_value: str,
        update_data: dict,
    ) -> BaseModel:
        update_data = {k: v for k, v in update_data.items() if v is not None}
        result = await self.mongo_client[self.db_name][collection_name].update_one(
            {f"{update_search_by}": update_search_value}, {"$set": update_data}
        )
        if result.modified_count >= 1:
            return JSONResponse(status_code=200, content={"message": True})

        raise HTTPException(status_code=404, detail="Not found!")

    async def delete(
        self, collection_name, delete_by: str = "username", delete_value: str = ""
    ) -> None:
        if await self.is_exists(collection_name, delete_by, delete_value):
            return None
        result = await self.mongo_client[self.db_name][collection_name].delete_one(
            {f"{delete_by}": delete_value}
        )
        if result.deleted_count != 1:
            raise HTTPException(status_code=404, detail="Object not found")

    async def get_all(
        self,
        collection_name: str,
        skip: int = 0,
        limit: int = 10,
        sort_by: Optional[str] = "created_at",
        model_cls: Optional[MongoModel] = None,
        query: Optional[dict] = None,
    ) -> List[MongoModel]:
        sort = []
        if sort_by:
            sort.append((sort_by, -1))
        result = self.mongo_client[self.db_name][collection_name].find(
            query, skip=skip, limit=limit, sort=sort
        )
        try:
            return [model_cls(**doc) async for doc in result]
        except Exception as e:
            print(e)
        


db_mongo = MongoService()
