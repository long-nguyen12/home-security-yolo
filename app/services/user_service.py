from app.config.mongo_service import db_mongo
from fastapi.responses import JSONResponse
from fastapi import status
from app.middlewares.auth_handler import JwtHandler
from app.schemas import user as user_model
from passlib.context import CryptContext
from .base import BaseService
from datetime import datetime


class UserService(BaseService):
    """
    Service class for Users collection in MongoDB.
    """

    def __init__(self):
        super().__init__("users", user_model.UserSchema)
        self.hasher = CryptContext(schemes=["bcrypt"])
        self.jwt_handler = JwtHandler()

    def encode_password(self, password: str):
        return self.hasher.hash(password)

    def verify_password(self, password: str, encoded_password: str):
        return self.hasher.verify(password, encoded_password)

    async def create(self, user) -> JSONResponse:
        hashed_password = self.encode_password(user["password"])
        user["password"] = hashed_password
        user["createdAt"] = datetime.utcnow()
        user["updatedAt"] = datetime.utcnow()
        new_user = await db_mongo.create(self.collection_name, user)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED
            if new_user is not None
            else status.HTTP_409_CONFLICT,
            content={"data": new_user},
        )

    async def check_login(
        self, entered_password: str, current_password: str, username: str
    ):
        if self.verify_password(entered_password, current_password):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=self.jwt_handler.sign_jwt(username),
            )
