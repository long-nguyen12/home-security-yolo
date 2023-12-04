from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.schemas import user, device_token
from app.config.mongo_service import db_mongo
from app.middlewares.auth_bearer import JwtBearer
from app.services.user_service import UserService
from app.services.device_token_service import DeviceTokenService

router = APIRouter()
user_service = UserService()
device_token_service = DeviceTokenService()


@router.post("/api/users/signup", tags=["Users"])
async def create_user(user: user.UserSchema = Body(...)):
    user_dict = jsonable_encoder(user)
    return await user_service.create(user_dict)


@router.post("/api/users/login", tags=["Users"])
async def user_login(user: user.UserLoginSchema = Body(...)):
    get_user = await user_service.get(search_by="username", search_value=user.username)
    if get_user:
        return await user_service.check_login(
            entered_password=user.password,
            current_password=get_user.password,
            username=user.username,
        )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Wrong username / password"},
    )


@router.get("/api/users/me", dependencies=[Depends(JwtBearer())], tags=["Users"])
async def current_user(username: str = Depends(JwtBearer())):
    return await user_service.get(search_by="username", search_value=username)


@router.put(
    "/api/users/update",
    dependencies=[Depends(JwtBearer())],
    response_model=user.UserSchema,
    tags=["Users"],
)
async def update_user(
    current_user: str = Depends(JwtBearer()),
    new_data: user.UserUpdateSchema = Body(...),
):
    return await user_service.update(
        search_value=current_user, new_data=jsonable_encoder(new_data)
    )


@router.post("/api/user/devicetoken", tags=["DeviceTokens"])
async def create_device_token(device_token: device_token.DeviceTokenSchema = Body(...)):
    device_token_dict = jsonable_encoder(device_token)
    search_value = str(device_token_dict["user_id"])
    try:
        user = await device_token_service.get(
            search_by="user_id", search_value=search_value
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"error": "Exist"},
        )
    except:
        return await device_token_service.create(device_token_dict)


@router.get("/api/user/devicetoken", tags=["DeviceTokens"])
async def get_device_token(
    username=Depends(JwtBearer()),
):
    user = await user_service.get(search_by="username", search_value=username)
    return await device_token_service.get(
        search_by="user_id", search_value=str(user.id)
    )
