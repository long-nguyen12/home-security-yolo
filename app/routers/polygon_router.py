from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.schemas import polygon
from app.config.mongo_service import db_mongo
from app.middlewares.auth_bearer import JwtBearer
from app.services.polygon_service import PolygonService
from app.services.user_service import UserService
from app.utils.detection_thread import DetectionThread

router = APIRouter()
polygon_service = PolygonService()
user_service = UserService()


@router.post("/api/polygon", dependencies=[Depends(JwtBearer())], tags=["Polygons"])
async def create_polygon(
    polygon: polygon.PolygonSchema = Body(...), username=Depends(JwtBearer())
):
    polygon_dict = jsonable_encoder(polygon)
    polygon_data = await polygon_service.create(polygon_dict)
    user = await user_service.get("username", username)
    point_1 = polygon_dict["point_1"]
    point_2 = polygon_dict["point_2"]
    point_3 = polygon_dict["point_3"]
    point_4 = polygon_dict["point_4"]
    detection_thread = DetectionThread()
    detection_thread.start_detecting(
        user.id,
        polygon.video,
        polygon.width,
        polygon.height,
        [int(point_1["x"]), int(point_1["y"])],
        [int(point_2["x"]), int(point_2["y"])],
        [int(point_3["x"]), int(point_3["y"])],
        [int(point_4["x"]), int(point_4["y"])],
    )
    return polygon_data
