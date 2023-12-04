from fastapi import APIRouter, Body, Depends, HTTPException, status, UploadFile
from pathlib import Path
import aiofiles
import os
from typing import Union
import time
from constants import Constants
import cv2

router = APIRouter()

SUB_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
PARENT_PATH = SUB_PATH.parent.absolute()


@router.post("/api/upload/file")
async def create(file: Union[UploadFile, None] = None):
    if not file:
        raise HTTPException(status_code=404, detail="Image not found")
    # elif not file.filename.lower().endswith((".mp4", ".mov", ".png", ".jpg")):
    #     raise HTTPException(status_code=403, detail="Wrong format")
    else:
        timestamp = int(time.time())
        det_filename = str(timestamp) + "_" + file.filename
        save_path = Constants.PUBLIC_FOLDER + det_filename

        file_copy = os.path.join(PARENT_PATH, save_path)
        try:
            async with aiofiles.open(file_copy, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)

            thumbnail_filename = str(timestamp) + "_" + file.filename[0:-4] + ".png"
            thumbnail_path = os.path.join(
                PARENT_PATH, Constants.THUMBNAIL_FOLDER + thumbnail_filename
            )

            cap = cv2.VideoCapture(file_copy)

            while True:
                success, frame = cap.read()
                if not success:
                    break
                else:
                    cv2.imwrite(thumbnail_path, frame)
                    break
            return {"path": det_filename, "thumbnail": thumbnail_filename}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500)
