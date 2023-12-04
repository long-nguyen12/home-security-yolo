import argparse
import datetime
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.backends.cudnn as cudnn
from yolov7.models.experimental import attempt_load
from numpy import random
from requests_toolbelt.multipart.encoder import MultipartEncoder
from yolov7.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov7.utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    TracedModel,
    load_classifier,
    select_device,
    time_synchronized,
)
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
from constants import Constants
from shapely.geometry import Point, Polygon
from app.config.mongo_service import db_mongo
from pydantic import BaseModel
from app.utils.env_service import env_service

PARENT_PATH = os.getcwd()


def interpolate_coords(coords, origin_shape, resized_shape):
    width_ratio = float(origin_shape[0] / resized_shape[0])
    height_ratio = float(origin_shape[1] / resized_shape[1])
    interpolations = []
    for coord in coords:
        interpolations.append(
            [int(coord[0] * width_ratio), int(coord[1] * height_ratio)]
        )
    return interpolations


def interpolate_point(point, origin_shape, resized_shape):
    # width_ratio = int(resized_shape[0] / origin_shape[0])
    # height_ratio = int(resized_shape[1] / origin_shape[1])
    width_ratio = float(origin_shape[0] / resized_shape[0])
    height_ratio = float(origin_shape[1] / resized_shape[1])
    x = point[0] * width_ratio
    y = point[1] * height_ratio
    return int(x), int(y)


def check_inside(ground_truth, pred):
    x = pred[0]
    y = pred[1]

    bbox_center = Point(x, y)
    polygon = Polygon(ground_truth)

    if polygon.contains(bbox_center):
        return True
    return False


class YoloDetect:
    def __init__(self, user_id, video, resized_shape, coords):
        self.alert_each = 60  # seconds
        self.last_alert = None
        self.conf_thres = 0.6
        self.iou_thres = 0.5
        self.augment = None
        self.user_id = user_id
        self.resized_shape = resized_shape
        self.coords = coords

        save_path = Constants.PUBLIC_FOLDER + video

        self.source = os.path.join(PARENT_PATH, save_path)
        print(self.source)
        self.weights, self.imgsz, self.trace = (
            "yolov7/weights/yolov7-tiny.pt",
            640,
            not False,
        )

        self.device = select_device("cpu")
        self.half = self.device.type != "cpu"

        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)

        if self.half:
            self.model.half()
        self.count_frame = 0
        self.extracted_image = []
        self.check = False

    def alert(self, img):
        if (self.last_alert is None) or (
            (datetime.datetime.utcnow() - self.last_alert).total_seconds()
            > self.alert_each
        ):
            self.last_alert = datetime.datetime.utcnow()
            file_name = (
                str(int(datetime.datetime.timestamp(datetime.datetime.now()))) + ".jpg"
            )

            # multipart_data = MultipartEncoder(
            #     fields={
            #         'file': (file_name, open(save_path, 'rb'))
            #     }
            # )
            # response = requests.post('http://localhost:8008/backend/api/notification',
            #                              data=multipart_data, headers={'Content-Type': multipart_data.content_type})
            # print(response)
            # thread.start()
            # thread.join()

            # thread = threading.Thread(target=send_telegram, args=[save_path])
            # multipart_data = MultipartEncoder(
            #     fields={"file": (file_name, open(save_path, "rb"))}
            # )
            # try:
            #     response = requests.post(
            #         "http://localhost:8008/backend/api/notification",
            #         data=multipart_data,
            #         headers={"Content-Type": multipart_data.content_type},
            #     )
            #     print(response)
            # except Exception as e:
            #     pass
            # thread.start()
        return img

    def alert_image(self, file_name):
        response = requests.post(
            f"http://{env_service.get_env_var('BASE_ADDRESS')}:8008/api/notification",
            json={"user_id": str(self.user_id), "detection_path": file_name},
        )
        if response.status_code == 200:
            print("Request successful")
            print("Response JSON:", response.json())
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response text:", response.text)

    def detection_thread(self):
        if (
            self.source.endswith(".png")
            or self.source.endswith(".jpg")
            or self.source.endswith(".jpeg")
        ):
            img = cv2.imread(self.source)
            img, checked, file_name = self.detect_single_image(img)

            response = requests.post(
                f"http://{env_service.get_env_var('BASE_ADDRESS')}:8008/api/history",
                json={
                    "user_id": str(self.user_id),
                    "result_path": file_name,
                    "status": checked,
                },
            )
            if response.status_code == 200 or response.status_code == 201:
                print("Request successful")
                print("Response JSON:", response.json())
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response text:", response.text)
        else:
            cap = cv2.VideoCapture(self.source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            now = datetime.datetime.now()
            file_name = str(int(datetime.datetime.timestamp(now))) + "_detection.mp4"
            save_path = os.path.join(
                PARENT_PATH, Constants.DETECTION_FOLDER + file_name
            )

            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    else:
                        img, attr = self.detect_image(frame)
                        vid_writer.write(img)

            except Exception as e:
                print(e)
                pass
            finally:
                cap.release()
                cv2.destroyAllWindows()

    def detect_image(self, img):
        img0 = img

        origin = img.shape
        origin_shape = [origin[1], origin[0]]
        _interpolation = interpolate_coords(
            self.coords, origin_shape, self.resized_shape
        )
        interpolation = np.array(_interpolation, dtype=np.int32)
        img = letterbox(img, self.imgsz, self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if self.device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        pred = self.model(img, augment=self.augment)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        for i, det in enumerate(pred):
            im0 = img0
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # for c in det[:, -1].unique():
                #     if int(c) > 0:
                #         self.count_frame += 1

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    x = xywh[0]
                    y = xywh[1]
                    x = int(x)
                    y = int(y)

                    # _x, _y = interpolate_point([x, y], origin_shape, self.resized_shape)
                    cv2.circle(im0, (x, y), 10, (255, 0, 0), thickness=2)

                    if check_inside(ground_truth=_interpolation, pred=[x, y]):
                        self.count_frame += 1

                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=3,
                        )

                    cv2.polylines(
                        im0,
                        [interpolation],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

        if self.count_frame == 30:
            self.check = True
            now = datetime.datetime.now()
            iso_date = now.isoformat()
            file_name = str(int(datetime.datetime.timestamp(now))) + ".jpg"
            alert_image = os.path.join(
                PARENT_PATH, Constants.DETECTION_FOLDER + file_name
            )
            cv2.imwrite(alert_image, im0)
            # self.extracted_image.append({"img": file_name, "time": iso_date})
            self.count_frame = 0
        return im0, self.extracted_image

    def detect_single_image(self, img):
        img0 = img

        origin = img.shape
        origin_shape = [origin[1], origin[0]]
        _interpolation = interpolate_coords(
            self.coords, origin_shape, self.resized_shape
        )
        interpolation = np.array(_interpolation, dtype=np.int32)
        img = letterbox(img, self.imgsz, self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if self.device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        pred = self.model(img, augment=self.augment)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        check = False

        for i, det in enumerate(pred):
            im0 = img0
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    if int(c) > 0:
                        self.count_frame += 1

                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        x = xywh[0]
                        y = xywh[1]
                        x = int(x)
                        y = int(y)

                        cv2.circle(im0, (x, y), 5, (255, 0, 0), thickness=1)
                        if check_inside(ground_truth=_interpolation, pred=[x, y]):
                            check = True
                            label = f"{names[int(cls)]} {conf:.2f}"
                            plot_one_box(
                                xyxy,
                                im0,
                                label=label,
                                color=(255, 0, 0),
                                line_thickness=1,
                            )
                        cv2.polylines(
                            im0,
                            [interpolation],
                            isClosed=True,
                            color=(0, 255, 0),
                            thickness=2,
                        )

        now = datetime.datetime.now()
        file_name = str(int(datetime.datetime.timestamp(now))) + ".jpg"
        _alert_image = os.path.join(PARENT_PATH, Constants.DETECTION_FOLDER + file_name)
        cv2.imwrite(_alert_image, im0)
        if check == True:
            self.alert_image(file_name)
        return im0, check, file_name
