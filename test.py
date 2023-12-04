import cv2

video = cv2.VideoCapture('public/files/3069441535869426896.mp4')

while True:
  success, frame = video.read()
  if not success:
    break
  cv2.imshow("", frame)
  cv2.imwrite("test.jpg", frame)
  # cv2.waitKey(1)
  break