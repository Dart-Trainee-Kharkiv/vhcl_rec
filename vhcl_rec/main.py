from classes import *

yolov3_tiny = YoloDetection(2)

video = cv2.VideoCapture(yolov3_tiny.get_source)

while True:
    flag, frame = video.read()
    yolov3_tiny.yolo_detection(frame)
   # yolov3_tiny.input()