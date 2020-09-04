import cv2
import numpy as np

class YoloDetection(object):

    def __init__(self, source_file=2, names_object="obj.names", weights_file="yolov3-tiny_last_now.weights", config_file="yolov3-tiny_last_cfg.cfg"):
        self.__weights = weights_file
        self.__config = config_file
        self.__frame = source_file
        self.__names = names_object

        if self.__frame == 0:  # for webcam
            self.__cap = cv2.VideoCapture(0)
        elif self.__frame == 1:  # for rtsp stream
            self.__cap = cv2.VideoCapture("rtsp://cam:jhg23dfc@178.150.141.135:1555/Streaming/Channels/101")
        elif self.__frame == 2:  # for img
            self.__cap = r"C:\Users\r.pedan\ComputerVison\car.jpg"
        else:
            raise Exception  # nado dopisat'

    def yoloDetection(self):
        # load YOLO pretrained weights and config file
        net = cv2.dnn.readNet(self.__weights, self.__config)
        classes = []

        with open(self.__names, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        if self.__frame == 2:
            frame = cv2.imread(self.__cap)
        else:
            flag, frame = self.__cap.read()

        height, width, channels = frame.shape
    # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[0]])
                color = colors[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 1, (255, 255, 255), 2)
                print(label)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    # def output(self, frame, outs, classes):
    #     colors = np.random.uniform(0, 255, size=(len(classes), 3))
    #     height, width, channels = frame.shape
    #     # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    #     class_ids = []
    #     confidences = []
    #     boxes = []
    #     for out in outs:
    #         for detection in out:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
    #             if confidence > 0.2:
    #                 # object detected
    #                 center_x = int(detection[0] * width)
    #                 center_y = int(detection[1] * height)
    #                 w = int(detection[2] * width)
    #                 h = int(detection[3] * height)
    #
    #                 # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
    #                 # rectangle co-ordinaters
    #                 x = int(center_x - w / 2)
    #                 y = int(center_y - h / 2)
    #                 # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #
    #                 boxes.append([x, y, w, h])  # put all rectangle areas
    #                 confidences.append(
    #                     float(confidence))  # how confidence was that object detected and show that percentage
    #                 class_ids.append(class_id)  # name of the object tha was detected
    #
    #     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    #
    #     font = cv2.FONT_HERSHEY_PLAIN
    #     for i in range(len(boxes)):
    #         if i in indexes:
    #             x, y, w, h = boxes[i]
    #             label = str(classes[class_ids[0]])
    #             color = colors[0]
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #             cv2.putText(frame, label, (x, y + 30), font, 1, (255, 255, 255), 2)
    #             print(label)
    #
    #     cv2.imshow("Frame", frame)
    #     cv2.waitKey(1)
