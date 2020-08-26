import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny_last (1).weights", "yolov3-tiny_last_cfg.cfg")
classes = []

with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


colors = np.random.uniform(0, 255, size=(len(classes), 3))


# loading image
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("rtsp://cam:jhg23dfc@178.150.141.135:1555/Streaming/Channels/101")
while True:

    flag, frame = cap.read()
    #cv2.imshow("Fraame", frame)
    #frame = cv2.imread(frame)
    #frame = cv2.resize(frame, None, fx=0.4, fy=0.3)
    height, width, channels = frame.shape

#img = cv2.imread(r"C:\Users\r.pedan\ComputerVison\cats.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.3)
#height, width, channels = img.shape

# detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)



# for b in blob:
#     for n,img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

    net.setInput(blob)
    outs = net.forward(outputlayers)
# print(outs[1])


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
                # onject detected
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
    #cv2.destroyAllWindows()
