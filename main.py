import cv2
import numpy as np
from numpy.lib.function_base import append

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# img = cv2.imread('image2.jpg')
cap = cv2.VideoCapture('people.mp4')
# img = cv2.resize(img, None, fx=2.0, fy=2.0)

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    boxes = []
    confidences = []
    class_ids = []


    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if(confidence>0.1):
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes)))
    x,y = 50,50
    offset = 35
    count = 0
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        if label == "person":
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y+20), font, 2, (0, 0, 0), 2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0, 0, 0), 2)
            # print(label)
            count = count + 1
    cv2.putText(img, "Number of person: " + str(count), (20, height - 20) ,font, 2, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()