from detections.people_detect import detectPeople
import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--input', '-i', type=str, default="", help="Path of a input file" )

argparser.add_argument('--output', '-o', type=str, default="", help="Path of a output file" )

argparser.add_argument('--display', '-d', type=int, default=0, help="output of the frame will diplayed" )

args = vars(argparser.parse_args())


labels_path = 'F:\\project\\social distance detector\\yolo\\coco.names'

labels = open(labels_path).read().strip().split("\n")

# load YOLO
net = cv2.dnn.readNet("F:\\project\\social distance detector\\yolo\\yolov3.cfg", "F:\\project\\social distance detector\\yolo\\yolov3.weights")

# getting the layers form network
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# loading video
cap = cv2.VideoCapture(args["input"] if args["input"] else 0)
# cap = cv2.VideoCapture("pedestrians.py")
out = None

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    results = detectPeople(frame, net, ln,)
    unsafe_persons = set()
    if len(results) >= 2:
        # extract the centroids and find the euclidian distance between 2
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j] < 60:
                    unsafe_persons.add(i)
                    unsafe_persons.add(j)
    for (i, (prob, box, centroid)) in enumerate(results):
        (sX, sY, eX, eY) = box
        (cX, cY) = centroid
        color = (0,255,0)
        if i in unsafe_persons:
            color = (0,0,255)
        # draw rectangle
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv2.circle(frame, (cX, cY), 2, (0,0,255), 2)
        text = "Social Distancing Violations: {}".format(len(unsafe_persons))
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)

    if args["display"] > 0:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if args["output"] != "" and out is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    if out is not None:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()
