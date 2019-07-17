# importing library
import cv2
import numpy as np
import argparse
from datetime import datetime
from imagezmq import imagezmq
import imutils
import csv

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="./MobileNetSSD_deploy.prototxt",
                help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", default="./MobileNetSSD_deploy.caffemodel",
                help="path to Caffe pretrained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize ImageHub
imageHub = imagezmq.ImageHub()

# initialize variabel
drawing = False
start = []
end = []
parkList = [] # parklist is list from parkSlot
carList = {} # carList is list contains car
modeDraw = "parkir"
ksg = ""

# named window for monitoring
cv2.namedWindow("Monitor Parkiran")

# object class parkSlot
class parkSlot():
    def __init__(self, no, x, y, xend, yend, size):
        self.no = no
        self.x = x
        self.y = y
        self.xend = xend
        self.yend = yend
        self.size = size
        self.kosong = True
        self.carID = None
        self.xandy = (0, 0)
        self.dateNow = None
        self.dateFill = None
        self.dateOut = None

    # assign a parkslot based on list of all car
    def assignSpot(self, carList):
        for i, car in carList.items():
            if car.xend in range(self.x, self.xend) and car.y in range(self.y, self.yend):
                self.kosong = False
                self.xandy = (car.xend, car.y)
                self.carID = car.no

    # scan spot if there is a car or not
    def scanSpot(self, carList):
        for i, car in carList.items():
            if self.xandy[0] != car.xend and self.xandy[1] != car.y:
                self.kosong = True
                self.carID = None

    def assignDate(self):
        self.dateNow = datetime.now().strftime('%H:%M:%S')

    def dateKosongTidak(self):
        if self.kosong == True:
            self.dateFill = None
            self.dateOut = datetime.now().strftime('%H:%M:%S')
        elif self.kosong == False:
            self.dateFill = datetime.now().strftime('%H:%M:%S')
            self.dateOut = None

# class object for car
class Car():
    def __init__(self, no, x, y, xend, yend):
        self.no = no
        self.x = x
        self.y = y
        self.xend = xend
        self.yend = yend
    
# initialize park object class by drawing with mouse
def initializePark(event, x, y, flags, param):
    global drawing, start, end, parkList

    # draw from position x, y from mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = [(x, y)]

    # draw is ended if mouse left button is up
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = [(x, y)]

        size_x = end[0][0] - start[0][0]
        size_y = end[0][1] - start[0][1]
        size = (size_x, size_y)

        no = len(parkList)+1

        parkList.append(parkSlot(no, start[0][0], start[0][1], end[0][0], end[0][1], size))
        parkList[-1].assignDate()


# calling method for drawing for window
def drawMode(event, x, y, flags, param):
    if modeDraw == "parkir":
        initializePark(event, x, y, flags, param)

# initialize list of class from MobileNet SSD was trained to
# detect then generate bounding box for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# print out info and model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# intitialize the consider for car and objcount dictionary
CONSIDER = set(["car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

# initialize dictionary which wil contain information regard the device active or not
lastActive = {}
lastActiveCheck = datetime.now()

ESTIMATED_NUM_PIS = 1
ACTIVATE_CHECK_PERIOD = 5
ACTIVATE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVATE_CHECK_PERIOD

# printout the detected obj in command line
print("[INFO] detecting: {}...".format(", ".join(obj for obj in CONSIDER)))

while True:
    # receive RPi name and frame from the RPi and ack
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    dtnow = datetime.now().strftime('%H:%M:%S %a-%d-%b')

    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))

    lastActive[rpiName] = datetime.now()

    # create frame from the client
    frame = imutils.resize(frame, width=480)
    (h, w) = frame.shape[:2]

    # blob for cnn
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # inserting blob to cnn
    net.setInput(blob)
    detections = net.forward()

    objCount = {obj: 0 for obj in CONSIDER}

    # detection
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
			# detections
            idx = int(detections[0, 0, i, 1])

            # check to see if the predicted class is in the set of
			# classes that need to be considered
            if CLASSES[idx] in CONSIDER:
                # increment the count of the particular object
				# detected in the frame
                
                objCount[CLASSES[idx]] += 1

                # compute the (x, y)-coordinates of the bounding box
				# for the object             
                # and draw the rectangle
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, xend, yend) = box.astype("int")
                cv2.rectangle(frame, (x, y), (xend, yend), (0, 0, 255), 2)

                # create object car
                nc = objCount[CLASSES[idx]]
                carList[nc] = Car(nc, x, y, xend, yend)
                # popout the unused car object
                if len(carList) > nc:
                    carList.popitem()

    # export data car to csv
    with open('car.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # looping for each car
        for ic, car in carList.items():

            xy = (car.x, car.y)
            xy_end = (car.xend, car.yend)

            # write to csv
            filewriter.writerow([dtnow, car.no, xy, xy_end])

    # draw window and export data parking to csv
    with open('parking.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # looping for each park slot
        for i in range(len(parkList)):
            parkList[i].scanSpot(carList)
            parkList[i].assignSpot(carList)
            parkList[i].dateKosongTidak()

            xy = (parkList[i].x, parkList[i].y)
            xy_end = (parkList[i].xend, parkList[i].yend)

            if parkList[i].kosong == True:
                ksg = "Iya"
            else:
                ksg = "Tidak"
            
            # write to csv
            filewriter.writerow([dtnow, i+1, xy, xy_end, ksg, parkList[i].carID, parkList[i].dateNow, parkList[i].dateFill, parkList[i].dateOut])

        # draw rectangle (bounding box) for park slot
            cv2.rectangle(frame, (parkList[i].x, parkList[i].y), (parkList[i].xend, parkList[i].yend), (0, 255, 0), 2)
        # put text
            cv2.putText(frame, "Parking Slot: {0}".format(parkList[i].no), (parkList[i].x, parkList[i].y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # name of webcam
    cv2.putText(frame, rpiName, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # date now
    cv2.putText(frame, dtnow, (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # obj counter
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
    cv2.putText(frame, label, (10, h - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

    frameDict[rpiName] = frame

    key = cv2.waitKey(50) & 0xFF

    # callback function for drawing parking slot
    cv2.setMouseCallback("Monitor Parkiran", drawMode)
    cv2.imshow("Monitor Parkiran", frame)
    if key == ord("q"):
        break

cv2.destroyAllWindows()