# import the necessary packages
import sys
import cv2
import numpy as np

# read the input file name, output file name and confidence threshold from the command line arguments
input_file = sys.argv[1]
output_file = sys.argv[2]
confidence_threshold = float(sys.argv[3])

# load the YOLO network and the class labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# get the output layer names of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# read the input image and get its dimensions
img = cv2.imread(input_file)
height, width, channels = img.shape

# create a blob from the image and pass it through the network
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# initialize the lists of class IDs, confidences and bounding boxes
class_ids = []
confidences = []
boxes = []

# loop over each of the output layer detections
for out in outs:
# loop over each of the detections
    for detection in out:
        # get the scores and the class ID
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # filter out weak detections by confidence
        if confidence > confidence_threshold:
            # get the center coordinates and the width and height of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # get the top-left coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # append the class ID, confidence and bounding box to the lists
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# apply non-maxima suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

# initialize a list of colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# loop over the indexes of the remaining boxes
for i in indexes:
#    i = i[0]
    # get the bounding box coordinates and the class label
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    # draw the bounding box and the label on the image
    color = colors[class_ids[i]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + str(round(confidences[i], 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

# write the output image to the output file
cv2.imwrite(output_file, img)

# show the input and output images
cv2.imshow("Input", cv2.imread(input_file))
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()