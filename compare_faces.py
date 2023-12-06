# import the necessary packages
import sys
import cv2
import numpy as np
import face_recognition

# read the face file names from the command line arguments
face1_file = sys.argv[1]
face2_file = sys.argv[2]

# load the YOLO network and the class labels
net = cv2.dnn.readNet("yolov3-face.weights", "yolov3-face.cfg")
classes = ["face"]

# get the output layer names of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize a color for the face class
color = (0, 255, 0)

# define a function to detect and compare faces in an image file
def detect_and_compare_faces(file):
    # read the image and get its dimensions
    img = cv2.imread(file)
    height, width, channels = img.shape

    # create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # initialize the lists of confidences and bounding boxes
    confidences = []
    boxes = []

    # loop over each of the output layer detections
    for out in outs:
    # loop over each of the detections
        for detection in out:
            # get the confidence score
            confidence = detection[4]
            # filter out weak detections by confidence
            if confidence > 0.5:
                # get the center coordinates and the width and height of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # get the top-left coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # append the confidence and bounding box to the lists
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-maxima suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # check if there is only one face detected
    if len(indexes) == 1:
        i = indexes[0]
        # get the bounding box coordinates and the confidence score
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        # draw the bounding box and the confidence score on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, "face " + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        # crop the face region from the image
        face = img[y:y+h, x:x+w]
        # encode the face features using face_recognition library
        face_encoding = face_recognition.face_encodings(face)[0]
        # return the image and the face encoding
        return img, face_encoding
    else:
        # return None if there is no face or more than one face detected
        return None, None

# detect and compare faces in the first image file
img1, face1_encoding = detect_and_compare_faces(face1_file)
# detect and compare faces in the second image file
img2, face2_encoding = detect_and_compare_faces(face2_file)

# check if both images have one face detected
if img1 is not None and img2 is not None:
    # compare the face encodings using face_recognition library
    similarity = face_recognition.compare_faces([face1_encoding], face2_encoding)[0]
    # write the similarity score on the images
    cv2.putText(img1, "Similarity: " + str(similarity), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img2, "Similarity: " + str(similarity), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # show the images with the boxes and similarity
    cv2.imshow("Face 1", img1)
    cv2.imshow("Face 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # print an error message if one or both images do not have one face detected
    print("Error: One or both images do not have one face detected.")