#! /usr/bin/env python
"""
Using the provided functions in dlib to detect the points of facial landmarks in an image
"""
import numpy as np
import cv2
import dlib
from constants.constants import debug_landmark_detection

# landmarks (from viewer perspective):
# index - (x,y)
CHIN_INDEX = 8
LEFT_EYE_INDEX = 37
LEFT_EAR_LEFT_INDEX = 0
RIGHT_EAR_LEFT_INDEX = 25
NOSE_INDEX = 30
RIGHT_EYE_INDEX = 44
LEFT_EAR_RIGHT_INDEX = 18
RIGHT_EAR_RIGHT_INDEX = 16
LEFT_CHEEK_INDEX = 4
RIGHT_CHEEK_INDEX = 12
# the common land mark points between the cat and human
COMMON_LANDMARK_INDEXES = [CHIN_INDEX, LEFT_EYE_INDEX, LEFT_EAR_LEFT_INDEX, RIGHT_EAR_LEFT_INDEX, NOSE_INDEX,
                           RIGHT_EYE_INDEX, LEFT_EAR_RIGHT_INDEX, RIGHT_EAR_RIGHT_INDEX, LEFT_CHEEK_INDEX,
                           RIGHT_CHEEK_INDEX]

# Pre-trained shape predictor from iBUG 300-W dataset
SHAPE_PREDICTOR = 'data/shape_predictor_68_face_landmarks.dat'

frontal_face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


# convenience function from imutils
def dlib_to_cv_bounding_box(box):
    # convert dlib bounding box for OpenCV display
    x = box.left()
    y = box.top()
    w = box.right() - x
    h = box.bottom() - y

    return x, y, w, h


# another conversion from imutils
def landmarks_to_numpy(landmarks):
    # initialize the matrix of (x, y)-coordinates with a row for each landmark
    coords = np.zeros((len(landmarks), 2), dtype=int)

    # convert each landmark to (x, y)
    for i in range(0, len(landmarks)):
        coords[i] = (landmarks[i][0], landmarks[i][1])

    # return the array of (x, y)-coordinates
    return coords


# only get the points we want from the face that correspond to points from the cat
def convert_to_cat_landmarks(facial_landmarks):
    human_landmarks_to_cat = []

    # this will copy landmarks with the same index order as the cat
    for landmark_index in COMMON_LANDMARK_INDEXES:
        human_landmarks_to_cat.append(facial_landmarks[landmark_index])

    return human_landmarks_to_cat


def show_face_annotated(faces, landmarks, img):

    for face in faces:
        # draw box for face
        x, y, w, h = dlib_to_cv_bounding_box(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw circles for landmarks
        for landmark_set in landmarks:
            for x, y in landmark_set:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_landmarks(img):
    # this list will contain the facial landmark points for each face detected
    points = []
    # second argument of 1 indicates the image will be upscaled once, upscaling creates a bigger image so it is easier
    # to detect the faces, can increase this number if there are troubles detecting faces
    # returns a bounding box around each face
    detected_faces = frontal_face_detector(img, 1)

    # now that we have the boxes containing the faces find the landmarks inside them
    for face in detected_faces:
        # use dlib to find the expected facial landmarks in the boxes around the detected faces
        landmarks = landmarks_predictor(img, face)

        landmark_list = []
        for i in range(0, landmarks.num_parts):
            landmark_list.append((landmarks.part(i).x, landmarks.part(i).y))

        # add the facial landmarks in a form we can use later without dlib
        points.append(landmarks_to_numpy(convert_to_cat_landmarks(landmark_list)))

    # show the bounding box
    if debug_landmark_detection:
        show_face_annotated(detected_faces, points, img)

    return points
