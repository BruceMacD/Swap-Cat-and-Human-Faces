#! /usr/bin/env python
"""
Run all the separate components of face swapping in an easily understandable high-level runner class
"""

import sys
import getopt
import cv2
from components.landmark_detection import detect_landmarks
from components.cat_frontal_face_detection import detect_cat_face
from components.convex_hull import find_convex_hull
from components.delaunay_triangulation import find_delauney_triangulation
from components.affine_transformation import apply_affine_transformation
from components.clone_mask import merge_mask_with_image

EXPECTED_NUM_IN = 2


def exit_error():
    print('Error: unexpected arguments')
    print('face_swap.py -i <path/to/humanFaceImage> -i <path/to/catFaceImage>')
    sys.exit()


def main(argv):
    in_imgs = []
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        exit_error()

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            in_imgs.append(arg)
        else:
            exit_error()

    # need specific number of ins
    if len(in_imgs) != EXPECTED_NUM_IN:
        exit_error()

    print('Input files', in_imgs)

    human_img = cv2.imread(in_imgs[0])
    cat_img = cv2.imread(in_imgs[1])

    landmarks_human = detect_landmarks(human_img)[0]
    landmarks_cat = detect_cat_face(cat_img)[0]

    hull_human, hull_cat = find_convex_hull(landmarks_human, landmarks_cat, human_img, cat_img)

    # divide the boundary of the face into triangular sections to morph
    delauney_human = find_delauney_triangulation(human_img, landmarks_human)
    delauney_cat = find_delauney_triangulation(cat_img, landmarks_cat)

    # warp the source triangles onto the target face
    img_1_face_to_img_2 = apply_affine_transformation(delauney_human, landmarks_human, landmarks_cat, human_img, cat_img)
    img_2_face_to_img_1 = apply_affine_transformation(delauney_cat, landmarks_cat, landmarks_human, cat_img, human_img)

    swap_1 = merge_mask_with_image(hull_cat, img_1_face_to_img_2, cat_img)
    swap_2 = merge_mask_with_image(hull_human, img_2_face_to_img_1, human_img)

    # show the results
    cv2.imshow("Face Swap 1: ", swap_1)
    cv2.imshow("Face Swap 2: ", swap_2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
