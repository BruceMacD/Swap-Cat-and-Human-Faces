#! /usr/bin/env python
"""
Morph the triangulation of one face onto another
Maps corresponding triangles between faces
An affine transformation is a transformation that preserves "collinearity and ratios of distances between collinear points"
"""

import cv2
import numpy as np
from constants.constants import debug_affine_transformation


POLY_FILL_COLOR = (1.0, 1.0, 1.0)


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def get_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


# morph a given triangular region one another image
# we are mapping the triangle from img_1 to img_2
def morph_triangular_region(triangle_1, triangle_2, img_1, img_2):
    # Find bounding rectangle for each triangle in the form <x, y, w, h>
    x_1, y_1, w_1, h_1 = cv2.boundingRect(np.float32([triangle_1]))
    x_2, y_2, w_2, h_2 = cv2.boundingRect(np.float32([triangle_2]))

    # Offset points by left top corner of the respective rectangles
    offset_triangle_1 = []
    offset_triangle_2 = []

    # for the <x,y> coordinates of each point the triangle find the offset
    # move this into a separate function if you need to do it a for a lot of triangles
    for coords in triangle_1:
        offset_triangle_1.append(((coords[0] - x_1), (coords[1] - y_1)))
    for coords in triangle_2:
        offset_triangle_2.append(((coords[0] - x_2), (coords[1] - y_2)))

    # get the mask by filling the triangle to mask pixels outside the desired area
    mask = np.zeros((h_2, w_2, 3))
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), POLY_FILL_COLOR)

    # get only the part of the image we are going to map within the bounding rectangle
    img_1_within_bounds = img_1[y_1:y_1 + h_1, x_1:x_1 + w_1]

    size_bounds_triangle_2 = (w_2, h_2)

    # apply the affine transform on img_1 based on the triangles
    transformed_area = get_affine_transform(img_1_within_bounds, offset_triangle_1, offset_triangle_2,
                                            size_bounds_triangle_2)

    # remove all parts of the transformed image outside the area we care about (triangle mask)
    transformed_triangle = transformed_area * mask

    # slice the current area out of the in the image we are mapping the face to
    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] * (POLY_FILL_COLOR - mask)
    # slice the transformed area back in its place
    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] + transformed_triangle

    return img_2


def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    # create a copy of image 2 that we will map the face from image 1 to
    img_2_with_face_1 = np.copy(img_2)

    # morph each triangular region one at a time
    for triangle in delauney:
        triangles_1 = []
        triangles_2 = []

        # get points within img_1 and img_2 corresponding to the triangle points previously found from the face in img_1
        for point in triangle:
            triangles_1.append(hull_1[point])
            triangles_2.append(hull_2[point])

        # once we have found the points in the landmarks corresponding to the triangle morph the triangular region from
        # img_1 to img_2 and return the result that we will modify again with the next triangle
        morph_triangular_region(triangles_1, triangles_2, img_1, img_2_with_face_1)

    if debug_affine_transformation:
        cv2.imshow("Affine transformation", img_2_with_face_1)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return img_2_with_face_1
