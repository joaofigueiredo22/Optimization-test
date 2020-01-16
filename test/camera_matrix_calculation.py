#!/usr/bin/env python2

import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
objp = objp * 100

# Arrays to store object points and image points from all the images.
objpoints_left = [] # 3d point in real world space from left camera
imgpoints_left = [] # 2d points in image plane from left camera.

objpoints_right = [] # 3d point in real world space from right camera
imgpoints_right = [] # 2d points in image plane from right camera.

images_left = glob.glob('top_left_camera_*.jpg')
images_right = glob.glob('top_right_camera_*.jpg')
images = [images_left, images_right]

def main():
    for index in range(2):
        for fname in images[index]:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                if index == 0:
                    objpoints_left.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints_left.append(corners2)

                elif index == 1:
                    objpoints_right.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints_right.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(20)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left, imgpoints_left,
                                                                                gray.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right, imgpoints_right,
                                                                                     gray.shape[::-1], None, None)
    print('\nK LEFT CAMERA:\n')
    print(mtx_left)
    print('\nDISTORTION LEFT CAMERA:\n')
    print(dist_left)
    print('\nK RIGHT CAMERA:\n')
    print(mtx_right)
    print('\nDISTORTION RIGHT CAMERA:\n')
    print(dist_right)


if __name__ == "__main__":
    main()