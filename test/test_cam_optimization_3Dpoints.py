#!/usr/bin/env python2

import numpy as np
import cv2
import glob
from scipy import optimize

def generate_chessboard(size, dimensions=(9, 6)):
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp = objp * size
    return objp

def objective_function(params):

    rvec_left, tvec_left = find_cam_chess_realpoints('./fotos_calibration/top_left_camera_4.jpg', 0)
    rvec_right, tvec_right = find_cam_chess_realpoints('./fotos_calibration/top_right_camera_4.jpg', 1)

    t2_t1 = np.zeros((4, 4), np.float32)
    t2_t1[0:3, 0:3], _ = cv2.Rodrigues(params[0:3])
    t2_t1[0:3, 3] = params[3:].T
    t2_t1[3, :] = [0, 0, 0, 1]

    t1_real = np.zeros((4, 4), np.float32)
    t1_real[0:3, 0:3], _ = cv2.Rodrigues(rvec_left)
    t1_real[0:3, 3] = tvec_left.T
    t1_real[3, :] = [0, 0, 0, 1]

    T2_chess_real = np.matmul(t2_t1, t1_real)
    r_cam2tochess_vector, _ = cv2.Rodrigues(T2_chess_real[0:3, 0:3])
    # t_cam2tochess = np.zeros((3, 1))
    t_cam2tochess = T2_chess_real[0:3, 3]
    imgpoints_right_optimize = cv2.projectPoints(objp, r_cam2tochess_vector, t_cam2tochess, k_right,
                                                 dist_right)

    img1 = cv2.imread('./fotos_calibration/top_right_camera_4.jpg')
    img = cv2.drawChessboardCorners(img1, (9, 6), imgpoints_right_optimize[0], True)
    cv2.imshow('img', img)
    cv2.waitKey(200)

    t2_real = np.zeros((3, 4), np.float32)
    t2_real[0:3, 0:3] = cv2.Rodrigues(rvec_right)[0]
    t2_real[0:3, 3] = tvec_right.T

    t2_chess = np.zeros((3, 4), np.float32)
    t2_chess[0:3, 0:3] = cv2.Rodrigues(r_cam2tochess_vector)[0]
    t2_chess[0:3, 3] = t_cam2tochess.T

    # projection_matrix_right = np.matmul(k_right,T2_chess)
    objp_right = np.zeros((4, dimensions[0] * dimensions[1]), np.float32)
    objp_right[0:3, :] = objp.T
    objp_right[3, :] = 1

    residuals = np.zeros((dimensions[0] * dimensions[1]))

    for i in range(dimensions[0] * dimensions[1]):
        points_op = np.matmul(np.array(t2_chess), objp_right[:,i].reshape((4,1)))
        points_real_right = np.matmul(t2_real, objp_right[:,i].reshape((4,1)))
        residuals[i] = np.sqrt((points_op[0] - points_real_right[0]) ** 2 + (points_op[1] - points_real_right[1]) ** 2 + (points_op[2] - points_real_right[2]) ** 2)

    return residuals

def find_cam_chess_realpoints(fname,left_or_right):

    objpoints_left = []
    imgpoints_left = []
    objpoints_right = []
    imgpoints_right = []
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        if left_or_right == 0:
            objpoints_left.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_left[0], imgpoints_left[0], k_left,
                                                        dist_left)  # calculating the rotation and translation vectors from left camera to chess board
        elif left_or_right == 1:
            objpoints_right.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_right[0], imgpoints_right[0], k_right,
                                              dist_right)  # calculating the rotation and translation vectors from left camera to chess board

    return rvec, tvec

def inv_tranformation_matrix(r, t):
    x_array = np.array(r[0])
    r_rot = []
    r_rot[0] = r[0].T * -1
    t_rot = np.matmul(r_rot, t[0][0:3])
    r[0]=r[0].T
    t[0][0:3] = t_rot
    return r, t

# Chessboard dimensions
dimensions = (6, 9)
size_board = 100

# K matrix and distortion coefficients from cameras
k_left = np.array([[1149.369, 0.0, 471.693],[0.0, 1153.728, 396.955],[0.0, 0.0, 1.0]])
dist_left = np.array([ -1.65512977e-01, -2.08184195e-01, -2.17490237e-03, -5.04628479e-04, 1.18772434e+00])

k_right = np.array([[1135.560, 0.0, 490.807],[0.0, 1136.240, 412.468],[0.0, 0.0, 1.0]])
dist_right = np.array([ -2.06069540e-01, -1.27768958e-01, 2.22591520e-03, 1.60327811e-03, 2.08236968e+00])

# Creating chessboard points
objp = generate_chessboard(100, (9, 6))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():


    # Arrays to store object points and image points from all the images.
    objpoints_left = []  # 3d point in real world space from left camera
    imgpoints_left = []  # 2d points in image plane from left camera.

    objpoints_right = []  # 3d point in real world space from right camera
    imgpoints_right = []  # 2d points in image plane from right camera.

    image_left = glob.glob('./fotos_calibration/top_left_camera_4.jpg')
    image_right = glob.glob('./fotos_calibration/top_right_camera_4.jpg')
    images = [image_left, image_right]

    for index in range(2):
        for fname in images[index]:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

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
                cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # params_initial =np.concatenate(( t_cam2tocam1.T,r_vector_cam2tocam1.T),axis=None)
    params_initial = [1.0, 0.0, 0.0, 105.0, -200.0, 150.0]

    # print(objective_function(params_initial))
    params_optimized = optimize.leastsq(objective_function, np.array(params_initial).reshape(1, 6), epsfcn=0.000001)
    print('Vetor de erros:\n')
    print(objective_function(params_optimized[0]))
    print()
    print('Erro medio:\n')
    print(np.mean(objective_function(params_optimized[0])))
    print('\nTRANSFORMACAO CAMERA 2 PARA CAMERA 1:')
    print('\nVetor de rotacoes:\n')
    print(params_optimized[0][0:3])
    print('\nVetor de translacoes:\n')
    print(params_optimized[0][3:])


if __name__ == "__main__":
    main()