import cv2, numpy as np, matplotlib.pyplot as plt
from cv2 import Stitcher
import os

### camera matrix, Homography - CAM0

mtx_cam0 = np.array([[351.169058, 0.000000, 330.784880],
                [0.000000, 351.657223, 239.471923],
                [0.000000, 0.000000, 1.000000]])

dist_cam0 = np.array([-0.325330, 0.091562, 0.000301, 0.000526, 0.000000])

H_cam0 = np.array([[-1.51455476e-01, -1.09276258e+00,  3.05878149e+02],
[-3.52454260e-02, -1.17820424e+00,  3.34907073e+02],
[-1.01942681e-04, -4.11129016e-03,  1.00000000e+00]])

#F0_pt1 = np.array([[147, 302], [511, 297], [211, 282], [450, 277]], dtype=np.float32) #[[147, 297], [511, 292], [211, 277], [450, 272]
#F0_pt2 = np.array([[180, 90], [360, 90], [180, 0], [360, 0]], dtype=np.float32)

F0_pt1 = np.array([[153, 355], [501, 345],[244, 296],[417, 290]], dtype=np.float32)
F0_pt2 = np.array([[230, 177], [315, 172], [230, 82],[315, 72]], dtype=np.float32)
H_cam0 = cv2.getPerspectiveTransform(F0_pt1, F0_pt2)

 ### camera matrix, Homography - CAM1

mtx_cam1 = np.array([[522.047976, 0.000000, 312.409819],
                [0.000000, 522.667306, 263.691174],
                [0.000000, 0.000000, 1.000000]])

dist_cam1 = np.array([0.052595, -0.175610, 0.002809, -0.001501, 0.000000])

H_cam1 = np.array([[-4.30081067e+00,  5.21520818e+00,  3.05229303e+02],
[-3.87074782e+00 , 4.64961780e+00, -2.48365117e+02],
[-1.66533940e-02,  1.67764342e-02 ,  1.00000000e+00]])

F1_pt1 = np.array([[98, 135], [295, 299], [134, 209], [363, 382]], dtype=np.float32)
F1_pt2 = np.array([[365, 0], [550, 0], [365, 90], [560, 90]], dtype=np.float32)
H_cam1 = cv2.getPerspectiveTransform(F1_pt1, F1_pt2)


 # Image Read
path_cam0 = "/home/milly/svm/multi2/cam00"
path_cam1 = "/home/milly/svm/multi2/cam11"
cam0_list = os.listdir(path_cam0)
cam1_list = os.listdir(path_cam1)
cam0_list.sort()
cam1_list.sort()


i = 0

for cam0, cam1 in zip(cam0_list, cam1_list):
    print(cam0,cam1)
    cam0_img = cv2.imread(path_cam0 + "/" + cam0, cv2.IMREAD_COLOR)
    und_cam0 = cv2.undistort(cam0_img, mtx_cam0, dist_cam0, None)
    dst_cam0 = cv2.warpPerspective(und_cam0, H_cam0, (540,540), None, cv2.INTER_CUBIC)    
    cam1_img = cv2.imread(path_cam1 + "/" + cam1, cv2.IMREAD_COLOR)
    und_cam1 = cv2.undistort(cam1_img, mtx_cam1, dist_cam1, None)
    rot_mat = cv2.getRotationMatrix2D((320,240), 315,1)
    rot_cam1 = cv2.warpAffine(und_cam1,rot_mat,(640,480))
    dst_cam1 = cv2.warpPerspective(rot_cam1, H_cam1, (540,540), None, cv2.INTER_CUBIC)
    
    cv2.imshow("und_Cam0",und_cam0)
    cv2.imshow("und_cam1",rot_cam1)
    
    dst_cam0[250:,:] = 0
    dst_cam1[250:,:] = 0
    #dst_all = dst_cam0 + dst_cam1

    cv2.imshow("dst_Cam0",dst_cam0)
    cv2.imshow("dst_cam1",dst_cam1)
    cv2.imshow("dst_sum",dst_cam0+dst_cam1)

    kernel = cv2.bitwise_and(dst_cam0,dst_cam1)
    kernel= cv2.cvtColor(kernel,cv2.COLOR_BGR2GRAY)
    _,kernel = cv2.threshold(kernel, 1, 255, cv2.THRESH_BINARY_INV)
    kernel= cv2.cvtColor(kernel,cv2.COLOR_GRAY2BGR)

	
    dst_cam0 = cv2.bitwise_and(dst_cam0,kernel)

    dst_all = dst_cam0+dst_cam1
    #cv2.imwrite("dst0" + str(i).zfill(3) + ".png",dst_all)
    i += 1
    cv2.imshow("dst_all",dst_cam0+dst_cam1)
    cv2.waitKey(0)
    
