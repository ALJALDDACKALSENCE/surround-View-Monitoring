import cv2, numpy as np, matplotlib.pyplot as plt

mtx = np.array([[351.169058, 0.000000, 330.784880],
                [0.000000, 351.657223, 239.471923],
                [0.000000, 0.000000, 1.000000]])
dist = np.array([-0.325330, 0.091562, 0.000301, 0.000526, 0.000000])


#### Left
left = cv2.imread("Left.png", cv2.IMREAD_COLOR)
und_left = cv2.undistort(left, mtx, dist, None)
rot_left = cv2.rotate(und_left, cv2.ROTATE_90_COUNTERCLOCKWISE)

L_pt1 = np.array([[355, 630], [275, 420], [275, 305], [355, 305]], dtype=np.float32)
L_pt2 = np.array([[180, 360], [0, 360], [0, 270], [180, 270]], dtype=np.float32)

H_L = cv2.getPerspectiveTransform(L_pt1, L_pt2)
dst_L = cv2.warpPerspective(rot_left, H_L, (540, 540), None, cv2.INTER_CUBIC)
dst_L[:,270:] = 0

#### Right
right = cv2.imread("Right.png", cv2.IMREAD_COLOR)
und_right = cv2.undistort(right, mtx, dist, None)
rot_right = cv2.rotate(und_right, cv2.ROTATE_90_CLOCKWISE)

R_pt1 = np.array([[122, -2], [200, 219], [202, 337], [123, 337]], dtype=np.float32)
R_pt2 = np.array([[360, 180], [540, 180], [540, 270], [360, 270]], dtype=np.float32)

H_R = cv2.getPerspectiveTransform(R_pt1, R_pt2)
dst_R = cv2.warpPerspective(rot_right, H_R, (540, 540), None, cv2.INTER_CUBIC)
dst_R[:,:270] = 0

#### Front
front = cv2.imread("front.png", cv2.IMREAD_COLOR)
und_front = cv2.undistort(front, mtx, dist, None)

# plt.imshow(und_front); plt.show()
F_pt1 = np.array([[163, 295], [221, 274], [455, 272], [511, 292]], dtype=np.float32)
F_pt2 = np.array([[180, 90], [180, 0], [360, 0], [360, 90]], dtype=np.float32)

H_F = cv2.getPerspectiveTransform(F_pt1, F_pt2)
dst_F = cv2.warpPerspective(und_front, H_F, (540, 540), None, cv2.INTER_CUBIC)
dst_F[270:,:] = 0

#### back
back = cv2.imread("Back.png", cv2.IMREAD_COLOR)
und_back = cv2.undistort(back, mtx, dist, None)
rot_back = cv2.rotate(und_back, cv2.ROTATE_180)

# plt.imshow(und_back); plt.show()
# plt.imshow(rot_back); plt.show()
B_pt1 = np.array([[470, 188], [389, 215], [208, 215], [122, 189]], dtype=np.float32)
B_pt2 = np.array([[360, 450], [360, 540], [180, 540], [180, 450]], dtype=np.float32)

H_B = cv2.getPerspectiveTransform(B_pt1, B_pt2)
dst_B = cv2.warpPerspective(rot_back, H_B, (540, 540), None, cv2.INTER_CUBIC)
dst_B[:360,:] = 0



dst_all = dst_F + dst_R + dst_L + dst_B
while cv2.waitKey(0) != 27:
    cv2.imshow("dst", dst_all)
    
# cv2.imshow("dst", dst_R); cv2.waitKey(0)
# plt.imshow(rot); plt.show()

print(H_L)