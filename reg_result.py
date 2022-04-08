import cv2
import numpy as np
import argparse

from reg_exp import show_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH_FIXED_O", "-fo", default="DATA/SW13L63_0.png")
    parser.add_argument("--PATH_FIXED", "-f", default="DATA/Fiji/SW13L63/SW13L63_0_e.png")
    parser.add_argument("--PATH_MOVING", "-m", default="DATA/SW13L63_1.png")
    parser.add_argument("--PATH_RESULT", "-r", default="DATA/Fiji/SW13L63/SW13L63_1_e.png")
    parser.add_argument("--NAME", "-n", default="elastic")
    parser.add_argument("--OUT", "-o", default="rst/reg/SW13L63")
    args = parser.parse_args()

    # main
    img_fixed_origin = cv2.imread(args.PATH_FIXED_O, cv2.IMREAD_GRAYSCALE)
    img_fixed = cv2.imread(args.PATH_FIXED, cv2.IMREAD_GRAYSCALE)
    img_moving = cv2.imread(args.PATH_MOVING, cv2.IMREAD_GRAYSCALE)
    img_result = cv2.imread(args.PATH_RESULT, cv2.IMREAD_GRAYSCALE)
    # remove black border
    # estimate affine
    sift = cv2.xfeatures2d_SIFT.create(1000)
    matcher = cv2.BFMatcher()
    kp_fixed_o, des_fixed_o = sift.detectAndCompute(img_fixed_origin, None)
    kp_fixed, des_fixed = sift.detectAndCompute(img_fixed, None)
    matches = matcher.match(des_fixed_o, des_fixed)
    src_pts = np.float32([kp_fixed_o[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_fixed[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)
    dx, dy = M[0, 2], M[1, 2]
    dx,dy = round(dx), round(dy)
    img_fixed = img_fixed[dy:dy + img_moving.shape[0], dx:dx + img_moving.shape[1]]
    img_result = img_result[dy:dy + img_moving.shape[0], dx:dx + img_moving.shape[1]]
    show_result(img_fixed, img_moving, img_result, args.NAME, args.OUT)
