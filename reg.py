import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import argparse
from reg_exp import ea_generate_field
from EA.warp import warp_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reg")
    parser.add_argument("--img_fixed", "-f", default="DATA/DOLW7_0.png")
    parser.add_argument("--img_moving", "-m", default="DATA/DOLW7_1.png")
    parser.add_argument("--img_mask", "-ma", default="DATA/DOLW7_m.png")
    parser.add_argument("--img_out", "-o")
    args = parser.parse_args()

    img_fixed = cv2.imread(args.img_fixed, cv2.IMREAD_GRAYSCALE)
    img_moving = cv2.imread(args.img_moving, cv2.IMREAD_GRAYSCALE)
    img_mask = cv2.imread(args.img_mask, cv2.IMREAD_GRAYSCALE)

    field = ea_generate_field(img_fixed, img_moving, img_moving, img_mask)
    img_result = warp_c(img_moving, field)
    cv2.imshow("img_result", img_result)
    cv2.imwrite(args.img_out, img_result)