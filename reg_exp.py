# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 22:36
# @Author  : XinTong
# @FileName: reg_exp.py
# @Software: PyCharm
import cv2
import torch
import numpy as np
import argparse
from EA.WrinkleReg import WrinkleReg
from EA.superpoint_wrinkle_reg import SuperPointReg as SuperPoint
from utils.Experiment import nmi, ncc, red_and_green, overlay_graph
from utils.easy_debug import *
from EA.grid import m2theta, sample
from EA.ransac import ransac
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from time import perf_counter
from utils.Flow import Flow


"""
the result include ssim, ncc, nmi and the red_green graph
"""


def show_result(reference, moving, finished, name, out_path):
    reference, moving = same_size(reference, moving)
    ssim_before = ssim(reference, moving)
    ssim_result = ssim(reference, finished)
    # print("ssim_before:%0.5f, ssim_result:%0.5f" % (ssim_before, ssim_result))
    ncc_before = ncc(reference, moving)
    ncc_result = ncc(reference, finished)
    # print("ncc_before:%0.5f, ncc_result:%0.5f" % (ncc_before, ncc_result))
    nmi_before = nmi(reference, moving)
    nmi_result = nmi(reference, finished)
    # print("nmi_before:%0.5f, nmi_result:%0.5f" % (nmi_before, nmi_result))
    psnr_before = psnr(reference, moving)
    psnr_result = psnr(reference, finished)
    # print("psnr_before:%0.5f, psnr_result:%0.5f" % (psnr_before, psnr_result))
    mse_before = mse(reference, moving)
    mse_result = mse(reference, finished)
    # print("mse_before:%0.5f, mse_result:%0.5f" % (mse_before, mse_result))
    print("{} before:\tssim\t&ncc\t&nmi\t&psnr\t&mse \\\\".format(name))
    print("\t\t{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}".format(ssim_before, ncc_before, nmi_before, psnr_before,
                                                                    mse_before))
    print("{} result:\tssim\t&ncc\t&nmi\t&psnr\t&mse \\\\".format(name))
    print("\t\t{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}".format(ssim_result, ncc_result, nmi_result, psnr_result,
                                                                    mse_result))
    rd_before = red_and_green(reference, moving)
    rd_result = red_and_green(reference, finished)
    ol_before = overlay_graph(reference, moving)
    ol_result = overlay_graph(reference, finished)
    cv2.imwrite(out_path + "/rd_before_{}.png".format(name), rd_before)
    cv2.imwrite(out_path + "/rd_result_{}.png".format(name), rd_result)
    cv2.imwrite(out_path + "/ol_before_{}.png".format(name), ol_before)
    cv2.imwrite(out_path + "/ol_result_{}.png".format(name), ol_result)
    return rd_before, rd_result


def draw_field(field):
    field = torch.tensor(field)
    field[0] = 2 * field[0] / field.shape[-2] - 1
    field[1] = 2 * field[1] / field.shape[-1] - 1
    field = field.unsqueeze(0)
    flow = Flow(field)
    flow_color = flow.draw_flow()
    return flow_color[0]


def ea_generate_field(img_fixed_kp, img_moving_kp, img_moving, img_tr=None):
    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(0.015, batch_sz=8, device="cpu")
    det.detectAndCompute(np.random.random([128, 128]))

    kp_fixed, des_fixed = det.detectAndComputeMean(img_fixed_kp, scale=1, mask=None)
    kp_moving, des_moving = det.detectAndComputeMean(img_moving_kp, scale=1, mask=None)
    if img_tr is None:
        img_tr = np.zeros_like(img_moving)
    reg_er = WrinkleReg(k=20, K=3, alpha=1, img_tr=img_tr, match_radius=10000, device="cpu", test=True)
    c_lst = reg_er.plant(kp_moving, kp_fixed, des_moving, des_fixed)
    t1 = perf_counter()
    field = reg_er.generate_field(c_lst, img_moving)
    t2 = perf_counter()
    print("ea time:{:.3f}s".format(t2 - t1))
    return field


def affine_generate_field(img_fixed_kp, img_moving_kp, img_moving):
    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(0.015, batch_sz=8, device="cpu")
    det.detectAndCompute(np.random.random([128, 128]))

    kp_fixed, des_fixed = det.detectAndComputeMean(img_fixed_kp, scale=1, mask=None)
    kp_moving, des_moving = det.detectAndComputeMean(img_moving_kp, scale=1, mask=None)

    img_tr = np.zeros_like(img_moving)
    reg_er = WrinkleReg(k=20, K=3, alpha=1, img_tr=img_tr,
                        match_radius=1000, device="cpu")

    kp_m = torch.tensor([kp.pt for kp in kp_moving]).to("cpu")
    kp_f = torch.tensor([kp.pt for kp in kp_fixed]).to("cpu")
    des_m = torch.tensor(des_moving).to("cpu")
    des_f = torch.tensor(des_fixed).to("cpu")
    match1, match2, ratio = reg_er.match_in_rad(kp_m, kp_f, des_m, des_f,
                                                True)  # 在一定的搜索框下，计算出最近邻匹配和ratio值
    kp_m = kp_m[match1]
    kp_f = kp_f[match2]
    height, width = img_moving.shape[0], img_moving.shape[1]
    m, inliers = ransac(kp_m, kp_f, device="cpu", rrt=3)
    # 生成形变场
    mat = m2theta(m, width, height)
    # 生成形变场
    t1 = perf_counter()
    affine_grid = F.affine_grid(mat.unsqueeze(0), (1, 1, height, width), align_corners=False)
    field = affine_grid[0, :, :, [1, 0]].permute([2, 0, 1]).cpu().numpy().astype(np.float32)
    field[0] = (field[0] + 1) * field.shape[-2] / 2
    field[1] = (field[1] + 1) * field.shape[-1] / 2
    t2 = perf_counter()
    print("affine time:{:.3f}s".format(t2 - t1))
    return field


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH_FIXED", "-f", default="DATA/SW13L63_0.png")
    parser.add_argument("--PATH_MOVING", "-m", default="DATA/SW13L63_1.png")
    parser.add_argument("--PATH_MASK", "-ma", default="DATA/SW13L63_m.png")
    parser.add_argument("--OUT", "-o", default="rst/reg/SW13L63")
    args = parser.parse_args()

    img_fixed = cv2.imread(args.PATH_FIXED, cv2.IMREAD_GRAYSCALE)
    img_moving = cv2.imread(args.PATH_MOVING, cv2.IMREAD_GRAYSCALE)
    img_mask = cv2.imread(args.PATH_MASK, cv2.IMREAD_GRAYSCALE)
    img_fixed, img_moving = same_size(img_fixed, img_moving)
    # equHist
    img_fixed = cv2.equalizeHist(img_fixed)
    img_moving = cv2.equalizeHist(img_moving)

    # RANSAC
    field = affine_generate_field(img_fixed, img_moving, img_moving)
    img_moving_finished = sample(img_moving, field)
    cv2.imwrite(args.OUT + "/RANSAC_Flow.tif", draw_field(field))
    show_result(img_fixed, img_moving, img_moving_finished, 'RANSAC', args.OUT)

    # EA
    field = ea_generate_field(img_fixed, img_moving, img_moving, img_mask)
    img_moving_finished = sample(img_moving, field)
    cv2.imwrite(args.OUT + "reg_exp/EA_Flow.tif", draw_field(field))
    # img_tr_finished = sample(img_tr, field)
    # img_moving_finished = (img_tr_finished < 1) * img_moving_finished
    show_result(img_fixed, img_moving, img_moving_finished, 'EA', args.OUT)
