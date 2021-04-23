import os
import os.path as osp
import glob
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np
import cv2
from PIL import Image
import imageio


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--data_dir", default="/UG2-2021/data/Track2.1/video/Train", type=str)
    parser.add_argument("--save_dir", default="/UG2-2021/results/optical_flow/tv-l1/Track2.1/raw", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    return args


def ToImg(raw_flow, bound):
    """
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    """
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= -bound
    flow *= 255 / float(2 * bound)
    return flow


def save_flows(image, flows, save_dir, video_name, num, bound):
    """
    To save the optical flow images and raw images
    :param save_dir: save_dir name
    :param flows: contains flow_x and flow_y
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: 0
    """
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)
    if not os.path.exists(osp.join(save_dir, video_name)):
        os.makedirs(osp.join(save_dir, video_name))

    # save images
    save_img = osp.join(save_dir, video_name, "img_{:05d}.jpg".format(num))
    imageio.imwrite(save_img, image)
    # save flows
    save_x = osp.join(save_dir, video_name, "flow_x_{:05d}.jpg".format(num))
    save_y = osp.join(save_dir, video_name, "flow_y_{:05d}.jpg".format(num))
    flow_x = flow_x.astype("uint8")
    flow_y = flow_x.astype("uint8")
    flow_x_img = Image.fromarray(flow_x)
    flow_y_img = Image.fromarray(flow_y)
    imageio.imwrite(save_x, flow_x_img)
    imageio.imwrite(save_y, flow_y_img)
    return 0


def dense_flow(video_fpath, save_dir, step=1, bound=15, gamma=None):
    fname = osp.splitext(osp.basename(video_fpath))[0]

    print(f"Processing - {fname}")

    # read video
    frame_list = []
    cap = cv2.VideoCapture(video_fpath)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        if gamma is not None:
            img = img.astype(np.float32)
            img = ((img / 255) ** gamma) * 255
            img = img.astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_list.append(img)

    len_frame = len(frame_list)
    frame_num, num0 = 0, 0
    image, prev_image, gray, prev_gray = None, None, None, None

    while True:
        if num0 >= len_frame:
            break
        frame = frame_list[num0]
        num0 += 1
        if frame_num == 0:
            image = np.zeros_like(frame)
            gray = np.zeros_like(frame)
            prev_gray = np.zeros_like(frame)
            prev_image = frame
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
            frame_num += 1
            # to pass the out of stepped frames
            step_t = step
            while step_t > 1:
                num0 += 1
                step_t -= 1
            continue

        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frame_0 = prev_gray
        frame_1 = gray
        # default choose the tvl1 algorithm
        dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)

        save_flows(image, flowDTVL1, save_dir, fname, frame_num, bound)

        prev_gray = gray
        prev_image = image
        frame_num += 1
        # to pass the out of stepped frames
        step_t = step
        while step_t > 1:
            num0 += 1
            step_t -= 1


def _main():
    args = parse_args()
    video_fpath_list = sorted(glob.glob(osp.join(args.data_dir, "*", "*")))

    # dense_flow(video_fpath_list[0], args.save_dir)
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(dense_flow, save_dir=args.save_dir, gamma=args.gamma), video_fpath_list)


if __name__ == "__main__":
    _main()
