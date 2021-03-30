import os
import os.path as osp
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import platform

import numpy as np
import cv2
from PIL import Image
import imageio


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument("--data_dir", default="/data/UG2-2021-Track2.1/video/Train", type=str)
    parser.add_argument("--save_dir", default="/results/optical_flow/tv-l1/UG2-2021-Track2.1/Train/raw", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
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


def save_flows(flows, save_dir, save_name, num, bound):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save the flows
    save_x = os.path.join(save_dir, f"{save_name}_x_{num:04d}.png")
    # save_y = os.path.join(save_dir, f"{save_name}_y_{num:04d}.png")
    flow_x = flow_x.astype("uint8")
    # flow_y = flow_x.astype("uint8")
    if (flow_x - flow_y).max() > 0:
        print(f"flow_x and flow_y is differenct on {save_name} - {num}")
    flow_x_img = Image.fromarray(flow_x)
    # flow_y_img = Image.fromarray(flow_y)
    imageio.imwrite(save_x, flow_x_img)
    # imageio.imwrite(save_y, flow_y_img)
    return 0


def dense_flow(video_fpath, save_dir, step=1, bound=15, save_as_image=False):
    fname = osp.splitext(osp.basename(video_fpath))[0]
    label = video_fpath.split(os.sep)[-2]

    print(f"Processing - {fname}")

    # read video
    frame_list = []
    cap = cv2.VideoCapture(video_fpath)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_list.append(img)

    len_frame = len(frame_list)
    frame_num, num0 = 0, 0
    image, prev_image, gray, prev_gray = None, None, None, None

    flow_list = []
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

        if save_as_image:
            save_flows(flowDTVL1, osp.join(save_dir, video_fpath.split(os.sep)[-2]), fname, frame_num, bound)
        else:
            flow_list.append(
                (
                    ToImg(flowDTVL1[..., 0], bound).astype("uint8"),
                    ToImg(flowDTVL1[..., 1], bound).astype("uint8"),
                )
            )

        prev_gray = gray
        prev_image = image
        frame_num += 1
        # to pass the out of stepped frames
        step_t = step
        while step_t > 1:
            num0 += 1
            step_t -= 1

    # save as video
    if not save_as_image:
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # size = flow_list[0].shape
        size = np.concatenate(
            (
                frame_list[0],
                cv2.cvtColor(flow_list[0][0], cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(flow_list[0][1], cv2.COLOR_GRAY2RGB),
            ),
            axis=1,
        ).shape[:2]

        if not os.path.exists(osp.join(save_dir, label)):
            os.makedirs(osp.join(save_dir, label))

        out = cv2.VideoWriter(osp.join(save_dir, label, f"{fname}.avi"), fourcc, fps, size[::-1])
        for img, flow in zip(frame_list, flow_list):
            frame = np.concatenate(
                (
                    img,
                    cv2.cvtColor(flow[0], cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(flow[1], cv2.COLOR_GRAY2RGB),
                ),
                axis=1,
            )
            out.write(frame)
        out.release()


def _main():
    args = parse_args()
    video_fpath_list = sorted(glob.glob(osp.join(args.data_dir, "*", "*")))

    # dense_flow(video_fpath_list[0], args.save_dir)
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(dense_flow, save_dir=args.save_dir), video_fpath_list)


if __name__ == "__main__":
    _main()
