import os
import os.path as osp
import glob
from shutil import copyfile

import cv2
from skimage.io import imread, imsave


root_dir = "/data/hyeokjae/results/UG2-2021/optical_flow/tv-l1/Track2.1-Test"
sid_dir = osp.join(root_dir, "sid")
gic_dir = osp.join(root_dir, "gic")
save_dir = osp.join(root_dir, "sid_gic")

samples = os.listdir(sid_dir)
for sample in samples:
    rgb_list = sorted(glob.glob(osp.join(sid_dir, sample, "img_*")))
    flow_list = sorted(glob.glob(osp.join(gic_dir, sample, "flow_*")))

    if 2 * len(rgb_list) - len(flow_list) != 0:
        print(sample)
        raise ValueError
    if not osp.isdir(osp.join(save_dir, sample)):
        os.makedirs(osp.join(save_dir, sample))

    for fpath in rgb_list:
        save_fpath = osp.join(save_dir, sample, osp.basename(fpath))
        # img = imread(fpath)
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # imsave(save_fpath, img)
        copyfile(fpath, save_fpath)

    for fpath in flow_list:
        save_fpath = osp.join(save_dir, sample, osp.basename(fpath))
        copyfile(fpath, save_fpath)
