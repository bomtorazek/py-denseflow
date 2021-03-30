# Py-denseflow

This is a python port of denseflow, which extract the videos' frames and **optical flow images** with **TVL1 algorithm** as default.

## Usage
```
python denseflow_hchoi.py --data_dir=/data/UG2-2021-Track2.1/video/Train --save_dir=/results/UG2-2021/optical_flow/tv-l1/Track2.1/Train/raw --num_workers=2
```

```
data_dir
----data
    ----UG2-2021-Track2.1
            ----video
                 ----Train
                       ----Run
                          ----Run_1_1.mp4
                          ....
                       ----Sit
                 
                  ----Validation
```
