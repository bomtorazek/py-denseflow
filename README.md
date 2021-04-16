# Py-denseflow


This is a python port of denseflow, which extract the videos' frames and **optical flow images** with **TVL1 algorithm** as default.

---

### Requirements:
- numpy
- cv2
- PIL.Image
- multiprocess
- scikit-video (optional)
- scipy


## Installation
#### Install the requirements:
```
pip install -r requirements.txt
conda 환경을 사용하실 경우:
conda env create -f environment_tvl1_linux.yml
conda env create -f environment_tvl1_windows.yml
```
위의 방법이 안 될 경우 requirements_linux.txt 또는 requirements_windows.txt에 있는 

제가 쓴 bash script를 복붙하시는 방법도 있습니다.

---

## Usage
The denseflow.py contains two modes including '**run**' and '**debug**'.


here 'debug' is built for debugging the video paths and video-read methods. ([IPython.embed](http://ipython.org/ipython-doc/dev/interactive/reference.html#embedding) suggested)

debug는 그냥 영상 하나만 잘 되는지 확인 해보는 모드입니다.

Just simply run the following code: 

data_root와 dataset을 밑의 경로를 보시며 적절히 수정해주세요

마지막 commit을 windows 환경에서는 검증하지 못했으니 확인 부탁드립니다.

```
python denseflow.py --data_root=/home/esuh/data/ --dataset=cvpr/Track2.1/Train --new_dir=flows --step=1 --mode=run

--rejection=completed.txt : 이미 완료된 항목을 배제합니다.
--validation : validation frame 추출시 동영상 경로가 조금 달라서 이 parse를 추가해야 합니다.
--modality: rgb, flow, both 중에 추출할 frames을 선택할 수 있습니다.
--semi: semi는 이름 경로가 달라서 추가 해줘야 합니다.

python denseflow.py --data_root=/home/esuh/data/ --dataset=cvpr/Track2.2/HMDB51 --new_dir=flows --step=1 --mode=run --modality=rgb --semi
python denseflow.py --data_root=/home/esuh/data/ --dataset=cvpr/Track2.2/Train --new_dir=flows --step=1 --mode=run --modality=rgb --semi --validation
python denseflow.py --data_root=/home/esuh/data/ --dataset=cvpr/Track2.2/Validation --new_dir=flows --step=1 --mode=run --modality=rgb --semi --validation

```

```
data
----cvpr
    ----Track2.1
            ----Train
                 ----videos (여기에 원본 클립 넣으시면 됩니다.)
                       ----Run
                          ----Run_1_1.mp4
                          ....
                       ----Sit
                       ...
                 ----flows (여기에 frame들이 만들어집니다. 폴더도 자동으로 만들어집니다.)
                 
            ----Validation
                 ----videos
                       ----0.mp4
                       ----1.mp4
                 ----flows (여기에 frame들이 만들어집니다. 폴더도 자동으로 만들어집니다.)
flows 아래에는 각 클립이름의 폴더가 생기며 그 아래에 img_00001.jpg, flow_x_00001.jpg, flow_y_00001.jpg가 만들어집니다.
```


While in 'run' mode, here we provide multi-process as well as multi-server with manually s_/e_ IDs setting.

for example:  server 0 need to process 3000 videos with 4 processes parallelly working:

```
python denseflow.py --new_dir=denseflow_py --num_workers=4 --step=1 --bound=20 --mode=run --s_=0 --e_=3000
여기 --s랑 --e는 시작 비디오 끝 비디오 선택하는 기능인데 필요없어서 코드 수정하고 주석 처리했습니다.
혹시 필요하신 분은 수정하셔서 사용하시면 됩니다.
```

---

Just feel free to let me know if any bugs exist.

