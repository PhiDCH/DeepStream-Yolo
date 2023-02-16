# DeepStream deployment

Support NVIDIA DeepStream SDK 6.1.1 / 6.1 / 6.0.1 / 6.0 docker images.

### Getting started

* [Tranning](#train)
* [mAP](#map)
* [DeepStream](#deepstream)
* [Tracker](#tracker)
* [Draw](#draw)
* [Export JSON](#exportjson)

### Suported models
* [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)


### Benchmarks

#### Config

```
device = NVIDIA RTX3060 12GB, A100
batch-size = 1
eval = val dataset, eval dataset
sample = 1920x1080 video and image
```


#### Results

**NOTE**: IoU=0.5, FPS = RTX3060 ; A100

| DeepStream           | Resolution | Val set      | Eval set|  FPS<br />(with display) |
|:--------------------:|:----------:|:------------:|:-------:|:------------------------:|
| yolov4               | 608        | 0.865        | 0.824   |                          |
| yolov4-fp32          | 608        |              |         | 40 ; 75                  |
| yolov4-fp16          | 608        |              |         | 85                       |
| yolov7               | 640        | 0.958        | 0.960   |                          |
| yolov7-fp32          | 640        |              |         | 50 ; 86                  |
| yolov7-fp16          | 640        |              |         | 140                      |

### Docker usage

* x86 platform

  ```
  nvcr.io/nvidia/deepstream:6.1.1-devel
  nvcr.io/nvidia/deepstream:6.1.1-triton
  ```

* Jetson platform

  ```
  nvcr.io/nvidia/deepstream-l4t:6.1.1-samples
  nvcr.io/nvidia/deepstream-l4t:6.1.1-triton
  ```

If watch realtime video, run
``` 
xhost + 
```
before  run container

```
docker run --gpus all -it --rm --net=host --privileged -v path-to-this-repo:/opt/nvidia/deepstream/deepstream-6.1/sources/DeepStream-Yolo -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.1 nvcr.io/nvidia/deepstream:6.1.1-devel
```
Inside container, install requirements
```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.4/pyds-1.1.4-py3-none-linux_x86_64.whl
pip3 install pyds-1.1.4-py3-none-linux_x86_64.whl
cd sources/team-winner-is-coming
pip3 install -e requirements.txt
```
Check cuda version by ```nvcc --version``` and build customparser C++ function 
```
export CUDA_VER=11.7
cd nvdsinfer_custombboxparser
make 
cd ..
cd nvdsinfer_custom_impl_Yolo
make 
cd ..
```
Then run deepstream app
```
python3 deepstream_app.py
```
