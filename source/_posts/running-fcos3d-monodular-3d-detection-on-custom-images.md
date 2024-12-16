---
title: Running FCOS3D Monodular 3D Detection on Custom Images
date: 2024-12-16 14:54:17
tags:
---

# Introduction 

[FCOS3D](https://arxiv.org/abs/2104.10956) (ICCV 2021) is a deep learning-based model for monocular 3D object detection in images. That is, other than common object detectors like YOLO, it estimates **three-dimensional** bounding boxes for objects, i.e. involving a depth component next to x and y positions. Moreover, it does so in a _monocular_ way, that is, from just a single image. Considering that most object detector use at least stereo images or even lidar point clouds, you can guess that this is an especially hard problem. On the one hand, this is super useful, because it could technically run on arbitrary pictures taken by your smartphone. On the other hand, you'll have to be aware that detection accuracy is, of course, much worse compared to detectors that utilize richer sensor data.

While FCOS3D, having been published in 2021 already, is arguably not actually state of the art anymore, I still wanted to use it as part of my research, especially because it's still one of the most widely adopted models in that realm. However, it took me quite a while to get it running on my own images. To save other people from similar struggles, here is a brief, hacky description of how to get it running locally.

Please note that this is the way that **I** managed to get the model running. Perhaps there are different or simpler ways, but this is what worked for me.

![](images/fcos3d.webp)

# Prerequisites
You'll need the [**intrinsic calibration**](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) of your camera as 3x3 matrix. You may use OpenCV to estimate it.

# Setup

FCOS3D is implemented as part of [MMDetection3D](https://mmdetection3d.readthedocs.io) framework, which, btw. supports a whole lot of other detection models in addition. The framework's code base is quite a mess and probably not particularly what you what consider well-structured and self-documenting code. Nevertheless, I luckily managed to dig my way through it. So here's what I did, roughly following MMDetection's [Getting Started](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) and their [docs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html#monocular-3d-demo) on inference.

First of all, I had to fall back to older versions of Python, PyTorch and CUDA to get things working. 

## Step 1: Clone repo and download pre-trained model
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
git reset --hard fe25f7a51d36e3702f961e198894580d83c4387b  # for reproducibility

cd demo/data
wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
```

We're using the model that was pre-trained on the [nuScenes](https://www.nuscenes.org/) dataset.

## Step 2: Use Python 3.8 and virtual environment

I used [pyenv](https://github.com/pyenv/pyenv) to install a separate Python distro alongside my system-wide installation. Alternatively, you may install Python 3.8 natively, or use a different version manager such as [asdf](https://github.com/asdf-vm/asdf).

```bash
pyenv install 3.8.19        # install 3.8
pyenv local 3.8.19          # use 3.8 for this project
python -m venv venv         # create virtual env
source venv/bin/activate    # activate virtual env
```

## Step 3: Install dependencies
From some GitHub issue (which I can't find anymore, unfortunately) I learned that I'd have to use PyTorch with CUDA 11.7. Also, I used the (outdated) MMDetection-specific versions mentioned in their docs.

```bash
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
pip install -v -e .  # install mmdet3d from local source
```

## Step 4: Inject custom config
This is probably the second-most hacky part of all, but I couldn't find a better way at first sight. We need to inject our custom calibration matrix into the (pickled, binary) config parameters file (aka. `ANNOTATION_FILE`). To do so, we load the nuScenes-specific config provided by the repo, modify it, and save it again.

```python
import pickle

custom_calib = [
        [ 1243.09579,    0.     ,  953.87192,   ],
        [ 0.     , 1245.01287,  558.13058,      ],
        [ 0.     ,    0.     ,    1.,           ],
    ]

with open('demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl', 'rb') as f:
    d = pickle.load(f)

d['data_list'][0]['images']['CAM_CUSTOM'] = {'cam2img': custom_calib}

with open('demo/data/custom.pkl', 'wb') as f:
    pickle.dump(d, f)
```

## Step 5: Apply some hacks
In addition, I had to apply a bunch of custom changes to the MMDetection3D, including:

1. Passing the `cam_type` command-line argument on to the inferencer. 
1. Ignoring lidar-specific parameters (see [#2868](https://github.com/open-mmlab/mmdetection3d/issues/2868))
1. Ignoring unneeded hard-coded image path param

Here's the according Git patch: [`mmdet3d_fixes.patch`](https://gist.github.com/muety/a53bbc5c7d896cb4bd6f6a25f63d15b6).

Apply it with `git am mmdet3d_fixes.patch`.

## Step 6: Run inference 🚀
```bash
python demo/mono_det_demo.py \
    /tmp/custom-image-0001.png \
    demo/data/custom.pkl \
    configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py \
    demo/data/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \
    --cam-type CAM_CUSTOM \
    --pred-score-thr 0.05 \
    --show 
```