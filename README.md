## Important update (2020/07/18)
The original webpage will not be accessible from 7/7 to 7/25. The temporary link to the webpage is below:

[Webpage](https://alex04072000.github.io/ObstructionRemoval/)

Sorry for the inconvenience.

# [CVPR 2020] Learning to See Through Obstructions

<img src='./teaser.png' width=1000>

We present a learning-based approach for removing unwanted obstructions, such as window reflections, fence occlusions or raindrops, from a short sequence of images captured by a moving camera. Our method leverages the motion differences between the background and the obstructing elements to recover both layers. Specifically, we alternate between estimating dense optical flow fields of the two layers and reconstructing each layer from the flowwarped images via a deep convolutional neural network. The learning-based layer reconstruction allows us to accommodate potential errors in the flow estimation and brittle assumptions such as brightness consistency. We show that training on synthetically generated data transfers well to real images. Our results on numerous challenging scenarios of reflection and fence removal demonstrate the effectiveness of the proposed method.

[[Project]](https://www.cmlab.csie.ntu.edu.tw/~yulunliu/ObstructionRemoval)

Paper

<a href="https://arxiv.org/abs/2004.01180" rel="Paper"><img src="thumb.jpg" alt="Paper" width="100%"></a>

## Overview
This is the author's reference implementation of the multi-image reflection removal using TensorFlow described in:
"Learning to See Through Obstructions"
[Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/), [Wei-Sheng Lai](https://www.wslai.net/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/) (National Taiwan University & Google & Virginia Tech & University of California at Merced & MediaTek Inc.)
in CVPR 2020.
If you find this code useful for your research, please consider citing the following paper.

Further information please contact [Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/).

## Requirements setup
* [TensorFlow](https://www.tensorflow.org/):

    * tested using TensorFlow 1.10.0

* [Pre-trained PWC-Net](https://github.com/philferriere/tfoptflow)
    * Please overwrite `tfoptflow/model_pwcnet.py` and `tfoptflow/model_base.py` using the ones in this repository.

* To download the pre-trained models:

    * [ckpt](https://www.cmlab.csie.ntu.edu.tw/~yulunliu/ObstructionRemoval_/ckpt.zip)

## Data Preparation
Please prepare 5 frames and follow the naming rule `XXXXX_I[0-4].png` as shown in `imgs` folder, and change the folder path in `run_reflection.py` or `test_fence.py`.

## Usage
* Run your own sequence (reflection removal):
``` bash
CUDA_VISIBLEDEVICES=0 python3 run_reflection.py
```

* Run your own sequence (fence removal):
``` bash
CUDA_VISIBLEDEVICES=0 python3 test_fence.py
```

## Citation
```
[1] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to See Through Obstructions. Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
```
