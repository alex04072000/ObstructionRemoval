# Learning to See Through Obstructions
Recovering a high dynamic range (HDR) image from asingle low dynamic range (LDR) input image is challenging due to missing details in under-/over-exposed regions caused by quantization and saturation of camera sensors. In contrast to existing learning-based methods, our core idea is to incorporate the domain knowledge of the LDR image formation pipeline into our model. We model the HDR-to-LDR image formation pipeline as the (1) dynamic range clipping, (2) non-linear mapping from a camera response function, and (3) quantization. We then propose to learn three specialized CNNs to reverse these steps. By decomposing the problem into specific sub-tasks, we impose effective physical constraints to facilitate the training of individual sub-networks. Finally, we jointly fine-tune the entire model end-to-end to reduce error accumulation. With extensive quantitative and qualitative experiments on diverse image datasets, we demonstrate that the proposed method performs favorably against state-of-the-art single-image HDR reconstruction algorithms. The source code, datasets, and pre-trained model are available at our project website.

[[Project]](https://www.cmlab.csie.ntu.edu.tw/~yulunliu/Obstruction)

Paper

<a href="http://www.cmlab.csie.ntu.edu.tw/~yulunliu/Obstruction_/.pdf" rel="Paper"><img src="thumb.png" alt="Paper" width="100%"></a>

## Overview
This is the author's reference implementation of the multi-image reflection removal using TensorFlow described in:
"Learning to See Through Obstructions"
[Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/), [Wei-Sheng Lai](https://www.wslai.net/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/) (National Taiwan University & Google & Virginia Tech & University of California at Merced & MediaTek Inc.)
in CVPR 2020.
Should you be making use of our work, please cite our paper [1]. 

<img src='./teaser.png' width=500>

Further information please contact [Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/).

## Requirements setup
* [TensorFlow](https://www.tensorflow.org/)

* [Pre-trained PWC-Net](https://github.com/philferriere/tfoptflow)
    * Please overwrite tfoptflow/model_pwcnet.py using the one in this repository.

* To download the pre-trained models:

    * [ckpt](https://drive.google.com/open?id=1OUjr1Cj-nHOUEONoqIMnQIL9qTElpE6r)

## Data Preparation
* [Deep Voxel Flow (DVF)](https://github.com/liuziwei7/voxel-flow)

## Usage
* Run your own sequence (with online optimization):
``` bash
CUDA_VISIBLEDEVICES=0 python3 run_reflection.py
```

* Run your own sequence (fence removal):
``` bash
CUDA_VISIBLEDEVICES=0 python3 test_fence.py
```

## Citation
```
[1]  @inproceedings{liu2019cyclicgen,
         author = {},
         title = {},
         booktitle = {},
         year = {}
     }
```
