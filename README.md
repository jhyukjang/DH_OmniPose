# OmniPose

  <a href="https://arxiv.org/abs/2103.10180">**OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation**</a>.
</p><br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/omnipose.png" title="OmniPose Architecture">
  Figure 1: OmniPose framework for multi-person pose estimation. The input color image of dimensions (HxW) is fed through the improvedHRNet backbone and WASPv2 module to generate one heatmap per joint, or class.
</p><br />

  
## Usage

### Settings
We assume that you already install conda environment from [here](https://github.com/ksos104/DH_HSSN#settings)
```
https://github.com/jhyukjang/DH_OmniPose.git
cd DH_OmniPose
conda activate hssn
pip install -r requirements.txt
```


### Structure
Follow this structure:
```
|-- datasets
        |-- Pascal Part Person
             |-- train
             |-- val
|             
|-- DH_HSSN
|-- DH_OmniPose
        |-- checkpoint.pth
```



### Inference
```
# parsing results are saved in ./visualization/pascal folder
./run_demo_pascal.sh
```



**Citation:**

<p align="justify"> Artacho, B.; Savakis, A. OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation. in ArXiv, 2021. <br />

```
@InProceedings{Artacho_2021_ArXiv,
  title = {OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation},
  author = {Artacho, Bruno and Savakis, Andreas},
  eprint={2103.10180},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  year = {2021},
}
```

<p align="justify"> Artacho, B.; Savakis, A. UniPose+: A unified framework for 2D and 3D human pose estimation in images and videos. on IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. <br />

```
@article{Artacho_2021_PAMI,
  title = {UniPose+: A unified framework for 2D and 3D human pose estimation in images and videos},
  author = {Artacho, Bruno and Savakis, Andreas},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2021},
}
```

<p align="justify"> Artacho, B.; Savakis, A. UniPose: Unified Human Pose Estimation in Single Images and Videos. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. <br />

```
@inproceedings{Artacho_2020_CVPR,
title = {UniPose: Unified Human Pose Estimation in Single Images and Videos},
author = {Artacho, Bruno and Savakis, Andreas},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```
