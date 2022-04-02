# Human 4D Modeling

### [Project Page](https://boyanjiang.github.io/H4D/) | [Video](https://youtu.be/ZT_3BsTOY9A) | [Paper](https://arxiv.org/pdf/2203.01247.pdf)
This is the official PyTorch implementation of the CVPR'2022 paper 
"H4D: Human 4D Modeling by Learning Neural Compositional Representation".

In this paper, we introduce a compact and compositional neural representation for 4D clothed human modeling, 
which supports 4D reconstruction and motion retargeting via forward prediction, 
and motion completion/extrapolation via auto-decoding.

If you have any question, please contact Boyan Jiang  <byjiang18@fudan.edu.cn>.

#### Citation
If you use our code for any purpose, please consider citing:
```bibtex
@inProceedings{jiang2022H4D,
  title={H4D: Human 4D Modeling by Learning Neural Compositional Representation},
  author={Jiang, Boyan and Zhang, Yinda and Wei, Xingkui and Xue, Xiangyang and Fu, Yanwei},
  booktitle={CVPR},
  year={2022}
}
```

## Prerequisites
1. PyTorch (test with Python 3.7, PyTorch 1.5.1, CUDA 10.1 on Ubuntu 16.04)
2. PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric)
3. PyTorch3D (https://github.com/facebookresearch/pytorch3d)
4. Chumpy
```
pip install ./chumpy
```
5. Other dependencies
```
pip install -r requirements.txt
```

## Data and Model
### Dataset
We use the [CAPE dataset](https://cape.is.tuebingen.mpg.de/media/upload/CAPE_paper.pdf) 
for training and evaluation. Please download from [this link](https://cape.is.tue.mpg.de),
and create a symbolic link via
```
ln -s <your data path> ./dataset/cape_release
```


### Pre-trained Model
We provide pre-trained model on the CAPE dataset. 
Please download from [this link](https://drive.google.com/file/d/1nou4Rh6Z1DGWiGy6SsbhSUOqzg8fJ3Wp/view?usp=sharing), 
and unzip to the `out/h4d_pretrained` folder.
```
mkdir -p out/h4d_pretrained
unzip h4d_model.zip -d out/h4d_pretrained
```

## Quick Demo
We prepare some mesh sequences for running the demo, which can be downloaded from [this link](https://drive.google.com/file/d/1Ye8UTSUu32LNm_qYW_FziZlGpaxb6ULl/view?usp=sharing).
Please unzip the demo data to the `dataset` folder. 
Then you can run H4D on various tasks via following instructions:

```
# Forward 4D Reconstruction
python generate.py h4d_pretrained --is_demo

# Backward Fitting
python fitting.py h4d_pretrained

# Motion Retargeting
python motion_retargeting.py h4d_pretrained
```


## Training
You can train H4D from scratch through the following steps:
1. Run Principal Component Analysis (PCA) on the training data using:
```
python lib/pca_motion_space.py h4d_stage1
```
The results are saved in `assets/pca_retrained.npz`,
remember to change the `$PCA_DICT` in `lib/models.py`.

2. Train Stage-1 via:
```
python train.py h4d_stage1
```

3. Train Stage-2 via:
```
python train.py h4d_stage2
```

By default, we use 4 NVIDIA GeForce RTX 2080Ti GPU cards for training,
you can reduce the batch size in `configs/h4d_stage*.yaml` 
for smaller memory footprint.


## Evaluation
We use the codes from [4D-CR](https://github.com/BoyanJIANG/4D-Compositional-Representation)
to evaluate the reconstructed shapes with IoU and Chamfer Distance. 
To evaluate the motion accuracy, you can use:
```
python eval_motion.py <.yaml>
```

## Further Information
### 4D Representations
This project is build upon 
[Occupancy Flow](https://github.com/autonomousvision/occupancy_flow) 
and 
[4D-CR](https://github.com/BoyanJIANG/4D-Compositional-Representation).
If you are interested in 4D representation, please check their project which are previous works in this area.

## License
Apache License Version 2.0