# Updates 
Arxvi link is available at [[here](https://arxiv.org/abs/2311.00660)]

Paper get accepted by WACV 2024!

# TPSeNCE
A PyTorch Implementatiion of TPSeNCE, an image rain generation model

# Abstract 
Rain generation algorithms have the potential to improve the generalization of deraining methods and scene understanding in rainy conditions. However, in practice, they produce artifacts and distortions and struggle to control the amount of rain generated due to a lack of proper constraints. In this paper, we propose an unpaired image-to-image translation framework for generating realistic rainy images. We first introduce a **Triangular Probability Similarity (TPS)** constraint to guide the generated images toward clear and rainy images in the discriminator manifold, thereby minimizing artifacts and distortions during rain generation. Unlike conventional contrastive learning approaches, which indiscriminately push negative samples away from the anchors, we propose a **Semantic Noise Contrastive Estimation (SeNCE)** strategy and reassess the pushing force of negative samples based on the semantic similarity between the clear and the rainy images and the feature similarity between the anchor and the negative samples. Experiments demonstrate  realistic **rain generation** with minimal artifacts and distortions, which benefits **image deraining** and **object detection in rain**. Furthermore, the method can be used to generate realistic **snowy and night** images, underscoring its potential for broader applicability.

# Visual Results (Video)

Rain Generation Video [[here](https://www.youtube.com/watch?v=eNS_8fuSLjc)]

Object Detection Video [[here](https://www.youtube.com/watch?v=RkGmAAORugc)]

# Visual Results (Images)


## Rain Generation (Clear to Rainy)
<p align="center">
  <img width="100%" src="figures/Rain_Generation.png">
</p>


## Deraining (Rainy to Clear)
<p align="center">
  <img width="100%" src="figures/Derain.png">
</p>


## Object Detection in Rain
<p align="center">
  <img width="100%" src="figures/Detect.png">
</p>


# Getting Started
```
git clone https://github.com/ShenZheng2000/TPSeNCE.git
```


# Dependencies
```
pip install -r requirements.txt
```


# Dataset Download

Download BDD100K from [[here](https://www.vis.xyz/bdd100k/)]

Download INIT from [[here](https://zhiqiangshen.com/projects/INIT/index.html)]

Download Boreas from [[here](https://www.boreas.utias.utoronto.ca/#/)]


TODO: provide dataset split files


# Dataset Explanations
Suppose we are translating clear images to rainy images, then we should put images under /path_to_your_dataset/ like this.

```
A: source images (e.g., clear images)
B: target images (e.g., rainy images)
S: sem. seg. maps of A
T: sem. seg. maps of B
```


# Dataset Folder Structure
```
/path_to_your_dataset/
    ├── trainA
    ├── trainB
    ├── trainS
    ├── trainT
    ├── testA
    ├── testB
    ├── testS
    ├── testT
```


## NOTE1: 
testS and testT is not used for training or testing. However, make sure to include images in the testS and testT folders to prevent them from being empty, as an empty folder cause error during training and testing. 

In convenience, we suggest that you use the following command to avoid empty folder.
```
cp -r testA testS
cp -r testB testT
```

## NOTE2: 
As ground truth semantic segmentation maps are not available for BDD100K, we estimate these maps using the [[ConvNeXt-XL](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/convnext)] model from the [[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/)] toolbox. If you are working with a dataset like [[Cityscapes](https://www.cityscapes-dataset.com/)] which already includes ground truth semantic segmentation maps, the semantic guidance can be expected to be more effective.


# Training from scratch
Run in terminal.
```
bash train.sh
```


# Testing with pretrained model
Download the checkpoints from [[here](https://drive.google.com/file/d/1NPWCHpljJhJcGTadYebyLf0sGbkJm3z4/view?usp=share_link)]

Unzip the checkpoints.

Create folder `bdd100k_1_20`, `INIT`, and `Boreas` under `./checkpoints` like below. 

```
/TPSeNCE/
    ├── checkpoints
    │   ├── bdd100k_1_20
    │   ├── INIT
    |   ├── Boreas
```

Run in terminal
```
bash test.sh
```

# Citation

If you find this work helpful, please cite

```
@inproceedings{zheng2024tpsnce,
  title={TPSeNCE: Towards Artifact-Free Realistic Rain Generation for Deraining and Object Detection in Rain},
  author={Zheng, Shen and Lu, Changjie and Narasimhan, Srinivasa G},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
```


# Acknowledgment

This repository is heavily based upon [[MoNCE](https://github.com/fnzhan/MoNCE)] and [[CUT](https://github.com/taesungp/contrastive-unpaired-translation)]. 


<!-- The specific lines of code are as follows.
```
python test.py \
--dataroot /root/autodl-tmp/Datasets/bdd100k_1_20 \
--results_dir ./results_test_a2b/ \
--num_test 4025 \
--gpu_ids 0 \
--name bdd100k_1_20 \
--phase test \
--preprocess scale_width \
--load_size 640
```

Here are some explanations:
```
dataroot: 
    path_to_your_dataset
results_dir: 
    path_for_output_images
num_test: 
    numbers of images you want to test
gpu_ids: 
    gpu id (only supports single gpu for testing)
name: 
    folder mame under ./checkpoints
phase: 
    train or test
preprocess: 
    scale_width (by default)
load_size: 
    640 for BDD, 572 for INIT, 430 for Boreas
``` -->
