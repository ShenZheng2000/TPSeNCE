# TPSeNCE
A PyTorch Implementatiion of TPSeNCE, an image rain generation model

# Abstract 
Rain generation algorithms have the potential to improve the generalization of deraining methods and scene understanding in rainy conditions. However, in practice, they produce artifacts and distortions and struggle to control the amount of rain generated due to a lack of proper constraints. In this paper, we propose an unpaired image-to-image translation framework for generating realistic rainy images. We first introduce a Triangular Probability Similarity (TPS) constraint to guide the generated images toward clear and rainy images in the discriminator manifold, thereby minimizing artifacts and distortions during rain generation. Unlike conventional contrastive learning approaches, which indiscriminately push negative samples away from the anchors, we propose a Semantic Noise Contrastive Estimation (SeNCE) strategy and reassess the pushing force of negative samples based on the semantic similarity between the clear and the rainy images and the feature similarity between the anchor and the negative samples. Experiments demonstrate  realistic rain generation with minimal artifacts and distortions, which benefits image deraining and object detection in rain. Furthermore, the method can be used to generate realistic snowy and night images, underscoring its potential for broader applicability.


# Dependencies
```
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
packaging
GPUtil>=1.4.0
```

# Datasets
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


# Dataset Explanations
Suppose we are translating clear images to rainy images, then we should put images under path_to_your_dataset like this.

```
A: source images (e.g., clear images)
B: target images (e.g., rainy images)
S: sem. seg. maps of A
T: sem. seg. maps of B
```

NOTE: 
testS and testT is not used for training or testing. However, make sure to include images in the testS and testT folders to prevent them from being empty, as an empty folder cause error during training and testing. 

In convenience, we suggest that you use the following command to avoid empty folder.
```
cp -r testA testS
cp -r testB testT
```


# Train
Run in terminal.
```
bash train.sh
```

Adjust the hyperparameters in the shell script, based on your requirements. 


# Test
Download the checkpoints from [here](https://drive.google.com/file/d/1NPWCHpljJhJcGTadYebyLf0sGbkJm3z4/view?usp=share_link)

Unzip the downloaded zip file.

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

The specific lines of code are as follows.
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
```
