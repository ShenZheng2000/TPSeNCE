import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import sys
import torchvision
import torch
import torchvision.transforms as transforms


def calculate_mPA(seg_map1, seg_map2, scale=False, sem_metrics='mPA', num_classes=19):
    """
    Calculate the mean Pixel Accuracy (mPA) between two semantic segmentation maps.
    
    Parameters:
    - seg_map1: A numpy array representing the first segmentation map.
    - seg_map2: A numpy array representing the second segmentation map.
    
    Returns:
    - mpa: The mPA between the two segmentation maps.
    """
    seg_map1 = seg_map1.cpu().numpy()
    seg_map2 = seg_map2.cpu().numpy()


    if sem_metrics == 'mIoU':
        # intersection = np.logical_and(seg_map1, seg_map2)
        # union = np.logical_or(seg_map1, seg_map2)
        # res = np.sum(intersection) / np.sum(union)
        IoU = []
        for c in range(num_classes):
            intersection = np.logical_and(seg_map1 == c, seg_map2 == c)
            union = np.logical_or(seg_map1 == c, seg_map2 == c)
            if np.sum(intersection) == 0 and np.sum(union) == 0:
                continue  # Skip this class if it doesn't exist in either the target or the prediction
            else:
                IoU.append(np.sum(intersection) / np.sum(union))
        
        res = np.mean(IoU)

    elif sem_metrics == 'mPA':      
        num_correct_pixels = np.sum(seg_map1 == seg_map2)
        total_pixels = np.prod(seg_map1.shape)
        res = num_correct_pixels / total_pixels

    else:
        print("sem_metrics should be either mIoU or mPA")
        

    if scale == True:
        return res * (2 - res)
    
    return res



class QuadDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_S = os.path.join(opt.dataroot, opt.phase + 'S') 
        self.dir_T = os.path.join(opt.dataroot, opt.phase + 'T') 

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.S_paths = sorted(make_dataset(self.dir_S, opt.max_dataset_size))
        self.T_paths = sorted(make_dataset(self.dir_T, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.S_size = len(self.S_paths)
        self.T_size = len(self.T_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        S_path = self.S_paths[index % self.S_size] 
        # print(f"A_path = {A_path}, S_path = {S_path}")
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        T_path = self.T_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        S_img = Image.open(S_path).convert('RGB')
        T_img = Image.open(T_path).convert('RGB')
        # S_img = Image.open(S_path).convert('L')
        # T_img = Image.open(T_path).convert('L')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        # NOTE: use dif strategies for A, S, B
        transform_params_S = get_params(self.opt, S_img.size)
        transform_params_T = get_params(self.opt, T_img.size)

        transform_A = get_transform(modified_opt, transform_params_S, normalize=True)
        transform_S = get_transform(modified_opt, transform_params_S, normalize=False)
        transform_B = get_transform(modified_opt, transform_params_T, normalize=True)
        transform_T = get_transform(modified_opt, transform_params_T, normalize=False)

        A = transform_A(A_img)
        S = transform_S(S_img)
        B = transform_B(B_img)
        T = transform_T(T_img)

        # visualize if they are really paired (DONE)
        # torchvision.utils.save_image(A, 'A_' + A_path.split('/')[-1])
        # torchvision.utils.save_image(B, 'B_' + B_path.split('/')[-1])
        # torchvision.utils.save_image(S, 'S_' + S_path.split('/')[-1])
        # torchvision.utils.save_image(T, 'T_' + T_path.split('/')[-1])
        # sys.exit(1)
        mPA = calculate_mPA(S, T, scale=False, sem_metrics=self.opt.sem_metrics)

        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        # return {'A': A, 'B': B, 'S': S, 'A_paths': A_path, 'B_paths': B_path, 'S_paths': S_path}
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'mPA': mPA}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
