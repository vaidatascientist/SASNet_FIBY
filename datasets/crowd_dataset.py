# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
from time import sleep
import skimage.transform
from torchvision import transforms
import re



class FIBY(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "fiby_train.list"
        self.eval_list = "fiby_test.list"
        
        # self.train_lists = 'shan_train.list'
        # self.eval_list = 'shan_test.list'
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')
            
        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        # get the image path
        img_path = self.img_list[index]
        
        # img_path = re.sub(r"(part_A_final/)", "", img_path, count=1)
        
        img, density_map = load_data(img_path)
        
        if isinstance(density_map, np.ndarray):
            density_map = torch.from_numpy(density_map)
        
        # perform data augumentation
        img = img.resize((128, 128), resample=Image.NEAREST)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                density_map = torch.nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=scale).squeeze(0).squeeze(0)
        # random crop augumentaiton
        if self.train and self.patch:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
            img = transforms.functional.crop(img, i, j, h, w)
            density_map = transforms.functional.crop(density_map, i, j, h, w)
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            img = transforms.functional.hflip(img)
            density_map = transforms.functional.hflip(density_map)

        img = torch.Tensor(img)
        
        density_map = density_map.numpy()
        
        resized_density_map = skimage.transform.resize(density_map, (128, 128))
        resized_density_map *= np.sum(density_map) / np.sum(resized_density_map)
        resized_density_map = torch.tensor(resized_density_map)
        resized_density_map = resized_density_map.float()
        resized_density_map = torch.unsqueeze(resized_density_map, 0)

        return img, resized_density_map

def load_data(img_path):
    # get the path of the ground truth
    gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
    # open the image
    img = Image.open(img_path).convert('RGB')
    # load the ground truth
    while True:
        try:
            gt_file = h5py.File(gt_path)
            break
        except:
            sleep(2)
    density_map = np.asarray(gt_file['density'])

    return img, density_map

# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den