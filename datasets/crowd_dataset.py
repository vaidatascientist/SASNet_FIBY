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
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import skimage
import torchvision.transforms as T

class FIBY(Dataset):
    def __init__(self, data_path:str, train=False):
        super().__init__()
        self.data_path = data_path
        self.train = train
        
        img_paths, density_paths = self.load_paths()

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.img_paths = img_paths
        self.density_paths = density_paths
        
    def load_paths(self):
        img_paths = []
        density_paths = []
        
        if self.train:
            split = 'train'
        else:
            split = 'test'
            
        split_file = f"{self.data_path}/fiby_{split}.list"
        # split_file = f"{self.data_path}/part_A_final/shan_{split}.list"
        
        with open(split_file, 'r') as f:
            for line in f:
                img_path, density_path = line.strip().split()
                img_paths.append(f"{self.data_path}/{img_path}")
                density_paths.append(f"{self.data_path}/{density_path}")

        return img_paths, density_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        density_path = self.density_paths[idx]
        
        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((128, 128), resample=Image.NEAREST)        
        # img = self.img_transform(img)

        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.resize((128, 128), resample=Image.NEAREST)
        img = self.img_transform(img)
        
        density_file = h5py.File(density_path)
        density_map = np.asarray(density_file['density'])
        
        if isinstance(density_map, np.ndarray):
            density_map = torch.from_numpy(density_map)
            
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
        if self.train:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(128, 128))
            img = T.functional.crop(img, i, j, h, w)
            density_map = T.functional.crop(density_map, i, j, h, w)
        # random flipping
        if random.random() > 0.5 and self.train:
            img = T.functional.hflip(img)
            density_map = T.functional.hflip(density_map)

        img = torch.Tensor(img)
        density_map = density_map.numpy()

        resized_density_map = skimage.transform.resize(density_map, (128, 128))
        resized_density_map *= np.sum(density_map) / np.sum(resized_density_map)
        resized_density_map = torch.tensor(resized_density_map)
        resized_density_map = resized_density_map.float()
        resized_density_map = torch.unsqueeze(resized_density_map, 0)

        return img, resized_density_map