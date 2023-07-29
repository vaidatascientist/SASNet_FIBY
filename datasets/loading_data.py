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
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from .crowd_dataset import FIBY
import pytorch_lightning as pl

# the function to return the dataloader 
def loading_data(data_root):
    # the augumentations
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])
    # create the dataset
    train_set = FIBY(data_root, train=True,
                     transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = FIBY(data_root, train=False, transform=transform)

    return train_set, val_set


class SASNet_Lightning(pl.LightningModule):
    def __init__(self, data_root, batch_size, num_workers, pin_memory):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage=None):
        self.train_set, self.val_set = loading_data(self.data_root)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)