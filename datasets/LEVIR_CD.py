import torch
import torchvision
import os
import glob
from PIL import Image
import numpy as np

def get_path(dataset_root, mode):
    image_t0_path = os.path.join(dataset_root, 'img_dir', mode, 'T0')
    image_t1_path = os.path.join(dataset_root, 'img_dir', mode, 'T1')
    label_path = os.path.join(dataset_root, 'ann_dir', mode)
    edge_label_path = os.path.join(dataset_root, 'ann_edge_dir', mode)
    return image_t0_path, image_t1_path, label_path, edge_label_path

class LEVIRCDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, mode='train'):
        self.mode = mode
        self.image_t0_path, self.image_t1_path, self.label_path, self.edge_label_path = get_path(dataset_path, self.mode)
        self.image_t0_list = sorted(glob.glob(os.path.join(self.image_t0_path, "*.png")))
        self.image_t1_list = sorted(glob.glob(os.path.join(self.image_t1_path, "*.png")))
        self.label_list = sorted(glob.glob(os.path.join(self.label_path, "*.png")))
        self.edge_label_list = sorted(glob.glob(os.path.join(self.edge_label_path, "*.png")))

        self.transforms = torchvision.transforms.Compose([
                                                        torchvision.transforms.RandomHorizontalFlip(0),
                                                        torchvision.transforms.RandomVerticalFlip(0),
        ])
        self.to_tensor = torchvision.transforms.ToTensor()
    
    def __getitem__(self, item):
        image_t0 = self.transforms(Image.open(self.image_t0_list[item]).convert('RGB'))
        image_t1 = self.transforms(Image.open(self.image_t1_list[item]).convert('RGB'))
        label = Image.open(self.label_list[item])
        label = torch.from_numpy(np.array(self.transforms(label))).long()
        if self.mode == 'test':
            return self.to_tensor(image_t0), self.to_tensor(image_t1), label, self.label_list[item].split('/')[-1].split('.')[0]
        else:
            return self.to_tensor(image_t0), self.to_tensor(image_t1), label
        
    def __len__(self):
        return len(os.listdir(self.image_t0_path))


