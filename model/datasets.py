import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from PIL import Image

class_list = ["scab", 
             "frog_eye_leaf_spot", 
             "rust",
             "powdery_mildew",
             "complex"]

class BasicDataset(Dataset):
    
    def __init__(self, img_path, label_path, transform = None, is_multi_class = False):
        self.is_multi_class = is_multi_class
        self.img_path = img_path
        self.img_names = os.listdir(img_path)
        self.label_file = pd.read_csv(label_path, header = 0)
        self.transform = transform
        self.label_dict()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        sample = dict(image = self.readin(os.path.join(self.img_path, img_name)),
                      label = self.labels[img_name])
        return sample

    def readin(self, path):
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img
                    
    def label_dict(self):
        self.labels = dict()
        
        def to_tensor(labels: list):
            return Tensor([1 if cls in labels else 0 for cls in class_list])

        for row in self.label_file.itertuples(index=False):
            self.labels[getattr(row, "image")] = to_tensor(getattr(row, "labels"))
        
        if self.is_multi_class:
            self.multi_class()

    def multi_class(self):
        self.label_cls = dict()
        for i, x in enumerate(set(self.labels.values())):
            self.label_cls[x] = i
        for m in self.labels.keys():
            self.labels[m] = self.label_cls[self.label_cls[m]]