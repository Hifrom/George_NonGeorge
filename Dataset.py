import torch.utils.data.dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        image = Image.open(img_name)
        label = self.annotation.iloc[index, 1]
        label = torch.tensor(label, dtype=torch.float32, requires_grad=False)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

class ShowImage():
    def __init__(self, image, label):
        self.image = image
        self.label = label
    def show_batch(self):
        image = self.image[0]
        print(self.label[0])
        image = image.permute(2, 1, 0)
        image = np.array(image)
        cv2.imshow('im', image)
        cv2.waitKey()