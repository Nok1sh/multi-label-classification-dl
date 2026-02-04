import pandas as pd
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class MultiLabelDataset(Dataset):

    CLASSES = {
        '/m/05czz6l': 'Mountain', 
        '/m/015p6': 'Forest', 
        '/m/0cgh4': 'Goose', 
        '/m/0k4j': 'Airplane', 
        '/m/01yrx': 'Person', 
        '/m/0bt9lr': 'Cat', 
        '/m/02_41': 'Dog', 
        '/m/02zr8': 'Water', 
        '/m/09ld4': 'Car', 
        '/m/0dbvp': 'Building', 
        '/m/09d_r': 'Snow', 
        '/m/01g317': 'Bird', 
        '/m/06gfj': 'Road', 
        '/m/06_dn': 'Train', 
        '/m/07jdr': 'Fire', 
        '/m/0838f': 'Frog'
        }
    
    TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    def __init__(self, path, transform=TRANSFORM) -> None:
        super().__init__()
        self.path = path
        self.transform = transform
        self.labels = pd.read_csv(f"data/labels/{path}/labels.csv", sep=",")
        self.full_path = os.path.join("data/images", path)
        
        self.dataset = []
        self.len_dataset = 0

        for index in range(self.labels.shape[0]):
            img, cls_multi_hot = self.labels.iloc[index, 0], self.labels.iloc[index, 1:].tolist()
            path_to_img = os.path.join(self.full_path, f"{img}.jpg")
            self.dataset.append((path_to_img, cls_multi_hot))
        
        self.len_dataset = len(self.dataset)


    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        path_to_img, cls_multi_hot = self.dataset[index]
        img = np.array(Image.open(path_to_img).convert('RGB'))

        classes = torch.tensor(cls_multi_hot, dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, classes
    


# data = MultiLabelDataset("train")

