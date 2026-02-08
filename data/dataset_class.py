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
        '/m/019jd': 'Goose',
        '/m/0bt_c3': 'Airplane',
        '/m/0cgh4': 'Person',
        '/m/0k4j': 'Cat',
        '/m/01yrx': 'Dog',
        '/m/01mzpv': 'Water',
        '/m/0csby': 'Car',
        '/m/01m3v': 'Building',
        '/m/0bt9lr': 'Snow',
        '/m/02_41': 'Bird',
        '/m/0ch_cf': 'Road',
        '/m/0c9ph5': 'Train',
        '/m/02wbm': 'Fire',
        '/m/02zr8': 'Frog',
        '/m/09ld4': 'Bridge',
        '/m/0dbvp': 'Cloud',
        '/m/08t9c_': 'Flower',
        '/m/03k3r': 'Boat',
        '/m/09d_r': 'Grass',
        '/m/01g317': 'Fish',
        '/m/06gfj': 'Chair',
        '/m/01bqvp': 'Table',
        '/m/06_dn': 'Book',
        '/m/06m_p': 'Computer',
        '/m/04bcr3': 'Food',
        '/m/07jdr': 'Sky',
        '/m/07j7r': 'Horse',
        '/m/0838f': 'Window',
        '/m/0d4v4': 'Tree'
        }
    
    TRANSFORM_TRAIN = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.RandomCrop(224),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomRotation(degrees=15),
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    TRANSFORM_VAL = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, path, transform=None) -> None:
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

