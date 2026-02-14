import pandas as pd
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class MultiLabelDataset(Dataset):

    CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }
    
    TRANSFORM_TRAIN = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
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
    
