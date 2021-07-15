import os
import cv2
import torch
import numpy as np

import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class VehicleX(Dataset):
    def __init__(self, root, gt_path):
        self.data_dir = root
        self.files = os.listdir(self.data_dir)
        self.len = len(self.files)

        tree = ET.ElementTree(file=gt_path)
        root = tree.find('Items')
        items = root.findall('Item')
        self.gt = {}
        for item in items:
            self.gt[item.attrib['imageName']] = int(item.attrib['typeID'])

    def __getitem__(self, file_index):
        file_name = self.files[file_index]
        file_path = os.path.join(self.data_dir, file_name)

        label = self.gt[file_name]
        label = torch.tensor([label], dtype=torch.long)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1)
        img /= 255. # normalize input image
        img = torch.tensor(img, dtype=torch.float)

        ret = {'input': img, 'label': label}
        return ret

    def __len__(self):
        return self.len
