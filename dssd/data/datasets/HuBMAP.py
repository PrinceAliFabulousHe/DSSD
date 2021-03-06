import os
import torch.utils.data
import numpy as np
from PIL import Image
import sqlite3

from dssd.structures.container import Container


class HuBMAP(torch.utils.data.Dataset):
    class_names = ('__background__', 'glom')

    def __init__(self, data_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        self.labels_dir = os.path.join(os.path.split(self.data_dir)[0], 'labels.db')

        for file in os.listdir(self.data_dir):
            if not file.endswith('.png'):
                print("Remove", file)
                os.remove(os.path.join(self.data_dir, file))
        self.imgs = list(sorted((os.listdir(self.data_dir))))

        self.db = sqlite3.connect(str(self.labels_dir))

    def __getitem__(self, index):
        # load the image as a PIL Image
        img_path = os.path.join(self.data_dir, self.imgs[index])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        cursor = self.db.cursor()
        cursor.execute("SELECT X1, Y1, X2, Y2 FROM label WHERE image_id = (SELECT ID FROM image where filename = ?)",
                       (self.imgs[index],))
        boxes = np.array(cursor.fetchall(), dtype=np.float32)
        cursor.close()

        labels = np.ones(shape=boxes.shape[0], dtype=np.int64)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # return the image, the targets and the index in your dataset
        return image, targets, index

    def __len__(self):
        return len(self.imgs)
