import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class UnetDatasetfull(Dataset):
    def __init__(self, annotation_lines, annotation_lines_pd, input_shape, num_classes, train, dataset_path):
        super(UnetDatasetfull, self).__init__()
        self.annotation_lines = annotation_lines
        self.annotation_lines_pd = annotation_lines_pd
        self.length = len(annotation_lines)
        self.length_pd = len(annotation_lines_pd)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length + self.length_pd

    def __getitem__(self, index):

        labeled_indices = []
        unlabeled_indices = []

        
        if index < self.length:
            annotation_line = self.annotation_lines[index]
            labeled_indices.append(index)
            fd = True
        else:
            
            annotation_line = self.annotation_lines_pd[index - self.length]
            unlabeled_indices.append(index)
            fd = False

        name = annotation_line.split()[0]

        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "image"), name + ".png"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "mk2"), name + ".png"))
        png_2 = Image.open(os.path.join(os.path.join(self.dataset_path, "point_label"), name + ".png"))
        U = np.load(os.path.join(os.path.join(self.dataset_path, "npy"), name + ".npy"))

        jpg, png, png_2 = self.get_random_data(jpg, png, png_2, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        png_2 = np.array(png_2)
        png_2[png_2 >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        

        return jpg, png, png_2, seg_labels, fd,labeled_indices,unlabeled_indices

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, label2, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        label2 = Image.fromarray(np.array(label2))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            label2 = label2.resize((nw, nh), Image.NEAREST)
            new_label2 = Image.new('L', [w, h], (0))
            new_label2.paste(label2, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label, new_label2

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label2 = label2.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_label2 = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_label2.paste(label2, (dx, dy))
        image = new_image
        label = new_label
        label2 = new_label2

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label, label2


# DataLoader中collate_fn使用
def unet_dataset_collatefull(batch):
    fd_images = []
    fd_pngs = []
    fd_pngs_2 = []
    fd_seg_labels = []
    pd_images = []
    pd_pngs = []
    pd_pngs_2 = []
    pd_seg_labels = []
    fds=[]
    
    for img, png, png_2, labels, fd in batch:

       
        if fd:
            fd_images.append(img)
            fd_pngs.append(png)
            fd_pngs_2.append(png_2)
            fd_seg_labels.append(labels)
        else:
            pd_images.append(img)
            pd_pngs.append(png)
            pd_pngs_2.append(png_2)
            pd_seg_labels.append(labels)

        fds.append(fd)

    fd_images = torch.from_numpy(np.array(fd_images)).type(torch.FloatTensor)
  
    fd_pngs = torch.from_numpy(np.array(fd_pngs)).long()
    fd_pngs_2 = torch.from_numpy(np.array(fd_pngs_2)).long()
    fd_seg_labels = torch.from_numpy(np.array(fd_seg_labels)).type(torch.FloatTensor)
    
    pd_images = torch.from_numpy(np.array(pd_images)).type(torch.FloatTensor)
    pd_pngs = torch.from_numpy(np.array(pd_pngs)).long()
    pd_pngs_2 = torch.from_numpy(np.array(pd_pngs_2)).long()
    pd_seg_labels = torch.from_numpy(np.array(pd_seg_labels)).type(torch.FloatTensor)

    
    
    return fd_images, fd_pngs, fd_pngs_2, fd_seg_labels, pd_images, pd_pngs, pd_pngs_2, pd_seg_labels,fds
