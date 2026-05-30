import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_U(U, save_path=None):
    # 确保 U 是一个 NumPy 数组
    if isinstance(U, torch.Tensor):
        U = U.cpu().detach().numpy()

    # 显示 U
    plt.figure(figsize=(10, 10))
    plt.imshow(U, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('U visualization')
    plt.savefig('2.png')

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 从文件中读取图像
        jpg = Image.open(os.path.join(self.dataset_path, "image", name + ".png"))
        png = Image.open(os.path.join(self.dataset_path, "point_label", name + ".png"))
        png_2 = Image.open(os.path.join(self.dataset_path, "superpixel", name + ".png"))
        U = np.load(os.path.join(self.dataset_path, "npy", name + ".npy"))

        # 数据增强
        jpg, png, png_2, U = self.get_random_data(jpg, png, png_2, U, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        png_2 = np.array(png_2)
        png_2[png_2 >= self.num_classes] = self.num_classes
        U=np.array(U)


        # 转化成one_hot的形式
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 应用权重到seg_labels
        weighted_labels = seg_labels * U[..., np.newaxis]

        return jpg, png, png_2, weighted_labels,U

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, label2, U, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        label2 = Image.fromarray(np.array(label2))
        U= (U * 255).astype(np.uint8)
        U=Image.fromarray(np.array(U))

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

            U = Image.resize((nw, nh), Image.NEAREST)
            new_U = Image.new('L', [w, h], (0))
            new_U.paste(U, ((w - nw) // 2, (h - nh) // 2))
            U = np.array(new_U)

            return new_image, new_label, new_label2, U

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
        U = U.resize((nw, nh), Image.NEAREST)


        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)
            U = U.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_label2 = Image.new('L', (w, h), (0))
        new_U = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_label2.paste(label2, (dx, dy))
        new_U.paste(U, (dx, dy))

        image = new_image
        label = new_label
        label2 = new_label2
        U = np.array(new_U)


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

        return image_data, label, label2, U


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    pngs_2 = []
    seg_labels = []
    Us=[]
    for img, png, png_2, labels,U in batch:
        images.append(img)
        pngs.append(png)
        pngs_2.append(png_2)
        seg_labels.append(labels)
        Us.append(U)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    pngs_2 = torch.from_numpy(np.array(pngs_2)).long()
    Us = torch.from_numpy(np.array(Us)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, pngs_2, seg_labels,Us
