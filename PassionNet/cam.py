import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from nets.utils import GradCAM, show_cam_on_image, center_crop_img
from nets.unet import Unet
from nets.vgg import VGG16


def main():

    model = Unet()
    model.load_state_dict(torch.load('/data1/dsy/WSY/unet-pytorch-main7/logs/best_epoch_weights.pth', map_location='cpu'))
    model.eval()
    # # model = models.mobilenet_v3_large(pretrained=True)
    target_layers = [model.features]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((128, 128)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = '/data1/dsy/WSY/pointretouch/image/1020_051.png'
    roi_path = '/data1/dsy/WSY/pointretouch/fluid/1020_051.png'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    roi = Image.open(roi_path).convert('RGB')
    roi = np.array(roi, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    roi_tensor = data_transform(roi)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    roi_tensor = torch.unsqueeze(roi_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 1 # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    print(img.shape)
    
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()