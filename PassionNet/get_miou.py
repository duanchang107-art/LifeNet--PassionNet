import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from nets.DBFnet import BFnet
from utils.utils_metrics import compute_mIoU, show_results
import matplotlib.pyplot as plt
import numpy as np
import cv2
def visualize_feature_maps(feature_maps,savename):
    """
    可视化特征图。这个函数假设feature_maps是一个形状为
    [C, H, W]的张量，其中C是通道数，H和W是特征图的高度和宽度。

    """
    fig = plt.figure(figsize=(16, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    n_channels = feature_maps.size(0)
    
    # 计算要在每行显示的图像数，以及总行数
    n_cols = np.ceil(np.sqrt(n_channels)).astype(int)
    n_rows = np.ceil(n_channels / n_cols).astype(int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n_channels:
            # 取第i个通道的特征图并转换为numpy数组
            feature_map = feature_maps[i].cpu().detach().numpy()
            # 显示特征图
            ax.imshow(feature_map, cmap='Blues')
            ax.axis('off')
        else:
            ax.axis('off')
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    plt.show()

def draw_features(width, height, x, savename):
   
    fig = plt.figure(figsize=(16, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        #x = np.array(x)
        print(x.shape)
       
        img = x[0, i, :, :].detach()
        img = img.cpu().numpy()  # 假设 img 已经是一个不需要梯度的张量
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 4
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","PED","SRF","IRF"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = "/data1/dsy/WSY/pointretouch8/"

    image_ids       = open(os.path.join(VOCdevkit_path, "3.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "crop_mk2")
    miou_out_path   = "miou_out0.5-0.3-25"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    feat_dir='/data1/dsy/WSY/unet-pytorch-main4/img/feat1'
    feat_dir2 = '/data1/dsy/WSY/unet-pytorch-main4/img/feat1roi'

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        #unet = BFnet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "crop_image",image_id+".png")
            image       = Image.open(image_path)
            image     = unet.get_miou_png(image)
            
            #feat1 = unet.get_miou_png(image)[1]

            #visualize_feature_maps(feat1.squeeze(0),os.path.join(feat_dir,image_id+".png"))
           # visualize_feature_maps(roifeat1.squeeze(0),os.path.join(feat_dir2,image_id+".png"))
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist,dice, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)