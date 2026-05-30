import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    # print(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n) )
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 



def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def per_class_dice(hist):
    # Calculate true positives, false negatives, and false positives
    true_positives = np.diag(hist)
    false_negatives = np.sum(hist, axis=1) - true_positives
    false_positives = np.sum(hist, axis=0) - true_positives

    # Calculate Dice coefficient for each class
    dice = 2 * true_positives / np.maximum((2 * true_positives + false_negatives + false_positives),1)
    return dice

def per_class_sen(hist):
    # Calculate true positives, false negatives, and false positives
    true_positives = np.diag(hist)
    false_negatives = np.sum(hist, axis=1) - true_positives
    false_positives = np.sum(hist, axis=0) - true_positives

    # Calculate Dice coefficient for each class
    sen = true_positives / np.maximum((true_positives + false_negatives),1)
    return sen
def per_class_spe(hist):
    # Calculate total elements in confusion matrix
    total_elements = np.sum(hist)
    
    # True Positives (TP) for each class
    true_positives = np.diag(hist)
    # False Negatives (FN) for each class
    false_negatives = np.sum(hist, axis=1) - true_positives
    # False Positives (FP) for each class
    false_positives = np.sum(hist, axis=0) - true_positives
    # True Negatives (TN) for each class
    true_negatives = total_elements - (true_positives + false_negatives + false_positives)
    
    # Specificity calculation
    spe = true_negatives / np.maximum((true_negatives + false_positives),1)
    return spe



# def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
#     print('Num classes', num_classes)  
#     #-----------------------------------------#
#     #   创建一个全是0的矩阵，是一个混淆矩阵
#     #-----------------------------------------#
#     hist = np.zeros((num_classes, num_classes))
    
#     #------------------------------------------------#
#     #   获得验证集标签路径列表，方便直接读取
#     #   获得验证集图像分割结果路径列表，方便直接读取
#     #------------------------------------------------#
#     gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
#     pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

#     #------------------------------------------------#
#     #   读取每一个（图片-标签）对
#     #------------------------------------------------#
#     for ind in range(len(gt_imgs)): 
#         #------------------------------------------------#
#         #   读取一张图像分割结果，转化成numpy数组
#         #------------------------------------------------#
#         pred = np.array(Image.open(pred_imgs[ind]))  
#         #------------------------------------------------#
#         #   读取一张对应的标签，转化成numpy数组
#         #------------------------------------------------#
#         label = np.array(Image.open(gt_imgs[ind]).convert('L')) 
        
#         #------------------------------------------------#
#         #   过滤掉多余的类别，将它们标记为 -1（错误分类）
#         #------------------------------------------------#
#         valid_classes = np.unique(label)  # 标签中的实际类别
#         pred_filtered = pred.copy()
#         pred_filtered[~np.isin(pred, valid_classes)] = 0  # 设置多余类别为 -1（错误分类）

#         # if not np.array_equal(np.unique(pred_filtered), np.unique(label)):
#         #     print(f"Different unique elements found: pred = {np.unique(pred_filtered)}, label = {np.unique(label)}") 

#         # 如果图像分割结果与标签的大小不一样，这张图片就不计算
#         if len(label.flatten()) != len(pred_filtered.flatten()):  
#             print(
#                 'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
#                     len(label.flatten()), len(pred_filtered.flatten()), gt_imgs[ind],
#                     pred_imgs[ind]))
#             continue

#         #------------------------------------------------#
#         #   对一张图片计算 hist 矩阵，并累加
#         #------------------------------------------------#
#         hist += fast_hist(label.flatten(), pred_filtered.flatten(), num_classes) 
#         # 检查标签和预测的最大值和最小值
       

#         # 每计算10张就输出一下目前已计算的图片中所有类别平均的 mIoU 值
#         if name_classes is not None and ind > 0 and ind % 10 == 0: 
#             print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
#                     ind, 
#                     len(gt_imgs),
#                     100 * np.nanmean(per_class_dice(hist)),
#                     100 * np.nanmean(per_class_iu(hist)),
#                     100 * np.nanmean(per_class_PA_Recall(hist)),
#                     100 * per_Accuracy(hist)
#                 )
#             )
#     #------------------------------------------------#
#     #   计算所有验证集图片的逐类别 mIoU 值
#     #------------------------------------------------#
#     print(hist)
#     dice = per_class_dice(hist)
#     IoUs        = per_class_iu(hist)
#     Spes        = per_class_spe(hist)
#     Sens        = per_class_sen(hist)
#     PA_Recall   = per_class_PA_Recall(hist)
#     Precision   = per_class_Precision(hist)
#     #------------------------------------------------#
#     #   逐类别输出一下 mIoU 值
#     #------------------------------------------------#
#     if name_classes is not None:
#         for ind_class in range(num_classes):
#             print('===>' + name_classes[ind_class] +':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
#                 + '; sen-' + str(round(Sens[ind_class] * 100, 2))+'; dice-' + str(round(dice[ind_class] * 100, 2))+ '; spe-' + str(round(Spes[ind_class] * 100, 2))+'; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

#     #-----------------------------------------------------------------#
#     #   在所有验证集图像上求所有类别平均的 mIoU 值，计算时忽略 NaN 值
#     #-----------------------------------------------------------------#
#     print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +'; dice: '+str(round(np.nanmean(dice) * 100, 2))+ '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
#     return np.array(hist, np.int), dice, IoUs, PA_Recall, Precision
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，记录全局的结果
    #-----------------------------------------#
    total_dice = np.zeros(num_classes)
    total_IoUs = np.zeros(num_classes)
    total_Spes = np.zeros(num_classes)
    total_Sens = np.zeros(num_classes)
    total_PA_Recall = np.zeros(num_classes)
    total_Precision = np.zeros(num_classes)
    image_count = np.zeros(num_classes)
    total_hist = np.zeros((num_classes, num_classes))

    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]).convert('L')) 
        
        #------------------------------------------------#
        #   过滤掉多余的类别，将它们标记为 0（错误分类）
        #------------------------------------------------#
        valid_classes = np.unique(label)  # 标签中的实际类别
        pred_filtered = pred.copy()
        pred_filtered[~np.isin(pred, valid_classes)] =0

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred_filtered.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred_filtered.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对当前图片计算 hist 矩阵
        #------------------------------------------------#
        hist = fast_hist(label.flatten(), pred_filtered.flatten(), num_classes)
        # total_hist +=fast_hist(label.flatten(), pred_filtered.flatten(), num_classes)

        #------------------------------------------------#
        #   单独计算当前图像的各项指标
        #------------------------------------------------#
        dice = per_class_dice(hist)
        IoUs = per_class_iu(hist)
        Spes = per_class_spe(hist)
        Sens = per_class_sen(hist)
        PA_Recall = per_class_PA_Recall(hist)
        Precision = per_class_Precision(hist)
        
        #------------------------------------------------#
        #   若某类别在实际标签中存在但预测中不存在，则将该类别指标设为 0
        #------------------------------------------------#
        for cls in range(num_classes):
            if cls in valid_classes:  # 标签中存在该类别
                if cls not in pred_filtered:  # 预测中不存在该类别
                    dice[cls] = 0
                    IoUs[cls] = 0
                    Spes[cls] = 0
                    Sens[cls] = 0
                    PA_Recall[cls] = 0
                    Precision[cls] = 0
                image_count[cls] +=1
            else:
                dice[cls] = 0
                IoUs[cls] = 0
                Spes[cls] = 0
                Sens[cls] = 0
                PA_Recall[cls] = 0
                Precision[cls] = 0
            
          
                
        #------------------------------------------------#
        #   累加每张图片的结果，按类别计算平均值
        #------------------------------------------------#
        
        total_hist += hist
        total_dice += dice
        total_IoUs += IoUs
        total_Spes += Spes
        total_Sens += Sens
        total_PA_Recall += PA_Recall
        total_Precision += Precision
        # image_count += (valid_classes != -1)  # 有效图片数量按类别记录
        # for cls in range(num_classes):
        #     if cls in valid_classes:  # 如果该类别在标签中存在
        #         image_count[cls] += 1

        #------------------------------------------------#
        #   打印每 10 张图片的进度
        #------------------------------------------------#
        # if name_classes is not None and ind > 0 and ind % 10 == 0: 
        #     print(f'{ind} / {len(gt_imgs)} images processed.')

    #------------------------------------------------#
    #   计算所有图片的逐类别平均指标
    #------------------------------------------------#
    
    avg_dice = total_dice / np.maximum(image_count, 1)
    avg_IoUs = total_IoUs / np.maximum(image_count, 1)
    avg_Spes = total_Spes / np.maximum(image_count, 1)
    avg_Sens = total_Sens / np.maximum(image_count, 1)
    avg_PA_Recall = total_PA_Recall /np.maximum(image_count, 1)
    avg_Precision = total_Precision / np.maximum(image_count, 1)

    #------------------------------------------------#
    #   输出逐类别结果
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] +':\tIou-' + str(round(avg_IoUs[ind_class] * 100, 2)) \
                + '; sen-' + str(round(avg_Sens[ind_class] * 100, 2))+'; dice-' + str(round(avg_dice[ind_class] * 100, 2))+ '; spe-' + str(round(avg_Spes[ind_class] * 100, 2))+'; Recall (equal to the PA)-' + str(round(avg_PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(avg_Precision[ind_class] * 100, 2)))

    #------------------------------------------------#
    #   输出整体平均结果
    #------------------------------------------------#
    print('===> Overall Results: ')
    print('mIoU: ' + str(round(((avg_IoUs[1]+avg_IoUs[2]+avg_IoUs[3])/3) * 100, 2)) + 
      '; dice: ' + str(round(((avg_dice[1]+avg_dice[2]+avg_dice[3])/3) * 100, 2)) + 
      '; sen: ' + str(round(((avg_Sens[1]+avg_Sens[2]+avg_Sens[3])/3) * 100, 2)) + 
      '; spe: ' + str(round(((avg_Spes[1]+avg_Spes[2]+avg_Spes[3])/3) * 100, 2)))
    return total_hist, avg_dice, avg_IoUs,  avg_PA_Recall, avg_Precision


def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))