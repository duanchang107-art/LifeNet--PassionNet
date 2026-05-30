import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F

import torch

import torch

import torch


def cfca_loss(v_p, v_seg):
    """
    计算交叉域交叉注意力相似性损失 (Loss_CFCA)

    参数：
    - v_p: [B×H/16×W/16, 256] 的参考向量（由互注意力计算得到）
    - v_seg: [B×H/16×W/16, 256] 的弱监督分割特征编码向量

    返回：
    - loss_cfca: 标量，CFCA 损失
    """
    # 计算向量的 L2 范数
    v_p_norm = torch.norm(v_p, p=2, dim=1, keepdim=True)  # [B×H/16×W/16, 1]
    v_seg_norm = torch.norm(v_seg, p=2, dim=1, keepdim=True)  # [B×H/16×W/16, 1]

    # 归一化向量
    v_p_normalized = v_p / (v_p_norm + 1e-8)  # 避免除以 0
    v_seg_normalized = v_seg / (v_seg_norm + 1e-8)

    # 计算余弦相似度
    cosine_similarity = torch.sum(v_p_normalized * v_seg_normalized, dim=1)  # [B×H/16×W/16]

    # 计算损失
    loss_cfca = 1 - torch.mean(cosine_similarity)  # 余弦相似度的平均值取反

    return loss_cfca


def Active_Contour_Loss(y_true, y_pred): 
    """
    length term
    """

    # 将多通道的预测转换为单通道类别标签（每个像素取最大概率的类别）
    y_pred_class = torch.argmax(y_pred, dim=1)  # 得到每个像素的类别标签

    # 获取 y_true 和 y_pred_class 的设备（GPU 或 CPU）
    device = y_true.device  # 获取 y_true 所在的设备

    # horizontal and vertical directions 
    x = y_pred_class[:,1:,:] - y_pred_class[:,:-1,:]
    y = y_pred_class[:,:,1:] - y_pred_class[:,:,:-1]

    delta_x = x[:,1:,:-2]**2
    delta_y = y[:,:-2,1:]**2
    delta_u = torch.abs(delta_x + delta_y)  # Use torch.abs instead of K.abs

    # length term: compute mean over the entire batch
    lenth = torch.mean(torch.sqrt(delta_u + 1e-8))  # Use torch.sqrt instead of K.sqrt

    """
    region term
    """
    # 确保 C_1 和 C_2 张量在相同设备上
    C_1 = torch.zeros((y_true.shape[1], y_true.shape[2]), dtype=torch.float32, device=device)  # 在相同设备上创建张量
    C_2 = torch.zeros((y_true.shape[1], y_true.shape[2]), dtype=torch.float32, device=device)

    region_in = torch.abs(torch.mean( y_pred_class * ((y_true - C_1)**2) ) )  # 直接使用y_true，因为它是3维的
    region_out = torch.abs(torch.mean( (1 - y_pred_class) * ((y_true - C_2)**2) ))  # 同上

    # lambdaP and mu are adjustable hyperparameters
    lambdaP = 1
    mu = 1
    
    return lenth + lambdaP * (mu * region_in + region_out)




def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def target_seg2target_cls(labels_batch):
    # labels_batch 是一个形状为 (batch_size, height, width) 的张量
    

    batch_size, _, _= labels_batch.shape

    cls_labels_batch = np.zeros(shape=(batch_size, 3), dtype=float)

    for batch_idx in range(batch_size):
        label_set = np.unique(labels_batch[batch_idx])
        

        for i in label_set:
            if i != 0:
                cls_labels_batch[batch_idx, i - 1] += 1

    cls_labels_batch = torch.from_numpy(cls_labels_batch).float()

    return cls_labels_batch
def clss_loss(class_label, target):
    #inputs_cls = class_label.detach().cpu().numpy()
    target_seg = target.cpu().numpy()
  

    
    target_cls = target_seg2target_cls(target_seg).cuda()
    loss_fn = F.multilabel_soft_margin_loss
   
    classloss=loss_fn(class_label,target_cls)
    return classloss

def CE_Loss2(inputs, target, cls_weights, U, num_classes=21):
    device = inputs.device
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # 插值输入以匹配目标的大小
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 重塑输入和目标以适应交叉熵损失
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # 加载像素权重
    U_final = U.cpu().numpy().astype(np.float32) / 255.0

    # 归一化 U_final 到 0-1 范围
    U_normalized = (U_final - U_final.min()) / (U_final.max() - U_final.min())
    pixel_weights = U_normalized

    # 确保像素权重的形状与目标对齐
    if pixel_weights.ndim == 2:
        pixel_weights = pixel_weights.flatten()
    elif pixel_weights.ndim == 3:
        pixel_weights = pixel_weights.transpose(0, 2, 1).reshape(-1)

    # 创建带有类权重和像素权重的交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')
    ce_loss = loss_fn(temp_inputs, temp_target)

    # 应用像素权重
    weighted_loss = ce_loss * torch.tensor(pixel_weights, dtype=torch.float32).to(device)

    # 计算最终的加权损失
    final_loss = weighted_loss.mean()
    print(final_loss)

    return final_loss
def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

   

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)

    return CE_loss

def entropy_loss(inputs, target, cls_weights, num_classes=21, lambda_ent=0.5):
    """
    计算交叉熵损失，并加入 Shannon 熵最小化损失。
    
    参数:
    - inputs: 网络的输出，形状为 (n, c, h, w)
    - target: 标签，形状为 (n, h, w)
    - cls_weights: 类别权重，用于加权交叉熵损失
    - num_classes: 类别总数
    - lambda_ent: Shannon 熵最小化损失的权重

    返回:
    - 联合损失 (交叉熵损失 + 熵最小化损失)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # 如果预测尺寸和标签尺寸不同，则插值匹配
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 将预测和标签展平
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # 计算交叉熵损失
    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)

    # 计算 Shannon 熵最小化损失
    softmax_outputs = F.softmax(inputs, dim=1)  # 对预测值进行 softmax
    log_softmax_outputs = F.log_softmax(inputs, dim=1)  # 对预测值进行 log softmax

    # 熵计算：H(p) = -p * log(p)
    entropy = -softmax_outputs * log_softmax_outputs
    entropy = entropy.sum(dim=1)  # 在类别维度上求和
    entropy_loss = entropy.mean()  # 对所有像素取平均值

    # 联合损失
    total_loss = lambda_ent * entropy_loss

    return total_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
