import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import os
import torch
from nets.unet_training import CE_Loss,CE_Loss2, Dice_loss, Focal_Loss, clss_loss,softmax_mse_loss,entropy_loss,cfca_loss
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from torch.autograd import Variable
import torch
torch.autograd.set_detect_anomaly(True)
import torchvision
import torchvision.transforms as transforms
global_step=0
def replace_and_save_labels(labels, filename):
    # 创建一个新的张量用于保存替换后的标签
    replaced_labels = labels.clone()

    # 替换像素值
    replaced_labels[labels == 1] = 105
    replaced_labels[labels == 2] = 190
    replaced_labels[labels == 3] = 255

    # 将标签转换为3通道的RGB图像

     # 将标签转换为0-1范围的浮点数
    replaced_labels = replaced_labels.float() / 255.0

    # 将标签转换为3通道的RGB图像
    replaced_labels = replaced_labels.unsqueeze(0)
    #replaced_labels = replaced_labels.unsqueeze(1).repeat(1, 1, 1, 1)
    torchvision.utils.save_image(replaced_labels, filename)
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))   
def get_current_consistency_weight(epoch):
   return 0.1* sigmoid_rampup(epoch,300)

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
    

    return F.mse_loss(input_softmax, target_softmax, reduction='mean') / num_classes

def semi_supervised_batch_loader(fully_labeled_loader, partially_labeled_loader):
                      for fully_labeled_batch, partially_labeled_batch in zip(fully_labeled_loader, partially_labeled_loader):
                          combined_data = torch.cat((fully_labeled_batch[0], partially_labeled_batch[0]), dim=0)
                          combined_labels = torch.cat((fully_labeled_batch[1], partially_labeled_batch[1]), dim=0)  # 使用0填充没有标签的数据
                          yield combined_data, combined_labels

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

last_iou = 0
def fit_one_epoch_mt(iter_num,fully_labeled_batch_size,model_train, model, teacher_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights,cls_weights2, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, ema_decay=0.99):
    global last_iou
    global global_step
    total_loss = 0
    total_L3 = 0
    total_L2 = 0
    total_L1 = 0
    total_f_score = 0
    consistency_weighttotal=0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    teacher_model.eval()
     


    #gen = semi_supervised_batch_loader(fully_labeled_loader, partially_labeled_loader)

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        #fd_images, fd_pngs, fd_pngs_2, fd_seg_labels, pd_images, pd_pngs, pd_pngs_2, pd_seg_labels
        #imgs,pngs,labels,fd_seg_labels,imgs_unlabeled,labels_unlabeled, pngs_unlabeled,pd_seg_labels,fds = batch
        
        imgs,pngs,pngs2 = batch

        target_var = torch.autograd.Variable(pngs.cuda())

        minibatch_size = len(target_var)

 

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            weights2 = torch.from_numpy(cls_weights2)

            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                pngs2 = pngs2.cuda(local_rank)
                weights = weights.cuda(local_rank)
                weights2 = weights2.cuda(local_rank)

        optimizer.zero_grad()

        fully_labeled_data = imgs[:fully_labeled_batch_size]
        
        fully_labeled_labels = pngs[:fully_labeled_batch_size]
 
        point_labeled_data = imgs[fully_labeled_batch_size:]
        
        point_labeled_labels = pngs[fully_labeled_batch_size:]

        supeipixel_labels = pngs2[fully_labeled_batch_size:]

        noise = torch.clamp(torch.randn_like(point_labeled_data) * 0.1, -0.2, 0.2)
        ema_inputs = point_labeled_data + noise
        

        # replace_and_save_labels(fully_labeled_labels[0], f'fully_labeled_labels_epoch{epoch}_batch{iteration}.png')
        # replace_and_save_labels(point_labeled_labels[0], f'point_labeled_labels_epoch{epoch}_batch{iteration}.png')
        
        if not fp16:
          L1 = L2 = L3 = torch.tensor(0.0).cuda(local_rank)  # 确保 L1, L2, L3 是张量

            # Forward pass for labeled data
          
          outputs,outputs2,sup4,sup3,sup2,k,v= model_train(fully_labeled_data)
            # L1: Cross-entropy loss for labeled data

      

        #   input_tensor_4d = fully_labeled_labels.unsqueeze(1)  # 添加一个 channel 维度 -> [8, 1, 64, 64]
        #   labels4 = F.interpolate(input_tensor_4d.float(), size=sup4.shape[2:], mode='nearest')
        #   labels3 = F.interpolate(input_tensor_4d.float(), size=sup3.shape[2:], mode='nearest')
        #   labels2 = F.interpolate(input_tensor_4d.float(), size=sup2.shape[2:], mode='nearest')
           
        #  # Step 3: 可选，还原为 [batch, target_height, target_width]
        #   labels4 = labels4.long().squeeze(1)  # 移除 channel 维度 -> [8, 128, 128]
        #   labels3 = labels3.long().squeeze(1) 
        #   labels2 = labels2.long().squeeze(1) 
        #  F.adaptive_max_pool2d(labels, output_size=(64, 64))
        #   labels4 =F.adaptive_max_pool2d(fully_labeled_labels.float(), output_size=sup4.shape[2:])
        #   labels3 = F.adaptive_max_pool2d(fully_labeled_labels.float(), output_size=sup3.shape[2:])
        #   labels2 = F.adaptive_max_pool2d(fully_labeled_labels.float(), output_size=sup2.shape[2:])

        #   labels4 = labels4.long().squeeze(1)  # 移除 channel 维度 -> [8, 128, 128]
        #   labels3 = labels3.long().squeeze(1) 
        #   labels2 = labels2.long().squeeze(1) 

         
          if focal_loss:
                L1 = Focal_Loss(outputs,fully_labeled_labels, weights2, num_classes=num_classes)
          else:
                L1 = CE_Loss(outputs,fully_labeled_labels, weights2, num_classes=num_classes)
                # + 0.2*CE_Loss(sup4, labels4, weights2, num_classes=num_classes)+0.3*CE_Loss(sup3, labels3, weights2, num_classes=num_classes)
                # +0.4*CE_Loss(sup2, labels2, weights2, num_classes=num_classes)

          if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                L1 =L1+ main_dice
         

            # Forward pass for unlabeled data
         
          outputs_unlabeled_student,outputs_unlabeled_student2,sup4,sup3,sup2,q,v_seg= model_train(point_labeled_data)
          outputs_soft2 = torch.softmax(outputs_unlabeled_student2, dim=1)
          ema_input_var = ema_inputs
          with torch.no_grad():      
             outputs_unlabeled_teacher, _,sup4,sup3,sup2,q2,vseg2= teacher_model(ema_input_var)
             ema_output_soft = torch.softmax(outputs_unlabeled_teacher, dim=1)
           
           
          labels4 =F.adaptive_max_pool2d(supeipixel_labels.float(), output_size=sup4.shape[2:])
          labels3 = F.adaptive_max_pool2d(supeipixel_labels.float(), output_size=sup3.shape[2:])
          labels2 = F.adaptive_max_pool2d(supeipixel_labels.float(), output_size=sup2.shape[2:])

          labels4 = labels4.long().squeeze(1)  # 移除 channel 维度 -> [8, 128, 128]
          labels3 = labels3.long().squeeze(1) 
          labels2 = labels2.long().squeeze(1) 
          

        #   CFCA
          attn = torch.matmul(q, k.T) 
          attn = F.softmax(attn, dim=-1)  # SoftMax归一化
          v_p = torch.matmul(attn, v)  # 得到参考向量 v_p


          if focal_loss:
                L2 = Focal_Loss(outputs_unlabeled_student, point_labeled_labels, weights, num_classes=num_classes)
          else:
                L2 = CE_Loss(outputs_unlabeled_student, point_labeled_labels, weights, num_classes=num_classes)
                
                # +0.1*cfca_loss(v_p, v_seg)
                # +0.2*CE_Loss(sup4, labels4, weights, num_classes=num_classes)
                # +0.3*CE_Loss(sup3, labels3, weights2, num_classes=num_classes)
                # +0.4*CE_Loss(sup2, labels2, weights2, num_classes=num_classes)

          if dice_loss:
                main_dice_unlabeled = Dice_loss(outputs_unlabeled_student, point_labeled_labels)
                L2 = L2+ main_dice_unlabeled
            
          #outputs_unlabeled_teacher = Variable(outputs_unlabeled_teacher.detach().data, requires_grad=False)
            # L3: MSE loss between student and teacher outputs for unlabeled data

           
          consistency_weight = get_current_consistency_weight(epoch)
           
          consistency_loss = softmax_mse_loss(outputs_unlabeled_student2, outputs_unlabeled_teacher) 
           
          
        

          L3 =  consistency_weight * (consistency_loss)

         
        # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_

          

          #L2 =  consistency_weight * (consistency_loss + args.weak_weight * weak_supervised_loss)
          #L3 =F.mse_loss(outputs_unlabeled_student2, outputs_unlabeled_teacher)
            # out,out2=outputs_unlabeled_student,outputs_unlabeled_student2

            # res_loss = 0.5* softmax_mse_loss(out,out2)

            # Total loss
          if iteration>=100:
            loss = L1 + L3+L2
          else:
            loss = L1+L2
        #   if imgs_unlabeled.numel() > 0:
           

        #     with torch.no_grad():
        #         _f_score = f_score(outputs_unlabeled_teacher, pngs_unlabeled)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          update_ema_variables(model_train, teacher_model, 0.999, global_step)
          global_step+=1
          #update_ema_variables(model, teacher_model, 0.99, 1000)

        #   loss.backward(retain_graph=True)
        #   optimizer.step()
        else:
            with autocast():
                outputs, cls_label = model_train(imgs)
                if focal_loss:
                    L1 = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    L1 = CE_Loss(outputs, pngs, weights, num_classes=num_classes) + clss_loss(cls_label, pngs)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    L1 += main_dice

                outputs_unlabeled_student, _ = model_train(imgs_unlabeled)
                outputs_unlabeled_teacher, _ = teacher_model(imgs_unlabeled)

                if focal_loss:
                    L2 = Focal_Loss(outputs_unlabeled_student, pngs_unlabeled, weights, num_classes=num_classes)
                else:
                    L2 = CE_Loss(outputs_unlabeled_student, pngs_unlabeled, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice_unlabeled = Dice_loss(outputs_unlabeled_student, labels_unlabeled)
                    L2 += main_dice_unlabeled

                L3 = F.mse_loss(outputs_unlabeled_student, outputs_unlabeled_teacher)
               

                loss = L1 + L2 + L3

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        #total_f_score += _f_score.item()
        total_L3 += L3.item()
        total_L2 += L2.item()
        total_L1 += L1.item()
        consistency_weighttotal += consistency_weight

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'total_l1': total_L1 / (iteration + 1),
                                'total_l2': total_L2 / (iteration + 1),
                                'total_l3': total_L3 / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

        # Update teacher model with EMA
        with torch.no_grad():
            for param, param_t in zip(model_train.parameters(), teacher_model.parameters()):
                param_t.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        # imgs,pngs,labels,fd_seg_labels,imgs_unlabeled,labels_unlabeled, pngs_unlabeled,pd_seg_labels,fds = batch

      
      
        

        # with torch.no_grad():
        #     weights = torch.from_numpy(cls_weights)
        #     if cuda:
        #         imgs = imgs.cuda(local_rank)
        #         pngs = pngs.cuda(local_rank)
        #         labels = labels.cuda(local_rank)
        #         imgs_unlabeled = imgs_unlabeled.cuda(local_rank)
        #         #pngs_unlabeled = pngs_unlabeled.cuda(local_rank)
        #         labels_unlabeled = labels_unlabeled.cuda(local_rank)
        #         pngs_unlabeled=labels_unlabeled.cuda(local_rank)
        #         weights = weights.cuda(local_rank)

        imgs,pngs,pngs2, labels = batch
        target_var = torch.autograd.Variable(pngs.cuda())

        minibatch_size = len(target_var)
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs, _,_,_,_,_,_ = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes) /minibatch_size

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss += main_dice

            #_f_score = f_score(outputs, labels)

            val_loss += loss.item()
            #val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        new_iou = eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        if (new_iou>last_iou):
            print('Save best model to best_epoch_weights.pth')
            last_iou =new_iou
            print(last_iou)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))