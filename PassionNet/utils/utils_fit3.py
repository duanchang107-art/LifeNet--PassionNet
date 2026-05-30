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
from nets.unet_training import CE_Loss,CE_Loss2, Dice_loss, Focal_Loss, clss_loss,softmax_mse_loss
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
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def fit_one_epoch_mt(model_train, model, teacher_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights,cls_weights2, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, ema_decay=0.99):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    teacher_model.eval()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        #fd_images, fd_pngs, fd_pngs_2, fd_seg_labels, pd_images, pd_pngs, pd_pngs_2, pd_seg_labels
        #imgs,pngs,labels,fd_seg_labels,imgs_unlabeled,labels_unlabeled, pngs_unlabeled,pd_seg_labels,fds = batch
        imgs,pngs = batch
 

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            weights2 = torch.from_numpy(cls_weights2)

            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                # labels = labels.cuda(local_rank)
                # imgs_unlabeled = imgs_unlabeled.cuda(local_rank)
                # #pngs_unlabeled = pngs_unlabeled.cuda(local_rank)
                # labels_unlabeled = labels_unlabeled.cuda(local_rank)
                # pngs_unlabeled=labels_unlabeled.cuda(local_rank)
                weights = weights.cuda(local_rank)
                weights2 = weights2.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
          L1 = L2 = L3 = torch.tensor(0.0).cuda(local_rank)  # 确保 L1, L2, L3 是张量

            # Forward pass for labeled data
          if imgs.numel() > 0:
            outputs, outputs2 = model_train(imgs)
            # L1: Cross-entropy loss for labeled data
           
            if focal_loss:
                L1 = Focal_Loss(outputs, pngs, weights2, num_classes=num_classes)
            else:
                L1 = CE_Loss(outputs, pngs, weights2, num_classes=num_classes) 

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                L1 =L1+ main_dice
         

            # Forward pass for unlabeled data
          if imgs_unlabeled.numel() > 0:
            
            outputs_unlabeled_student,outputs_unlabeled_student2 = model_train(imgs_unlabeled)
            with torch.no_grad():
                   ema_input_var = imgs_unlabeled
            outputs_unlabeled_teacher, _ = teacher_model(ema_input_var)

            # L2: Weighted cross-entropy loss for unlabeled data
            if focal_loss:
                L2 = Focal_Loss(outputs_unlabeled_student, pngs_unlabeled, weights, num_classes=num_classes)
            else:
                L2 = CE_Loss(outputs_unlabeled_student, pngs_unlabeled, weights, num_classes=num_classes)

            if dice_loss:
                main_dice_unlabeled = Dice_loss(outputs_unlabeled_student, labels_unlabeled)
                L2 = L2+ main_dice_unlabeled
            
            outputs_unlabeled_teacher = Variable(outputs_unlabeled_teacher.detach().data, requires_grad=False)
            # L3: MSE loss between student and teacher outputs for unlabeled data
            L3 =F.mse_loss(outputs_unlabeled_student2, outputs_unlabeled_teacher)
            # out,out2=outputs_unlabeled_student,outputs_unlabeled_student2

            # res_loss = 0.5* softmax_mse_loss(out,out2)

            # Total loss
          loss = L1 + L2 + L3
        #   if imgs_unlabeled.numel() > 0:
           

        #     with torch.no_grad():
        #         _f_score = f_score(outputs_unlabeled_teacher, pngs_unlabeled)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
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

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                
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
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs, cls_label = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes) 

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
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # Save weights
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
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